"""Synthesis agent (component 10): research refs → ranked, budget-respecting EvidenceBundle.

The only agent allowed to read raw source content, and it still never holds the corpus
at once:

1. resolve   every kept ref through SourceResolver (web pages and PDF chunks);
2. select    within each source, the passages most similar to the sub-task query
             (local embeddings), so a 5k-token article contributes ~3 paragraphs;
3. extract   one small model call per sub-task turns those passages into atomic claims,
             each tied to exactly one passage (and so one source);
4. dedupe    same entity + near-identical claim (cosine) → one item; the highest-credibility
             source becomes primary, the others count as corroboration;
5. score     confidence: conflict → LOW; 2+ independent sources → HIGH; one source with
             credibility ≥ threshold → MEDIUM; else LOW;
6. rank      recency (month) → credibility → relevance to the brief;
7. compress  while the rendered bundle is over budget, summarize the lowest-ranked tail of the
             largest section into one item (then drop, as a last resort); every cut is logged.

Steps 4–7 are code, so the bundle's structure and confidence labels are reproducible.
"""

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from insightforge.agents.prompts import COMPRESS_PROMPT, SYNTHESIS_CLAIMS_PROMPT
from insightforge.graph.state import (
    Confidence,
    Cut,
    EntityGraph,
    EvidenceBundle,
    EvidenceItem,
    EvidenceSection,
    ResearchResult,
    SourceMeta,
    SubTaskSpec,
)
from insightforge.guardrails.untrusted import wrap
from insightforge.llm.embeddings import embed_documents, embed_query
from insightforge.llm.gateway import ModelGateway, Role, TokenBudgetExceeded
from insightforge.llm.schema import clip
from insightforge.llm.tokens import count_tokens
from insightforge.store.models import SourceKind, SubTaskStatus
from insightforge.store.run_log import RunLog
from insightforge.store.sources import ResolvedSource, SourceResolver
from insightforge.tools.credibility import registered_host
from insightforge.tools.entities import numbers_in
from insightforge.tools.lexical import shortlist

AGENT = "synthesis"
PASSAGE_MIN, PASSAGE_MAX = 300, 1_200
DEDUPE_SIMILARITY = 0.9


# --- model I/O ---------------------------------------------------------------------


class PassageClaim(BaseModel):
    passage_id: str = Field(description='The passage the claim comes from, e.g. "p-3"')
    claim: Annotated[str, clip(400)] = Field(max_length=400)


class SubTaskClaims(BaseModel):
    theme: Annotated[str, clip(80)] = Field(max_length=80, description="Short section heading for this sub-task")
    claims: Annotated[list[PassageClaim], clip(20)] = Field(default_factory=list, max_length=20)


class CompressedClaim(BaseModel):
    summary: Annotated[str, clip(600)] = Field(max_length=600)


# --- passages ------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    passage_id: str
    source_id: str
    text: str


def split_passages(text: str) -> list[str]:
    """Paragraphs merged up to PASSAGE_MIN and split down to PASSAGE_MAX characters."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    merged: list[str] = []
    for p in paras:
        if merged and len(merged[-1]) < PASSAGE_MIN:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    out = []
    for p in merged:
        while len(p) > PASSAGE_MAX:
            cut = p.rfind(". ", PASSAGE_MIN, PASSAGE_MAX)
            cut = cut + 1 if cut > 0 else PASSAGE_MAX
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


async def select_passages(sources: list[ResolvedSource], query: str, per_source: int,
                          shortlist_per_source: int = 8) -> list[Passage]:
    """The passages of each source most similar to the query (a PDF chunk is already one passage).

    Two stages: a keyword shortlist per source, then embeddings on the shortlist only.
    """
    candidates: list[tuple[str, str]] = []
    for s in sources:
        parts = [s.text] if s.kind is SourceKind.PDF_CHUNK else split_passages(s.text)
        parts = [parts[i] for i in shortlist(query, parts, shortlist_per_source)]
        candidates += [(s.source_id, p) for p in parts]
    if not candidates:
        return []
    q = np.array(await embed_query(query))
    vecs = np.array(await embed_documents([text for _, text in candidates]))
    scores = _cos(vecs, q[None, :])[:, 0]
    best: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, (sid, _) in enumerate(candidates):
        best[sid].append((float(scores[i]), i))
    chosen = sorted(i for sid in best for _, i in sorted(best[sid], reverse=True)[:per_source])
    return [Passage(f"p-{n + 1}", candidates[i][0], candidates[i][1]) for n, i in enumerate(chosen)]


# --- claims → items ------------------------------------------------------------------


@dataclass
class Candidate:
    sub_task_id: str
    claim: str
    passage: str
    source_ids: list[str]
    vector: np.ndarray | None = None
    entity_keys: list[str] = field(default_factory=list)
    relevance: float = 0.0


def independent_count(source_ids: list[str], sources: dict[str, SourceMeta]) -> int:
    """Distinct publishers: registered domain for web pages, file name for PDF chunks."""
    owners = set()
    for sid in source_ids:
        meta = sources.get(sid)
        if meta is None:
            continue
        owners.add(registered_host(meta.url) if meta.url else meta.label.split(" p.")[0])
    return len(owners)


def entity_keys_for(text: str, graph: EntityGraph | None) -> list[str]:
    if not graph:
        return []
    low = text.lower()
    return sorted({e.key for e in graph.entities
                   if len(e.name) >= 3 and re.search(rf"(?<!\w){re.escape(e.name.lower())}(?!\w)", low)})


def touches_conflict(c: Candidate, graph: EntityGraph | None) -> bool:
    """The claim is about a conflicting fact: same entity, a source in the conflict, and a matching value."""
    if not graph or not graph.conflicts:
        return False
    nums, low = numbers_in(c.claim), c.claim.lower()
    for conflict in graph.conflicts:
        if conflict.subject not in c.entity_keys:
            continue
        involved = {sid for v in conflict.values for sid in v.source_ids}
        if not involved & set(c.source_ids):
            continue
        for v in conflict.values:
            if nums & numbers_in(v.object) or v.object.lower() in low:
                return True
    return False


def confidence_for(c: Candidate, sources: dict[str, SourceMeta], conflict: bool, high_cred: float) -> Confidence:
    if conflict:
        return Confidence.LOW
    if independent_count(c.source_ids, sources) >= 2:
        return Confidence.HIGH
    cred = max((sources[s].credibility or 0.0) for s in c.source_ids if s in sources)
    return Confidence.MEDIUM if cred >= high_cred else Confidence.LOW


def section_confidence(items: list[EvidenceItem]) -> Confidence:
    if not items:
        return Confidence.LOW
    counts = {c: sum(i.confidence is c for i in items) for c in Confidence}
    if counts[Confidence.LOW] * 2 > len(items) or any(i.conflict for i in items[:3]):
        return Confidence.LOW
    if counts[Confidence.HIGH] * 2 >= len(items):
        return Confidence.HIGH
    return Confidence.MEDIUM


def dedupe(cands: list[Candidate], sources: dict[str, SourceMeta]) -> list[Candidate]:
    """Merge near-identical claims about the same entities; the most credible source leads."""
    def cred(c: Candidate) -> float:
        return max((sources[s].credibility or 0.0) for s in c.source_ids if s in sources)

    ordered = sorted(cands, key=cred, reverse=True)
    kept: list[Candidate] = []
    for c in ordered:
        for k in kept:
            same_entities = set(c.entity_keys) == set(k.entity_keys) or bool(set(c.entity_keys) & set(k.entity_keys))
            # Embeddings barely separate "costs SGD 3" from "costs SGD 5": different numbers are different facts.
            same_numbers = numbers_in(c.claim) == numbers_in(k.claim)
            if same_entities and same_numbers and float(_cos(c.vector, k.vector)) >= DEDUPE_SIMILARITY:
                k.source_ids += [s for s in c.source_ids if s not in k.source_ids]
                k.relevance = max(k.relevance, c.relevance)
                break
        else:
            kept.append(c)
    return kept


def rank_key(item: EvidenceItem) -> tuple:
    d = item.published_date
    month = (d.year, d.month) if d else (0, 0)  # unknown dates rank as oldest
    return (month, item.credibility, item.relevance)


# --- rendering + compression ------------------------------------------------------------


def render_item(item: EvidenceItem, sources: dict[str, SourceMeta]) -> str:
    labels = ", ".join(dict.fromkeys(sources[s].label for s in item.source_ids if s in sources))
    date = item.published_date.isoformat() if item.published_date else "n.d."
    flags = " | CONFLICTING REPORTS" if item.conflict else ""
    flags += " | summary of lower-priority items" if item.compressed else ""
    return (f"{item.evidence_id} | {item.confidence.value} | {date} | cred {item.credibility:.2f} | "
            f"{labels}{flags}\n  {item.claim}")


def render_bundle(sections: list[EvidenceSection], sources: dict[str, SourceMeta]) -> str:
    """Exactly what the Writer sees; the token budget is measured on this text."""
    blocks = []
    for s in sections:
        lines = [f"## {s.theme} (sub-task {s.sub_task_id}, section confidence {s.confidence.value})"]
        lines += [render_item(i, sources) for i in s.evidence]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@dataclass
class SynthesisResult:
    bundle: EvidenceBundle
    corpus_tokens: int  # raw tokens of every resolved source (for the context-engineering writeup)
    passage_tokens: int  # tokens of the passages actually sent to the claim-extraction model


class Synthesizer:
    def __init__(
        self,
        gateway: ModelGateway,
        resolver: SourceResolver,
        log: RunLog,
        *,
        token_budget: int = 6_000,
        passages_per_source: int = 3,
        high_credibility: float = 0.6,
    ):
        self.gateway = gateway
        self.resolver = resolver
        self.log = log
        self.token_budget = token_budget
        self.passages_per_source = passages_per_source
        self.high_credibility = high_credibility

    async def run(
        self,
        brief: str,
        specs: list[SubTaskSpec],
        results: list[ResearchResult],
        entity_graph: EntityGraph | None,
    ) -> SynthesisResult:
        done = [r for r in results if r.status is SubTaskStatus.DONE and r.refs]
        all_ids = list(dict.fromkeys(ref.source_id for r in done for ref in r.refs))
        resolved = {s.source_id: s for s in await self.resolver.resolve_many(all_ids)}
        sources = {sid: _meta(s) for sid, s in resolved.items()}
        spec_by_id = {s.id: s for s in specs}

        per_task = await asyncio.gather(*(
            self._claims_for(spec_by_id[r.sub_task_id], [resolved[ref.source_id] for ref in r.refs
                                                          if ref.source_id in resolved])
            for r in done if r.sub_task_id in spec_by_id
        ))
        themes = {task_id: theme for task_id, theme, _, _ in per_task}
        cands = [c for _, _, cs, _ in per_task for c in cs]
        passage_tokens = sum(t for *_, t in per_task)

        if cands:
            vecs = np.array(await embed_documents([c.claim for c in cands]))
            brief_vec = np.array(await embed_query(brief))
            rel = _cos(vecs, brief_vec[None, :])[:, 0]
            for c, v, r in zip(cands, vecs, rel, strict=True):
                c.vector, c.relevance = v, float(max(0.0, min(1.0, r)))
                c.entity_keys = entity_keys_for(c.claim, entity_graph)
        merged = dedupe(cands, sources)

        sections = self._sections(merged, sources, entity_graph, themes, [r.sub_task_id for r in done])
        sections, cuts = await self._compress(sections, sources)
        bundle = EvidenceBundle(sections=sections, sources=sources, cuts=cuts, token_budget=self.token_budget,
                                token_count=count_tokens(render_bundle(sections, sources)))
        corpus = sum(count_tokens(s.text) for s in resolved.values())
        await self.log.progress(
            AGENT,
            f"evidence bundle: {sum(len(s.evidence) for s in sections)} items in {len(sections)} sections, "
            f"{bundle.token_count:,}/{self.token_budget:,} tokens ({len(cuts)} cut(s))",
            corpus_tokens=corpus, passage_tokens=passage_tokens, candidates=len(cands), merged=len(merged),
            bundle_tokens=bundle.token_count,
        )
        return SynthesisResult(bundle, corpus, passage_tokens)

    async def _claims_for(self, spec: SubTaskSpec, sources: list[ResolvedSource]):
        passages = await select_passages(sources, f"{spec.query}. {spec.scope}", self.passages_per_source)
        if not passages:
            return spec.id, spec.query[:80], [], 0
        by_id = {p.passage_id: p for p in passages}
        body = "\n\n".join(wrap(p.passage_id, p.text, url=p.source_id) for p in passages)
        prompt = f"Sub-task: {spec.query}\nScope: {spec.scope}\nTime horizon: {spec.time_horizon}\n\n{body}"
        out: SubTaskClaims = await self.gateway.structured(Role.SYNTHESIS, SubTaskClaims).ainvoke(
            [SystemMessage(SYNTHESIS_CLAIMS_PROMPT), HumanMessage(prompt)]
        )
        cands = [
            Candidate(spec.id, pc.claim.strip(), by_id[pc.passage_id].text, [by_id[pc.passage_id].source_id])
            for pc in out.claims if pc.passage_id in by_id and pc.claim.strip()
        ]
        dropped = len(out.claims) - len(cands)
        if dropped:
            await self.log.emit(AGENT, "error", f"{spec.id}: dropped {dropped} claim(s) citing unknown passages")
        await self.log.progress(AGENT, f"{spec.id}: {len(cands)} claim(s) from {len(passages)} passage(s) "
                                       f"of {len(sources)} source(s)", sub_task_id=spec.id)
        return spec.id, out.theme or spec.query[:80], cands, count_tokens(prompt)

    def _sections(self, merged, sources, graph, themes, order) -> list[EvidenceSection]:
        by_task: dict[str, list[EvidenceItem]] = defaultdict(list)
        for c in merged:
            primary = sorted(c.source_ids, key=lambda s: sources[s].credibility or 0.0, reverse=True)
            conflict = touches_conflict(c, graph)
            dates = [sources[s].published_date for s in primary if sources[s].published_date]
            by_task[c.sub_task_id].append(EvidenceItem(
                evidence_id="ev-0", claim=c.claim, source_ids=primary,
                published_date=max(dates) if dates else None,
                credibility=sources[primary[0]].credibility or 0.0, relevance=round(c.relevance, 3),
                confidence=confidence_for(c, sources, conflict, self.high_credibility),
                conflict=conflict, entity_keys=c.entity_keys, passage=c.passage,
            ))
        sections, n = [], 0
        for task_id in order:
            items = sorted(by_task.get(task_id, []), key=rank_key, reverse=True)
            if not items:
                continue
            for item in items:
                n += 1
                item.evidence_id = f"ev-{n}"
            sections.append(EvidenceSection(theme=themes.get(task_id, task_id), sub_task_id=task_id,
                                            evidence=items, confidence=section_confidence(items)))
        return sections

    async def _compress(self, sections, sources) -> tuple[list[EvidenceSection], list[Cut]]:
        cuts: list[Cut] = []
        next_id = max((int(i.evidence_id[3:]) for s in sections for i in s.evidence), default=0)
        for _ in range(50):  # bounded: each pass removes at least one item
            size = count_tokens(render_bundle(sections, sources))
            if size <= self.token_budget:
                break
            target = max(sections, key=lambda s: count_tokens(render_bundle([s], sources)))
            plain = [i for i in target.evidence if not i.compressed]
            if len(plain) >= 3:
                tail = plain[len(plain) // 2:]  # lowest-ranked half (items are sorted best-first)
                next_id += 1
                summary = await self._summarize(tail, f"ev-{next_id}")
                kept = [i for i in target.evidence if i not in tail]
                target.evidence = [*kept, summary] if summary else kept
                action = "summarized" if summary else "dropped"
                reason = (f"bundle {size:,} > {self.token_budget:,} tokens: lowest-ranked {len(tail)} item(s) of "
                          f"'{target.theme}' (older / less credible / less relevant)")
            else:
                worst = min((i for s in sections for i in s.evidence), key=rank_key)
                target = next(s for s in sections if worst in s.evidence)
                tail = [worst]
                target.evidence = [i for i in target.evidence if i is not worst]
                action, reason = "dropped", f"bundle {size:,} > {self.token_budget:,} tokens after summarizing"
            saved = size - count_tokens(render_bundle(sections, sources))
            cut = Cut(evidence_ids=[i.evidence_id for i in tail], action=action, reason=reason, tokens_saved=saved)
            cuts.append(cut)
            await self.log.emit(AGENT, "cut", f"{action} {len(tail)} item(s): {reason}", **cut.model_dump())
            target.confidence = section_confidence(target.evidence)
        sections = [s for s in sections if s.evidence]
        return sections, cuts

    async def _summarize(self, items: list[EvidenceItem], evidence_id: str) -> EvidenceItem | None:
        listing = "\n".join(f"- {i.claim}" for i in items)
        try:
            out: CompressedClaim = await self.gateway.structured(Role.SUMMARIZER, CompressedClaim).ainvoke(
                [SystemMessage(COMPRESS_PROMPT), HumanMessage(listing)]
            )
        except TokenBudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - a failed summary falls back to dropping the items (logged as a cut)
            return None
        order = list(Confidence)
        return EvidenceItem(
            evidence_id=evidence_id, claim=out.summary,
            source_ids=list(dict.fromkeys(s for i in items for s in i.source_ids)),
            published_date=max((i.published_date for i in items if i.published_date), default=None),
            credibility=max(i.credibility for i in items), relevance=max(i.relevance for i in items),
            confidence=max((i.confidence for i in items), key=order.index),  # the weakest (LOW is last)
            compressed=True, conflict=any(i.conflict for i in items),
            entity_keys=sorted({k for i in items for k in i.entity_keys}),
        )


def _meta(s: ResolvedSource) -> SourceMeta:
    label = s.label if s.kind is SourceKind.PDF_CHUNK else (registered_host(s.url) if s.url else s.source_id)
    return SourceMeta(source_id=s.source_id, kind=s.kind, label=label, url=s.url, title=s.title,
                      published_date=s.published_date, credibility=s.credibility)
