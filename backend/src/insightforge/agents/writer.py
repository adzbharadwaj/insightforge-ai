"""Writer agent (component 11) and the deterministic report renderer.

The model returns a WriterOutput: sentences, each listing the evidence ids it relies on.
Code renders the Markdown, so the required structure, the citation format
`[Source: {domain}, {date}, credibility: {score}]`, the per-section confidence labels
(taken from the bundle, not from the model) and `[UNVERIFIED]` marks are guaranteed and
can never cite a source the evidence doesn't contain.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from insightforge.agents.prompts import WRITER_PROMPT
from insightforge.agents.synthesis import render_bundle
from insightforge.graph.state import (
    ClaimRef,
    Confidence,
    Draft,
    EvidenceBundle,
    ResearchResult,
    WriterOutput,
    WrittenSentence,
)
from insightforge.llm.gateway import ModelGateway, Role

SUMMARY_MAX_WORDS = 150
_HAS_NUMBER = re.compile(r"\d")


async def write_report(
    gateway: ModelGateway,
    *,
    brief: str,
    bundle: EvidenceBundle,
    failed: list[ResearchResult],
    constraints: list[str] | None = None,
) -> WriterOutput:
    parts = [f"Research brief:\n{brief}", "Evidence bundle:\n" + render_bundle(bundle.sections, bundle.sources)]
    notes = [f"- sub-task {r.sub_task_id} failed: {r.error or 'no usable sources'}" for r in failed]
    notes += [f"- {n}" for n in bundle.notes]
    if bundle.cuts:
        n = sum(len(c.evidence_ids) for c in bundle.cuts)
        notes.append(f"- {n} lower-priority evidence item(s) were summarized or dropped to fit the evidence budget")
    if notes:
        parts.append("Research notes (use for gaps):\n" + "\n".join(notes))
    if constraints:
        parts.append("Your previous draft broke the rules. Fix all of these:\n" + "\n".join(f"- {c}" for c in constraints))
    return await gateway.structured(Role.WRITER, WriterOutput).ainvoke(
        [SystemMessage(WRITER_PROMPT), HumanMessage("\n\n".join(parts))]
    )


def citation(evidence_id: str, bundle: EvidenceBundle) -> list[tuple[str, str]]:
    """(source_id, citation text) for an evidence item's primary source."""
    item = bundle.items().get(evidence_id)
    if item is None:
        return []
    meta = bundle.sources.get(item.source_ids[0])
    if meta is None:
        return []
    date = meta.published_date.isoformat() if meta.published_date else "n.d."
    cred = f"{meta.credibility:.2f}" if meta.credibility is not None else "n/a"
    return [(meta.source_id, f"[Source: {meta.label}, {date}, credibility: {cred}]")]


def _cite(sentence: WrittenSentence, bundle: EvidenceBundle) -> str:
    seen, out = set(), []
    for eid in sentence.evidence_ids:
        for sid, text in citation(eid, bundle):
            if sid not in seen:
                seen.add(sid)
                out.append(text)
    return " ".join(out)


def trim_summary(sentences: list[WrittenSentence], max_words: int = SUMMARY_MAX_WORDS) -> list[WrittenSentence]:
    kept, words = [], 0
    for s in sentences:
        n = len(s.text.split())
        if kept and words + n > max_words:
            break
        kept.append(s)
        words += n
    return kept


def render_report(out: WriterOutput, bundle: EvidenceBundle, unverified: frozenset[str] = frozenset()) -> Draft:
    claims: list[ClaimRef] = []

    def claim_for(sentence: WrittenSentence, section: str, loc: str) -> str | None:
        if not sentence.evidence_ids and not _HAS_NUMBER.search(sentence.text):
            return None
        cid = f"c-{len(claims) + 1}"
        claims.append(ClaimRef(claim_id=cid, section=section, text=sentence.text,
                               evidence_ids=sentence.evidence_ids, loc=loc))
        return cid

    def mark(cid: str | None) -> str:
        return " **[UNVERIFIED]**" if cid and cid in unverified else ""

    confidence = {s.sub_task_id: s.confidence for s in bundle.sections}
    lines = [f"# {out.title}", "", "## Executive Summary", ""]
    summary = []
    for i, s in enumerate(out.executive_summary):
        cid = claim_for(s, "Executive Summary", f"summary:{i}")
        summary.append(s.text.strip() + mark(cid))
    lines += [" ".join(summary), ""]

    for si, section in enumerate(out.sections):
        conf = confidence.get(section.sub_task_id, Confidence.LOW)
        lines += [f"## {section.heading}", "", f"**Confidence: {conf.value}**", ""]
        rendered = []
        for i, s in enumerate(section.sentences):
            cid = claim_for(s, section.heading, f"section:{si}:{i}")
            cite = _cite(s, bundle)
            rendered.append(s.text.strip() + (f" {cite}" if cite else "") + mark(cid))
        for start in range(0, len(rendered), 3):  # short paragraphs read better than one block
            lines += [" ".join(rendered[start:start + 3]), ""]

    lines += ["## Gaps & Limitations", ""]
    lines += [f"- {g}" for g in out.gaps] or ["- None identified."]
    return Draft(report=out, markdown="\n".join(lines).strip() + "\n", claims=claims)
