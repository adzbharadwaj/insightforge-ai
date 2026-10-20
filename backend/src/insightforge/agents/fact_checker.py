"""Fact-Checker (component 12): the last verification pass before a report is delivered.

- Samples max(20%, 5) of the report's section claims (seeded by run id, so a re-run of the
  same report samples the same claims).
- Re-fetches each cited web source live (through the same SSRF-guarded fetcher). If the
  re-fetch fails, the copy stored during research is used and the check says so; PDF chunks
  are checked against the stored document text.
- Splits each source into 1–3 sentence windows and claim-sized word windows (12/20/30 words),
  shortlists them by keyword overlap with the claims, embeds them into a temporary Qdrant
  collection `factcheck_{run}` and takes the best cosine similarity between the claim and its
  own sources' windows. Below the threshold (default 0.85) → `[UNVERIFIED]`.
- A claim whose sources yield no text fails closed. The temporary collection is always dropped.
"""

import asyncio
import math
import random
import re
import uuid
from itertools import pairwise

from insightforge.graph.state import ClaimCheck, ClaimRef, Draft, EvidenceBundle, FactCheckSummary
from insightforge.llm.embeddings import embed_documents
from insightforge.store.models import SourceKind
from insightforge.store.qdrant import VectorStore, factcheck_collection
from insightforge.store.run_log import RunLog
from insightforge.store.sources import SourceResolver
from insightforge.tools.fetch import FetchedPage, PageFetcher
from insightforge.tools.lexical import shortlist

AGENT = "fact_checker"
MAX_WINDOWS_PER_SOURCE = 400
WINDOWS_PER_SOURCE = 60  # keyword shortlist before embedding
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def sentence_windows(text: str) -> list[str]:
    """1-, 2- and 3-sentence windows: a claim often condenses neighbouring sentences."""
    sentences = [s.strip() for line in text.splitlines() for s in _SENTENCE.split(line) if len(s.strip()) >= 20]
    windows = sentences + [f"{a} {b}" for a, b in pairwise(sentences)]
    windows += [" ".join(sentences[i:i + 3]) for i in range(len(sentences) - 2)]  # claims that merge 3 sentences
    return list(dict.fromkeys(windows))[:MAX_WINDOWS_PER_SOURCE]


def word_windows(text: str, sizes: tuple[int, ...] = (12, 20, 30), stride: int = 4) -> list[str]:
    """Sliding windows about as long as a claim. An atomic claim is often part of a longer sentence, and the
    extra clauses dilute the cosine: measured on a live run, this lifted 4 of 5 failing true claims over 0.85."""
    words = text.split()
    out = [" ".join(words[i:i + n]) for n in sizes for i in range(0, max(1, len(words) - n + 1), stride)]
    return list(dict.fromkeys(out))


def sample_claims(claims: list[ClaimRef], ratio: float, minimum: int, seed: str) -> list[ClaimRef]:
    eligible = [c for c in claims if c.evidence_ids and c.section != "Executive Summary"]
    n = min(len(eligible), max(math.ceil(ratio * len(eligible)), minimum))
    picked = {c.claim_id for c in random.Random(seed).sample(eligible, n)}
    return [c for c in eligible if c.claim_id in picked]


class FactChecker:
    def __init__(
        self,
        vectors: VectorStore,
        resolver: SourceResolver,
        log: RunLog,
        *,
        fetcher: PageFetcher | None,
        sample_ratio: float = 0.2,
        min_claims: int = 5,
        threshold: float = 0.85,
    ):
        self.vectors = vectors
        self.resolver = resolver
        self.log = log
        self.fetcher = fetcher
        self.sample_ratio = sample_ratio
        self.min_claims = min_claims
        self.threshold = threshold

    async def _source_text(self, source_id: str) -> tuple[str, str]:
        src = await self.resolver.resolve(source_id)
        if src is None:
            return "", "unavailable"
        if src.kind is SourceKind.PDF_CHUNK:
            return src.text, "document"
        if self.fetcher and src.url:
            page = await self.fetcher.fetch(src.url)
            if isinstance(page, FetchedPage):
                return page.body_text, "live"
            await self.log.emit(AGENT, "skipped_url", f"re-fetch failed, using stored copy: {src.url}",
                                url=src.url, reason=page.reason)
        return src.text, "stored_copy"

    async def run(self, draft: Draft, bundle: EvidenceBundle) -> FactCheckSummary:
        run_id = self.resolver.run_id
        sampled = sample_claims(draft.claims, self.sample_ratio, self.min_claims, str(run_id))
        items = bundle.items()
        claim_sources = {c.claim_id: list(dict.fromkeys(s for e in c.evidence_ids if e in items
                                                        for s in items[e].source_ids)) for c in sampled}
        source_ids = list(dict.fromkeys(s for ids in claim_sources.values() for s in ids))
        await self.log.progress(AGENT, f"checking {len(sampled)} of {len(draft.claims)} claim(s) "
                                       f"against {len(source_ids)} source(s)")

        collection = factcheck_collection(run_id)
        await self.vectors.ensure_collection(collection)
        try:
            texts = await asyncio.gather(*(self._source_text(s) for s in source_ids))
            against = {sid: how for sid, (_, how) in zip(source_ids, texts, strict=True)}
            windows = []
            for sid, (text, _) in zip(source_ids, texts, strict=True):
                # Embed only the windows that share words with a claim citing this source (embedding is the
                # slow part); the cosine threshold still decides.
                citing = " ".join(c.text for c in sampled if sid in claim_sources[c.claim_id])
                all_windows = sentence_windows(text) + word_windows(text)
                windows += [(sid, all_windows[i]) for i in shortlist(citing, all_windows, WINDOWS_PER_SOURCE)]
            if windows:
                vecs = await embed_documents([w for _, w in windows])
                await self.vectors.upsert(
                    collection, user_id=self.resolver.user_id, vectors=vecs,
                    payloads=[{"run_id": str(run_id), "source_id": sid, "text": w} for sid, w in windows],
                    ids=[str(uuid.uuid4()) for _ in windows],
                )
            claim_vecs = await embed_documents([c.text for c in sampled])
            checks = []
            for claim, vec in zip(sampled, claim_vecs, strict=True):
                checks.append(await self._check(collection, claim, vec, claim_sources[claim.claim_id], against))
        finally:
            await self.vectors.drop_collection(collection)

        passed = sum(c.verified for c in checks)
        summary = FactCheckSummary(
            total_checked=len(checks), passed=passed, failed=len(checks) - passed,
            pass_rate=passed / len(checks) if checks else 1.0,
            unverified_claims=[c.text for c in checks if not c.verified], checks=checks,
        )
        await self.log.progress(AGENT, f"fact-check: {passed}/{len(checks)} passed",
                                **summary.model_dump(mode="json", exclude={"checks"}))
        return summary

    async def _check(self, collection, claim: ClaimRef, vec, sources: list[str], against: dict[str, str]) -> ClaimCheck:
        usable = [s for s in sources if against.get(s) != "unavailable"]
        if not usable:
            return ClaimCheck(claim_id=claim.claim_id, text=claim.text, verified=False,
                              checked_against="unavailable", note="source unreachable")
        hits = await self.vectors.search(collection, vec, user_id=self.resolver.user_id, limit=1,
                                         match_any={"source_id": usable})
        if not hits:
            return ClaimCheck(claim_id=claim.claim_id, text=claim.text, verified=False,
                              checked_against=against[usable[0]], note="no text found in source")
        best = hits[0]
        sid = best.payload["source_id"]
        ok = best.score >= self.threshold
        return ClaimCheck(
            claim_id=claim.claim_id, text=claim.text, verified=ok, similarity=round(best.score, 4), source_id=sid,
            checked_against=against[sid],
            note="" if ok else f"best match {best.score:.2f} < {self.threshold}: \"{best.payload['text'][:160]}\"",
        )
