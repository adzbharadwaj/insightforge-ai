"""Grounding check (part of output guardrails 01b): every claim must trace to the bundle.

Two layers, cheapest first:

1. deterministic  cited ids exist in the bundle; a sentence with a number cites evidence; every
                  number in a claim appears in its cited evidence (claim text, source passage, or
                  source date); sections match the bundle; summary ≤ 150 words.
2. entailment     one model call per batch of claims that passed (1), each with only the evidence
                  it cites, judging whether the evidence states the claim.

On failure the orchestrator retries the Writer with the reasons as explicit constraints
(max retries from config). If claims are still unsupported after that, they are removed and
the removal is reported under Gaps & Limitations: fail safe, never ship an ungrounded claim.
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from insightforge.agents.prompts import ENTAILMENT_PROMPT
from insightforge.agents.writer import SUMMARY_MAX_WORDS, trim_summary
from insightforge.graph.state import (
    ClaimRef,
    Draft,
    EntailmentVerdicts,
    EvidenceBundle,
    GroundingResult,
    WriterOutput,
)
from insightforge.llm.gateway import ModelGateway, Role
from insightforge.tools.entities import numbers_in

ENTAILMENT_BATCH = 15  # ~40 output tokens per verdict; keeps each call far from max_tokens
REPORT_KEY = "report"  # reasons that are about the report as a whole, not one claim


def _evidence_numbers(claim: ClaimRef, bundle: EvidenceBundle) -> set[float]:
    items = bundle.items()
    allowed: set[float] = set()
    for eid in claim.evidence_ids:
        item = items.get(eid)
        if item is None:
            continue
        allowed |= numbers_in(item.claim) | numbers_in(item.passage or "")
        for sid in item.source_ids:
            meta = bundle.sources.get(sid)
            if meta and meta.published_date:
                d = meta.published_date
                allowed |= {float(d.year), float(d.month), float(d.day)}
    return allowed


def structural_issues(draft: Draft, bundle: EvidenceBundle) -> dict[str, str]:
    reasons: dict[str, str] = {}
    known = bundle.evidence_ids()
    for c in draft.claims:
        unknown = [e for e in c.evidence_ids if e not in known]
        if unknown:
            reasons[c.claim_id] = f"cites evidence ids that do not exist: {unknown}"
        elif not c.evidence_ids:
            reasons[c.claim_id] = "states a number but cites no evidence"
        else:
            missing = numbers_in(c.text) - _evidence_numbers(c, bundle)
            if missing:
                shown = ", ".join(f"{n:g}" for n in sorted(missing))
                reasons[c.claim_id] = f"number(s) {shown} do not appear in the cited evidence"

    report = []
    words = sum(len(s.text.split()) for s in draft.report.executive_summary)
    if words > SUMMARY_MAX_WORDS:
        report.append(f"executive summary is {words} words; the limit is {SUMMARY_MAX_WORDS}")
    expected = [s.sub_task_id for s in bundle.sections]
    written = [s.sub_task_id for s in draft.report.sections]
    if written != expected:
        report.append(f"sections must be exactly {expected} in that order (got {written})")
    if report:
        reasons[REPORT_KEY] = "; ".join(report)
    return reasons


async def entailment_issues(gateway: ModelGateway, claims: list[ClaimRef], bundle: EvidenceBundle) -> dict[str, str]:
    if not claims:
        return {}
    items = bundle.items()

    def publishers(e: str) -> str:
        # The Writer attributes facts to publishers ("Outsail lists ..."); the judge must see who published what.
        labels = [bundle.sources[s].label for s in items[e].source_ids if s in bundle.sources]
        return ", ".join(dict.fromkeys(labels)) or "unknown"

    def block(c: ClaimRef) -> str:
        ev = "\n".join(f"  {e} [published by {publishers(e)}]: {items[e].claim}"
                       + (f"\n    passage: {items[e].passage[:300]}" if items[e].passage else "")
                       for e in c.evidence_ids if e in items)
        return f"claim_id: {c.claim_id}\nclaim: {c.text}\nevidence:\n{ev}"

    batches = [claims[i:i + ENTAILMENT_BATCH] for i in range(0, len(claims), ENTAILMENT_BATCH)]
    results = await asyncio.gather(*(
        gateway.structured(Role.GROUNDING, EntailmentVerdicts).ainvoke(
            [SystemMessage(ENTAILMENT_PROMPT), HumanMessage("\n\n".join(block(c) for c in batch))]
        )
        for batch in batches
    ))
    asked = {c.claim_id for c in claims}
    reasons = {}
    for verdicts in results:
        for v in verdicts.verdicts:
            if v.claim_id in asked and not v.supported:
                reasons[v.claim_id] = v.reason or "not supported by the cited evidence"
    return reasons


async def check_grounding(gateway: ModelGateway, draft: Draft, bundle: EvidenceBundle) -> GroundingResult:
    reasons = structural_issues(draft, bundle)
    to_check = [c for c in draft.claims if c.claim_id not in reasons and c.evidence_ids]
    reasons |= await entailment_issues(gateway, to_check, bundle)
    unsupported = [c.claim_id for c in draft.claims if c.claim_id in reasons]
    return GroundingResult(ok=not reasons, unsupported_claim_ids=unsupported, reasons=reasons)


def retry_constraints(result: GroundingResult, draft: Draft) -> list[str]:
    by_id = {c.claim_id: c for c in draft.claims}
    out = [f'Remove or fix: "{by_id[cid].text}" ({result.reasons[cid]}). Only state what the cited items say.'
           for cid in result.unsupported_claim_ids if cid in by_id]
    if REPORT_KEY in result.reasons:
        out.append(result.reasons[REPORT_KEY])
    return out


def drop_unsupported(draft: Draft, result: GroundingResult, bundle: EvidenceBundle) -> WriterOutput:
    """Final fallback: remove claims that are still unsupported and say so under Gaps."""
    locs = {c.loc for c in draft.claims if c.claim_id in set(result.unsupported_claim_ids)}
    out = draft.report.model_copy(deep=True)
    out.executive_summary = trim_summary([s for i, s in enumerate(out.executive_summary)
                                          if f"summary:{i}" not in locs])
    for si, section in enumerate(out.sections):
        section.sentences = [s for i, s in enumerate(section.sentences) if f"section:{si}:{i}" not in locs]
    expected = {s.sub_task_id for s in bundle.sections}
    out.sections = [s for s in out.sections if s.sub_task_id in expected and s.sentences]
    if locs:
        out.gaps.append(f"{len(locs)} statement(s) were removed because the gathered evidence did not support them.")
    return out
