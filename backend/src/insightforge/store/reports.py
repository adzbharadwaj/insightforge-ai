"""Persist a finished report: Markdown, evidence bundle, fact-check summary and claims."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.graph.state import Draft, EvidenceBundle, FactCheckSummary
from insightforge.store.models import Claim, Report

SUMMARY_CHARS = 800


async def save_report(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    run_id: uuid.UUID,
    user_id: uuid.UUID,
    draft: Draft,
    markdown: str,
    bundle: EvidenceBundle,
    fact_check: FactCheckSummary,
) -> uuid.UUID:
    checks = {c.claim_id: c for c in fact_check.checks}
    report = Report(run_id=run_id, user_id=user_id, title=draft.report.title, markdown=markdown,
                    evidence_bundle=bundle.model_dump(mode="json"),
                    factcheck_summary=fact_check.model_dump(mode="json"))
    report.claims = [
        Claim(claim_key=c.claim_id, section=c.section, text=c.text, evidence_ids=c.evidence_ids,
              sampled=c.claim_id in checks,
              verified=checks[c.claim_id].verified if c.claim_id in checks else None,
              similarity=checks[c.claim_id].similarity if c.claim_id in checks else None,
              verify_note=checks[c.claim_id].note if c.claim_id in checks else None)
        for c in draft.claims
    ]
    async with sessionmaker() as session:
        session.add(report)
        await session.commit()
        return report.id


def session_summary(brief: str, draft: Draft) -> str:
    """The compressed memory of this run that future Planners read (no extra model call)."""
    summary = " ".join(s.text for s in draft.report.executive_summary)
    return f"Brief: {brief[:200]} | Report: {draft.report.title} | {summary}"[:SUMMARY_CHARS]
