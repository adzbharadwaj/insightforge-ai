"""Response models and the queries that build them. Every query is scoped by user id."""

import datetime as dt
import uuid

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.store.models import Report, Run, RunStatus, SubTask

TERMINAL = frozenset({RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED})

# Measured on live runs: planning ~15 s, research/entities/synthesis/writing scale with sub-tasks.
ETA_BASE_S, ETA_PER_SUBTASK_S, ETA_PER_PDF_S = 60, 25, 15


def estimate_runtime_s(sub_tasks: int, pdfs: int = 0) -> int:
    return ETA_BASE_S + ETA_PER_SUBTASK_S * sub_tasks + ETA_PER_PDF_S * pdfs


class SubTaskView(BaseModel):
    id: str
    query: str
    source_type: str
    priority: int
    status: str
    summary: str | None = None
    error: str | None = None


class RunSnapshot(BaseModel):
    run_id: uuid.UUID
    status: RunStatus
    brief: str
    created_at: dt.datetime
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    elapsed_s: float | None
    sub_tasks: list[SubTaskView]
    clarifying_question: str | None
    review: dict | None = None  # conflicts awaiting the analyst (AWAITING_REVIEW)
    error: str | None
    report_id: uuid.UUID | None
    tokens_used: int
    attempts: int
    eta_seconds: int


async def owned_run(sessionmaker: async_sessionmaker[AsyncSession], run_id: uuid.UUID, user_id: uuid.UUID) -> Run | None:
    async with sessionmaker() as session:
        run = await session.get(Run, run_id)
    return run if run is not None and run.user_id == user_id else None


async def run_snapshot(sessionmaker: async_sessionmaker[AsyncSession], run: Run) -> RunSnapshot:
    async with sessionmaker() as session:
        run = await session.get(Run, run.id)  # fresh status
        tasks = (await session.scalars(select(SubTask).where(SubTask.run_id == run.id)
                                       .order_by(SubTask.priority, SubTask.task_key))).all()
        report_id = await session.scalar(select(Report.id).where(Report.run_id == run.id))
    end = run.finished_at or (None if run.status in TERMINAL else dt.datetime.now(dt.UTC))
    elapsed = (end - run.started_at).total_seconds() if run.started_at and end else None
    n = len(tasks) or 5
    return RunSnapshot(
        run_id=run.id, status=run.status, brief=run.brief, created_at=run.created_at, started_at=run.started_at,
        finished_at=run.finished_at, elapsed_s=round(elapsed, 1) if elapsed is not None else None,
        sub_tasks=[SubTaskView(id=t.task_key, query=t.query, source_type=t.source_type.value, priority=t.priority,
                               status=t.status.value, summary=t.summary, error=t.error) for t in tasks],
        clarifying_question=run.clarifying_question if run.status is RunStatus.AWAITING_CLARIFICATION else None,
        review=run.review if run.status is RunStatus.AWAITING_REVIEW else None,
        error=run.error, report_id=report_id, tokens_used=run.tokens_used, attempts=run.attempts,
        eta_seconds=estimate_runtime_s(n),
    )


class ReportSummary(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    title: str
    brief: str
    created_at: dt.datetime
    pass_rate: float | None


class EvidenceView(BaseModel):
    evidence_id: str
    claim: str
    passage: str | None
    source_ids: list[str]
    confidence: str
    published_date: str | None
    credibility: float
    conflict: bool
    entity_keys: list[str] = []


class ReportView(BaseModel):
    id: uuid.UUID
    title: str
    brief: str
    created_at: dt.datetime
    markdown: str
    structure: dict | None  # WriterOutput + claims, for clickable citations
    fact_check: dict | None
    sections: list[dict]  # theme, sub_task_id, confidence
    sources: dict[str, dict]  # citation drawer: label, url, title, date, credibility
    evidence: dict[str, EvidenceView]
    entity_graph: dict | None = None  # entities, relationships, conflicts (entity graph visualizer)


def report_view(report: Report, brief: str, entity_graph: dict | None = None) -> ReportView:
    bundle = report.evidence_bundle or {}
    evidence = {
        item["evidence_id"]: EvidenceView(
            evidence_id=item["evidence_id"], claim=item["claim"], passage=item.get("passage"),
            source_ids=item["source_ids"], confidence=item["confidence"], published_date=item.get("published_date"),
            credibility=item["credibility"], conflict=item.get("conflict", False),
            entity_keys=item.get("entity_keys", []))
        for section in bundle.get("sections", []) for item in section["evidence"]
    }
    sections = [{"theme": s["theme"], "sub_task_id": s["sub_task_id"], "confidence": s["confidence"]}
                for s in bundle.get("sections", [])]
    return ReportView(id=report.id, title=report.title, brief=brief, created_at=report.created_at,
                      markdown=report.markdown, structure=report.structure, fact_check=report.factcheck_summary,
                      sections=sections, sources=bundle.get("sources", {}), evidence=evidence,
                      entity_graph=entity_graph)
