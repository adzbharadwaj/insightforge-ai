from sqlalchemy import select

from insightforge.store.models import (
    Claim,
    Report,
    Run,
    RunEvent,
    RunStatus,
    ScratchSource,
    SourceKind,
    SourceType,
    SubTask,
    SubTaskStatus,
    User,
)


async def test_run_with_subtasks_scratch_and_report(db_session):
    user = User(email="analyst@example.com")
    db_session.add(user)
    await db_session.flush()

    run = Run(user_id=user.id, brief="B2B payroll software in Southeast Asia", source_prefs={"web": True})
    run.sub_tasks.append(
        SubTask(task_key="st-1", query="pricing", source_type=SourceType.WEB, time_horizon="18 months", priority=1)
    )
    db_session.add(run)
    await db_session.flush()

    assert run.status is RunStatus.QUEUED
    assert run.sub_tasks[0].status is SubTaskStatus.QUEUED

    db_session.add(
        ScratchSource(
            run_id=run.id,
            source_id="src-001",
            kind=SourceKind.WEB_PAGE,
            url="https://example.com/a",
            body_text="full article body",
            credibility=0.7,
        )
    )
    db_session.add(RunEvent(run_id=run.id, agent="synthesis", kind="cut", message="summarized 3 low-priority items"))
    report = Report(run_id=run.id, user_id=user.id, title="t", markdown="# t", evidence_bundle={"sections": []})
    report.claims.append(Claim(claim_key="c-1", section="Pricing", text="x", evidence_ids=["src-001"]))
    db_session.add(report)
    await db_session.flush()

    src = await db_session.scalar(select(ScratchSource).where(ScratchSource.source_id == "src-001"))
    assert src.kind is SourceKind.WEB_PAGE and src.body_text == "full article body"

    claim = await db_session.scalar(select(Claim).where(Claim.report_id == report.id))
    assert claim.evidence_ids == ["src-001"] and claim.verified is None
