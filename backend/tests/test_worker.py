"""Worker job guards and housekeeping: jobs act only on runs in the state they were queued for."""

import datetime as dt

import pytest
from sqlalchemy import update

from insightforge.store.db import get_engine
from insightforge.store.models import Run, RunEvent, RunStatus
from insightforge.worker import fail_orphaned_runs, resume_run, retry_run, start_run


@pytest.fixture(autouse=True)
async def _fresh_pool():
    # Worker jobs use the process-wide engine; its pooled connections belong to earlier tests' event loops.
    await get_engine().dispose(close=False)


class ExplodingRuntime:
    def __getattr__(self, name):
        raise AssertionError(f"runtime.{name} must not be called")


async def set_status(maker, run_id, status, **values):
    async with maker() as s:
        await s.execute(update(Run).where(Run.id == run_id).values(status=status, **values))
        await s.commit()


async def test_redelivered_job_never_restarts_a_run_in_progress(committed_run):
    """Live incident: arq re-delivered a dead worker's start_run 40 minutes later and restarted the run."""
    maker, _, run_id = committed_run
    ctx = {"runtime": ExplodingRuntime()}
    await set_status(maker, run_id, RunStatus.RESEARCHING)
    assert await start_run(ctx, str(run_id)) == "RESEARCHING"
    await set_status(maker, run_id, RunStatus.COMPLETE)
    assert await start_run(ctx, str(run_id)) == "COMPLETE"
    assert await retry_run(ctx, str(run_id)) == "COMPLETE"  # only FAILED/QUEUED runs are retried
    await set_status(maker, run_id, RunStatus.WRITING)
    assert await resume_run(ctx, str(run_id), "answer") == "WRITING"  # only right after the clarify endpoint
    await set_status(maker, run_id, RunStatus.CANCELLED)
    assert await start_run(ctx, str(run_id)) == "CANCELLED"


async def test_orphaned_runs_are_failed_but_live_and_finished_runs_are_left_alone(committed_run):
    maker, user_id, run_id = committed_run
    long_ago = dt.datetime.now(dt.UTC) - dt.timedelta(hours=3)
    async with maker() as s:
        live = Run(user_id=user_id, brief="live", status=RunStatus.WRITING)
        done = Run(user_id=user_id, brief="done", status=RunStatus.COMPLETE, created_at=long_ago)
        s.add_all([live, done])
        await s.flush()
        s.add(RunEvent(run_id=live.id, agent="writer", kind="progress", message="writing", data={}))
        s.add(RunEvent(run_id=run_id, agent="planner", kind="progress", message="old", data={}, created_at=long_ago))
        await s.commit()
    await set_status(maker, run_id, RunStatus.RESEARCHING, created_at=long_ago)

    assert await fail_orphaned_runs({}) >= 1
    async with maker() as s:
        orphan, live_row, done_row = await s.get(Run, run_id), await s.get(Run, live.id), await s.get(Run, done.id)
    assert orphan.status is RunStatus.FAILED and "retry to resume" in orphan.error
    assert live_row.status is RunStatus.WRITING and done_row.status is RunStatus.COMPLETE
