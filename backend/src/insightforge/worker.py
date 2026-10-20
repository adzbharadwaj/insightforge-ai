"""Run worker (component 04): executes queued runs outside the API process.

    uv run arq insightforge.worker.WorkerSettings

- `start_run`  ingests the run's uploads, then runs the pipeline to its next stop.
- `resume_run` continues a run paused for clarification with the user's answer.
- `cancel_stale_clarifications` (every minute) cancels runs whose question has waited
  longer than `clarification_timeout_s` (30 min), with a notice in the run log.

`max_jobs` caps concurrent runs per worker; the job timeout sits just above the run's own
time budget, which `execute()` enforces (so a run ends FAILED with a reason, not killed).
"""

import datetime as dt
import logging
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from arq import cron
from arq.connections import RedisSettings
from sqlalchemy import func, select, update

from insightforge.config import get_settings
from insightforge.llm.gateway import ModelGateway
from insightforge.runtime import Runtime, Upload
from insightforge.store.db import get_sessionmaker
from insightforge.store.models import Run, RunEvent, RunStatus
from insightforge.store.run_log import RunLog
from insightforge.watchlist.service import process_watch_result

logger = logging.getLogger(__name__)
TERMINAL = (RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED)
ACTIVE = (RunStatus.PLANNING, RunStatus.RESEARCHING, RunStatus.SYNTHESIZING, RunStatus.WRITING,
          RunStatus.GROUNDING, RunStatus.FACT_CHECKING)
ORPHAN_GRACE_S = 300


async def startup(ctx: dict) -> None:
    stack = AsyncExitStack()
    ctx["stack"] = stack
    ctx["runtime"] = await stack.enter_async_context(Runtime.open())


async def shutdown(ctx: dict) -> None:
    await ctx["stack"].aclose()


async def _expect(run_id: uuid.UUID, job: str, allowed: tuple[RunStatus, ...]) -> RunStatus | None:
    """Only act on a run in the state this job was queued for; return its status otherwise.

    arq re-delivers a job whose worker died mid-run, possibly much later. Re-running it would restart the run
    from scratch (and spend its budget again) without anyone asking; recovering it is an explicit retry, which
    resumes from the checkpoint. Cancelled runs are skipped the same way.
    """
    async with get_sessionmaker()() as session:
        run = await session.get(Run, run_id)
    if run is None:
        return RunStatus.CANCELLED
    if run.status not in allowed:
        logger.warning("%s skipped for run %s: status is %s", job, run_id, run.status.value)
        return run.status
    return None


async def start_run(ctx: dict, run_id: str, uploads: list[dict] | None = None) -> str:
    rid = uuid.UUID(run_id)
    if skipped := await _expect(rid, "start_run", (RunStatus.QUEUED,)):
        return skipped.value
    files = [Upload(u["filename"], Path(u["path"])) for u in uploads or []]
    runtime: Runtime = ctx["runtime"]
    outcome = await runtime.run_with_auto_resume(await runtime.start(rid, files), rid)
    if outcome.status is RunStatus.COMPLETE:
        await _after_complete(runtime, rid)
    return outcome.status.value


async def _after_complete(runtime: Runtime, run_id: uuid.UUID) -> None:
    """Watchlist runs: diff against the previous report and alert. A failure here never fails the run."""
    try:
        await process_watch_result(runtime.sessionmaker, ModelGateway(runtime.settings), runtime.http, run_id)
    except Exception:
        logger.exception("diff/alert failed for run %s", run_id)
        await RunLog(runtime.sessionmaker, run_id).emit("diff", "error", "change detection failed; see worker log")


async def resume_run(ctx: dict, run_id: str, answer: str) -> str:
    rid = uuid.UUID(run_id)
    if skipped := await _expect(rid, "resume_run", (RunStatus.PLANNING,)):  # the clarify endpoint sets PLANNING
        return skipped.value
    runtime: Runtime = ctx["runtime"]
    outcome = await runtime.run_with_auto_resume(await runtime.resume(rid, answer), rid)
    if outcome.status is RunStatus.COMPLETE:
        await _after_complete(runtime, rid)
    return outcome.status.value


async def retry_run(ctx: dict, run_id: str) -> str:
    """User-requested retry: resume a failed run from its last checkpoint."""
    rid = uuid.UUID(run_id)
    if skipped := await _expect(rid, "retry_run", (RunStatus.QUEUED, RunStatus.FAILED)):  # the endpoint sets QUEUED
        return skipped.value
    runtime: Runtime = ctx["runtime"]
    outcome = await runtime.retry(rid)
    if outcome.status is RunStatus.COMPLETE:
        await _after_complete(runtime, rid)
    return outcome.status.value


async def cancel_stale_clarifications(ctx: dict) -> int:
    s = get_settings()
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=s.clarification_timeout_s)
    maker = get_sessionmaker()
    async with maker() as session:
        stale = (await session.scalars(
            select(Run.id).where(Run.status == RunStatus.AWAITING_CLARIFICATION, Run.clarification_asked_at < cutoff)
        )).all()
        if stale:
            await session.execute(update(Run).where(Run.id.in_(stale)).values(
                status=RunStatus.CANCELLED, finished_at=dt.datetime.now(dt.UTC),
                error="cancelled: the clarifying question was not answered within 30 minutes"))
            await session.commit()
    for rid in stale:
        await RunLog(maker, rid).emit("orchestrator", "progress",
                                      "Run cancelled: the clarifying question was not answered within 30 minutes.")
    return len(stale)


async def fail_orphaned_runs(ctx: dict) -> int:
    """Runs left "in progress" by a process that died (no run-log activity for longer than a whole attempt may
    take) become FAILED, so they stop counting as active and can be retried from their checkpoint. Not resumed
    automatically: nobody is waiting on them, and resuming would spend budget unasked."""
    s = get_settings()
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=s.run_time_budget_s + ORPHAN_GRACE_S)
    maker = get_sessionmaker()
    last_event = select(func.max(RunEvent.created_at)).where(RunEvent.run_id == Run.id).scalar_subquery()
    async with maker() as session:
        orphans = (await session.scalars(select(Run.id).where(
            Run.status.in_(ACTIVE), func.coalesce(last_event, Run.created_at) < cutoff))).all()
        if orphans:
            await session.execute(update(Run).where(Run.id.in_(orphans)).values(
                status=RunStatus.FAILED, finished_at=dt.datetime.now(dt.UTC),
                error="interrupted: no progress (the worker stopped mid-run); retry to resume from the last checkpoint"))
            await session.commit()
    for rid in orphans:
        await RunLog(maker, rid).emit("orchestrator", "error", "Run marked failed: no progress since its worker stopped.")
    return len(orphans)


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    functions = (start_run, resume_run, retry_run)
    cron_jobs = (cron(cancel_stale_clarifications, second=0),  # every minute
                 cron(fail_orphaned_runs, minute=set(range(0, 60, 5)), second=30))
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = redis_settings()
    max_jobs = get_settings().max_concurrent_runs
    job_timeout = get_settings().run_time_budget_s + 120
    keep_result = 3600
    allow_abort_jobs = True
