"""Watchlist scheduler (component 21): its own process, never inside the API's web workers.

    uv run insightforge-scheduler

- APScheduler with a SQLAlchemy job store on Postgres, so schedules survive restarts.
- One job per active watchlist item (`watch:<id>`), interval = cadence (daily / weekly).
  `max_instances=1` and `coalesce=True`: a slow run is never doubled up, and runs missed
  while the scheduler was down collapse into one (fired if within the misfire grace).
- A reconcile pass every minute syncs jobs with the watchlist table: the API only writes
  rows; it never talks to the scheduler. New items run right away to set the baseline.
- A job only enqueues `start_run` on the same arq queue as manual briefs: same pipeline.
"""

import asyncio
import datetime as dt
import logging
import signal
import uuid

from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from arq import create_pool
from sqlalchemy import select, update

from insightforge.config import Settings, get_settings
from insightforge.store.db import get_sessionmaker
from insightforge.store.models import WatchlistItem
from insightforge.watchlist.service import interval, start_watch_run
from insightforge.worker import redis_settings

logger = logging.getLogger(__name__)
JOB_PREFIX = "watch:"
JOB_TABLE = "apscheduler_jobs"
_queue = None  # set in main(); jobs are stored by reference, so they reach shared clients via the module


def jobstore_url(settings: Settings) -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)


async def fire(item_id: str) -> None:
    """The scheduled job: queue one run for a watchlist item."""
    run_id = await start_watch_run(get_sessionmaker(), _queue, uuid.UUID(item_id))
    logger.info("watchlist %s: %s", item_id, f"queued run {run_id}" if run_id else "skipped (inactive or busy)")


async def reconcile(scheduler: AsyncIOScheduler) -> None:
    maker = get_sessionmaker()
    now = dt.datetime.now(dt.UTC)
    async with maker() as session:
        items = (await session.scalars(select(WatchlistItem).where(WatchlistItem.active.is_(True)))).all()
    wanted = {f"{JOB_PREFIX}{i.id}": i for i in items}

    for job in scheduler.get_jobs(jobstore="default"):
        if job.id.startswith(JOB_PREFIX) and job.id not in wanted:
            job.remove()  # deleted or paused in the UI
    for job_id, item in wanted.items():
        every = interval(item.cadence.value)
        job = scheduler.get_job(job_id, jobstore="default")
        if job is not None and getattr(job.trigger, "interval", None) == every:
            continue
        # Never run → fire right away (baseline). Otherwise anchor to the last run, so changing the cadence (or
        # restarting the scheduler) doesn't trigger an extra run.
        soon = now + dt.timedelta(seconds=2)
        first = soon if item.last_run is None else max(soon, item.last_run + every)
        scheduler.add_job(fire, IntervalTrigger(seconds=int(every.total_seconds()), start_date=first),
                          args=[str(item.id)], id=job_id, jobstore="default", replace_existing=True)

    async with maker() as session:  # mirror next run times for the dashboard
        for job in scheduler.get_jobs(jobstore="default"):
            if job.id in wanted and job.next_run_time:
                await session.execute(update(WatchlistItem).where(WatchlistItem.id == wanted[job.id].id)
                                      .values(next_run=job.next_run_time))
        await session.commit()


def build_scheduler(settings: Settings) -> AsyncIOScheduler:
    return AsyncIOScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=jobstore_url(settings), tablename=JOB_TABLE),
                   "local": MemoryJobStore()},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": settings.scheduler_misfire_grace_s},
        timezone=dt.UTC,
    )


async def run() -> None:
    global _queue
    settings = get_settings()
    _queue = await create_pool(redis_settings())
    scheduler = build_scheduler(settings)
    scheduler.start()
    scheduler.add_job(reconcile, IntervalTrigger(seconds=settings.scheduler_reconcile_s), args=[scheduler],
                      id="reconcile", jobstore="local", next_run_time=dt.datetime.now(dt.UTC))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    logger.info("scheduler started (reconcile every %ss)", settings.scheduler_reconcile_s)
    await stop.wait()
    scheduler.shutdown(wait=False)
    await _queue.aclose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run())
