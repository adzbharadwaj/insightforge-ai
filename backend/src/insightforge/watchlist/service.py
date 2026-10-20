"""Watchlist service: start a run for a watched topic, and turn a finished run into an alert.

Scheduled runs use the same queue and pipeline as a manual brief; only the trigger differs.
After a watchlist run completes, the worker calls `process_watch_result`: the Diff Agent
compares it with the previous completed report for the same item, and a material change
becomes an in-app alert plus (optionally) a signed webhook POST.
"""

import datetime as dt
import logging
import uuid
from typing import Any, Protocol

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.agents.diff import ChangeSummary, diff_claims
from insightforge.config import Settings, get_settings
from insightforge.guardrails.webhooks import deliver, is_slack, slack_payload
from insightforge.llm.gateway import ModelGateway
from insightforge.store.models import Alert, Claim, Report, Run, RunStatus, WatchlistItem
from insightforge.store.run_log import RunLog

logger = logging.getLogger(__name__)
WATCH_PREFS = {"web": True, "docs": False, "memory": True}
SUMMARY_SECTION = "Executive Summary"


class Queue(Protocol):
    async def enqueue_job(self, function: str, *args: Any, _job_id: str | None = None) -> Any: ...


def interval(cadence: str, settings: Settings | None = None) -> dt.timedelta:
    s = settings or get_settings()
    if s.watchlist_interval_override_s:  # demos and E2E tests: e.g. every 60 s
        return dt.timedelta(seconds=s.watchlist_interval_override_s)
    return dt.timedelta(days=1 if cadence == "daily" else 7)


async def start_watch_run(sessionmaker: async_sessionmaker[AsyncSession], queue: Queue,
                          item_id: uuid.UUID) -> uuid.UUID | None:
    """Create and queue a run for one watchlist item. Skips inactive items and items with a run in flight."""
    now = dt.datetime.now(dt.UTC)
    async with sessionmaker() as session:
        item = await session.get(WatchlistItem, item_id)
        if item is None or not item.active:
            return None
        busy = await session.scalar(select(Run.id).where(
            Run.watchlist_item_id == item.id,
            Run.status.not_in([RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED])).limit(1))
        if busy:
            return None  # never stack runs for one item (the scheduler also uses max_instances=1)
        run = Run(user_id=item.user_id, brief=item.topic, source_prefs=WATCH_PREFS, watchlist_item_id=item.id)
        session.add(run)
        item.last_run = now
        item.next_run = now + interval(item.cadence.value)
        await session.commit()
    await RunLog(sessionmaker, run.id).progress("scheduler", "watchlist run queued", watchlist_item_id=str(item_id))
    await queue.enqueue_job("start_run", str(run.id), [], _job_id=f"run:{run.id}")
    return run.id


async def _claims(session: AsyncSession, report_id: uuid.UUID) -> list[str]:
    rows = await session.scalars(select(Claim.text).where(Claim.report_id == report_id,
                                                          Claim.section != SUMMARY_SECTION))
    return list(rows)


async def process_watch_result(
    sessionmaker: async_sessionmaker[AsyncSession],
    gateway: ModelGateway,
    http: httpx.AsyncClient,
    run_id: uuid.UUID,
) -> Alert | None:
    async with sessionmaker() as session:
        run = await session.get(Run, run_id)
        if run is None or run.watchlist_item_id is None or run.status is not RunStatus.COMPLETE:
            return None
        item = await session.get(WatchlistItem, run.watchlist_item_id)
        new_report = await session.scalar(select(Report).where(Report.run_id == run.id))
        prev_report = await session.scalar(
            select(Report).join(Run, Run.id == Report.run_id)
            .where(Run.watchlist_item_id == item.id, Run.status == RunStatus.COMPLETE, Report.run_id != run.id)
            .order_by(Report.created_at.desc()).limit(1))
        if item is None or new_report is None:
            return None
        new_claims = await _claims(session, new_report.id)
        old_claims = await _claims(session, prev_report.id) if prev_report else []

    log = RunLog(sessionmaker, run.id)
    summary: ChangeSummary = await diff_claims(gateway, old_claims, new_claims)
    if prev_report is None:
        await log.progress("diff", "first report for this watchlist item: saved as the baseline")
        return None
    counts = (len(summary.added_findings), len(summary.changed_findings), len(summary.removed_findings))
    await log.progress("diff", f"diff vs previous report: {counts[0]} added, {counts[1]} changed, {counts[2]} removed",
                       previous_report_id=str(prev_report.id))
    if not summary.alertable:
        return None

    alert = Alert(user_id=run.user_id, watchlist_item_id=item.id, run_id=run.id,
                  change_summary=summary.model_dump(mode="json"))
    async with sessionmaker() as session:
        session.add(alert)
        await session.commit()
    if item.webhook_url:
        link = f"{get_settings().app_base_url}/report/{new_report.id}"
        data = summary.model_dump(mode="json")
        payload = (slack_payload(item.topic, data, link) if is_slack(item.webhook_url) else
                   {"event": "watchlist.changed", "topic": item.topic, "watchlist_item_id": str(item.id),
                    "report_id": str(new_report.id), "report_url": link, "change_summary": data})
        result = await deliver(http, item.webhook_url, payload)
        async with sessionmaker() as session:
            await session.execute(update(Alert).where(Alert.id == alert.id).values(webhook_status=result.status))
            await session.commit()
        alert.webhook_status = result.status
        await log.emit("diff", "progress" if result.status == "sent" else "error",
                       f"webhook {result.status} after {result.attempts} attempt(s)", detail=result.detail)
    return alert
