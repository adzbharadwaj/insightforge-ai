"""Live progress stream for one run (Server-Sent Events).

Tails the run log in Postgres (written by the worker, so it works across processes)
and sends:

- `snapshot`  run status + every sub-task row (sent first, then whenever it changes)
- `log`       one run-log event (agent, kind, message); its id is the event id, so a
              reconnecting client resumes with `Last-Event-ID` and misses nothing
- `end`       the final snapshot when the run is COMPLETE / FAILED / CANCELLED

Progress only: the report is never streamed. The client fetches it from /reports once
the final snapshot carries a report_id, i.e. after grounding and fact-checking.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.api.views import TERMINAL, run_snapshot
from insightforge.store.models import Run, RunEvent

POLL_S = 0.75
BATCH = 200


def _log_event(e: RunEvent) -> dict:
    payload = {"agent": e.agent, "kind": e.kind, "message": e.message, "at": e.created_at.isoformat()}
    return {"event": "log", "id": str(e.id), "data": json.dumps(payload)}


async def run_event_stream(
    sessionmaker: async_sessionmaker[AsyncSession],
    run: Run,
    last_event_id: int = 0,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    poll_s: float = POLL_S,
) -> AsyncIterator[dict]:
    last_id = last_event_id
    last_snapshot: str | None = None
    while True:
        if is_disconnected and await is_disconnected():
            return
        async with sessionmaker() as session:
            rows = (await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run.id, RunEvent.id > last_id)
                .order_by(RunEvent.id).limit(BATCH))).all()
        for e in rows:
            last_id = e.id
            yield _log_event(e)

        snap = await run_snapshot(sessionmaker, run)
        encoded = snap.model_dump_json(exclude={"elapsed_s"})  # elapsed changes every poll; not a real change
        if encoded != last_snapshot:
            last_snapshot = encoded
            yield {"event": "snapshot", "data": snap.model_dump_json()}
        if snap.status in TERMINAL and len(rows) < BATCH:
            yield {"event": "end", "data": snap.model_dump_json()}
            return
        if len(rows) < BATCH:
            await asyncio.sleep(poll_s)
