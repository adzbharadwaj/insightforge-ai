"""Run log: append-only events per run (Postgres `run_events`).

Feeds the live activity stream in the UI (Phase 5 adds Redis pub/sub on top)
and is the audit trail for compression cuts, skipped URLs, retries and errors.
"""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.store.models import RunEvent


class RunLog:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], run_id: uuid.UUID):
        self._sessionmaker = sessionmaker
        self.run_id = run_id

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        return self._sessionmaker

    async def emit(self, agent: str, kind: str, message: str, **data: Any) -> None:
        async with self._sessionmaker() as session:
            session.add(RunEvent(run_id=self.run_id, agent=agent, kind=kind, message=message, data=data))
            await session.commit()

    async def progress(self, agent: str, message: str, **data: Any) -> None:
        await self.emit(agent, "progress", message, **data)
