"""Scratch store (component 17): keyed storage for raw fetched content.

Researchers write full page bodies and PDF chunk text here and hand back only
the `source_id`. Downstream agents pull content on demand with `read_source`,
so the corpus never sits in a prompt or in conversation history.
"""

import hashlib
import uuid
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.graph.state import SourceRef
from insightforge.llm.tokens import count_tokens
from insightforge.store.models import ScratchSource, SourceKind


def make_source_id(run_id: uuid.UUID, key: str) -> str:
    """Stable per run: the same URL fetched by two sub-tasks maps to one source."""
    return "src-" + hashlib.sha256(f"{run_id}:{key}".encode()).hexdigest()[:12]


class ScratchStore:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], run_id: uuid.UUID):
        self._sessionmaker = sessionmaker
        self.run_id = run_id

    async def put(
        self,
        *,
        kind: SourceKind,
        key: str,
        body_text: str,
        summary: str | None = None,
        sub_task_key: str | None = None,
        url: str | None = None,
        title: str | None = None,
        published_date: date | None = None,
        credibility: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Store one source and return its id. Re-putting the same key updates it in place."""
        source_id = make_source_id(self.run_id, key)
        values = {
            "run_id": self.run_id,
            "source_id": source_id,
            "kind": kind,
            "sub_task_key": sub_task_key,
            "url": url,
            "title": title,
            "published_date": published_date,
            "body_text": body_text,
            "summary": summary,
            "credibility": credibility,
            "token_count": count_tokens(body_text),
            "meta": meta or {},
        }
        stmt = insert(ScratchSource).values(**values)
        update_cols = {k: stmt.excluded[k] for k in values if k not in ("run_id", "source_id")}
        stmt = stmt.on_conflict_do_update(index_elements=["run_id", "source_id"], set_=update_cols)
        async with self._sessionmaker() as session:
            await session.execute(stmt)
            await session.commit()
        return source_id

    async def get(self, source_id: str) -> ScratchSource | None:
        async with self._sessionmaker() as session:
            return await session.get(ScratchSource, (self.run_id, source_id))

    async def get_many(self, source_ids: list[str]) -> list[ScratchSource]:
        if not source_ids:
            return []
        async with self._sessionmaker() as session:
            rows = await session.scalars(
                select(ScratchSource).where(
                    ScratchSource.run_id == self.run_id, ScratchSource.source_id.in_(source_ids)
                )
            )
            by_id = {r.source_id: r for r in rows}
        return [by_id[i] for i in source_ids if i in by_id]

    async def total_tokens(self) -> int:
        """Size of the whole corpus for this run (used in the context-engineering writeup)."""
        async with self._sessionmaker() as session:
            rows = await session.scalars(
                select(ScratchSource.token_count).where(ScratchSource.run_id == self.run_id)
            )
            return sum(rows)

    @staticmethod
    def to_ref(row: ScratchSource) -> SourceRef:
        return SourceRef(
            source_id=row.source_id,
            kind=row.kind,
            summary=(row.summary or row.title or "")[:300],
            credibility=row.credibility,
        )
