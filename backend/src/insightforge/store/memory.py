"""Session memory: compressed summaries of past runs in Qdrant `report_summaries`.

The Planner loads only the most recent few (default 3), never full transcripts.
Summaries are written at the end of each run (Phase 4).
"""

import time
import uuid

from qdrant_client import models

from insightforge.llm.embeddings import embed_documents
from insightforge.store.qdrant import REPORT_SUMMARIES, VectorStore


async def load_recent_summaries(store: VectorStore, user_id: uuid.UUID, n: int) -> list[str]:
    points, _ = await store.client.scroll(
        REPORT_SUMMARIES,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))
        ]),
        order_by=models.OrderBy(key="created_at", direction=models.Direction.DESC),
        limit=n,
        with_payload=True,
    )
    return [p.payload["summary"] for p in points if p.payload and p.payload.get("summary")]


async def save_summary(store: VectorStore, user_id: uuid.UUID, run_id: uuid.UUID, summary: str) -> None:
    [vector] = await embed_documents([summary])
    await store.upsert(
        REPORT_SUMMARIES,
        user_id=user_id,
        vectors=[vector],
        payloads=[{"summary": summary, "run_id": str(run_id), "created_at": time.time()}],
        ids=[str(run_id)],
    )
