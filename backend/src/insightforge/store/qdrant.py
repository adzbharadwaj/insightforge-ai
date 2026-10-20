"""Qdrant vector store (component 18).

Collections:
- doc_chunks        Docling chunks of user PDFs
- entities          entity embeddings, reused across runs
- report_summaries  compressed session summaries (Planner loads the last 3)
- factcheck_{run}   temporary, re-fetched source pages; deleted after the run

Every point carries `user_id`, and every search requires it: a caller cannot
query without the ACL filter.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from insightforge.config import get_settings

DOC_CHUNKS = "doc_chunks"
ENTITIES = "entities"
REPORT_SUMMARIES = "report_summaries"
PERSISTENT_COLLECTIONS = (DOC_CHUNKS, ENTITIES, REPORT_SUMMARIES)


def factcheck_collection(run_id: uuid.UUID | str) -> str:
    return f"factcheck_{str(run_id).replace('-', '')}"


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


def make_client() -> AsyncQdrantClient:
    s = get_settings()
    api_key = s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None
    return AsyncQdrantClient(url=s.qdrant_url, api_key=api_key)


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, dim: int | None = None):
        self.client = client
        self.dim = dim or get_settings().embedding_dim

    async def ensure_collection(self, name: str) -> None:
        if not await self.client.collection_exists(name):
            await self.client.create_collection(
                name,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
        # Index creation is idempotent, so collections created by older code get new indexes too.
        indexes = {"user_id": models.PayloadSchemaType.KEYWORD, "run_id": models.PayloadSchemaType.KEYWORD}
        if name.startswith("factcheck_"):
            indexes["source_id"] = models.PayloadSchemaType.KEYWORD
        if name == REPORT_SUMMARIES:  # memory loads "most recent first", which needs an ordered index
            indexes["created_at"] = models.PayloadSchemaType.FLOAT
        for field, schema in indexes.items():
            await self.client.create_payload_index(name, field, schema)

    async def ensure_collections(self) -> None:
        for name in PERSISTENT_COLLECTIONS:
            await self.ensure_collection(name)

    async def upsert(
        self,
        collection: str,
        *,
        user_id: uuid.UUID | str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")
        ids = ids or [str(uuid.uuid4()) for _ in vectors]
        points = [
            models.PointStruct(id=pid, vector=vec, payload={**payload, "user_id": str(user_id)})
            for pid, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        await self.client.upsert(collection, points=points, wait=True)
        return ids

    async def search(
        self,
        collection: str,
        vector: list[float],
        *,
        user_id: uuid.UUID | str,
        limit: int = 8,
        match: dict[str, str] | None = None,
        match_any: dict[str, list[str]] | None = None,
        score_threshold: float | None = None,
    ) -> list[Hit]:
        must = [models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
        for key, value in (match or {}).items():
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        for key, values in (match_any or {}).items():
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
        result = await self.client.query_points(
            collection,
            query=vector,
            query_filter=models.Filter(must=must),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [Hit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in result.points]

    async def retrieve(self, collection: str, ids: list[str], *, user_id: uuid.UUID | str) -> list[Hit]:
        """Resolve point IDs (e.g. chunk IDs) to payloads, dropping any the user does not own."""
        points = await self.client.retrieve(collection, ids=ids, with_payload=True)
        return [
            Hit(id=str(p.id), score=1.0, payload=p.payload or {})
            for p in points
            if (p.payload or {}).get("user_id") == str(user_id)
        ]

    async def drop_collection(self, name: str) -> None:
        if name in PERSISTENT_COLLECTIONS:
            raise ValueError(f"refusing to drop persistent collection {name!r}")
        await self.client.delete_collection(name)
