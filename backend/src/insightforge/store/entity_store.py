"""Entity persistence: the run's graph in Postgres, entity embeddings in Qdrant `entities`.

Embeddings are keyed by (user, entity key), so the same company seen in a later run
updates one point instead of piling up duplicates. The Planner retrieves related
past entities by embedding the brief; only short fact strings travel, never sources.
"""

import time
import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.graph.state import EntityGraph, EntityType, Relationship
from insightforge.llm.embeddings import embed_documents, embed_query
from insightforge.store.models import Run
from insightforge.store.qdrant import ENTITIES, VectorStore

_ENTITY_NS = uuid.UUID("0b9f3a52-7c1e-4d0a-8f6b-3e2a9c5d1f47")
# Money, dates and rounds are attributes of other entities, not useful to recall on their own.
RECALLED_TYPES = frozenset({EntityType.COMPANY, EntityType.PERSON, EntityType.PRODUCT})
MAX_FACTS = 5


def describe(r: Relationship) -> str:
    fact = f"{r.predicate.value.replace('_', ' ')} {r.object}"
    if r.qualifier:
        fact += f" ({r.qualifier})"
    if r.date:
        fact += f" [{r.date.isoformat()}]"
    return fact + (" [conflicting reports]" if r.conflict else "")


async def save_entity_graph(sessionmaker: async_sessionmaker[AsyncSession], run_id: uuid.UUID,
                            graph: EntityGraph) -> None:
    async with sessionmaker() as session:
        await session.execute(update(Run).where(Run.id == run_id).values(entity_graph=graph.model_dump(mode="json")))
        await session.commit()


async def upsert_entity_embeddings(vectors: VectorStore, user_id: uuid.UUID, run_id: uuid.UUID,
                                   graph: EntityGraph) -> int:
    facts: dict[str, list[str]] = {}
    for r in graph.relationships:
        facts.setdefault(r.subject, []).append(describe(r))
    entities = [e for e in graph.entities if e.type in RECALLED_TYPES]
    if not entities:
        return 0
    texts = [f"{e.name} ({e.type.value}): " + "; ".join(facts.get(e.key, [])[:MAX_FACTS]) for e in entities]
    payloads = [
        {"key": e.key, "name": e.name, "type": e.type.value, "run_id": str(run_id),
         "facts": facts.get(e.key, [])[:MAX_FACTS], "updated_at": time.time()}
        for e in entities
    ]
    ids = [str(uuid.uuid5(_ENTITY_NS, f"{user_id}:{e.key}")) for e in entities]
    await vectors.upsert(ENTITIES, user_id=user_id, vectors=await embed_documents(texts), payloads=payloads, ids=ids)
    return len(entities)


async def load_related_entities(vectors: VectorStore, user_id: uuid.UUID, query: str, n: int,
                                min_score: float = 0.55) -> list[str]:
    """Short lines like "Salary Hero (company): raised US$7.6M (series a)" for the Planner prompt."""
    hits = await vectors.search(ENTITIES, await embed_query(query), user_id=user_id, limit=n, score_threshold=min_score)
    lines = []
    for h in hits:
        facts = "; ".join(h.payload.get("facts", [])[:3])
        lines.append(f"{h.payload.get('name')} ({h.payload.get('type')})" + (f": {facts}" if facts else ""))
    return lines
