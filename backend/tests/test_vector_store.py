import uuid

import pytest

from insightforge.llm.embeddings import embed_documents, embed_query
from insightforge.store.qdrant import DOC_CHUNKS, VectorStore, factcheck_collection, make_client


async def test_embeddings_shape_and_similarity():
    docs = await embed_documents(["Rippling raised a Series F at a $13.5B valuation.", "Bananas are yellow."])
    assert len(docs) == 2 and len(docs[0]) == 384
    q = await embed_query("How much funding did Rippling raise?")

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5)

    assert cos(q, docs[0]) > cos(q, docs[1])


@pytest.fixture
async def store():
    client = make_client()
    try:
        await client.get_collections()
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        await client.close()
        pytest.skip("Qdrant not reachable (run `docker compose up -d`)")
    yield VectorStore(client)
    await client.close()


async def test_search_is_filtered_by_user(store):
    run_id = uuid.uuid4()
    name = factcheck_collection(run_id)
    alice, bob = uuid.uuid4(), uuid.uuid4()
    await store.ensure_collection(name)
    try:
        texts = ["Deel expanded payroll to Vietnam in 2025.", "Deel expanded payroll to Vietnam in 2025."]
        vectors = await embed_documents(texts)
        await store.upsert(name, user_id=alice, vectors=vectors[:1], payloads=[{"text": texts[0]}])
        await store.upsert(name, user_id=bob, vectors=vectors[1:], payloads=[{"text": texts[1]}])

        q = await embed_query("Deel payroll Vietnam")
        hits = await store.search(name, q, user_id=alice, limit=10)
        assert len(hits) == 1 and hits[0].payload["user_id"] == str(alice)

        # Bob cannot resolve Alice's point by ID either.
        assert await store.retrieve(name, [hits[0].id], user_id=bob) == []
    finally:
        await store.drop_collection(name)


async def test_persistent_collections_cannot_be_dropped(store):
    with pytest.raises(ValueError):
        await store.drop_collection(DOC_CHUNKS)
