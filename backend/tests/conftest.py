import uuid

import pytest
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from insightforge.config import get_settings
from insightforge.store.models import Run, User
from insightforge.store.qdrant import VectorStore, make_client


@pytest.fixture
async def db_session():
    """A session inside a transaction that is rolled back after the test.

    Needs `docker compose up -d` and `alembic upgrade head`; skips otherwise.
    """
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as probe:
            await probe.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        await engine.dispose()
        pytest.skip("Postgres not reachable (run `docker compose up -d`)")

    async with engine.connect() as conn:
        trans = await conn.begin()
        session = async_sessionmaker(bind=conn, expire_on_commit=False)()
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
    await engine.dispose()


@pytest.fixture
async def committed_run():
    """A committed user + run for code that opens its own sessions (e.g. ScratchStore).

    Yields (sessionmaker, user_id, run_id); deleting the user afterwards cascades to everything.
    """
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as probe:
            await probe.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        await engine.dispose()
        pytest.skip("Postgres not reachable (run `docker compose up -d`)")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        user = User(email=f"test-{uuid.uuid4().hex[:8]}@example.com")
        session.add(user)
        await session.flush()
        run = Run(user_id=user.id, brief="test brief")
        session.add(run)
        await session.commit()
        user_id, run_id = user.id, run.id
    try:
        yield maker, user_id, run_id
    finally:
        async with maker() as session:
            await session.execute(delete(User).where(User.id == user_id))
            await session.commit()
        await engine.dispose()


@pytest.fixture
async def vectors():
    """A VectorStore on the dev Qdrant with persistent collections ensured; skips if Qdrant is down."""
    client = make_client()
    try:
        await client.get_collections()
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        await client.close()
        pytest.skip("Qdrant not reachable (run `docker compose up -d`)")
    store = VectorStore(client)
    await store.ensure_collections()
    yield store
    await client.close()
