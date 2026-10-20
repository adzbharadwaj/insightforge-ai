"""FastAPI application (component 02). Routers for research, reports and watchlist are added in Phase 5."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from redis.asyncio import Redis
from sqlalchemy import text

from insightforge.config import get_settings
from insightforge.store.db import get_engine
from insightforge.store.qdrant import VectorStore, make_client

HEALTH_TIMEOUT_S = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.qdrant = make_client()
    app.state.vectors = VectorStore(app.state.qdrant)
    app.state.redis = Redis.from_url(settings.redis_url)
    await app.state.vectors.ensure_collections()
    try:
        yield
    finally:
        await app.state.qdrant.close()
        await app.state.redis.aclose()
        await get_engine().dispose()


app = FastAPI(title="InsightForge AI", version="0.1.0", lifespan=lifespan)


async def _check(coro) -> str:
    try:
        await asyncio.wait_for(coro, HEALTH_TIMEOUT_S)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - report any dependency failure as "down"
        return f"down: {type(exc).__name__}"


async def _ping_postgres() -> None:
    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))


@app.get("/health")
async def health(response: Response) -> dict:
    postgres, redis, qdrant = await asyncio.gather(
        _check(_ping_postgres()),
        _check(app.state.redis.ping()),
        _check(app.state.qdrant.get_collections()),
    )
    checks = {"postgres": postgres, "redis": redis, "qdrant": qdrant}
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": checks}
