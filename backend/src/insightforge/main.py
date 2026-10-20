"""FastAPI application (component 02): validates, queues and serves. Runs execute in the worker.

    uv run uvicorn insightforge.main:app --reload     API on :8000 (docs at /docs)
    uv run arq insightforge.worker.WorkerSettings     run worker (separate process)
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from arq import create_pool
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import text

from insightforge.api import reports, research, users, watchlist
from insightforge.config import get_settings, require_secret
from insightforge.llm.gateway import ModelGateway
from insightforge.store.db import get_engine, get_sessionmaker
from insightforge.store.qdrant import VectorStore, make_client
from insightforge.worker import redis_settings

HEALTH_TIMEOUT_S = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    require_secret(settings.jwt_secret, "JWT_SECRET")  # fail at startup, not on the first login
    app.state.qdrant = make_client()
    app.state.vectors = VectorStore(app.state.qdrant)
    app.state.redis = Redis.from_url(settings.redis_url)
    app.state.sessionmaker = get_sessionmaker()
    app.state.queue = await create_pool(redis_settings())
    # The pre-run scope check is not part of any run, so it has no run budget (no meter).
    app.state.guardrail_gateway = ModelGateway(settings)
    await app.state.vectors.ensure_collections()
    try:
        yield
    finally:
        await app.state.queue.aclose()
        await app.state.qdrant.close()
        await app.state.redis.aclose()
        await get_engine().dispose()


app = FastAPI(title="InsightForge AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[get_settings().app_base_url], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(users.router)
app.include_router(research.router)
app.include_router(reports.router)
app.include_router(watchlist.router)


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
