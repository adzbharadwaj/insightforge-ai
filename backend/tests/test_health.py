import httpx
import pytest
from asgi_lifespan import LifespanManager

from insightforge.main import app


@pytest.fixture
async def client():
    try:
        async with LifespanManager(app) as manager:
            transport = httpx.ASGITransport(app=manager.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                yield c
    except OSError:
        pytest.skip("infrastructure not reachable (run `docker compose up -d`)")


async def test_health_reports_every_dependency(client):
    r = await client.get("/health")
    body = r.json()
    assert set(body["checks"]) == {"postgres", "redis", "qdrant"}
    assert r.status_code == 200, body
    assert body["status"] == "ok"
