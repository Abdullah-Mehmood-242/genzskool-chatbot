from __future__ import annotations

import httpx

from app.main import create_app
from app.settings import Settings


async def test_health_endpoints() -> None:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    transport = httpx.ASGITransport(app=app)
    # readiness waits for the startup phase of auto-ingest, which runs in the lifespan
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        health = await client.get("/healthz")
        ready = await client.get("/readyz")
        metrics = await client.get("/metrics")
    assert health.status_code == 200
    assert health.json()["service"] == "ingestion"
    assert ready.status_code == 200
    assert metrics.status_code == 200
