from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from common.errors import NotFound, ProviderError
from common.logging import REQUEST_ID_HEADER
from common.observability import create_service_app


class Body(BaseModel):
    text: str


def _app(ready: bool = True) -> FastAPI:
    async def check() -> bool:
        return ready

    async def boom() -> bool:
        raise RuntimeError("db down")

    app = create_service_app(
        service="t",
        version="9.9.9",
        readiness_checks={"dep": check, "flaky": boom} if not ready else {"dep": check},
        max_body_bytes=100,
    )

    @app.get("/missing")
    async def missing() -> None:
        raise NotFound("no such doc")

    @app.get("/provider")
    async def provider() -> None:
        raise ProviderError(details={"provider": "fake"})

    @app.get("/crash")
    async def crash() -> None:
        raise RuntimeError("secret internals at /etc/passwd")

    @app.post("/echo")
    async def echo(body: Body) -> Body:
        return body

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


async def test_healthz_and_request_id_generated() -> None:
    async with _client(_app()) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "t", "version": "9.9.9", "checks": {}}
    assert len(r.headers[REQUEST_ID_HEADER]) == 32


async def test_request_id_is_propagated() -> None:
    async with _client(_app()) as c:
        r = await c.get("/healthz", headers={REQUEST_ID_HEADER: "abc12345-def"})
    assert r.headers[REQUEST_ID_HEADER] == "abc12345-def"


async def test_invalid_request_id_is_replaced() -> None:
    async with _client(_app()) as c:
        r = await c.get("/healthz", headers={REQUEST_ID_HEADER: "<script>"})
    assert r.headers[REQUEST_ID_HEADER] != "<script>"


@pytest.mark.parametrize(("ready", "status"), [(True, 200), (False, 503)])
async def test_readyz(ready: bool, status: int) -> None:
    async with _client(_app(ready)) as c:
        r = await c.get("/readyz")
    assert r.status_code == status
    if not ready:
        assert r.json()["checks"] == {"dep": False, "flaky": False}


async def test_app_error_schema() -> None:
    async with _client(_app()) as c:
        r = await c.get("/missing")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "not_found"
    assert body["message"] == "no such doc"
    assert body["request_id"] == r.headers[REQUEST_ID_HEADER]


async def test_provider_error_is_502() -> None:
    async with _client(_app()) as c:
        r = await c.get("/provider")
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "provider_error"


async def test_unhandled_error_hides_internals() -> None:
    async with _client(_app()) as c:
        r = await c.get("/crash")
    assert r.status_code == 500
    assert "passwd" not in r.text
    assert "Traceback" not in r.text
    assert r.json()["error"]["code"] == "internal_error"


async def test_validation_error_does_not_echo_input() -> None:
    async with _client(_app()) as c:
        r = await c.post("/echo", json={"wrong": "sk-SUPERSECRET"})
    assert r.status_code == 422
    assert "SUPERSECRET" not in r.text
    assert r.json()["error"]["code"] == "validation_error"


async def test_unknown_route_uses_error_schema() -> None:
    async with _client(_app()) as c:
        r = await c.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_body_size_limit_by_header() -> None:
    async with _client(_app()) as c:
        r = await c.post("/echo", json={"text": "x" * 500})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


async def test_metrics_exposed() -> None:
    async with _client(_app()) as c:
        await c.get("/healthz")
        r = await c.get("/metrics")
    assert r.status_code == 200
    assert 'http_requests_total{method="GET",route="/healthz",service="t",status="200"}' in r.text
