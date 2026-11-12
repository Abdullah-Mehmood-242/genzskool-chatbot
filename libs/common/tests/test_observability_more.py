"""Request-id validation, access logging, metrics accounting, readiness aggregation, isolation."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI

from common.logging import REQUEST_ID_HEADER, get_request_id
from common.observability import create_service_app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


def _app(**kw: object) -> FastAPI:
    app = create_service_app(service="o", version="2.0", **kw)  # type: ignore[arg-type]

    @app.get("/work")
    async def work() -> dict[str, str | None]:
        return {"rid": get_request_id()}

    @app.get("/crash")
    async def crash() -> None:
        raise RuntimeError("x")

    return app


@pytest.mark.parametrize(
    ("incoming", "kept"),
    [
        ("a" * 7, False),
        ("a" * 8, True),
        ("a" * 64, True),
        ("a" * 65, False),
        ("abc-123-def-456", True),
        ("abc_123_def", False),
        ("abc 123 def", False),
        ("", False),
        ("ąęśćżźółń", False),
    ],
)
async def test_request_id_acceptance_rules(incoming: str, kept: bool) -> None:
    async with _client(_app()) as c:
        r = await c.get("/work", headers=[(REQUEST_ID_HEADER.encode(), incoming.encode("utf-8"))])
    rid = r.headers[REQUEST_ID_HEADER]
    assert (rid == incoming) is kept
    assert r.json()["rid"] == rid


async def test_request_ids_do_not_leak_between_requests() -> None:
    async with _client(_app()) as c:
        first = (await c.get("/work")).headers[REQUEST_ID_HEADER]
        second = (await c.get("/work")).headers[REQUEST_ID_HEADER]
    assert first != second


async def test_concurrent_requests_keep_their_own_request_ids() -> None:
    async with _client(_app()) as c:
        ids = [f"concurrent-{i:04d}" for i in range(20)]
        responses = await asyncio.gather(*(c.get("/work", headers={REQUEST_ID_HEADER: i}) for i in ids))
    assert [r.json()["rid"] for r in responses] == ids


async def test_access_log_for_business_routes_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="access"):
        async with _client(_app()) as c:
            await c.get("/healthz")
            await c.get("/readyz")
            await c.get("/metrics")
            await c.get("/work")
    access = [r for r in caplog.records if r.name == "access"]
    assert len(access) == 1
    rec = access[0]
    assert rec.method == "GET"  # type: ignore[attr-defined]
    assert rec.path == "/work"  # type: ignore[attr-defined]
    assert rec.status == 200  # type: ignore[attr-defined]
    assert rec.latency_ms >= 0  # type: ignore[attr-defined]


async def test_metrics_count_statuses_including_errors_and_413() -> None:
    app = _app(max_body_bytes=10)
    async with _client(app) as c:
        await c.get("/work")
        await c.get("/crash")
        await c.post("/work", content=b"x" * 100)
        text = (await c.get("/metrics")).text
    assert 'route="/work",service="o",status="200"' in text
    assert 'route="/crash",service="o",status="500"' in text
    assert 'status="413"' in text
    assert "http_request_duration_seconds_bucket" in text


async def test_non_numeric_content_length_is_ignored_by_precheck() -> None:
    app = _app(max_body_bytes=10)
    async with _client(app) as c:
        r = await c.get("/work", headers={"Content-Length": "abc"})
    assert r.status_code != 413


async def test_apps_have_isolated_metric_registries() -> None:
    a, b = _app(), _app()
    async with _client(a) as ca, _client(b) as cb:
        await ca.get("/work")
        await ca.get("/work")
        text_b = (await cb.get("/metrics")).text
    assert 'route="/work"' not in text_b


async def test_readiness_aggregation() -> None:
    async def yes() -> bool:
        return True

    async def no() -> bool:
        return False

    app = create_service_app(service="o", version="2.0", readiness_checks={"a": yes, "b": no})
    async with _client(app) as c:
        r = await c.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "service": "o", "version": "2.0", "checks": {"a": True, "b": False}}


async def test_readiness_checks_can_be_added_after_creation() -> None:
    app = create_service_app(service="o", version="2.0")

    async def no() -> bool:
        return False

    app.state.readiness_checks["late"] = no
    async with _client(app) as c:
        assert (await c.get("/readyz")).status_code == 503


async def test_health_is_ok_even_when_not_ready() -> None:
    async def no() -> bool:
        return False

    app = create_service_app(service="o", version="2.0", readiness_checks={"x": no})
    async with _client(app) as c:
        assert (await c.get("/healthz")).status_code == 200


async def test_openapi_metadata() -> None:
    async with _client(_app()) as c:
        spec = (await c.get("/openapi.json")).json()
    assert spec["info"] == {"title": "genz-o", "version": "2.0"}
    assert "/healthz" in spec["paths"]
    assert "/readyz" in spec["paths"]
    assert "/metrics" not in spec["paths"]


@pytest.mark.parametrize(
    "raw_header",
    [b"\xff\xfe\xfd-invalid-utf8", b"\xc3\x28-bad-continuation", b"abc\x00defgh12"],
)
async def test_invalid_header_bytes_never_crash(raw_header: bytes) -> None:
    async with _client(_app()) as c:
        r = await c.get("/work", headers=[(REQUEST_ID_HEADER.encode(), raw_header), (b"x-other", raw_header)])
    assert r.status_code == 200
    assert r.headers[REQUEST_ID_HEADER].isascii()
    assert len(r.headers[REQUEST_ID_HEADER]) == 32


@pytest.mark.parametrize("value", [b"\xb2", b"\xb9\xb2\xb3", b"1e5", b"-5", b" 12", b"9" * 40])
async def test_odd_content_length_values_never_crash(value: bytes) -> None:
    app = _app(max_body_bytes=10)

    async def call() -> int:
        # Raw ASGI call: httpx would refuse to send a malformed Content-Length itself.
        statuses: list[int] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.start":
                statuses.append(int(message["status"]))  # type: ignore[call-overload]

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/work",
            "raw_path": b"/work",
            "query_string": b"",
            "headers": [(b"content-length", value)],
            "http_version": "1.1",
            "scheme": "http",
            "server": ("t", 80),
            "client": ("c", 1),
            "root_path": "",
            "asgi": {"version": "3.0"},
        }
        await app(scope, receive, send)  # type: ignore[arg-type]
        return statuses[0]

    status = await call()
    # Only plain ASCII digits are parsed; an absurdly large declared size is rejected up front.
    assert status == (413 if value == b"9" * 40 else 200)
