"""Middleware edge cases: chunked bodies, streaming responses, lifespan scopes, metrics labels."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from common.logging import REQUEST_ID_HEADER, get_request_id
from common.observability import create_service_app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


def _app(**kw: object) -> FastAPI:
    app = create_service_app(service="obs", version="1", max_body_bytes=50, **kw)  # type: ignore[arg-type]

    @app.post("/raw")
    async def raw(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen() -> AsyncIterator[str]:
            for i in range(3):
                yield f"data: {i}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/rid")
    async def rid() -> dict[str, str | None]:
        return {"rid": get_request_id()}

    return app


async def _chunks(total: int) -> AsyncIterator[bytes]:
    for _ in range(total // 10):
        yield b"x" * 10


async def test_chunked_body_over_limit_is_rejected_without_content_length() -> None:
    async with _client(_app()) as c:
        r = await c.post("/raw", content=_chunks(200))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"
    assert REQUEST_ID_HEADER in r.headers


async def test_chunked_body_under_limit_passes() -> None:
    async with _client(_app()) as c:
        r = await c.post("/raw", content=_chunks(40))
    assert r.status_code == 200
    assert r.json() == {"size": 40}


async def test_streaming_response_passes_through_with_request_id() -> None:
    async with _client(_app()) as c:
        r = await c.get("/stream", headers={REQUEST_ID_HEADER: "stream-req-0001"})
    assert r.status_code == 200
    assert r.text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
    assert r.headers[REQUEST_ID_HEADER] == "stream-req-0001"


async def test_request_id_is_visible_to_handlers() -> None:
    async with _client(_app()) as c:
        r = await c.get("/rid", headers={REQUEST_ID_HEADER: "handler-rid-01"})
    assert r.json() == {"rid": "handler-rid-01"}


async def test_lifespan_scope_passes_through_middleware() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        events.append("start")
        yield
        events.append("stop")

    app = _app(lifespan=lifespan)
    async with app.router.lifespan_context(app), _client(app) as c:
        assert (await c.get("/healthz")).status_code == 200
    assert events == ["start", "stop"]


async def test_metrics_use_route_templates_not_raw_paths() -> None:
    app = _app()

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    async with _client(app) as c:
        await c.get("/items/123")
        await c.get("/items/456")
        await c.get("/nope/xyz")
        text = (await c.get("/metrics")).text
    assert 'route="/items/{item_id}"' in text
    assert "/items/123" not in text
    assert 'route="unmatched"' in text


async def test_docs_can_be_disabled() -> None:
    app = create_service_app(service="obs", version="1", docs=False)
    async with _client(app) as c:
        assert (await c.get("/docs")).status_code == 404


async def test_non_http_scopes_bypass_middleware_logic() -> None:
    from common.observability import RequestContextMiddleware, ServiceMetrics

    seen: list[str] = []

    async def inner(scope: dict[str, object], receive: object, send: object) -> None:
        seen.append(str(scope["type"]))

    mw = RequestContextMiddleware(inner, metrics=ServiceMetrics("x"), max_body_bytes=1)  # type: ignore[arg-type]
    await mw({"type": "websocket"}, None, None)  # type: ignore[arg-type]
    assert seen == ["websocket"]
