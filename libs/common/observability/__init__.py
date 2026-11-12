"""Service app factory: request-id + access-log + metrics middleware, health endpoints, error handlers."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from common.contracts.errors import HealthStatus
from common.errors.http import error_response, install_error_handlers
from common.logging.context import REQUEST_ID_HEADER, new_request_id, set_request_id

ReadinessCheck = Callable[[], Awaitable[bool]]
access_logger = logging.getLogger("access")
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9-]{8,64}")
_ASCII_DIGITS_RE = re.compile(r"[0-9]+")


class ServiceMetrics:
    """Per-app Prometheus registry (avoids duplicate registration across test apps)."""

    def __init__(self, service: str) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total", "HTTP requests", ["service", "method", "route", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "http_request_duration_seconds", "HTTP latency", ["service", "route"], registry=self.registry
        )
        self.service = service


class RequestContextMiddleware:
    """Pure ASGI middleware (streaming-safe): request id, body-size limit, access log, metrics."""

    def __init__(self, app: ASGIApp, *, metrics: ServiceMetrics, max_body_bytes: int) -> None:
        self.app = app
        self.metrics = metrics
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # HTTP header bytes are latin-1 (RFC 9110); decoding as UTF-8 would crash on arbitrary client bytes.
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        incoming = headers.get(REQUEST_ID_HEADER.lower(), "")
        # ASCII-only allow-list: the id is echoed in headers and logs, so never trust arbitrary input.
        rid = incoming if _REQUEST_ID_RE.fullmatch(incoming) else new_request_id()
        set_request_id(rid)
        start = time.perf_counter()
        status_holder = {"status": 500}

        declared = headers.get("content-length", "")
        if _ASCII_DIGITS_RE.fullmatch(declared) and int(declared) > self.max_body_bytes:
            response = error_response(413, "payload_too_large", "The request body is too large.")
            response.headers[REQUEST_ID_HEADER] = rid
            await response(scope, receive, send)
            self._observe(scope, 413, start)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                raw = list(message.get("headers", []))
                raw.append((REQUEST_ID_HEADER.encode(), rid.encode()))
                message = {**message, "headers": raw}
            await send(message)

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except _BodyTooLarge:
            response = error_response(413, "payload_too_large", "The request body is too large.")
            response.headers[REQUEST_ID_HEADER] = rid
            await response(scope, receive, send)
            status_holder["status"] = 413
        finally:
            self._observe(scope, status_holder["status"], start)

    def _observe(self, scope: Scope, status: int, start: float) -> None:
        elapsed = time.perf_counter() - start
        route = scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        self.metrics.requests.labels(self.metrics.service, scope["method"], route_path, str(status)).inc()
        self.metrics.latency.labels(self.metrics.service, route_path).observe(elapsed)
        if route_path not in ("/healthz", "/readyz", "/metrics"):
            access_logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": route_path,
                    "status": status,
                    "latency_ms": round(elapsed * 1000, 1),
                },
            )


class _BodyTooLarge(Exception):
    pass


def create_service_app(
    *,
    service: str,
    version: str,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    max_body_bytes: int = 64_000,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
    docs: bool = True,
) -> FastAPI:
    """Create a FastAPI app with the cross-cutting concerns every service needs."""
    app = FastAPI(
        title=f"genz-{service}",
        version=version,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
    )
    metrics = ServiceMetrics(service)
    app.state.metrics = metrics
    app.state.readiness_checks = dict(readiness_checks or {})
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, metrics=metrics, max_body_bytes=max_body_bytes)

    @app.get("/healthz", response_model=HealthStatus, tags=["ops"])
    async def healthz() -> HealthStatus:
        return HealthStatus(status="ok", service=service, version=version)

    @app.get("/readyz", response_model=HealthStatus, tags=["ops"])
    async def readyz() -> JSONResponse:
        checks: dict[str, bool] = {}
        for name, check in app.state.readiness_checks.items():
            try:
                checks[name] = bool(await check())
            except Exception:
                checks[name] = False
        ready = all(checks.values())
        body = HealthStatus(status="ok" if ready else "unavailable", service=service, version=version, checks=checks)
        return JSONResponse(status_code=200 if ready else 503, content=body.model_dump())

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


__all__ = ["ReadinessCheck", "RequestContextMiddleware", "ServiceMetrics", "create_service_app"]
