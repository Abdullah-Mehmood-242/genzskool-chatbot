"""Every error class: code, status, message semantics, hierarchy and HTTP mapping."""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI, HTTPException

import common.errors as errors_module
from common.errors import (
    AppError,
    CircuitOpen,
    ConfigurationError,
    Forbidden,
    NotFound,
    PayloadTooLarge,
    ProviderError,
    RateLimited,
    ServiceUnavailable,
    TransientProviderError,
    Unauthorized,
    UpstreamError,
    UpstreamTimeout,
    ValidationFailed,
)
from common.observability import create_service_app

CASES: list[tuple[type[AppError], str, int]] = [
    (AppError, "internal_error", 500),
    (ValidationFailed, "validation_error", 422),
    (NotFound, "not_found", 404),
    (Unauthorized, "unauthorized", 401),
    (Forbidden, "forbidden", 403),
    (PayloadTooLarge, "payload_too_large", 413),
    (RateLimited, "rate_limited", 429),
    (UpstreamError, "upstream_error", 502),
    (ProviderError, "provider_error", 502),
    (TransientProviderError, "provider_unavailable", 503),
    (UpstreamTimeout, "upstream_timeout", 504),
    (ServiceUnavailable, "service_unavailable", 503),
    (CircuitOpen, "circuit_open", 503),
    (ConfigurationError, "configuration_error", 500),
]


@pytest.mark.parametrize(("cls", "code", "status"), CASES, ids=lambda v: getattr(v, "__name__", str(v)))
def test_code_and_status(cls: type[AppError], code: str, status: int) -> None:
    exc = cls()
    assert exc.code == code
    assert exc.status_code == status
    assert exc.message == cls.default_message
    assert str(exc) == cls.default_message
    assert exc.details == {}


@pytest.mark.parametrize(("cls", "_code", "_status"), CASES, ids=lambda v: getattr(v, "__name__", str(v)))
def test_custom_message_and_details(cls: type[AppError], _code: str, _status: int) -> None:
    exc = cls("custom", details={"k": 1})
    assert exc.message == "custom"
    assert str(exc) == "custom"
    assert exc.details == {"k": 1}


def test_details_are_not_shared_between_instances() -> None:
    a, b = NotFound(), NotFound()
    a.details["x"] = 1
    assert b.details == {}


def test_codes_are_unique() -> None:
    codes = [code for _, code, _ in CASES]
    assert len(codes) == len(set(codes))


def test_hierarchy() -> None:
    assert issubclass(ProviderError, UpstreamError)
    assert issubclass(TransientProviderError, ProviderError)
    assert issubclass(UpstreamTimeout, UpstreamError)
    assert issubclass(CircuitOpen, ServiceUnavailable)
    assert all(issubclass(cls, AppError) for cls, _, _ in CASES)
    assert all(issubclass(cls, Exception) for cls, _, _ in CASES)


def test_every_error_class_is_exported() -> None:
    exported = set(errors_module.__all__)
    assert {cls.__name__ for cls, _, _ in CASES} == exported


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


@pytest.mark.parametrize(("cls", "code", "status"), CASES, ids=lambda v: getattr(v, "__name__", str(v)))
async def test_every_error_maps_to_http(cls: type[AppError], code: str, status: int) -> None:
    app = create_service_app(service="e", version="1")

    @app.get("/raise")
    async def raise_it() -> None:
        raise cls(details={"hint": "h"})

    async with _client(app) as c:
        r = await c.get("/raise")
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["message"] == cls.default_message
    assert body["error"]["details"] == {"hint": "h"}
    assert body["error"]["request_id"] == r.headers["X-Request-ID"]


async def test_method_not_allowed_uses_schema() -> None:
    app = create_service_app(service="e", version="1")
    async with _client(app) as c:
        r = await c.post("/healthz")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "method_not_allowed"


async def test_http_exception_variants() -> None:
    app = create_service_app(service="e", version="1")

    @app.get("/teapot")
    async def teapot() -> None:
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get("/dict-detail")
    async def dict_detail() -> None:
        raise HTTPException(status_code=400, detail={"internal": "do not leak"})

    async with _client(app) as c:
        teapot_r = await c.get("/teapot")
        dict_r = await c.get("/dict-detail")
    assert teapot_r.status_code == 418
    assert teapot_r.json()["error"] == {
        "code": "http_error",
        "message": "short and stout",
        "request_id": teapot_r.headers["X-Request-ID"],
        "details": {},
    }
    assert dict_r.json()["error"]["code"] == "bad_request"
    assert dict_r.json()["error"]["message"] == "HTTP error."
    assert "do not leak" not in dict_r.text


async def test_server_errors_are_logged_client_errors_are_not(caplog: pytest.LogCaptureFixture) -> None:
    app = create_service_app(service="e", version="1")

    @app.get("/nf")
    async def nf() -> None:
        raise NotFound

    @app.get("/up")
    async def up() -> None:
        raise UpstreamError

    @app.get("/crash")
    async def crash() -> None:
        raise ZeroDivisionError("token=abc123 internal detail")

    with caplog.at_level(logging.WARNING, logger="common.errors"):
        async with _client(app) as c:
            await c.get("/nf")
            await c.get("/up")
            await c.get("/crash")
    records = [r for r in caplog.records if r.name == "common.errors"]
    assert [r.getMessage() for r in records] == ["app_error", "unhandled_exception"]
    assert records[0].error_code == "upstream_error"  # type: ignore[attr-defined]
    assert records[1].error_type == "ZeroDivisionError"  # type: ignore[attr-defined]
    assert all("abc123" not in r.getMessage() for r in records)
