"""Map exceptions to the shared JSON error schema. Never leaks stack traces."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from common.contracts.errors import ErrorDetail, ErrorResponse
from common.errors import AppError
from common.logging.context import get_request_id

logger = logging.getLogger("common.errors")

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
}


def error_response(status_code: int, code: str, message: str, details: dict[str, object] | None = None) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=get_request_id(), details=dict(details or {}))
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


async def _app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    if exc.status_code >= 500:
        logger.warning("app_error", extra={"error_code": exc.code, "status": exc.status_code})
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Only expose field locations and messages, never the submitted values.
    errors = [{"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))} for e in exc.errors()]
    return error_response(422, "validation_error", "The request is invalid.", {"errors": errors})


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
    return error_response(exc.status_code, code, message)


async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", extra={"error_type": type(exc).__name__})
    return error_response(500, "internal_error", "An internal error occurred.")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)
