"""Request-scoped context (request id) carried through async code via contextvars."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def outbound_headers() -> dict[str, str]:
    """Headers to propagate on calls to other services."""
    rid = get_request_id()
    return {REQUEST_ID_HEADER: rid} if rid else {}
