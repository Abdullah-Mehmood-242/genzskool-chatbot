"""Error and envelope schemas shared by every HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable message, safe to show to clients.")
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """The single error schema returned by every service."""

    error: ErrorDetail


class Envelope[T](BaseModel):
    """Success envelope used by the public gateway."""

    data: T
    request_id: str | None = None


class HealthStatus(BaseModel):
    status: str = Field(description="'ok' or 'degraded' or 'unavailable'.")
    service: str
    version: str
    checks: dict[str, bool] = Field(default_factory=dict)
