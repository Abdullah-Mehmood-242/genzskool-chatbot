"""Typed exception hierarchy shared by all services.

Every error carries a stable machine-readable ``code`` and an HTTP ``status_code``.
The HTTP mapping lives in :mod:`common.errors.http` so this module stays framework-free.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected, client-presentable errors."""

    code: str = "internal_error"
    status_code: int = 500
    default_message: str = "An internal error occurred."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class ValidationFailed(AppError):
    code = "validation_error"
    status_code = 422
    default_message = "The request is invalid."


class NotFound(AppError):
    code = "not_found"
    status_code = 404
    default_message = "The requested resource was not found."


class Unauthorized(AppError):
    code = "unauthorized"
    status_code = 401
    default_message = "Missing or invalid API key."


class Forbidden(AppError):
    code = "forbidden"
    status_code = 403
    default_message = "You are not allowed to perform this action."


class PayloadTooLarge(AppError):
    code = "payload_too_large"
    status_code = 413
    default_message = "The request body is too large."


class RateLimited(AppError):
    code = "rate_limited"
    status_code = 429
    default_message = "Too many requests. Please slow down."


class UpstreamError(AppError):
    """A downstream service or provider returned an error."""

    code = "upstream_error"
    status_code = 502
    default_message = "A downstream dependency failed."


class ProviderError(UpstreamError):
    """An LLM / embedding / reranker provider failed."""

    code = "provider_error"
    default_message = "The AI provider request failed."


class UpstreamTimeout(UpstreamError):
    code = "upstream_timeout"
    status_code = 504
    default_message = "A downstream dependency timed out."


class ServiceUnavailable(AppError):
    code = "service_unavailable"
    status_code = 503
    default_message = "The service is temporarily unavailable."


class CircuitOpen(ServiceUnavailable):
    code = "circuit_open"
    default_message = "A downstream dependency is failing; requests are paused briefly."


class ConfigurationError(AppError):
    """Raised at startup when configuration is invalid (fail fast)."""

    code = "configuration_error"
    status_code = 500
    default_message = "The service is misconfigured."


__all__ = [
    "AppError",
    "CircuitOpen",
    "ConfigurationError",
    "Forbidden",
    "NotFound",
    "PayloadTooLarge",
    "ProviderError",
    "RateLimited",
    "ServiceUnavailable",
    "Unauthorized",
    "UpstreamError",
    "UpstreamTimeout",
    "ValidationFailed",
]
