"""Structured JSON logging with secret and PII redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from datetime import UTC, datetime
from typing import Any

from common.logging.context import get_request_id

# Pakistani mobile numbers (03xx-xxxxxxx, +92 3xx ...), generic long digit runs, emails.
_PHONE_RE = re.compile(r"(?:\+?92[\s-]?|0)3\d{2}[\s-]?\d{7}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_BEARER_RE = re.compile(r"(?i)(bearer\s+|api[_-]?key[\"'=:\s]+)[A-Za-z0-9._\-]{8,}")

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class _SecretRegistry:
    """Process-wide set of secret values that must never appear in logs."""

    def __init__(self) -> None:
        self._values: set[str] = set()
        self._lock = threading.Lock()

    def register(self, *values: str | None) -> None:
        with self._lock:
            for v in values:
                if v and len(v) >= 6:
                    self._values.add(v)

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            # Longest first, so overlapping secrets are fully masked.
            return tuple(sorted(self._values, key=len, reverse=True))


secret_registry = _SecretRegistry()


def redact(text: str) -> str:
    """Mask registered secrets, API-key-looking tokens, phone numbers and emails."""
    for secret in secret_registry.snapshot():
        text = text.replace(secret, "***")
    text = _BEARER_RE.sub(lambda m: m.group(1) + "***", text)
    text = _PHONE_RE.sub("[phone]", text)
    return _EMAIL_RE.sub("[email]", text)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(_redact_value(payload), default=str, ensure_ascii=False))


def configure_logging(service: str, level: str = "INFO") -> None:
    """Install a single JSON stdout handler on the root logger (idempotent)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_genz_json", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    handler._genz_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
