from __future__ import annotations

import io
import json
import logging

import pytest

from common.logging import (
    REQUEST_ID_HEADER,
    JsonFormatter,
    configure_logging,
    outbound_headers,
    secret_registry,
    set_request_id,
)


def _capture() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter("svc"))
    logger = logging.getLogger("extra-test")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


def test_outbound_headers_propagate_request_id() -> None:
    set_request_id(None)
    assert outbound_headers() == {}
    set_request_id("abcdef123456")
    try:
        assert outbound_headers() == {REQUEST_ID_HEADER: "abcdef123456"}
    finally:
        set_request_id(None)


def test_exception_logging_is_structured_and_redacted() -> None:
    secret = "sk-exception-secret-987654"
    secret_registry.register(secret)
    logger, stream = _capture()
    try:
        raise RuntimeError(f"provider rejected key {secret}")
    except RuntimeError:
        logger.exception("provider_failed")
    record = json.loads(stream.getvalue())
    assert record["exc_type"] == "RuntimeError"
    assert "Traceback" in record["exc"]
    assert secret not in stream.getvalue()


def test_nested_extras_are_redacted() -> None:
    logger, stream = _capture()
    logger.info("evt", extra={"contacts": ["0301-1112223", {"mail": "a.b@example.org"}], "n": 3})
    record = json.loads(stream.getvalue())
    assert record["contacts"] == ["[phone]", {"mail": "[email]"}]
    assert record["n"] == 3


def test_short_values_are_not_registered_as_secrets() -> None:
    # Very short strings would mask ordinary words, so they are ignored.
    secret_registry.register("ab", None, "")
    assert "ab" not in secret_registry.snapshot()


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("svc-a", "debug")
        configure_logging("svc-b", "WARNING")
        ours = [h for h in root.handlers if getattr(h, "_genz_json", False)]
        assert len(ours) == 1
        assert root.level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        logging.getLogger("idem").warning("hello")
        out = capsys.readouterr().out.strip().splitlines()[-1]
        assert json.loads(out)["service"] == "svc-b"
    finally:
        for h in list(root.handlers):
            if getattr(h, "_genz_json", False):
                root.removeHandler(h)
        for h in before:
            if h not in root.handlers:
                root.addHandler(h)


def test_uvicorn_loggers_are_routed_to_json(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    uv = logging.getLogger("uvicorn")
    plain = logging.StreamHandler()  # what uvicorn's default log config installs
    uv.handlers = [plain]
    uv.propagate = False
    try:
        configure_logging("svc-uv", "INFO")
        assert uv.handlers == []
        assert uv.propagate is True
        assert logging.getLogger("uvicorn.error").handlers == []
        logging.getLogger("uvicorn.error").info("Started server process [8]")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        record = json.loads(line)
        assert record["msg"] == "Started server process [8]"
        assert record["logger"] == "uvicorn.error"
        assert record["service"] == "svc-uv"
    finally:
        for h in list(root.handlers):
            if getattr(h, "_genz_json", False):
                root.removeHandler(h)
        for h in before:
            if h not in root.handlers:
                root.addHandler(h)
