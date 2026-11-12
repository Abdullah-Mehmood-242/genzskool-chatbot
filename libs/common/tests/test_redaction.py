"""Redaction rules applied to every log line (secrets, Pakistani mobiles, emails, bearer tokens)."""

from __future__ import annotations

import io
import json
import logging

import pytest

from common.logging import JsonFormatter, new_request_id, redact, secret_registry, set_request_id


@pytest.mark.parametrize(
    "phone",
    [
        "0309-4190245",
        "03094190245",
        "0309 4190245",
        "+92-309-4190245",
        "+92 309 4190245",
        "923094190245",
        "+923094190245",
    ],
)
def test_mobile_formats_are_masked(phone: str) -> None:
    out = redact(f"call {phone} now")
    assert out == "call [phone] now"


@pytest.mark.parametrize(
    "text",
    ["Rs. 30,000", "30000", "24 lessons / 9h 20m", "GS-DM-001", "2026-11-12", "29 seats left", "index_version 12"],
)
def test_ordinary_numbers_are_not_masked(text: str) -> None:
    assert redact(text) == text


@pytest.mark.parametrize("email", ["info@genzskool.com", "a.b+tag@sub.example.co.uk", "X_Y@ex-ample.org"])
def test_emails_are_masked(email: str) -> None:
    assert redact(f"mail {email}.") == "mail [email]."


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer abcdEFGH1234",
        "bearer abcdEFGH1234",
        "api_key=abcdEFGH1234",
        "api-key: abcdEFGH1234",
        'API_KEY="abcdEFGH1234"',
    ],
)
def test_token_patterns_are_masked(text: str) -> None:
    assert "abcdEFGH1234" not in redact(text)
    assert "***" in redact(text)


def test_overlapping_secrets_are_fully_masked() -> None:
    secret_registry.register("overlap-secret", "overlap-secret-longer-version")
    assert redact("x overlap-secret-longer-version y") == "x *** y"
    assert redact("x overlap-secret y") == "x *** y"


def test_placeholder_like_short_values_ignored() -> None:
    secret_registry.register("abc")
    assert redact("abc def") == "abc def"


def _record(**extra: object) -> dict[str, object]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter("svc"))
    logger = logging.getLogger("redaction-json")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("m", extra=extra)
    return json.loads(stream.getvalue())  # type: ignore[no-any-return]


def test_urdu_text_is_preserved_unescaped() -> None:
    rec = _record(text="شاپیفائی کورس")
    assert rec["text"] == "شاپیفائی کورس"


def test_non_serialisable_extras_fall_back_to_str() -> None:
    class Thing:
        def __str__(self) -> str:
            return "thing!"

    assert _record(obj=Thing())["obj"] == "thing!"


def test_no_request_id_key_when_unset() -> None:
    set_request_id(None)
    assert "request_id" not in _record()


def test_log_record_core_fields() -> None:
    rec = _record()
    assert set(rec) >= {"ts", "level", "service", "logger", "msg"}
    assert rec["level"] == "INFO"
    assert str(rec["ts"]).endswith("+00:00")


def test_request_ids_are_unique_hex() -> None:
    ids = {new_request_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)
