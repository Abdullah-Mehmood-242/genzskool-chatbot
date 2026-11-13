"""Output and storage hygiene: PII never leaves or persists except the business's own public contacts."""

from __future__ import annotations

import re
from collections.abc import Iterable

_PHONE = re.compile(r"(?<!\d)(?:\+?92[\s-]?|0)3\d{2}[\s-]?\d{7}(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)[-10:]  # compare on the last 10 digits (0309... vs +92309...)


def redact_pii(text: str, allowed: Iterable[str] = ()) -> str:
    """Mask phone numbers and emails, except those in ``allowed`` (the official Genz Skool contacts)."""
    allowed = list(allowed)
    allowed_phones = {_digits(p) for a in allowed for p in _PHONE.findall(a)}
    allowed_emails = {e.lower() for a in allowed for e in _EMAIL.findall(a)}

    def phone(m: re.Match[str]) -> str:
        return m.group(0) if _digits(m.group(0)) in allowed_phones else "[phone removed]"

    def email(m: re.Match[str]) -> str:
        return m.group(0) if m.group(0).lower() in allowed_emails else "[email removed]"

    return _EMAIL.sub(email, _PHONE.sub(phone, text))
