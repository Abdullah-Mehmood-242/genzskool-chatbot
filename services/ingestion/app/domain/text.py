"""Small pure text helpers: ids, slugs, fees, levels, token counts, splitting."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid

from common.contracts import Level

_CHUNK_NAMESPACE = uuid.UUID("8f2d3c1e-6b7a-4c5d-9e0f-1a2b3c4d5e6f")
_FEE_RE = re.compile(r"(?i)\b(?:rs\.?|pkr)\s*([0-9][0-9,]*)(?:\s*/-)?")
_WS_RE = re.compile(r"\s+")


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def doc_id_from_file(file: str) -> str:
    """'GenZ Skool KB (v2).pdf' -> 'genz_skool_kb_v2'. Stable and safe for ids and file names."""
    stem = file.rsplit(".", 1)[0] if file.lower().endswith(".pdf") else file
    normalized = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    if not slug:
        slug = "doc_" + sha256_hex(file)[:12]
    return slug[:80]


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")


def chunk_id(doc_id: str, index_version: int, section: str, text: str) -> str:
    """Deterministic UUID (valid Qdrant point id). Includes the version so blue/green copies never collide."""
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_id}\x1f{index_version}\x1f{section}\x1f{sha256_hex(text)}"))


def parse_fee_pkr(text: str | None) -> int | None:
    """'Rs. 30,000' -> 30000; 'Rs 1,25,000' -> 125000. Anything unpublished/unparseable -> None.

    Never invents a number: 'Free', 'Not published', '' all return None.
    """
    if not text:
        return None
    match = _FEE_RE.search(text)
    if not match:
        return None
    digits = match.group(1).replace(",", "")
    if not digits.isdigit():
        return None
    value = int(digits)
    return value if value > 0 else None


def parse_level(text: str | None) -> Level | None:
    if not text:
        return None
    lowered = text.strip().lower()
    for level in Level:
        if lowered.startswith(level.value):
            return level
    return None


def collapse_ws(text: str | None) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def count_tokens(text: str) -> int:
    """Whitespace token estimate (ADR-0004). Deliberately simple and model-independent."""
    return len(text.split())


def split_by_tokens(paragraphs: list[str], *, max_tokens: int, overlap_tokens: int, min_tokens: int) -> list[str]:
    """Greedy-pack paragraphs into parts of <= max_tokens, splitting oversized paragraphs on words.

    Each part after the first starts with the last ``overlap_tokens`` words of the previous part.
    A trailing part smaller than ``min_tokens`` is merged into the previous part when it fits.
    """
    words_per_para = [p.split() for p in paragraphs if p.strip()]
    budget = max_tokens - overlap_tokens
    units: list[list[str]] = []
    for words in words_per_para:
        for i in range(0, len(words), budget):
            units.append(words[i : i + budget])

    parts: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and len(current) + len(unit) > budget:
            parts.append(current)
            current = []
        current = [*current, *(["\n"] if current else []), *unit]
    if current:
        parts.append(current)

    if (
        len(parts) >= 2
        and _word_count(parts[-1]) < min_tokens
        and _word_count(parts[-2]) + _word_count(parts[-1]) <= budget
    ):
        parts[-2] = [*parts[-2], "\n", *parts[-1]]
        parts.pop()

    out: list[str] = []
    for i, part in enumerate(parts):
        words = part
        if i and overlap_tokens:
            tail = [w for w in parts[i - 1] if w != "\n"][-overlap_tokens:]
            words = [*tail, "\n", *part]
        out.append(_join(words))
    return out


def _word_count(words: list[str]) -> int:
    return sum(1 for w in words if w != "\n")


def _join(words: list[str]) -> str:
    text = " ".join(words)
    return text.replace(" \n ", "\n").strip()
