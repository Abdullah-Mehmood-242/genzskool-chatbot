from __future__ import annotations

import uuid

import pytest

from app.domain.models import ChunkingConfig, PageContent, ParsedDocument, is_scanned
from app.domain.text import (
    chunk_id,
    collapse_ws,
    count_tokens,
    doc_id_from_file,
    parse_fee_pkr,
    parse_level,
    sha256_hex,
    slugify,
    split_by_tokens,
)
from common.contracts import Level


@pytest.mark.parametrize(
    ("text", "fee"),
    [
        ("Rs. 30,000", 30000),
        ("Rs. 30,000 (instalments)", 30000),
        ("Rs 28000", 28000),
        ("rs.35,000", 35000),
        ("PKR 32,000", 32000),
        ("Rs. 1,25,000", 125000),  # South-Asian digit grouping
        ("Fee: Rs. 30,000, instalments available", 30000),
        ("Rs. 30,000/-", 30000),
        ("Not published: fees and batch dates on WhatsApp", None),
        ("Free", None),
        ("Rs. 0", None),  # never record a zero fee: that would read as "free"
        ("30,000", None),  # no currency marker -> not a fee
        ("", None),
        (None, None),
    ],
)
def test_parse_fee_never_invents(text: str | None, fee: int | None) -> None:
    assert parse_fee_pkr(text) == fee


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Beginner", Level.BEGINNER),
        ("intermediate", Level.INTERMEDIATE),
        (" Beginner ", Level.BEGINNER),
        ("Advanced", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_level(text: str | None, level: Level | None) -> None:
    assert parse_level(text) == level


@pytest.mark.parametrize(
    ("file", "doc_id"),
    [
        ("genzskool_knowledge_base.pdf", "genzskool_knowledge_base"),
        ("GenZ Skool KB (v2).PDF", "genz_skool_kb_v2"),
        ("café-menü.pdf", "cafe_menu"),
        ("---.pdf", None),
        ("شاپیفائی.pdf", None),
    ],
)
def test_doc_id_is_safe_and_stable(file: str, doc_id: str | None) -> None:
    result = doc_id_from_file(file)
    assert result == doc_id_from_file(file)
    assert result.replace("_", "").isalnum()
    assert result.islower() or result.startswith("doc_")
    if doc_id:
        assert result == doc_id
    else:
        assert result.startswith("doc_")


def test_doc_id_length_capped() -> None:
    assert len(doc_id_from_file("x" * 300 + ".pdf")) <= 80


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Shopify store setup", "shopify-store-setup"),
        ("TikTok Shop: setup, affiliates", "tiktok-shop-setup-affiliates"),
        ("  ", ""),
    ],
)
def test_slugify(text: str, slug: str) -> None:
    assert slugify(text) == slug


def test_chunk_ids_are_uuids_deterministic_and_version_scoped() -> None:
    a = chunk_id("kb", 1, "Course: Shopify", "text")
    assert uuid.UUID(a)
    assert a == chunk_id("kb", 1, "Course: Shopify", "text")
    assert a != chunk_id("kb", 2, "Course: Shopify", "text"), "blue/green versions must not overwrite each other"
    assert a != chunk_id("kb", 1, "Course: eBay", "text")
    assert a != chunk_id("kb", 1, "Course: Shopify", "text2")
    assert a != chunk_id("other", 1, "Course: Shopify", "text")


def test_sha256_accepts_str_and_bytes() -> None:
    assert sha256_hex("abc") == sha256_hex(b"abc")
    assert len(sha256_hex(b"")) == 64


def test_collapse_ws_and_tokens() -> None:
    assert collapse_ws(" a\n b\t c ") == "a b c"
    assert collapse_ws(None) == ""
    assert count_tokens("one two  three\nfour") == 4
    assert count_tokens("") == 0


# ---------- split_by_tokens ----------


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_split_small_input_is_single_part() -> None:
    assert split_by_tokens(["a b c"], max_tokens=50, overlap_tokens=5, min_tokens=2) == ["a b c"]


def test_split_respects_max_tokens_including_overlap() -> None:
    paras = [_words(30, f"p{i}_") for i in range(10)]
    parts = split_by_tokens(paras, max_tokens=100, overlap_tokens=10, min_tokens=5)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 100 for p in parts)


def test_split_overlap_repeats_previous_tail() -> None:
    parts = split_by_tokens([_words(60, "a"), _words(60, "b")], max_tokens=70, overlap_tokens=5, min_tokens=1)
    assert len(parts) >= 2
    tail = parts[0].split()[-5:]
    assert parts[1].split()[:5] == tail


def test_split_without_overlap() -> None:
    parts = split_by_tokens([_words(50, "a"), _words(50, "b")], max_tokens=60, overlap_tokens=0, min_tokens=1)
    assert parts[1].split()[0] == "b0"


def test_split_oversized_single_paragraph_is_cut_on_words() -> None:
    parts = split_by_tokens([_words(250)], max_tokens=100, overlap_tokens=0, min_tokens=1)
    assert [count_tokens(p) for p in parts] == [100, 100, 50]
    assert " ".join(parts).split() == _words(250).split()  # nothing lost or duplicated


def test_split_tiny_tail_is_merged() -> None:
    parts = split_by_tokens([_words(50, "a"), _words(3, "z")], max_tokens=60, overlap_tokens=0, min_tokens=10)
    assert len(parts) == 1
    assert parts[0].endswith("z2")


def test_split_preserves_paragraph_breaks() -> None:
    parts = split_by_tokens(["a b", "c d"], max_tokens=50, overlap_tokens=0, min_tokens=1)
    assert parts == ["a b\nc d"]


def test_split_ignores_blank_paragraphs() -> None:
    assert split_by_tokens(["", "  ", "x"], max_tokens=50, overlap_tokens=0, min_tokens=1) == ["x"]


# ---------- models ----------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 10},
        {"overlap_tokens": -1},
        {"max_tokens": 100, "overlap_tokens": 50},
        {"min_chunk_tokens": -1},
        {"max_tokens": 100, "min_chunk_tokens": 100},
    ],
)
def test_chunking_config_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be"):
        ChunkingConfig(**kwargs)


def _doc(*texts: str, tables: int = 0) -> ParsedDocument:
    pages = [PageContent(page=i + 1, text=t, tables=[[["cell"] * 3]] * tables) for i, t in enumerate(texts)]
    return ParsedDocument(doc_id="d", source_file="d.pdf", content_hash="h", pages=pages)


def test_is_scanned() -> None:
    assert is_scanned(_doc("", "", ""))
    assert is_scanned(_doc("  ", "x"))
    assert not is_scanned(_doc("A real paragraph of text that is clearly not a scan."))
    assert not is_scanned(ParsedDocument(doc_id="d", source_file="d.pdf", content_hash="h", pages=[]))
    assert not is_scanned(_doc("", tables=3)), "table text counts as extractable text"
    assert is_scanned(_doc("short"), min_chars_per_page=10)
    assert not is_scanned(_doc("short"), min_chars_per_page=5)
