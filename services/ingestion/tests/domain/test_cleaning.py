from __future__ import annotations

import pytest

from app.domain.cleaning import StandardTextCleaner, normalize_text
from app.domain.models import PageContent, ParsedDocument


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  hello   world  ", "hello world"),
        ("a b", "a b"),  # non-breaking space
        ("zero​width", "zerowidth"),
        ("soft­hyphen", "softhyphen"),
        ("﻿BOM", "BOM"),
        ("ﬁle", "file"),  # ligature (NFKC)
        ("ＡＢＣ", "ABC"),  # full-width (NFKC)
        ("Mon–Sat — 10am", "Mon-Sat - 10am"),
        ("‘quoted’ “double”", "'quoted' \"double\""),
        ("tab\tseparated", "tab separated"),
        ("شاپیفائی کورس", "شاپیفائی کورس"),  # Urdu survives
    ],
)
def test_normalize_text(raw: str, clean: str) -> None:
    assert normalize_text(raw) == clean


def _doc(pages: list[PageContent]) -> ParsedDocument:
    return ParsedDocument(doc_id="d", source_file="d.pdf", content_hash="h", pages=pages, warnings=["w1"])


def test_lines_keep_page_numbers_and_drop_blanks_and_footers() -> None:
    cleaner = StandardTextCleaner((r"^Footer Page \d+$",))
    doc = _doc([PageContent(1, "First\n\n  \nSecond\nFooter Page 1"), PageContent(2, "Third\nFooter Page 2")])
    out = cleaner.clean(doc)
    assert [(ln.page, ln.text) for ln in out.lines] == [(1, "First"), (1, "Second"), (2, "Third")]
    assert out.warnings == ["w1"]
    assert (out.doc_id, out.source_file, out.content_hash) == ("d", "d.pdf", "h")


def test_strip_patterns_only_remove_matching_lines() -> None:
    cleaner = StandardTextCleaner((r"^Page \d+$",))
    out = cleaner.clean(_doc([PageContent(1, "Page 3\nPage 3 of the guide says hi")]))
    assert [ln.text for ln in out.lines] == ["Page 3 of the guide says hi"]


def test_table_cells_are_normalised_and_empty_rows_dropped() -> None:
    table = [["Course", "Fee"], ["Shopify\nstore", "Rs. 30,000\n(instalments)"], [None, ""], ["  ", None]]
    out = StandardTextCleaner().clean(_doc([PageContent(2, "", tables=[table])]))
    assert out.tables == [(2, [["Course", "Fee"], ["Shopify store", "Rs. 30,000 (instalments)"]])]


def test_none_cells_survive_as_none() -> None:
    out = StandardTextCleaner().clean(_doc([PageContent(1, "", tables=[[["A", None, "B"]]])]))
    assert out.tables[0][1] == [["A", None, "B"]]


def test_fully_empty_tables_dropped() -> None:
    out = StandardTextCleaner().clean(_doc([PageContent(1, "x", tables=[[[None, ""]]])]))
    assert out.tables == []


def test_invalid_pattern_fails_at_construction() -> None:
    import re

    with pytest.raises(re.error):
        StandardTextCleaner(("(unclosed",))
