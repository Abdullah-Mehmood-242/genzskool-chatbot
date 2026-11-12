"""Chunker behaviour on layouts the real PDF does not exercise: fallbacks, splitting, odd tables."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain.chunking import CourseFacts, StructureAwareChunker, paragraphs, parse_catalogue
from app.domain.models import ChunkingConfig, CleanedDocument, Line, Table
from app.domain.text import count_tokens
from common.contracts import Chunk, DocType, Level

NOW = datetime(2026, 1, 1, tzinfo=UTC)
HEADER: list[str | None] = ["Course", "Category", "Lessons / Length", "Level", "Mentor", "Fee"]


def _doc(lines: list[str | tuple[int, str]], tables: list[tuple[int, Table]] | None = None) -> CleanedDocument:
    out = [Line(page=x[0], text=x[1]) if isinstance(x, tuple) else Line(page=1, text=x) for x in lines]
    return CleanedDocument(doc_id="d", source_file="d.pdf", content_hash="h", lines=out, tables=tables or [])


def _chunk(doc: CleanedDocument, **cfg: object) -> tuple[list[Chunk], list[str]]:
    result = StructureAwareChunker(ChunkingConfig(**cfg)).chunk(doc, index_version=3, ingested_at=NOW)  # type: ignore[arg-type]
    return result.chunks, result.warnings


# ---------- paragraphs ----------


def test_paragraphs_join_soft_wraps_and_split_bullets() -> None:
    lines = ["First sentence that", "wraps here.", "- bullet one", "continues", "- bullet two"]
    assert paragraphs(lines) == ["First sentence that wraps here.", "- bullet one continues", "- bullet two"]


def test_paragraphs_detect_subheadings_before_bullets() -> None:
    lines = ["Intro text.", "Why choose us", "- reason one", "- reason two"]
    assert paragraphs(lines) == ["Intro text.", "Why choose us", "- reason one", "- reason two"]


def test_short_bullet_continuation_before_next_bullet_is_joined() -> None:
    assert paragraphs(["- Phone: 0300", "(fastest)", "- Email: x"]) == ["- Phone: 0300 (fastest)", "- Email: x"]


def test_finished_bullet_then_subheading() -> None:
    assert paragraphs(["- done.", "Next topic", "- a"]) == ["- done.", "Next topic", "- a"]


def test_paragraphs_empty() -> None:
    assert paragraphs([]) == []


# ---------- sectioning ----------


def test_numbering_must_be_sequential() -> None:
    doc = _doc(["1. About Us", "We teach.", "3. Not a heading because 2 is missing", "2. How it works", "- Step 1"])
    chunks, _ = _chunk(doc)
    sections = [c.metadata.section for c in chunks]
    assert sections == ["About Genz Skool", "How it works"]
    assert "3. Not a heading" in chunks[0].text


def test_numbered_list_items_are_not_headings() -> None:
    doc = _doc(["1. About Us", "1. this lower-case item is not a heading", "We teach."])
    chunks, _ = _chunk(doc)
    assert len(chunks) == 1


def test_no_headings_falls_back_to_per_page_chunks() -> None:
    doc = _doc([(1, "Just some text."), (1, "More."), (2, "Page two text.")])
    chunks, warnings = _chunk(doc)
    assert [c.metadata.section for c in chunks] == ["Page 1", "Page 2"]
    assert [c.metadata.page for c in chunks] == [1, 2]
    assert all(c.metadata.doc_type == DocType.POLICY for c in chunks)
    assert any("fell back" in w for w in warnings)


def test_empty_document_has_no_chunks() -> None:
    chunks, _ = _chunk(_doc([]))
    assert chunks == []


def test_unknown_section_titles_become_policy_chunks_without_parenthetical() -> None:
    doc = _doc(["1. Refund policy (summary)", "Refunds within 7 days."])
    (chunk,), _ = _chunk(doc)
    assert chunk.metadata.section == "Refund policy"
    assert chunk.text.startswith("Genz Skool | Section: Refund policy\n")


def test_preamble_without_about_section_kept_as_overview() -> None:
    doc = _doc(["Title page", "Crawled yesterday.", "1. Enrollment", "- Fill the form."])
    chunks, _ = _chunk(doc)
    assert chunks[0].metadata.section == "Document overview"
    assert "Crawled yesterday." in chunks[0].text


def test_empty_sections_dropped() -> None:
    chunks, _ = _chunk(_doc(["1. About Us", "2. Enrollment", "- Fill the form."]))
    assert [c.metadata.section for c in chunks] == ["Enrollment"]


# ---------- catalogue ----------


def test_catalogue_columns_found_by_name_not_position() -> None:
    table: Table = [["Fee", "Mentor", "Course", "Level"], ["Rs. 10,000", "A B", "Alpha course", "Beginner"]]
    (facts,) = parse_catalogue([(1, table)])
    assert (facts.title, facts.fee_pkr, facts.mentor, facts.level_text) == ("Alpha course", 10000, "A B", "Beginner")
    assert facts.category is None


def test_non_catalogue_tables_ignored() -> None:
    assert parse_catalogue([(1, [["Day", "Time"], ["Mon", "7pm"]])]) == []
    assert parse_catalogue([(1, [])]) == []


def test_repeated_headers_and_duplicate_rows_skipped() -> None:
    rows: Table = [HEADER, ["A (New)", "Cat", "1 / 1h", "Beginner", "M N", "Rs. 1,000"]]
    facts = parse_catalogue([(1, rows), (2, [HEADER, *rows[1:]]), (2, [HEADER, ["a", "x", "", "", "", ""]])])
    assert len(facts) == 1
    assert facts[0].is_new
    assert facts[0].title == "A"


def test_short_rows_and_blank_titles_tolerated() -> None:
    facts = parse_catalogue([(1, [HEADER, ["Only title"], ["", "x", "y", "z", "w", "Rs. 5"]])])
    assert [f.title for f in facts] == ["Only title"]
    assert facts[0].fee_text is None


def test_course_facts_fee_property() -> None:
    f = CourseFacts("t", False, None, None, None, None, "Not published", 1)
    assert f.fee_pkr is None


def _kb(detail_lines: list[str], rows: list[list[str | None]]) -> CleanedDocument:
    lines = [
        "1. About X",
        "We teach.",
        "2. Course catalogue",
        "Prices in Rs.",
        "3. Course details",
        "Intro to courses.",
        *detail_lines,
    ]
    return _doc(lines, [(1, [HEADER, *rows])])


ROW_ALPHA: list[str | None] = ["Alpha course (New)", "Cat A", "10 lessons / 2h", "Beginner", "Ann Bee", "Rs. 12,000"]


def test_course_matched_to_detail() -> None:
    doc = _kb(["Alpha course", "Page: https://x.test/courses/alpha", "Learn: A, B. Mentor: Ann Bee."], [ROW_ALPHA])
    chunks, warnings = _chunk(doc)
    assert warnings == []
    course = next(c for c in chunks if c.metadata.doc_type == DocType.COURSE)
    assert course.metadata.course_slug == "alpha"
    assert course.metadata.fee_pkr == 12000
    assert "Learn: A, B." in course.text
    overview = next(c for c in chunks if c.metadata.doc_type == DocType.OVERVIEW)
    assert "- Alpha course (New); Fee: Rs. 12,000" in overview.text
    structure = next(c for c in chunks if c.metadata.section == "Course structure")
    assert "Intro to courses." in structure.text
    assert "Alpha course" not in structure.text  # the title line moved into the course block


def test_course_without_detail_uses_catalogue_only() -> None:
    chunks, warnings = _chunk(_kb([], [ROW_ALPHA]))
    course = next(c for c in chunks if c.metadata.doc_type == DocType.COURSE)
    assert course.metadata.course_slug == "alpha-course"
    assert course.metadata.fee_pkr == 12000
    assert any("No detail section" in w for w in warnings)


def test_detail_without_catalogue_row_parses_text_conservatively() -> None:
    lines = [
        "Beta course",
        "Page: https://x.test/courses/beta",
        "Learn: B. Mentor: Carl Dee (Expert). Level: Intermediate. Fee: Rs. 9,500, instalments available.",
    ]
    chunks, warnings = _chunk(_kb(lines, [ROW_ALPHA]))
    beta = next(c for c in chunks if c.metadata.course_slug == "beta")
    assert beta.metadata.mentor == "Carl Dee"
    assert beta.metadata.level == Level.INTERMEDIATE
    assert beta.metadata.fee_pkr == 9500
    assert beta.metadata.category is None
    assert any("no catalogue row" in w for w in warnings)


def test_detail_without_fee_gives_no_fee() -> None:
    lines = ["Gamma", "Page: https://x.test/courses/gamma", "Fee and batch dates are given on WhatsApp only."]
    chunks, _ = _chunk(_kb(lines, [ROW_ALPHA]))
    gamma = next(c for c in chunks if c.metadata.course_slug == "gamma")
    assert gamma.metadata.fee_pkr is None
    assert gamma.metadata.mentor is None


def test_prefix_title_matching() -> None:
    row: list[str | None] = ["Delta course: full title", "C", "", "Beginner", "D E", "Rs. 1,000"]
    lines = ["Delta course: full title extended", "Page: https://x.test/courses/delta", "Learn: D."]
    chunks, warnings = _chunk(_kb(lines, [row]))
    assert next(c for c in chunks if c.metadata.doc_type == DocType.COURSE).metadata.course_slug == "delta"
    assert warnings == []


def test_catalogue_without_table_kept_as_text() -> None:
    doc = _doc(["1. About X", "We teach.", "2. Course catalogue", "Alpha Rs. 1,000"])
    chunks, warnings = _chunk(doc)
    assert any(c.metadata.section == "Course catalogue" and "Alpha Rs. 1,000" in c.text for c in chunks)
    assert not any(c.metadata.doc_type == DocType.OVERVIEW for c in chunks)
    assert any("table not found" in w for w in warnings)


# ---------- mentors ----------


def test_mentor_lines_parsed_and_unparsed_kept() -> None:
    doc = _doc(
        [
            "1. Mentors",
            "All mentors review work.",
            "- Ann Bee - Designer - teaches Design.",
            "- Someone without the expected format",
        ]
    )
    chunks, warnings = _chunk(doc)
    mentor = next(c for c in chunks if c.metadata.doc_type == DocType.MENTOR)
    assert mentor.metadata.mentor == "Ann Bee"
    assert mentor.text.splitlines()[0] == "Mentor: Ann Bee | Role: Designer | Teaches: Design"
    fallback = next(c for c in chunks if c.metadata.section == "Mentors")
    assert "Someone without the expected format" in fallback.text
    assert any("mentor line" in w for w in warnings)


def test_mentor_section_with_no_bullets_is_policy() -> None:
    chunks, _ = _chunk(_doc(["1. Mentors", "Our mentors are experts."]))
    assert [c.metadata.section for c in chunks] == ["Mentors"]


# ---------- size limits, headers, dedup ----------


def test_long_section_is_split_into_parts_each_with_header() -> None:
    long_lines = [f"- Item {i} " + " ".join(f"word{i}_{j}" for j in range(30)) for i in range(40)]
    chunks, _ = _chunk(_doc(["1. Enrollment", *long_lines]), max_tokens=120, overlap_tokens=10)
    assert len(chunks) > 5
    n = len(chunks)
    assert [c.metadata.section for c in chunks] == [f"Enrollment (part {i}/{n})" for i in range(1, n + 1)]
    for c in chunks:
        assert c.text.startswith("Genz Skool | Section: Enrollment\n")
        assert count_tokens(c.text) <= 120
    assert len({c.id for c in chunks}) == n


def test_context_header_can_be_disabled() -> None:
    (chunk,), _ = _chunk(_doc(["1. Enrollment", "- Fill the form."]), context_header=False)
    assert chunk.text == "- Fill the form."


def test_duplicate_chunks_are_skipped_with_warning() -> None:
    doc = _doc(["1. Enrollment", "- Same.", "2. Enrollment", "- Same."])
    chunks, warnings = _chunk(doc)
    assert len(chunks) == 1
    assert any("Duplicate chunk skipped" in w for w in warnings)


def test_header_too_big_for_budget_raises() -> None:
    title = " ".join(["Very"] * 18)  # 89 chars: still a heading (<= 100), but 22 header tokens
    with pytest.raises(ValueError, match="no room for the header"):
        _chunk(_doc([f"1. {title}", "- text"]), max_tokens=40, overlap_tokens=0, min_chunk_tokens=0)


def test_metadata_carries_version_and_time() -> None:
    (chunk,), _ = _chunk(_doc(["1. Enrollment", "- Fill the form."]))
    assert chunk.metadata.index_version == 3
    assert chunk.metadata.ingested_at == NOW
    assert chunk.metadata.doc_id == "d"


def test_warnings_reset_between_runs() -> None:
    chunker = StructureAwareChunker(ChunkingConfig())
    first = chunker.chunk(_doc(["no headings"]), index_version=1, ingested_at=NOW)
    second = chunker.chunk(_doc(["1. Enrollment", "- ok"]), index_version=1, ingested_at=NOW)
    assert first.warnings
    assert second.warnings == []
