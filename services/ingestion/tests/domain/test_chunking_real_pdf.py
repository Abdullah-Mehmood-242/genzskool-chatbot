"""Chunking tests against the REAL knowledge-base PDF (the source of truth).

These pin every business fact the advisor depends on. If the PDF is regenerated with a different
layout, these fail loudly (see scripts/refresh_kb.md) instead of the advisor quietly degrading.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import FOOTER, REAL_PDF

from app.domain.chunking import StructureAwareChunker, parse_catalogue
from app.domain.cleaning import StandardTextCleaner, normalize_text
from app.domain.models import ChunkingConfig, CleanedDocument, SourceFile
from app.domain.text import count_tokens
from app.infrastructure.pdf_loader import PdfPlumberLoader
from common.contracts import Chunk, DocType, Level

INGESTED_AT = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
SOURCE = SourceFile(file=REAL_PDF.name, doc_id="genzskool_knowledge_base")


@pytest.fixture(scope="module")
def cleaned() -> CleanedDocument:
    doc = PdfPlumberLoader().load(REAL_PDF.read_bytes(), SOURCE, "hash")
    return StandardTextCleaner((FOOTER,)).clean(doc)


@pytest.fixture(scope="module")
def chunks(cleaned: CleanedDocument) -> list[Chunk]:
    result = StructureAwareChunker(ChunkingConfig(strip_patterns=(FOOTER,))).chunk(
        cleaned, index_version=1, ingested_at=INGESTED_AT
    )
    assert result.warnings == []
    return result.chunks


@pytest.fixture(scope="module")
def courses(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.metadata.course_slug: c for c in chunks if c.metadata.doc_type == DocType.COURSE}  # type: ignore[misc]


def _by_type(chunks: list[Chunk], doc_type: DocType) -> list[Chunk]:
    return [c for c in chunks if c.metadata.doc_type == doc_type]


def test_chunk_counts_per_type(chunks: list[Chunk]) -> None:
    counts = {t: len(_by_type(chunks, t)) for t in DocType}
    assert counts == {DocType.COURSE: 8, DocType.MENTOR: 6, DocType.POLICY: 6, DocType.CONTACT: 1, DocType.OVERVIEW: 1}


# (slug, title, fee, level, mentor, category, page)
COURSES = [
    ("shopify", "Shopify store setup to first sale", 30000, Level.BEGINNER, "Hamza Ali", "E-commerce", 3),
    (
        "tiktok-shop",
        "TikTok Shop: setup, affiliates and live selling",
        30000,
        Level.BEGINNER,
        "Sana Iqbal",
        "E-commerce",
        3,
    ),
    ("ebay", "eBay product research and listings", 28000, Level.BEGINNER, "Hamza Ali", "E-commerce", 3),
    (
        "video-editing",
        "Video editing with Premiere Pro and CapCut",
        30000,
        Level.BEGINNER,
        "Bilal Ahmed",
        "Editing and Design",
        3,
    ),
    (
        "digital-marketing",
        "Digital marketing: SEO, social and ads",
        35000,
        Level.BEGINNER,
        "Mahnoor Raza",
        "Marketing",
        3,
    ),
    (
        "graphic-design",
        "Graphic designing in Photoshop and Illustrator",
        28000,
        Level.BEGINNER,
        "Zara Malik",
        "Editing and Design",
        3,
    ),
    (
        "ai-automation",
        "AI automation with ChatGPT and Make.com",
        32000,
        Level.INTERMEDIATE,
        "Usman Tariq",
        "AI Automation",
        4,
    ),
    (
        "tiktok-automation",
        "TikTok automation: faceless videos that earn",
        None,
        Level.INTERMEDIATE,
        "Sana Iqbal",
        "AI Automation / Social",
        4,
    ),
]


@pytest.mark.parametrize(
    ("slug", "title", "fee", "level", "mentor", "category", "page"), COURSES, ids=[c[0] for c in COURSES]
)
def test_course_facts(
    courses: dict[str, Chunk],
    slug: str,
    title: str,
    fee: int | None,
    level: Level,
    mentor: str,
    category: str,
    page: int,
) -> None:
    chunk = courses[slug]
    m = chunk.metadata
    assert m.section == f"Course: {title}"
    assert m.fee_pkr == fee
    assert m.level == level
    assert m.mentor == mentor
    assert m.category == category
    assert m.page == page
    header = chunk.text.splitlines()[0]
    assert header.startswith(f"Course: {title} | Category: {category} | Level: {level.value.capitalize()}")
    assert f"Mentor: {mentor}" in header
    assert f"https://genzskool.com/courses/{slug}" in chunk.text


def test_tiktok_automation_fee_is_not_invented(courses: dict[str, Chunk]) -> None:
    chunk = courses["tiktok-automation"]
    assert chunk.metadata.fee_pkr is None
    assert "Not published" in chunk.text
    assert "WhatsApp" in chunk.text
    assert not re.search(r"Rs\.?\s*\d", chunk.text)


@pytest.mark.parametrize(
    ("slug", "length"),
    [
        ("shopify", "24 lessons / 9h 20m"),
        ("tiktok-shop", "20 lessons / 7h 45m"),
        ("ebay", "16 lessons / 6h 05m"),
        ("video-editing", "28 lessons / 11h 30m"),
        ("digital-marketing", "26 lessons / 10h 10m"),
        ("graphic-design", "22 lessons / 8h 40m"),
        ("ai-automation", "18 lessons / 6h 10m"),
        ("tiktok-automation", "15 lessons / 5h 30m"),
    ],
)
def test_course_lengths_from_catalogue(courses: dict[str, Chunk], slug: str, length: str) -> None:
    assert length in courses[slug].text


def test_new_courses_flagged(courses: dict[str, Chunk]) -> None:
    flagged = {slug for slug, c in courses.items() if "(New)" in c.text}
    assert flagged == {"tiktok-shop", "ai-automation"}


def test_batch_details_and_known_defects_preserved(courses: dict[str, Chunk]) -> None:
    dm, gd = courses["digital-marketing"].text, courses["graphic-design"].text
    assert "GS-DM-001" in dm
    assert "29 seats left" in dm
    assert "Portal Test Batch" in gd  # data defect kept verbatim so the advisor can caveat it
    assert "only 4 seats left" in gd


def test_course_modules_and_learn_lists(courses: dict[str, Chunk]) -> None:
    assert "Modules: 1 Store setup, 2 Themes, 3 Dropshipping, 4 Marketing (3 lessons each)" in courses["shopify"].text
    assert "Learn: ChatGPT, Make.com, Workflows, AI tools." in courses["ai-automation"].text


MENTORS = {
    "Hamza Ali": ("E-commerce mentor", "Shopify, eBay"),
    "Sana Iqbal": ("Social commerce", "TikTok Shop, TikTok automation"),
    "Bilal Ahmed": ("Video editor / video editing mentor", "Video editing"),
    "Mahnoor Raza": ("Growth marketer", "Digital marketing"),
    "Zara Malik": ("Brand designer", "Graphic design"),
    "Usman Tariq": ("Automation engineer", "AI automation"),
}


@pytest.mark.parametrize("name", MENTORS)
def test_mentor_chunks(chunks: list[Chunk], name: str) -> None:
    role, teaches = MENTORS[name]
    found = [c for c in _by_type(chunks, DocType.MENTOR) if c.metadata.mentor == name]
    assert len(found) == 1
    chunk = found[0]
    assert chunk.metadata.section == f"Mentor: {name}"
    assert chunk.text.splitlines()[0] == f"Mentor: {name} | Role: {role} | Teaches: {teaches}"
    assert "reviews student assignments personally" in chunk.text
    assert chunk.metadata.page == 4


def test_every_course_mentor_has_a_mentor_chunk(chunks: list[Chunk], courses: dict[str, Chunk]) -> None:
    mentors = {c.metadata.mentor for c in _by_type(chunks, DocType.MENTOR)}
    assert {c.metadata.mentor for c in courses.values()} <= mentors


@pytest.mark.parametrize(
    ("section", "page", "must_contain"),
    [
        (
            "About Genz Skool",
            1,
            ["Sargodha, Pakistan", "8 practical courses", "Crawled 12 Nov 2026", "Lifetime access"],
        ),
        ("Course structure", 3, ["3-4 modules of 3 lessons", "Walkthrough (module topic)"]),
        ("How it works", 4, ["Step 1 Pick a skill", "Step 6 Start earning", "weekly live classes"]),
        ("Enrollment", 4, ["/enroll", "No payment is taken on the website", "Under 18"]),
        ("FAQ and certificate", 4, ["Will I get a certificate? Yes.", "Fiverr or Upwork"]),
        (
            "Known data limitations",
            5,
            [
                "Portal Test Batch",
                "must not promise earnings",
                "0309-4190245",
                "TikTok automation has no published fee",
                "placeholder names",
            ],
        ),
    ],
)
def test_policy_chunks(chunks: list[Chunk], section: str, page: int, must_contain: list[str]) -> None:
    found = [c for c in _by_type(chunks, DocType.POLICY) if c.metadata.section == section]
    assert len(found) == 1, section
    chunk = found[0]
    assert chunk.metadata.page == page
    for fact in must_contain:
        assert fact in chunk.text, f"{section} missing: {fact}"


def test_faq_spans_page_break(chunks: list[Chunk]) -> None:
    faq = next(c for c in chunks if c.metadata.section == "FAQ and certificate")
    assert "Questions listed:" in faq.text  # page 4
    assert "Only one answer was visible" in faq.text  # page 5


def test_contact_chunk(chunks: list[Chunk]) -> None:
    (contact,) = _by_type(chunks, DocType.CONTACT)
    for fact in [
        "Farhan Plaza 1st Floor, Office No. 6, Near 47 Pull, Mall Road, Sargodha",
        "0309-4190245",
        "https://wa.me/923094190245",
        "info@genzskool.com",
        "Mon-Sat, 10am-8pm",
    ]:
        assert fact in contact.text


def test_overview_lists_all_courses_with_fees(chunks: list[Chunk]) -> None:
    (overview,) = _by_type(chunks, DocType.OVERVIEW)
    assert "The catalogue lists 8 courses:" in overview.text
    for _, title, fee, *_ in COURSES:
        line = next(ln for ln in overview.text.splitlines() if ln.startswith(f"- {title}"))
        if fee is None:
            assert "Not published" in line
        else:
            assert f"Rs. {fee:,}" in line
    assert overview.metadata.fee_pkr is None  # overview carries no single fee: filters must not match it


def test_chunk_size_limits(chunks: list[Chunk]) -> None:
    assert all(count_tokens(c.text) <= 400 for c in chunks)
    assert all(count_tokens(c.text) >= 15 for c in chunks)


def test_every_chunk_has_context_header(chunks: list[Chunk]) -> None:
    for c in chunks:
        header = c.text.splitlines()[0]
        assert header.startswith(("Course: ", "Mentor: ", "Genz Skool | Section: "))
        assert " | " in header


def test_no_page_furniture_or_table_garble(chunks: list[Chunk]) -> None:
    for c in chunks:
        assert "crawled 12 Nov 2026 from genzskool.com Page" not in c.text
        assert "Lessons / Level Mentor Fee" not in c.text  # raw table header text must not leak


def test_ids_unique_and_metadata_consistent(chunks: list[Chunk]) -> None:
    assert len({c.id for c in chunks}) == len(chunks)
    assert len({c.metadata.section for c in chunks}) == len(chunks)
    for c in chunks:
        m = c.metadata
        assert m.doc_id == "genzskool_knowledge_base"
        assert m.source_file == REAL_PDF.name
        assert m.index_version == 1
        assert m.ingested_at == INGESTED_AT
        assert len(m.content_hash) == 64


def test_deterministic_across_runs(cleaned: CleanedDocument, chunks: list[Chunk]) -> None:
    again = StructureAwareChunker(ChunkingConfig()).chunk(cleaned, index_version=1, ingested_at=INGESTED_AT).chunks
    assert [c.model_dump() for c in again] == [c.model_dump() for c in chunks]


def test_new_version_changes_ids_but_not_text(cleaned: CleanedDocument, chunks: list[Chunk]) -> None:
    v2 = StructureAwareChunker(ChunkingConfig()).chunk(cleaned, index_version=2, ingested_at=INGESTED_AT).chunks
    assert [c.text for c in v2] == [c.text for c in chunks]
    assert not {c.id for c in v2} & {c.id for c in chunks}


_PUNCT = "()[];,.:'\"|"


def _norm_word(word: str) -> str:
    return word.strip(_PUNCT)


def _source_words() -> set[str]:
    doc = PdfPlumberLoader().load(REAL_PDF.read_bytes(), SOURCE, "hash")
    text = " ".join([p.text for p in doc.pages] + [c or "" for p in doc.pages for t in p.tables for r in t for c in r])
    return {_norm_word(w) for w in normalize_text(text).split()}


# Words the chunker itself adds as field labels or section names (never facts).
_LABELS = {
    "Course",
    "Mentor",
    "Role",
    "Teaches",
    "Category",
    "Level",
    "Fee",
    "Genz",
    "Skool",
    "Section",
    "Catalogue",
    "entry",
    "Lessons",
    "/",
    "length",
    "Length",
    "The",
    "catalogue",
    "lists",
    "courses",
    "-",
    "Page",
    "New",
    "8",
    "Mentors",
    "structure",
    "overview",
    "FAQ",
    "and",
    "certificate",
    "Known",
    "data",
    "limitations",
    "Enrollment",
    "How",
    "it",
    "works",
    "About",
    "Contact",
    "location",
    "",
}


def test_no_invented_words(chunks: list[Chunk]) -> None:
    """Every word in every chunk comes from the PDF (or is one of the chunker's own field labels)."""
    source = _source_words()
    for c in chunks:
        for word in c.text.split():
            core = _norm_word(word)
            assert core in source or core in _LABELS, f"'{word}' in {c.metadata.section} not found in the source PDF"


def test_no_invented_numbers(chunks: list[Chunk]) -> None:
    """Stricter: every number (fees, counts, durations, phone) must appear in the raw PDF text verbatim."""
    doc = PdfPlumberLoader().load(REAL_PDF.read_bytes(), SOURCE, "hash")
    raw = normalize_text(
        " ".join([p.text for p in doc.pages] + [c or "" for p in doc.pages for t in p.tables for r in t for c in r])
    )
    raw = re.sub(r"\s+", " ", raw)
    for c in chunks:
        for number in re.findall(r"\d[\d,]*\d|\d", c.text):
            assert number in raw, f"number {number} in {c.metadata.section} is not in the PDF"


def test_catalogue_parser_reads_both_pages(cleaned: CleanedDocument) -> None:
    facts = parse_catalogue(cleaned.tables)
    assert len(facts) == 8
    assert {f.page for f in facts} == {1, 2}
    assert all(f.title and f.fee_text for f in facts)
