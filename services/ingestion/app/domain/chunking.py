"""Structure-aware chunking (ADR-0004): one chunk per course, mentor and policy section.

Pure domain logic. Every value placed in metadata or a context header is copied from the source;
nothing is inferred or invented. When the layout is not recognised the chunker falls back to
per-page chunks and records a warning instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from app.domain.models import ChunkingConfig, ChunkResult, CleanedDocument, Line, Table
from app.domain.ports import Chunker
from app.domain.text import (
    chunk_id,
    collapse_ws,
    count_tokens,
    parse_fee_pkr,
    parse_level,
    sha256_hex,
    slugify,
    split_by_tokens,
)
from common.contracts import Chunk, ChunkMetadata, DocType, Level

_HEADING_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z].{1,100})$")
_COURSE_URL_RE = re.compile(r"^Page:\s*(https?://\S+/courses/([a-z0-9-]+))\s*$", re.IGNORECASE)
_MENTOR_RE = re.compile(r"^-\s*(?P<name>.+?)\s+-\s+(?P<role>.+?)\s+-\s+teaches\s+(?P<teaches>.+?)\.?$")
_DETAIL_MENTOR_RE = re.compile(r"Mentor:\s*([A-Z][\w'-]+(?:\s[A-Z][\w'-]+)+)")
_DETAIL_LEVEL_RE = re.compile(r"Level:\s*(\w+)")
_NEW_RE = re.compile(r"\s*\(New\)\s*$", re.IGNORECASE)
_TERMINAL = (".", ":", ",", ";", "?", "!", ")")

# Canonical section names (stable citation targets), chosen by keywords in the PDF heading.
_SECTION_KINDS: list[tuple[str, str, str]] = [
    ("about", "about", "About Genz Skool"),
    ("contact", "contact", "Contact and location"),
    ("catalog", "catalogue", "Course catalogue overview"),
    ("course detail", "details", "Course structure"),
    ("mentor", "mentors", "Mentors"),
    ("how it works", "policy", "How it works"),
    ("enrol", "policy", "Enrollment"),
    ("faq", "policy", "FAQ and certificate"),
    ("limitation", "policy", "Known data limitations"),
]


@dataclass
class _Section:
    number: int
    heading: str
    page: int
    lines: list[Line] = field(default_factory=list)

    @property
    def kind_and_name(self) -> tuple[str, str]:
        lowered = self.heading.lower()
        for keyword, kind, name in _SECTION_KINDS:
            if keyword in lowered:
                return kind, name
        return "policy", re.sub(r"\s*\(.*\)\s*$", "", self.heading)


@dataclass(frozen=True)
class CourseFacts:
    title: str
    is_new: bool
    category: str | None
    length: str | None
    level_text: str | None
    mentor: str | None
    fee_text: str | None
    page: int

    @property
    def fee_pkr(self) -> int | None:
        return parse_fee_pkr(self.fee_text)


@dataclass(frozen=True)
class _CourseDetail:
    title: str
    url: str
    slug: str
    page: int
    body: str


@dataclass(frozen=True)
class _Draft:
    """A chunk before ids/versions are assigned."""

    section: str
    doc_type: DocType
    page: int
    header: str
    body: str
    course_slug: str | None = None
    category: str | None = None
    level: Level | None = None
    fee_pkr: int | None = None
    mentor: str | None = None


# ------------------------------------------------------------------------------------------------
# Line -> paragraph helpers
# ------------------------------------------------------------------------------------------------


def _is_bullet(text: str) -> bool:
    return text.startswith("- ")


def _is_subheading(text: str, nxt: str | None) -> bool:
    return (
        len(text.split()) <= 8
        and not text.endswith(_TERMINAL)
        and not _is_bullet(text)
        and nxt is not None
        and _is_bullet(nxt)
    )


def paragraphs(lines: list[str]) -> list[str]:
    """Re-join soft-wrapped PDF lines. Bullets and sub-headings start new paragraphs."""
    out: list[str] = []
    for i, text in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        # A line right after an unfinished bullet is that bullet's wrapped continuation, never a sub-heading.
        continues_bullet = bool(out) and _is_bullet(out[-1]) and not out[-1].endswith(_TERMINAL)
        starts_new = (
            not out
            or _is_bullet(text)
            or (not continues_bullet and _is_subheading(text, nxt))
            or _is_subheading(out[-1], text)
        )
        if starts_new:
            out.append(text)
        else:
            out[-1] = f"{out[-1]} {text}"
    return out


def _norm_title(title: str) -> str:
    return collapse_ws(_NEW_RE.sub("", title)).lower()


# ------------------------------------------------------------------------------------------------
# Catalogue table
# ------------------------------------------------------------------------------------------------

_COLUMNS = {
    "title": ("course",),
    "category": ("category",),
    "length": ("lesson", "length"),
    "level": ("level",),
    "mentor": ("mentor",),
    "fee": ("fee", "price"),
}


def _column_map(header: list[str | None]) -> dict[str, int] | None:
    names = [(c or "").lower() for c in header]
    mapping: dict[str, int] = {}
    for key, keywords in _COLUMNS.items():
        for idx, name in enumerate(names):
            if any(k in name for k in keywords) and idx not in mapping.values():
                mapping[key] = idx
                break
    return mapping if {"title", "fee"} <= mapping.keys() else None


def parse_catalogue(tables: list[tuple[int, Table]]) -> list[CourseFacts]:
    """Read course rows from every table whose header looks like the catalogue (repeated headers ok)."""
    courses: list[CourseFacts] = []
    seen: set[str] = set()
    for page, table in tables:
        if not table:
            continue
        mapping = _column_map(table[0])
        if mapping is None:
            continue
        for row in table[1:]:

            def cell(key: str, row: list[str | None] = row, mapping: dict[str, int] = mapping) -> str | None:
                idx = mapping.get(key)
                value = row[idx] if idx is not None and idx < len(row) else None
                return collapse_ws(value) or None

            raw_title = cell("title")
            if not raw_title or _column_map(row) is not None:  # empty row or a repeated header row
                continue
            key = _norm_title(raw_title)
            if key in seen:
                continue
            seen.add(key)
            courses.append(
                CourseFacts(
                    title=collapse_ws(_NEW_RE.sub("", raw_title)),
                    is_new=bool(_NEW_RE.search(raw_title)),
                    category=cell("category"),
                    length=cell("length"),
                    level_text=cell("level"),
                    mentor=cell("mentor"),
                    fee_text=cell("fee"),
                    page=page,
                )
            )
    return courses


# ------------------------------------------------------------------------------------------------
# The chunker
# ------------------------------------------------------------------------------------------------


class StructureAwareChunker(Chunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.warnings: list[str] = []

    # ---- public API ----

    def chunk(self, doc: CleanedDocument, *, index_version: int, ingested_at: datetime) -> ChunkResult:
        self.warnings = []
        drafts = self.drafts(doc)
        chunks = self._finalise(doc, drafts, index_version=index_version, ingested_at=ingested_at)
        return ChunkResult(chunks=chunks, warnings=list(self.warnings))

    def drafts(self, doc: CleanedDocument) -> list[_Draft]:
        preamble, sections = self._split_sections(doc.lines)
        if not sections:
            self.warnings.append("No numbered sections found; fell back to per-page chunks.")
            return self._per_page(doc.lines)

        courses = parse_catalogue(doc.tables)
        details: list[_CourseDetail] = []
        drafts: list[_Draft] = []
        for section in sections:
            kind, name = section.kind_and_name
            texts = [ln.text for ln in section.lines]
            if kind == "about":
                body_lines = [ln.text for ln in preamble] + texts
                drafts.append(self._policy(name, section.page, body_lines))
            elif kind == "contact":
                drafts.append(self._policy(name, section.page, texts, doc_type=DocType.CONTACT))
            elif kind == "catalogue":
                drafts.extend(self._catalogue(name, section, courses))
            elif kind == "details":
                intro, details = self._split_details(section)
                if intro:
                    drafts.append(self._policy(name, section.page, intro))
            elif kind == "mentors":
                drafts.extend(self._mentors(section))
            else:
                drafts.append(self._policy(name, section.page, texts))

        drafts.extend(self._courses(courses, details))
        if not any(s.kind_and_name[0] == "about" for s in sections) and preamble:
            drafts.insert(0, self._policy("Document overview", preamble[0].page, [ln.text for ln in preamble]))
        return [d for d in drafts if d.body.strip()]

    # ---- sectioning ----

    def _split_sections(self, lines: list[Line]) -> tuple[list[Line], list[_Section]]:
        preamble: list[Line] = []
        sections: list[_Section] = []
        for line in lines:
            match = _HEADING_RE.match(line.text)
            expected = sections[-1].number + 1 if sections else 1
            if match and int(match.group(1)) == expected:
                sections.append(_Section(number=expected, heading=match.group(2).strip(), page=line.page))
            elif sections:
                sections[-1].lines.append(line)
            else:
                preamble.append(line)
        return preamble, sections

    def _per_page(self, lines: list[Line]) -> list[_Draft]:
        pages: dict[int, list[str]] = {}
        for ln in lines:
            pages.setdefault(ln.page, []).append(ln.text)
        return [self._policy(f"Page {page}", page, texts) for page, texts in sorted(pages.items())]

    # ---- builders ----

    def _policy(self, name: str, page: int, texts: list[str], doc_type: DocType = DocType.POLICY) -> _Draft:
        return _Draft(
            section=name,
            doc_type=doc_type,
            page=page,
            header=f"Genz Skool | Section: {name}",
            body="\n".join(paragraphs(texts)),
        )

    def _catalogue(self, name: str, section: _Section, courses: list[CourseFacts]) -> list[_Draft]:
        prose = [ln.text for ln in section.lines]
        if not courses:
            self.warnings.append("Course catalogue table not found; catalogue kept as plain text.")
            return [self._policy("Course catalogue", section.page, prose)]
        rows = [
            "- "
            + "; ".join(
                part
                for part in (
                    f"{c.title}{' (New)' if c.is_new else ''}",
                    f"Fee: {c.fee_text}" if c.fee_text else None,
                    f"Level: {c.level_text}" if c.level_text else None,
                    f"Mentor: {c.mentor}" if c.mentor else None,
                    f"Category: {c.category}" if c.category else None,
                    f"Length: {c.length}" if c.length else None,
                )
                if part
            )
            for c in courses
        ]
        body = "\n".join([*paragraphs(prose), f"The catalogue lists {len(courses)} courses:", *rows])
        return [
            _Draft(
                section=name,
                doc_type=DocType.OVERVIEW,
                page=section.page,
                header=f"Genz Skool | Section: {name}",
                body=body,
            )
        ]

    def _split_details(self, section: _Section) -> tuple[list[str], list[_CourseDetail]]:
        intro: list[str] = []
        blocks: list[tuple[str, str, str, int, list[str]]] = []  # title, url, slug, page, lines
        lines = section.lines
        for i, line in enumerate(lines):
            match = _COURSE_URL_RE.match(line.text)
            if match:
                title = lines[i - 1].text if i else ""
                # the title line was appended to the previous block (or intro); move it
                if blocks and blocks[-1][4] and blocks[-1][4][-1] == title:
                    blocks[-1][4].pop()
                elif intro and intro[-1] == title:
                    intro.pop()
                blocks.append(
                    (title, match.group(1), match.group(2).lower(), lines[i - 1].page if i else line.page, [])
                )
            elif blocks:
                blocks[-1][4].append(line.text)
            else:
                intro.append(line.text)
        details = [
            _CourseDetail(
                title=title,
                url=url,
                slug=slug,
                page=page,
                body="\n".join([f"Page: {url}", *paragraphs(body_lines)]),
            )
            for title, url, slug, page, body_lines in blocks
        ]
        return intro, details

    def _mentors(self, section: _Section) -> list[_Draft]:
        paras = paragraphs([ln.text for ln in section.lines])
        intro = [p for p in paras if not _is_bullet(p)]
        drafts: list[_Draft] = []
        unparsed: list[str] = []
        for p in paras:
            if not _is_bullet(p):
                continue
            match = _MENTOR_RE.match(p)
            if not match:
                unparsed.append(p)
                continue
            name, role, teaches = (collapse_ws(match.group(g)) for g in ("name", "role", "teaches"))
            drafts.append(
                _Draft(
                    section=f"Mentor: {name}",
                    doc_type=DocType.MENTOR,
                    page=section.page,
                    header=f"Mentor: {name} | Role: {role} | Teaches: {teaches}",
                    body="\n".join([p, *intro]),
                    mentor=name,
                )
            )
        if unparsed or not drafts:
            if unparsed:
                self.warnings.append(f"{len(unparsed)} mentor line(s) not in 'Name - Role - teaches X' form.")
            drafts.append(self._policy("Mentors", section.page, [*intro, *unparsed]))
        return drafts

    def _courses(self, courses: list[CourseFacts], details: list[_CourseDetail]) -> list[_Draft]:
        by_title = {_norm_title(d.title): d for d in details}
        used: set[str] = set()
        drafts: list[_Draft] = []
        for facts in courses:
            key = _norm_title(facts.title)
            detail = by_title.get(key) or next(
                (d for k, d in by_title.items() if k not in used and (k.startswith(key) or key.startswith(k))), None
            )
            if detail is None:
                self.warnings.append(f"No detail section for course '{facts.title}'; using catalogue row only.")
            else:
                used.add(_norm_title(detail.title))
            drafts.append(self._course_draft(facts, detail))
        for key, detail in by_title.items():
            if key not in used:
                self.warnings.append(f"Course detail '{detail.title}' has no catalogue row; metadata from text only.")
                drafts.append(self._course_draft(None, detail))
        return drafts

    def _course_draft(self, facts: CourseFacts | None, detail: _CourseDetail | None) -> _Draft:
        assert facts or detail
        title = facts.title if facts else detail.title  # type: ignore[union-attr]
        detail_body = detail.body if detail else ""
        mentor = facts.mentor if facts and facts.mentor else _first(_DETAIL_MENTOR_RE, detail_body)
        level_text = facts.level_text if facts and facts.level_text else _first(_DETAIL_LEVEL_RE, detail_body)
        fee_text = facts.fee_text if facts else None
        fee_pkr = facts.fee_pkr if facts else parse_fee_pkr(_fee_sentence(detail_body))
        header_parts = [f"Course: {title}"]
        if facts and facts.category:
            header_parts.append(f"Category: {facts.category}")
        if level_text:
            header_parts.append(f"Level: {level_text}")
        if mentor:
            header_parts.append(f"Mentor: {mentor}")
        if fee_text:
            header_parts.append(f"Fee: {fee_text}")
        catalogue_line = None
        if facts:
            catalogue_line = "Catalogue entry: " + "; ".join(
                p
                for p in (
                    f"{facts.title}{' (New)' if facts.is_new else ''}",
                    f"Category: {facts.category}" if facts.category else None,
                    f"Lessons / length: {facts.length}" if facts.length else None,
                    f"Level: {facts.level_text}" if facts.level_text else None,
                    f"Mentor: {facts.mentor}" if facts.mentor else None,
                    f"Fee: {facts.fee_text}" if facts.fee_text else None,
                )
                if p
            )
        return _Draft(
            section=f"Course: {title}",
            doc_type=DocType.COURSE,
            page=detail.page if detail else facts.page,  # type: ignore[union-attr]
            header=" | ".join(header_parts),
            body="\n".join(p for p in (catalogue_line, detail_body) if p),
            course_slug=detail.slug if detail else slugify(title),
            category=facts.category if facts else None,
            level=parse_level(level_text),
            fee_pkr=fee_pkr,
            mentor=mentor,
        )

    # ---- finalisation: headers, size limits, ids, dedup ----

    def _finalise(
        self, doc: CleanedDocument, drafts: list[_Draft], *, index_version: int, ingested_at: datetime
    ) -> list[Chunk]:
        cfg = self.config
        chunks: list[Chunk] = []
        seen_text: set[str] = set()
        for draft in drafts:
            header = draft.header if cfg.context_header else ""
            header_tokens = count_tokens(header)
            budget = cfg.max_tokens - header_tokens
            if budget < 20:
                raise ValueError(f"max_tokens={cfg.max_tokens} leaves no room for the header of '{draft.section}'")
            if count_tokens(draft.body) <= budget:
                parts = [draft.body]
            else:
                parts = split_by_tokens(
                    draft.body.split("\n"),
                    max_tokens=budget,
                    overlap_tokens=min(cfg.overlap_tokens, budget // 4),
                    min_tokens=cfg.min_chunk_tokens,
                )
            for i, part in enumerate(parts, start=1):
                section = draft.section if len(parts) == 1 else f"{draft.section} (part {i}/{len(parts)})"
                text = f"{header}\n{part}" if header else part
                text_hash = sha256_hex(text)
                if text_hash in seen_text:
                    self.warnings.append(f"Duplicate chunk skipped: {section}")
                    continue
                seen_text.add(text_hash)
                meta = ChunkMetadata(
                    doc_id=doc.doc_id,
                    source_file=doc.source_file,
                    page=draft.page,
                    section=section,
                    doc_type=draft.doc_type,
                    course_slug=draft.course_slug,
                    category=draft.category,
                    level=draft.level,
                    fee_pkr=draft.fee_pkr,
                    mentor=draft.mentor,
                    content_hash=text_hash,
                    ingested_at=ingested_at,
                    index_version=index_version,
                )
                chunks.append(Chunk(id=chunk_id(doc.doc_id, index_version, section, text), text=text, metadata=meta))
        return chunks


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _fee_sentence(text: str) -> str | None:
    match = re.search(r"Fee:\s*([^.]*(?:\.\s*\d[^.]*)?)", text)
    return match.group(1) if match else None
