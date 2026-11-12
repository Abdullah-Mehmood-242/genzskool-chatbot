"""Ingestion domain models. Pure Python: no I/O, no third-party imports beyond pydantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from common.contracts import Chunk

Table = list[list[str | None]]


@dataclass(frozen=True)
class SourceFile:
    """A PDF available for ingestion."""

    file: str  # file name inside the PDF directory, e.g. "genzskool_knowledge_base.pdf"
    doc_id: str  # stable id derived from the file name


@dataclass(frozen=True)
class SourceFingerprint:
    """Cheap change-detection signature of a source file (name, size, mtime)."""

    file: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class PageContent:
    page: int  # 1-based
    text: str  # text OUTSIDE table regions
    tables: list[Table] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    source_file: str
    content_hash: str
    pages: list[PageContent]
    warnings: list[str] = field(default_factory=list)

    @property
    def text_chars(self) -> int:
        return sum(len(p.text.strip()) for p in self.pages) + sum(
            len(c or "") for p in self.pages for t in p.tables for row in t for c in row
        )


@dataclass(frozen=True)
class Line:
    """One cleaned line of text with the page it came from."""

    page: int
    text: str


@dataclass(frozen=True)
class CleanedDocument:
    doc_id: str
    source_file: str
    content_hash: str
    lines: list[Line]
    tables: list[tuple[int, Table]]  # (page, table)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkResult:
    chunks: list[Chunk]
    warnings: list[str] = field(default_factory=list)


def is_scanned(doc: ParsedDocument, min_chars_per_page: int = 20) -> bool:
    """A PDF whose pages carry (almost) no extractable text is most likely a scan (OCR is out of scope)."""
    return bool(doc.pages) and doc.text_chars < min_chars_per_page * len(doc.pages)


@dataclass(frozen=True)
class IndexRecord:
    """Registry entry: which index version of a document is active (the blue/green pointer)."""

    doc_id: str
    source_file: str
    content_hash: str
    active_index_version: int
    chunks: int
    ingested_at: datetime


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 400
    overlap_tokens: int = 40
    min_chunk_tokens: int = 15
    context_header: bool = True
    strip_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens < 20:
            raise ValueError("max_tokens must be >= 20")
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError("overlap_tokens must be >= 0 and < max_tokens / 2")
        if not 0 <= self.min_chunk_tokens < self.max_tokens:
            raise ValueError("min_chunk_tokens must be >= 0 and < max_tokens")
