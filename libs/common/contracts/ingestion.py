"""Ingestion service API contracts (v1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    file: str | None = Field(default=None, description="File name inside PDF_DIR; omit to ingest all PDFs.")
    force: bool = Field(default=False, description="Re-ingest even if the content hash is unchanged.")


class FileOutcome(StrEnum):
    INGESTED = "ingested"
    UNCHANGED = "unchanged"
    FAILED = "failed"
    SKIPPED_SCANNED = "skipped_scanned"


class FileReport(BaseModel):
    file: str
    doc_id: str
    outcome: FileOutcome
    chunks: int = 0
    index_version: int | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class IngestReport(BaseModel):
    files: list[FileReport]
    started_at: datetime
    finished_at: datetime

    @property
    def ok(self) -> bool:
        return all(f.outcome != FileOutcome.FAILED for f in self.files)


class DocumentStatus(BaseModel):
    doc_id: str
    source_file: str
    content_hash: str
    active_index_version: int
    chunks: int
    ingested_at: datetime


class IngestStatus(BaseModel):
    documents: list[DocumentStatus]
