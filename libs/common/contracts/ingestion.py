"""Ingestion service API contracts (v1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


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
    pruned: list[str] = Field(default_factory=list, description="doc_ids removed because their PDF was deleted.")
    started_at: datetime
    finished_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """False if any file failed. Unchanged and scanned-skipped files are not failures."""
        return all(f.outcome != FileOutcome.FAILED for f in self.files)


class DocumentStatus(BaseModel):
    doc_id: str
    source_file: str
    content_hash: str
    active_index_version: int
    chunks: int
    ingested_at: datetime


class AutoIngestTrigger(StrEnum):
    STARTUP = "startup"
    FILE_CHANGE = "file_change"
    INDEX_DRIFT = "index_drift"
    RETRY = "retry"


class AutoIngestStatus(BaseModel):
    """State of the background auto-ingester (startup run + PDF folder watcher)."""

    on_startup: bool
    watch_interval_s: float = Field(description="0 means the folder watcher is disabled.")
    prune_deleted: bool
    runs: int = 0
    last_run_at: datetime | None = None
    last_trigger: AutoIngestTrigger | None = None
    last_ok: bool | None = None
    last_error: str | None = None
    consecutive_failures: int = 0


class IngestStatus(BaseModel):
    documents: list[DocumentStatus]
    auto_ingest: AutoIngestStatus | None = None
