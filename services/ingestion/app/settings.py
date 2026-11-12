from __future__ import annotations

from pathlib import Path

from pydantic import Field

from common.config import EmbeddingSettings, VectorStoreSettings


class Settings(EmbeddingSettings, VectorStoreSettings):
    service_name: str = "ingestion"
    pdf_dir: Path = Path("data/pdfs")
    max_body_bytes: int = 4_096
    max_pdf_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    qdrant_timeout_s: float = Field(default=10.0, gt=0)

    # Embedding calls
    embed_batch_size: int = Field(default=64, ge=1, le=512)
    embed_retry_attempts: int = Field(default=3, ge=1, le=10)
    embed_timeout_s: float = Field(default=30.0, gt=0)

    # Auto-ingest (ADR-0011)
    auto_ingest_on_startup: bool = True
    auto_ingest_interval_s: float = Field(default=30.0, ge=0, description="0 disables the PDF folder watcher.")
    auto_ingest_prune: bool = Field(default=True, description="Remove documents whose PDF was deleted.")
    scanned_min_chars_per_page: int = Field(default=20, ge=0)
