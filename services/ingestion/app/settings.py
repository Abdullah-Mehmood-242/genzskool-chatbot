from __future__ import annotations

from pathlib import Path

from common.config import EmbeddingSettings, VectorStoreSettings


class Settings(EmbeddingSettings, VectorStoreSettings):
    service_name: str = "ingestion"
    pdf_dir: Path = Path("data/pdfs")
    max_body_bytes: int = 4_096
