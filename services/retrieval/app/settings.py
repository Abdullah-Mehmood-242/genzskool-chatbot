from __future__ import annotations

from common.config import EmbeddingSettings, VectorStoreSettings


class Settings(EmbeddingSettings, VectorStoreSettings):
    service_name: str = "retrieval"
    max_body_bytes: int = 8_192
