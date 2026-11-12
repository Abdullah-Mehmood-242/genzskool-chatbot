from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr

from common.config import EmbeddingSettings, VectorStoreSettings, require_real_secret


class Settings(EmbeddingSettings, VectorStoreSettings):
    service_name: str = "retrieval"
    max_body_bytes: int = 8_192
    qdrant_timeout_s: float = Field(default=10.0, gt=0)
    embed_timeout_s: float = Field(default=15.0, gt=0)

    # Optional reranker (config/retrieval.yaml use_reranker must also be true)
    reranker_provider: Literal["none", "cohere", "voyage", "cross_encoder"] = "none"
    reranker_api_key: SecretStr = SecretStr("")
    reranker_model: str = ""
    reranker_timeout_s: float = Field(default=15.0, gt=0)

    def validate_runtime(self) -> None:
        super().validate_runtime()
        require_real_secret(
            "RERANKER_API_KEY", self.reranker_api_key, when=self.reranker_provider in ("cohere", "voyage")
        )
        require_real_secret("RERANKER_MODEL", self.reranker_model, when=self.reranker_provider != "none")
