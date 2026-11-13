from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from common.config import LLMSettings, require_real_secret


class Settings(LLMSettings):
    service_name: str = "advisor"
    max_body_bytes: int = 16_384
    prompts_dir: Path = Path("prompts")

    # LLM (provider/key/model/max tokens/temperature come from LLMSettings)
    llm_effort: Literal["", "low", "medium", "high", "xhigh", "max"] = Field(
        default="low", description="Anthropic effort; '' omits it (for models without effort support)."
    )
    llm_refusal_fallback: bool = Field(default=True, description="Server-side refusal fallback on supported models.")
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # Retrieval service
    retrieval_url: str = "http://retrieval:8002"
    retrieval_timeout_s: float = Field(default=8.0, gt=0)

    # Sessions (ADR-0007)
    session_store: Literal["memory", "redis"] = "memory"
    redis_url: str = ""
    session_ttl_s: int = Field(default=3600, ge=60)
    session_max: int = Field(default=10_000, ge=10)

    def validate_runtime(self) -> None:
        super().validate_runtime()
        require_real_secret("REDIS_URL", self.redis_url, when=self.session_store == "redis")
