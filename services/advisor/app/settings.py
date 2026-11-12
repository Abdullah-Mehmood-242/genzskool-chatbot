from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from common.config import LLMSettings, require_real_secret


class Settings(LLMSettings):
    service_name: str = "advisor"
    retrieval_url: str = "http://retrieval:8002"
    prompts_dir: Path = Path("prompts")
    session_store: Literal["memory", "redis"] = "memory"
    redis_url: str = ""
    session_ttl_s: int = Field(default=3600, ge=60)
    max_body_bytes: int = 16_384

    def validate_runtime(self) -> None:
        super().validate_runtime()
        require_real_secret("REDIS_URL", self.redis_url, when=self.session_store == "redis")
