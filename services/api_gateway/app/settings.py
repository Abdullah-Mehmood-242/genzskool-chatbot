from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field, SecretStr

from common.config import ServiceSettings, is_placeholder
from common.errors import ConfigurationError

logger = logging.getLogger(__name__)


class Settings(ServiceSettings):
    service_name: str = "api_gateway"
    app_env: Literal["dev", "prod"] = "dev"
    gateway_api_key: SecretStr = SecretStr("")
    admin_api_key: SecretStr = SecretStr("")
    rate_limit_per_minute: int = Field(default=30, ge=1)
    cors_origins: str = "http://localhost:3000"
    advisor_url: str = "http://advisor:8003"
    ingestion_url: str = "http://ingestion:8001"
    upstream_timeout_s: float = Field(default=30.0, gt=0)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip() and o.strip() != "*"]

    def validate_runtime(self) -> None:
        super().validate_runtime()
        weak = [
            name
            for name, value in (("GATEWAY_API_KEY", self.gateway_api_key), ("ADMIN_API_KEY", self.admin_api_key))
            if is_placeholder(value) or len(value.get_secret_value()) < 16
        ]
        if weak and self.app_env == "prod":
            raise ConfigurationError(f"{', '.join(weak)} must be set to a strong value (>= 16 chars) in prod.")
        if weak:
            # Dev only: the gateway still requires the configured key; it just warns that it is weak.
            logger.warning("weak_gateway_keys_dev_only", extra={"fields": weak})
