"""BaseSettings helpers: env validation, placeholder detection, YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.errors import ConfigurationError
from common.logging.setup import secret_registry

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def is_placeholder(value: str | SecretStr | None) -> bool:
    """True for empty values and the ``YOUR_..._HERE`` placeholders from .env.example."""
    if value is None:
        return True
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    raw = raw.strip()
    return not raw or (raw.startswith("YOUR_") and raw.endswith("_HERE"))


class ServiceSettings(BaseSettings):
    """Settings every service shares. Services subclass and add their own fields."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "service"
    service_version: str = "0.1.0"
    log_level: LogLevel = "INFO"
    config_dir: Path = Path("config")
    request_timeout_s: float = 10.0
    max_body_bytes: int = 64_000

    def register_secrets(self) -> None:
        """Register every SecretStr field so the log formatter masks it."""
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr) and not is_placeholder(value):
                secret_registry.register(value.get_secret_value())

    def validate_runtime(self) -> None:
        """Cross-field checks (e.g. real provider needs a real key). Subclasses extend; call super()."""


EmbeddingProviderName = Literal["fake", "voyage", "openai", "sentence_transformers"]
LLMProviderName = Literal["fake", "anthropic", "openai", "ollama"]
VectorStoreName = Literal["qdrant", "memory"]


class EmbeddingSettings(ServiceSettings):
    """Shared by ingestion and retrieval: both MUST use the same provider, model and dimension."""

    embedding_provider: EmbeddingProviderName = "fake"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = ""
    embedding_dim: int = Field(default=1024, ge=8, le=8192)
    embedding_base_url: str | None = None

    def validate_runtime(self) -> None:
        super().validate_runtime()
        real = self.embedding_provider != "fake"
        require_real_secret(
            "EMBEDDING_API_KEY", self.embedding_api_key, when=self.embedding_provider in ("voyage", "openai")
        )
        require_real_secret("EMBEDDING_MODEL", self.embedding_model, when=real)


class VectorStoreSettings(ServiceSettings):
    vector_store: VectorStoreName = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = Field(default="genzskool_kb", pattern=r"^[A-Za-z0-9_\-]+$")
    processed_dir: Path = Path("data/processed")


class LLMSettings(ServiceSettings):
    llm_provider: LLMProviderName = "fake"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    # Thinking tokens count toward this on current Claude models: a low cap truncates answers.
    llm_max_tokens: int = Field(default=16000, ge=64, le=128000)
    llm_base_url: str | None = None

    def validate_runtime(self) -> None:
        super().validate_runtime()
        require_real_secret("LLM_API_KEY", self.llm_api_key, when=self.llm_provider in ("anthropic", "openai"))
        require_real_secret("LLM_MODEL", self.llm_model, when=self.llm_provider != "fake")


def load_settings[S: ServiceSettings](cls: type[S], **overrides: Any) -> S:
    """Instantiate and validate settings; any problem becomes a fail-fast ConfigurationError.

    Error messages name the offending fields but never echo their values (they may be secrets).
    """
    try:
        settings = cls(**overrides)
    except ValidationError as exc:
        fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()]
        raise ConfigurationError(f"Invalid configuration for fields: {', '.join(fields)}") from None
    settings.validate_runtime()
    settings.register_secrets()
    return settings


def require_real_secret(name: str, value: SecretStr | str | None, *, when: bool) -> None:
    """Fail fast when a real provider is selected but its key is still a placeholder."""
    if when and is_placeholder(value):
        raise ConfigurationError(f"{name} is required for the selected provider but is unset or a placeholder.")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Config file not found: {path.name}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file must contain a mapping: {path.name}")
    return data


def load_yaml_model[M: BaseModel](path: Path, model: type[M]) -> M:
    """Load a YAML file and validate it against a Pydantic model (fail fast)."""
    try:
        return model.model_validate(load_yaml(path))
    except ValidationError as exc:
        fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()]
        raise ConfigurationError(f"Invalid values in {path.name}: {', '.join(fields)}") from None


__all__ = [
    "EmbeddingProviderName",
    "EmbeddingSettings",
    "LLMProviderName",
    "LLMSettings",
    "LogLevel",
    "ServiceSettings",
    "VectorStoreName",
    "VectorStoreSettings",
    "is_placeholder",
    "load_settings",
    "load_yaml",
    "load_yaml_model",
    "require_real_secret",
]
