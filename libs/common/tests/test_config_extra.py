"""Settings loading edge cases: .env files, env precedence, secret registration, YAML corner cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from common.config import (
    EmbeddingSettings,
    LLMSettings,
    ServiceSettings,
    VectorStoreSettings,
    is_placeholder,
    load_settings,
    load_yaml,
    require_real_secret,
)
from common.errors import ConfigurationError
from common.logging import redact, secret_registry


class _S(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")
    api_key: SecretStr = SecretStr("")
    count: int = 1


class _All(LLMSettings, EmbeddingSettings, VectorStoreSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "API_KEY",
        "COUNT",
        "LOG_LEVEL",
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "SERVICE_NAME",
    ):
        monkeypatch.delenv(var, raising=False)


def test_service_settings_defaults() -> None:
    s = load_settings(_S)
    assert s.log_level == "INFO"
    assert s.config_dir == Path("config")
    assert s.request_timeout_s == 10.0
    assert s.max_body_bytes == 64_000


def test_reads_dotenv_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("COUNT=7\nLOG_LEVEL=DEBUG\nUNRELATED=ignored\n", encoding="utf-8")
    s = load_settings(_S, _env_file=env)
    assert s.count == 7
    assert s.log_level == "DEBUG"


def test_environment_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("COUNT=7\n", encoding="utf-8")
    monkeypatch.setenv("COUNT", "9")
    assert load_settings(_S, _env_file=env).count == 9


def test_env_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("count", "4")
    assert load_settings(_S).count == 4


def test_invalid_log_level_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigurationError, match="log_level"):
        load_settings(_S)


def test_error_message_never_contains_the_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "sk-looks-like-a-secret-123")
    with pytest.raises(ConfigurationError) as exc:
        load_settings(_All)
    assert "sk-looks-like-a-secret-123" not in str(exc.value)
    assert "embedding_dim" in str(exc.value)


def test_real_secrets_are_registered_for_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "registered-secret-value-42")
    load_settings(_S)
    assert "registered-secret-value-42" in secret_registry.snapshot()
    assert redact("key registered-secret-value-42") == "key ***"


def test_placeholder_secrets_are_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "YOUR_SOMETHING_HERE")
    load_settings(_S)
    assert "YOUR_SOMETHING_HERE" not in secret_registry.snapshot()


def test_all_provider_mixins_compose() -> None:
    s = load_settings(_All)
    assert (s.llm_provider, s.embedding_provider, s.vector_store) == ("fake", "fake", "qdrant")
    assert s.embedding_dim == 1024


def test_fake_providers_ignore_placeholder_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "YOUR_CHAT_MODEL_NAME_HERE")
    monkeypatch.setenv("EMBEDDING_MODEL", "YOUR_EMBEDDING_MODEL_NAME_HERE")
    load_settings(_All)


def test_both_real_providers_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "sk-real-llm-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-x")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        load_settings(_All)
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-real-emb-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-x")
    assert load_settings(_All).embedding_provider == "openai"


@pytest.mark.parametrize("value", ["YOUR_A_HERE", "YOUR__HERE", "  YOUR_X_HERE  "])
def test_placeholder_variants(value: str) -> None:
    assert is_placeholder(value)


@pytest.mark.parametrize("value", ["YOUR_KEY", "MY_KEY_HERE", "your_key_here", "sk-YOUR_X_HERE"])
def test_non_placeholders(value: str) -> None:
    assert not is_placeholder(value)


def test_require_real_secret_accepts_plain_strings() -> None:
    require_real_secret("X", "real-value", when=True)
    with pytest.raises(ConfigurationError, match="X"):
        require_real_secret("X", None, when=True)


def test_load_yaml_empty_file_is_empty_mapping(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    assert load_yaml(f) == {}


def test_load_yaml_rejects_unsafe_tags(tmp_path: Path) -> None:
    import yaml

    f = tmp_path / "evil.yaml"
    f.write_text("x: !!python/object/apply:os.system ['echo pwned']\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_yaml(f)


def test_load_yaml_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_yaml(tmp_path)


def test_repo_config_files_load() -> None:
    root = Path(__file__).resolve().parents[3] / "config"
    for name in ("chunking", "retrieval", "advisor", "evaluation"):
        assert load_yaml(root / f"{name}.yaml")
