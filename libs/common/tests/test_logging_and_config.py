from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr
from pydantic_settings import SettingsConfigDict

from common.config import (
    EmbeddingSettings,
    LLMSettings,
    ServiceSettings,
    is_placeholder,
    load_settings,
    load_yaml_model,
    require_real_secret,
)
from common.errors import ConfigurationError
from common.logging import JsonFormatter, redact, set_request_id


class _Settings(ServiceSettings):
    llm_api_key: SecretStr = SecretStr("YOUR_LLM_API_KEY_HERE")
    retries: int = 3


def _capture(logger_name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter("test"))
    logger = logging.getLogger(logger_name)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("YOUR_LLM_API_KEY_HERE", True),
        ("", True),
        (None, True),
        ("  ", True),
        ("sk-real-123", False),
        (SecretStr("YOUR_X_HERE"), True),
    ],
)
def test_is_placeholder(value: str | SecretStr | None, expected: bool) -> None:
    assert is_placeholder(value) is expected


def test_require_real_secret_fails_fast() -> None:
    with pytest.raises(ConfigurationError) as exc:
        require_real_secret("LLM_API_KEY", SecretStr("YOUR_LLM_API_KEY_HERE"), when=True)
    assert "LLM_API_KEY" in str(exc.value)
    require_real_secret("LLM_API_KEY", SecretStr("YOUR_LLM_API_KEY_HERE"), when=False)


def test_load_settings_invalid_value_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIES", "not-a-number")
    with pytest.raises(ConfigurationError) as exc:
        load_settings(_Settings)
    assert "retries" in str(exc.value)
    assert "not-a-number" not in str(exc.value)


def test_secrets_are_never_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-ant-THIS-IS-A-REAL-LOOKING-KEY-123456"
    monkeypatch.setenv("LLM_API_KEY", secret)
    settings = load_settings(_Settings)
    logger, stream = _capture("secret-test")
    logger.info("calling provider with %s", secret, extra={"headers": {"x-api-key": secret}, "cfg": repr(settings)})
    logger.info(f"raw f-string {secret}")
    out = stream.getvalue()
    assert secret not in out
    assert "***" in out
    # repr(SecretStr) is already masked by pydantic
    assert "**********" in out


def test_pii_is_redacted_from_logs() -> None:
    logger, stream = _capture("pii-test")
    logger.info("user said: call me at 0300-1234567 or +92 321 7654321, mail ali.khan@example.com")
    out = stream.getvalue()
    assert "1234567" not in out
    assert "7654321" not in out
    assert "ali.khan@example.com" not in out
    assert "[phone]" in out
    assert "[email]" in out


def test_json_log_has_request_id_and_extras() -> None:
    logger, stream = _capture("json-test")
    set_request_id("req-123456789")
    try:
        logger.info("hello", extra={"latency_ms": 12.5})
    finally:
        set_request_id(None)
    record = json.loads(stream.getvalue())
    assert record["msg"] == "hello"
    assert record["request_id"] == "req-123456789"
    assert record["latency_ms"] == 12.5
    assert record["service"] == "test"


def test_redact_bearer_tokens() -> None:
    assert "abcdefghijkl" not in redact("Authorization: Bearer abcdefghijkl")
    assert "zyxwvutsrqpo" not in redact('api_key="zyxwvutsrqpo"')


class _Cfg(BaseModel):
    top_k: int
    threshold: float


def test_load_yaml_model(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("top_k: 5\nthreshold: 0.2\n", encoding="utf-8")
    assert load_yaml_model(good, _Cfg) == _Cfg(top_k=5, threshold=0.2)

    bad = tmp_path / "bad.yaml"
    bad.write_text("top_k: five\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_yaml_model(bad, _Cfg)

    with pytest.raises(ConfigurationError):
        load_yaml_model(tmp_path / "missing.yaml", _Cfg)

    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_yaml_model(lst, _Cfg)


class _ProviderSettings(LLMSettings, EmbeddingSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_fake_providers_need_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    s = load_settings(_ProviderSettings)
    assert s.llm_provider == "fake"
    assert s.embedding_provider == "fake"


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "YOUR_LLM_API_KEY_HERE", "LLM_MODEL": "m"}, "LLM_API_KEY"),
        (
            {"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "sk-real-key-1", "LLM_MODEL": "YOUR_CHAT_MODEL_NAME_HERE"},
            "LLM_MODEL",
        ),
        ({"EMBEDDING_PROVIDER": "voyage", "EMBEDDING_API_KEY": "", "EMBEDDING_MODEL": "m"}, "EMBEDDING_API_KEY"),
    ],
)
def test_real_provider_with_placeholder_fails_fast(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], missing: str
) -> None:
    _clear_provider_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ConfigurationError, match=missing):
        load_settings(_ProviderSettings)


def test_ollama_needs_model_but_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3")
    assert load_settings(_ProviderSettings).llm_provider == "ollama"
