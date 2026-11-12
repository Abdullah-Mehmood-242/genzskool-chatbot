from __future__ import annotations

import pytest

from app.settings import Settings
from common.config import load_settings
from common.errors import ConfigurationError


def _load(**kw: object) -> Settings:
    return load_settings(Settings, _env_file=None, **kw)


def test_defaults_run_offline() -> None:
    s = _load(llm_provider="fake")
    assert s.service_name == "advisor"
    assert s.session_store == "memory"


def test_redis_store_requires_url() -> None:
    with pytest.raises(ConfigurationError, match="REDIS_URL"):
        _load(llm_provider="fake", session_store="redis")
    assert _load(llm_provider="fake", session_store="redis", redis_url="redis://r:6379/0").session_store == "redis"


def test_anthropic_requires_real_key_and_model() -> None:
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        _load(llm_provider="anthropic", llm_api_key="YOUR_LLM_API_KEY_HERE", llm_model="m")
    s = _load(llm_provider="anthropic", llm_api_key="sk-ant-real-key", llm_model="claude-x")
    assert s.llm_provider == "anthropic"


@pytest.mark.parametrize(
    ("field", "value"),
    [("llm_temperature", 1.5), ("llm_max_tokens", 10), ("session_ttl_s", 5), ("llm_provider", "gpt5000")],
)
def test_out_of_range_values_fail_fast(field: str, value: object) -> None:
    with pytest.raises(ConfigurationError, match=field):
        _load(**{field: value})
