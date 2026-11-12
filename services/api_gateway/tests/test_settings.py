from __future__ import annotations

import pytest

from app.settings import Settings
from common.config import load_settings
from common.errors import ConfigurationError

STRONG = "x" * 32


def _load(**kw: object) -> Settings:
    return load_settings(Settings, _env_file=None, **kw)


def test_defaults_are_dev_friendly() -> None:
    s = _load()
    assert s.service_name == "api_gateway"
    assert s.app_env == "dev"
    assert s.rate_limit_per_minute == 30


def test_dev_allows_placeholder_keys_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        _load(gateway_api_key="YOUR_INTERNAL_GATEWAY_KEY_HERE")
    assert any("weak_gateway_keys_dev_only" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("gateway", "admin", "missing"),
    [
        ("YOUR_INTERNAL_GATEWAY_KEY_HERE", STRONG, "GATEWAY_API_KEY"),
        (STRONG, "", "ADMIN_API_KEY"),
        ("short-key", STRONG, "GATEWAY_API_KEY"),
    ],
)
def test_prod_rejects_weak_keys(gateway: str, admin: str, missing: str) -> None:
    with pytest.raises(ConfigurationError, match=missing):
        _load(app_env="prod", gateway_api_key=gateway, admin_api_key=admin)


def test_prod_accepts_strong_keys() -> None:
    s = _load(app_env="prod", gateway_api_key=STRONG, admin_api_key=STRONG + "y")
    assert s.gateway_api_key.get_secret_value() == STRONG


def test_cors_list_parsing_drops_wildcard_and_blanks() -> None:
    s = _load(cors_origins=" http://a.test , *, ,https://b.test")
    assert s.cors_origin_list == ["http://a.test", "https://b.test"]


def test_invalid_rate_limit_fails_fast() -> None:
    with pytest.raises(ConfigurationError, match="rate_limit_per_minute"):
        _load(rate_limit_per_minute=0)


def test_secret_never_in_repr() -> None:
    s = _load(gateway_api_key=STRONG)
    assert STRONG not in repr(s)
    assert STRONG not in str(s.model_dump())
