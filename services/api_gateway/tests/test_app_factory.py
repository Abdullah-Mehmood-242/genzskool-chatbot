"""Composition root and startup behaviour of the api_gateway service."""

from __future__ import annotations

import httpx
import pytest

from app.api.dependencies import Container, build_container
from app.main import create_app
from app.settings import Settings
from common.errors import ConfigurationError

SERVICE = "api_gateway"


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def _client(app: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")  # type: ignore[arg-type]


def test_settings_identify_the_service() -> None:
    assert _settings().service_name == SERVICE


def test_container_holds_settings() -> None:
    s = _settings()
    container = build_container(s)
    assert isinstance(container, Container)
    assert container.settings is s
    assert isinstance(container.readiness_checks(), dict)
    assert isinstance(container.routers(), list)


def test_app_state_is_wired() -> None:
    s = _settings()
    app = create_app(s)
    assert app.state.settings is s
    assert app.state.container.settings is s


async def test_version_and_title_come_from_settings() -> None:
    app = create_app(_settings(service_version="9.8.7"))
    async with _client(app) as c:
        health = (await c.get("/healthz")).json()
        spec = (await c.get("/openapi.json")).json()
    assert health == {"status": "ok", "service": SERVICE, "version": "9.8.7", "checks": {}}
    assert spec["info"]["title"] == f"genz-{SERVICE}"


def test_create_app_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE_VERSION", "1.2.3")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    assert create_app().state.settings.service_version == "1.2.3"


def test_create_app_fails_fast_on_bad_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigurationError, match="log_level"):
        create_app()


async def test_unknown_route_and_error_schema() -> None:
    async with _client(create_app(_settings())) as c:
        r = await c.get("/v1/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    assert "X-Request-ID" in r.headers


async def test_oversized_body_rejected_at_service_limit() -> None:
    s = _settings()
    async with _client(create_app(s)) as c:
        r = await c.post("/healthz", content=b"x" * (s.max_body_bytes + 1))
    assert r.status_code == 413


def test_settings_warnings_are_logged_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.setenv("GATEWAY_API_KEY", "YOUR_INTERNAL_GATEWAY_KEY_HERE")
    create_app()
    out = [ln for ln in capsys.readouterr().out.splitlines() if "weak_gateway_keys_dev_only" in ln]
    assert out, "warning not emitted on stdout"
    assert json.loads(out[-1])["service"] == "api_gateway"
