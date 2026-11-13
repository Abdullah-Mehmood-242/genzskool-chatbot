"""HTTP retrieval client (retry, breaker, request id, errors) and the versioned prompt loader."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import PROMPTS_DIR, kb_chunks

from app.infrastructure.prompts import REQUIRED_MESSAGES, SystemClock, load_prompt_set
from app.infrastructure.retrieval_client import HttpRetrievalClient
from common.contracts import SearchFilters, SearchRequest, SearchResponse
from common.errors import CircuitOpen, ConfigurationError, UpstreamError, UpstreamTimeout
from common.logging import set_request_id
from common.resilience import CircuitBreaker, RetryPolicy

BASE = "http://retrieval.test"
OK = SearchResponse(chunks=kb_chunks()[:2]).model_dump(mode="json")


def client(handler: object, *, attempts: int = 2, threshold: int = 5) -> HttpRetrievalClient:
    router = respx.Router(base_url=BASE)
    router.post("/v1/search").mock(side_effect=handler)
    router.get("/healthz").mock(return_value=httpx.Response(200))
    return HttpRetrievalClient(
        BASE,
        transport=httpx.MockTransport(router.handler),
        retry=RetryPolicy(
            attempts=attempts, base_delay_s=0, max_delay_s=0, retry_on=HttpRetrievalClient(BASE).retry.retry_on
        ),
        breaker=CircuitBreaker("retrieval", failure_threshold=threshold, reset_after_s=60),
    )


async def test_success_and_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    set_request_id("req-propagated-01")
    try:
        resp = await client(handler).search(
            SearchRequest(query="ebay", top_k=3, filters=SearchFilters(course_slugs=["ebay"]))
        )
    finally:
        set_request_id(None)
    assert len(resp.chunks) == 2
    body = json.loads(seen[0].content)
    assert body == {"query": "ebay", "top_k": 3, "filters": {"course_slugs": ["ebay"]}, "debug": False}
    assert seen[0].headers["X-Request-ID"] == "req-propagated-01"


async def test_5xx_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, json=OK)

    assert len((await client(handler).search(SearchRequest(query="q"))).chunks) == 2
    assert calls["n"] == 2


async def test_timeouts_retried_then_raised() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    with pytest.raises(UpstreamTimeout):
        await client(handler, attempts=3).search(SearchRequest(query="q"))
    assert calls["n"] == 3


async def test_connection_error_raises_upstream_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(UpstreamError, match="unreachable"):
        await client(handler).search(SearchRequest(query="q"))


async def test_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"error": {}})

    with pytest.raises(UpstreamError, match="rejected"):
        await client(handler, attempts=3).search(SearchRequest(query="q"))
    assert calls["n"] == 1


async def test_invalid_response_body() -> None:
    with pytest.raises(UpstreamError, match="invalid response"):
        await client(lambda r: httpx.Response(200, json={"nope": 1})).search(SearchRequest(query="q"))


async def test_circuit_opens_after_repeated_failures() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    c = client(handler, attempts=1, threshold=2)
    for _ in range(2):
        with pytest.raises(UpstreamError):
            await c.search(SearchRequest(query="q"))
    with pytest.raises(CircuitOpen):
        await c.search(SearchRequest(query="q"))
    assert calls["n"] == 2  # the open circuit did not call retrieval again


async def test_healthy() -> None:
    assert await client(lambda r: httpx.Response(200, json=OK)).healthy() is True
    assert await HttpRetrievalClient("http://127.0.0.1:1", timeout_s=0.5).healthy() is False


# ---------------------------------------------------------------- prompt loader


def test_loads_v1() -> None:
    ps = load_prompt_set(PROMPTS_DIR, "v1")
    assert ps.version == "v1"
    assert "Genz Skool" in ps.templates.system
    assert "${message}" in ps.templates.answer
    assert set(ps.messages.templates) >= REQUIRED_MESSAGES


def _copy(tmp_path: Path) -> Path:
    shutil.copytree(PROMPTS_DIR / "v1", tmp_path / "v9")
    return tmp_path


@pytest.mark.parametrize("missing", ["system.md", "answer.md", "regenerate.md", "messages.yaml"])
def test_missing_file_fails_fast(tmp_path: Path, missing: str) -> None:
    (_copy(tmp_path) / "v9" / missing).unlink()
    with pytest.raises(ConfigurationError, match=missing):
        load_prompt_set(tmp_path, "v9")


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("- a\n- b\n", "mapping"),
        ("greeting: {en: hi}\n", "missing"),
        ("\n".join(f"{k}: {{roman_ur: x}}" for k in sorted(REQUIRED_MESSAGES)), "'en' variant"),
    ],
)
def test_invalid_messages_fail_fast(tmp_path: Path, content: str, match: str) -> None:
    (_copy(tmp_path) / "v9" / "messages.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=match):
        load_prompt_set(tmp_path, "v9")


def test_unknown_version(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_prompt_set(tmp_path, "v404")


def test_system_clock_is_utc() -> None:
    assert SystemClock().now().utcoffset() is not None
