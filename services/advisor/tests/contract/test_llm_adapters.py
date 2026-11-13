"""LLMClient port contract: fake, Anthropic (official SDK), OpenAI, Ollama behave alike (Liskov)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import PROMPTS_DIR, kb_chunks

from app.domain.compose import AnswerComposer, GenerationContext
from app.domain.models import Language
from app.domain.ports import LLMClient, LLMRequest
from app.infrastructure.llm import (
    REFUSAL_FALLBACK_BETA,
    AnthropicLLM,
    FakeLLM,
    OllamaLLM,
    OpenAILLM,
    build_llm,
)
from app.infrastructure.prompts import load_prompt_set
from common.contracts import Intent
from common.errors import ConfigurationError, ProviderError, TransientProviderError, UpstreamTimeout

KEY = "-".join(["sk", "test", "llm", "key", "0123456789"])  # built at runtime: no key literal in git
MESSAGES = load_prompt_set(PROMPTS_DIR, "v1").messages
CONTEXT = GenerationContext(
    intent=Intent.CONTACT,
    language=Language.EN,
    user_message="whatsapp?",
    evidence=tuple(c for c in kb_chunks() if c.metadata.section.startswith("Contact")),
)
REQUEST = LLMRequest(system="You are helpful.", messages=(("user", "What is the WhatsApp number?"),), context=CONTEXT)

Handler = Callable[[Any], Any]


def anthropic_message(text: str = "Call 0309-4190245.", stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def anthropic_llm(handler: Handler, model: str = "claude-opus-5-5", **kw: Any) -> AnthropicLLM:
    client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    return AnthropicLLM(api_key=KEY, model=model, max_tokens=16000, max_retries=0, http_client=client, **kw)


def openai_llm(handler: Handler) -> OpenAILLM:
    router = respx.Router(base_url="https://api.openai.com/v1")
    router.post("/chat/completions").mock(side_effect=handler)
    return OpenAILLM(
        api_key=KEY,
        model="gpt-x",
        max_tokens=500,
        temperature=0.2,
        timeout_s=5,
        transport=httpx.MockTransport(router.handler),
    )


def ollama_llm(handler: Handler) -> OllamaLLM:
    router = respx.Router(base_url="http://ollama.test:11434")
    router.post("/api/chat").mock(side_effect=handler)
    return OllamaLLM(
        base_url="http://ollama.test:11434",
        model="llama3",
        max_tokens=500,
        temperature=0.2,
        timeout_s=5,
        transport=httpx.MockTransport(router.handler),
    )


def ok_anthropic(_: Any) -> Any:
    return httpx2.Response(200, json=anthropic_message())


def ok_openai(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-x",
            "choices": [{"message": {"content": "Call 0309-4190245."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        },
    )


def ok_ollama(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "llama3",
            "message": {"content": "Call 0309-4190245."},
            "prompt_eval_count": 12,
            "eval_count": 7,
            "done_reason": "stop",
        },
    )


FACTORIES: dict[str, Callable[[], LLMClient]] = {
    "fake": lambda: FakeLLM(AnswerComposer(MESSAGES)),
    "anthropic": lambda: anthropic_llm(ok_anthropic),
    "openai": lambda: openai_llm(ok_openai),
    "ollama": lambda: ollama_llm(ok_ollama),
}


@pytest.fixture(params=sorted(FACTORIES))
def llm(request: pytest.FixtureRequest) -> LLMClient:
    return FACTORIES[request.param]()


# ---------------------------------------------------------------- contract


async def test_contract_returns_text_and_usage(llm: LLMClient) -> None:
    r = await llm.generate(REQUEST)
    assert "0309-4190245" in r.text
    assert r.input_tokens > 0
    assert r.output_tokens > 0
    assert r.refused is False
    assert r.text == r.text.strip()


async def test_contract_has_a_name(llm: LLMClient) -> None:
    assert llm.name


# ---------------------------------------------------------------- fake


async def test_fake_is_deterministic_and_reports_citations() -> None:
    fake = FakeLLM(AnswerComposer(MESSAGES))
    a, b = await fake.generate(REQUEST), await fake.generate(REQUEST)
    assert a == b
    assert a.cited == (CONTEXT.evidence[0].id,)


async def test_fake_needs_context() -> None:
    with pytest.raises(ProviderError, match="context"):
        await FakeLLM(AnswerComposer(MESSAGES)).generate(LLMRequest(system="s", messages=(("user", "x"),)))


# ---------------------------------------------------------------- anthropic request shape


async def test_anthropic_request_shape_with_refusal_fallback() -> None:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return httpx2.Response(200, json=anthropic_message())

    llm = anthropic_llm(handler)
    r = await llm.generate(REQUEST)
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages"
    assert KEY in req.headers["authorization"]  # SDK 2.x sends the key as a bearer credential
    assert req.headers["anthropic-version"] == "2026-10-21"  # this API version rejects temperature everywhere
    assert REFUSAL_FALLBACK_BETA in req.headers.get("anthropic-beta", "")
    assert body["fallbacks"] == "default"
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16000
    assert body["system"] == "You are helpful."
    assert body["messages"] == [{"role": "user", "content": "What is the WhatsApp number?"}]
    assert body["output_config"] == {"effort": "low"}
    assert "temperature" not in body  # current models reject sampling parameters
    assert "thinking" not in body
    assert (r.input_tokens, r.output_tokens, r.stop_reason) == (12, 7, "end_turn")
    assert llm.name == "anthropic:claude-opus-5-5"


async def test_anthropic_older_model_without_fallback_or_effort() -> None:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return httpx2.Response(200, json=anthropic_message())

    await anthropic_llm(handler, model="claude-haiku-4-5", effort="").generate(REQUEST)
    body = json.loads(seen[0].content)
    assert "fallbacks" not in body
    assert "output_config" not in body
    assert REFUSAL_FALLBACK_BETA not in seen[0].headers.get("anthropic-beta", "")


async def test_anthropic_fallback_can_be_disabled() -> None:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return httpx2.Response(200, json=anthropic_message())

    await anthropic_llm(handler, refusal_fallback=False).generate(REQUEST)
    assert "fallbacks" not in json.loads(seen[0].content)


async def test_anthropic_refusal_is_reported() -> None:
    llm = anthropic_llm(lambda r: httpx2.Response(200, json=anthropic_message(text="", stop="refusal")))
    r = await llm.generate(REQUEST)
    assert r.refused is True
    assert r.text == ""


async def test_anthropic_ignores_non_text_blocks() -> None:
    msg = anthropic_message()
    msg["content"] = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "Answer."}]
    r = await anthropic_llm(lambda req: httpx2.Response(200, json=msg)).generate(REQUEST)
    assert r.text == "Answer."


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, TransientProviderError),
        (500, TransientProviderError),
        (503, TransientProviderError),
        (529, TransientProviderError),
        (401, ProviderError),
        (403, ProviderError),
        (400, ProviderError),
        (404, ProviderError),
    ],
)
async def test_anthropic_error_mapping(status: int, error: type[ProviderError]) -> None:
    body = {"type": "error", "error": {"type": "x", "message": f"bad {KEY}"}}
    llm = anthropic_llm(lambda r: httpx2.Response(status, json=body))
    with pytest.raises(error) as info:
        await llm.generate(REQUEST)
    assert type(info.value) is error
    assert KEY not in str(info.value)


async def test_anthropic_timeout_and_connection_errors() -> None:
    def timeout(_: Any) -> Any:
        raise httpx2.ReadTimeout("slow")

    def refused(_: Any) -> Any:
        raise httpx2.ConnectError("down")

    with pytest.raises(UpstreamTimeout):
        await anthropic_llm(timeout).generate(REQUEST)
    with pytest.raises(TransientProviderError):
        await anthropic_llm(refused).generate(REQUEST)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"api_key": ""}, "LLM_API_KEY"), ({"model": ""}, "LLM_MODEL"), ({"effort": "extreme"}, "LLM_EFFORT")],
)
def test_anthropic_config_validation(kwargs: dict[str, str], match: str) -> None:
    base: dict[str, Any] = {"api_key": KEY, "model": "claude-opus-5-5", "max_tokens": 100}
    with pytest.raises(ConfigurationError, match=match):
        AnthropicLLM(**{**base, **kwargs})


# ---------------------------------------------------------------- openai / ollama


async def test_openai_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_openai(request)

    await openai_llm(handler).generate(REQUEST)
    body = json.loads(seen[0].content)
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"
    assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert body["messages"][1]["role"] == "user"
    assert body["temperature"] == 0.2
    assert body["max_completion_tokens"] == 500


async def test_openai_refusal_field() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": None, "refusal": "no"}, "finish_reason": "stop"}]}
        )

    r = await openai_llm(handler).generate(REQUEST)
    assert r.refused is True


async def test_ollama_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_ollama(request)

    await ollama_llm(handler).generate(REQUEST)
    body = json.loads(seen[0].content)
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.2, "num_predict": 500}
    assert "Authorization" not in seen[0].headers


@pytest.mark.parametrize("make", [openai_llm, ollama_llm], ids=["openai", "ollama"])
@pytest.mark.parametrize(
    ("status", "error"),
    [(429, TransientProviderError), (502, TransientProviderError), (401, ProviderError), (400, ProviderError)],
)
async def test_http_llm_status_mapping(
    make: Callable[[Handler], LLMClient], status: int, error: type[ProviderError]
) -> None:
    llm = make(lambda r: httpx.Response(status, text=KEY))
    with pytest.raises(error) as info:
        await llm.generate(REQUEST)
    assert type(info.value) is error
    assert KEY not in str(info.value)


@pytest.mark.parametrize("make", [openai_llm, ollama_llm], ids=["openai", "ollama"])
async def test_http_llm_transport_errors_and_bad_json(make: Callable[[Handler], LLMClient]) -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    def refused(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(UpstreamTimeout):
        await make(timeout).generate(REQUEST)
    with pytest.raises(TransientProviderError):
        await make(refused).generate(REQUEST)
    for bad in (httpx.Response(200, text="<html>"), httpx.Response(200, json=[1]), httpx.Response(200, json={})):
        with pytest.raises(ProviderError, match="malformed"):
            await make(lambda r, b=bad: b).generate(REQUEST)


# ---------------------------------------------------------------- factory


def _build(provider: str, **kw: Any) -> LLMClient:
    args: dict[str, Any] = {
        "composer": AnswerComposer(MESSAGES),
        "api_key": KEY,
        "model": "m",
        "max_tokens": 100,
        "temperature": 0.2,
        "effort": "low",
        "refusal_fallback": True,
        "base_url": None,
        "timeout_s": 5,
        "max_retries": 0,
    }
    return build_llm(provider, **{**args, **kw})


def test_factory_builds_each_provider() -> None:
    assert isinstance(_build("fake"), FakeLLM)
    assert isinstance(_build("anthropic"), AnthropicLLM)
    assert isinstance(_build("openai"), OpenAILLM)
    assert isinstance(_build("ollama"), OllamaLLM)


def test_factory_errors() -> None:
    with pytest.raises(ConfigurationError, match="Unknown LLM_PROVIDER"):
        _build("gemini")
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        _build("openai", api_key="")
    with pytest.raises(ConfigurationError, match="LLM_MODEL"):
        _build("ollama", model="")
