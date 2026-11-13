"""LLM adapters behind the LLMClient port (ADR-0005, ADR-0013).

- fake:      deterministic, built from the evidence (offline tests, CI, first `docker compose up`)
- anthropic: official `anthropic` SDK (Messages API)
- openai:    Chat Completions over httpx (also any OpenAI-compatible endpoint via LLM_BASE_URL)
- ollama:    local models over httpx

Every adapter maps failures to the same errors: TransientProviderError / UpstreamTimeout (worth
retrying) or ProviderError (not). API keys never appear in messages.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anthropic
import httpx

from app.domain.compose import AnswerComposer
from app.domain.ports import LLMClient, LLMRequest, LLMResponse
from common.errors import ConfigurationError, ProviderError, TransientProviderError, UpstreamTimeout

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
REFUSAL_FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class FakeLLM(LLMClient):
    """Deterministic: answers with the evidence-built composition (grounded by construction)."""

    def __init__(self, composer: AnswerComposer) -> None:
        self.composer = composer

    @property
    def name(self) -> str:
        return "fake"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.context is None:
            raise ProviderError("The fake LLM needs a generation context.")
        composed = self.composer.compose(request.context)
        prompt_words = len(request.system.split()) + sum(len(c.split()) for _, c in request.messages)
        return LLMResponse(
            text=composed.text,
            model="fake",
            input_tokens=prompt_words,
            output_tokens=len(composed.text.split()),
            stop_reason="end_turn",
            cited=composed.used,
        )


class AnthropicLLM(LLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        effort: str = "low",
        refusal_fallback: bool = True,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        http_client: anthropic.DefaultAsyncHttpxClient | None = None,
    ) -> None:
        if not api_key or not model:
            raise ConfigurationError("LLM_API_KEY and LLM_MODEL are required for LLM_PROVIDER=anthropic.")
        if effort and effort not in EFFORTS:
            raise ConfigurationError(f"LLM_EFFORT must be one of {sorted(EFFORTS)} or empty.")
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.use_fallback = refusal_fallback and model in REFUSAL_FALLBACK_MODELS
        # The SDK retries connection errors, 408/409/429 and 5xx with backoff (max_retries).
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=max_retries, http_client=http_client
        )

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": request.system,
            "messages": [{"role": role, "content": content} for role, content in request.messages],
        }
        # No temperature: current Claude models reject sampling parameters. Depth is set with effort.
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        try:
            if self.use_fallback:
                message: Any = await self.client.beta.messages.create(
                    betas=[REFUSAL_FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                message = await self.client.messages.create(**params)
        except anthropic.APITimeoutError:
            raise UpstreamTimeout("The Anthropic API timed out.") from None
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
            raise TransientProviderError(f"The Anthropic API is unavailable ({type(exc).__name__}).") from None
        except anthropic.AuthenticationError:
            raise ProviderError("The Anthropic API rejected the credentials; check LLM_API_KEY.") from None
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"The Anthropic API returned {exc.status_code}.") from None
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        stop_reason = getattr(message, "stop_reason", None)
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text.strip(),
            model=str(getattr(message, "model", self.model)),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            stop_reason=stop_reason,
            refused=stop_reason == "refusal",
        )


class _HttpLLM(LLMClient):
    provider = "http"

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model:
            raise ConfigurationError(f"LLM_MODEL is required for LLM_PROVIDER={self.provider}.")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s, transport=transport
        )

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self.client.post(path, json=body)
        except httpx.TimeoutException:
            raise UpstreamTimeout(f"{self.provider} timed out.") from None
        except httpx.TransportError as exc:
            raise TransientProviderError(f"{self.provider} unreachable ({type(exc).__name__}).") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientProviderError(f"{self.provider} returned {resp.status_code}.")
        if resp.status_code in (401, 403):
            raise ProviderError(f"{self.provider} rejected the credentials; check LLM_API_KEY.")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.provider} returned {resp.status_code}.")
        try:
            data = resp.json()
        except ValueError:
            raise ProviderError(f"{self.provider} returned a malformed response.") from None
        if not isinstance(data, dict):
            raise ProviderError(f"{self.provider} returned a malformed response.")
        return data

    @staticmethod
    def _chat(request: LLMRequest) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": request.system},
            *({"role": role, "content": content} for role, content in request.messages),
        ]

    async def aclose(self) -> None:
        await self.client.aclose()


class OpenAILLM(_HttpLLM):
    provider = "openai"

    def __init__(self, *, api_key: str, base_url: str | None = None, **kw: Any) -> None:
        if not api_key:
            raise ConfigurationError("LLM_API_KEY is required for LLM_PROVIDER=openai.")
        super().__init__(
            base_url=base_url or "https://api.openai.com/v1", headers={"Authorization": f"Bearer {api_key}"}, **kw
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        data = await self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": self._chat(request),
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
            },
        )
        try:
            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage") or {}
            refusal = message.get("refusal")
            return LLMResponse(
                text=(message.get("content") or "").strip(),
                model=str(data.get("model", self.model)),
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                stop_reason=choice.get("finish_reason"),
                refused=bool(refusal),
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("openai returned a malformed response.") from None


class OllamaLLM(_HttpLLM):
    provider = "ollama"

    def __init__(self, *, base_url: str | None = None, **kw: Any) -> None:
        super().__init__(base_url=base_url or "http://host.docker.internal:11434", headers={}, **kw)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        data = await self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": self._chat(request),
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            },
        )
        try:
            return LLMResponse(
                text=str(data["message"]["content"]).strip(),
                model=str(data.get("model", self.model)),
                input_tokens=int(data.get("prompt_eval_count", 0)),
                output_tokens=int(data.get("eval_count", 0)),
                stop_reason=str(data.get("done_reason", "stop")),
            )
        except (KeyError, TypeError):
            raise ProviderError("ollama returned a malformed response.") from None


def build_llm(
    provider: str,
    *,
    composer: AnswerComposer,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    effort: str,
    refusal_fallback: bool,
    base_url: str | None,
    timeout_s: float,
    max_retries: int,
) -> LLMClient:
    factories: dict[str, Callable[[], LLMClient]] = {
        "fake": lambda: FakeLLM(composer),
        "anthropic": lambda: AnthropicLLM(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            effort=effort,
            refusal_fallback=refusal_fallback,
            timeout_s=timeout_s,
            max_retries=max_retries,
        ),
        "openai": lambda: OpenAILLM(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        ),
        "ollama": lambda: OllamaLLM(
            base_url=base_url, model=model, max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s
        ),
    }
    if provider not in factories:
        raise ConfigurationError(f"Unknown LLM_PROVIDER '{provider}'. Options: {sorted(factories)}")
    return factories[provider]()
