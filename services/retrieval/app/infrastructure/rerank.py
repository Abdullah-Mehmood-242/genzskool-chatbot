"""Rerankers (optional, ADR-0003) and the default no-op query rewriter."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import httpx

from app.domain.models import NormalizedQuery
from app.domain.ports import QueryRewriter, Reranker
from common.contracts import Chunk
from common.errors import ConfigurationError, ProviderError, TransientProviderError, UpstreamTimeout


class NoopReranker(Reranker):
    @property
    def enabled(self) -> bool:
        return False

    async def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        return [1.0] * len(chunks)


class NoopRewriter(QueryRewriter):
    """Default. An LLM-based rewriter can be added behind this port (disabled by default per spec)."""

    async def rewrite(self, query: NormalizedQuery) -> NormalizedQuery:
        return query


class HttpReranker(Reranker):
    """Cohere (v2/rerank) or Voyage (v1/rerank). Both return (index, relevance_score) pairs."""

    ENDPOINTS: ClassVar[dict[str, tuple[str, str]]] = {
        "cohere": ("https://api.cohere.com/v2", "results"),
        "voyage": ("https://api.voyageai.com/v1", "data"),
    }

    def __init__(
        self,
        provider: str,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if provider not in self.ENDPOINTS:
            raise ConfigurationError(f"Unknown RERANKER_PROVIDER '{provider}'.")
        if not api_key:
            raise ConfigurationError("RERANKER_API_KEY is required for the selected reranker.")
        if not model:
            raise ConfigurationError("RERANKER_MODEL is required for the selected reranker.")
        base, self._results_key = self.ENDPOINTS[provider]
        self.provider = provider
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_s, transport=transport
        )

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        if not chunks:
            return []
        payload: dict[str, Any] = {"query": query, "documents": [c.text for c in chunks], "model": self.model}
        try:
            resp = await self._client.post("/rerank", json=payload)
        except httpx.TimeoutException:
            raise UpstreamTimeout(f"{self.provider} rerank timed out.") from None
        except httpx.TransportError:
            raise TransientProviderError(f"{self.provider} rerank unreachable.") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientProviderError(f"{self.provider} rerank returned {resp.status_code}.")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.provider} rerank returned {resp.status_code}.")
        try:
            items = resp.json()[self._results_key]
            scores = [0.0] * len(chunks)
            for item in items:
                scores[int(item["index"])] = float(item["relevance_score"])
        except (ValueError, KeyError, TypeError, IndexError):
            raise ProviderError(f"{self.provider} returned a malformed rerank response.") from None
        return [min(1.0, max(0.0, s)) for s in scores]

    async def aclose(self) -> None:
        await self._client.aclose()


class CrossEncoderReranker(Reranker):
    """Local cross-encoder (optional package `sentence-transformers`); logits squashed to [0, 1]."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ConfigurationError(
                "RERANKER_PROVIDER=cross_encoder needs the optional package 'sentence-transformers'."
            ) from None
        self._model = CrossEncoder(model)

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        if not chunks:
            return []
        raw = await asyncio.to_thread(self._model.predict, [(query, c.text) for c in chunks])
        return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]


def build_reranker(provider: str, *, api_key: str, model: str, timeout_s: float = 15.0) -> Reranker:
    if provider == "none":
        return NoopReranker()
    if provider == "cross_encoder":
        return CrossEncoderReranker(model)
    return HttpReranker(provider, api_key=api_key, model=model, timeout_s=timeout_s)
