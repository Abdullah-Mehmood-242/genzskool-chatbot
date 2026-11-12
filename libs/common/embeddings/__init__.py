"""Embedding clients shared by ingestion and retrieval (ADR-0012).

Both services MUST embed text identically (same provider, model, dimension and, for the fake
provider, the same hashing), or dense retrieval silently breaks. Keeping one implementation here
makes that impossible to get wrong. This is infrastructure plumbing, not business logic.

Adding a provider = one new class + one entry in ``_REGISTRY`` (Open/Closed).
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from common.errors import ConfigurationError, ProviderError, TransientProviderError, UpstreamTimeout

Vector = list[float]


class Embedder(ABC):
    """Turns text into fixed-size vectors. Documents and queries may be embedded differently."""

    name: str
    dim: int

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> Vector: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook, default no-op
        """Release network resources."""

    def _check(self, vectors: list[Vector], expected: int) -> list[Vector]:
        if len(vectors) != expected:
            raise ProviderError(f"{self.name} returned {len(vectors)} vectors for {expected} inputs.")
        for v in vectors:
            if len(v) != self.dim:
                raise ProviderError(
                    f"{self.name} returned dimension {len(v)}, expected {self.dim}. Set EMBEDDING_DIM to match the "
                    "model and re-ingest."
                )
        return vectors


# ---------------------------------------------------------------------------------------------
# Fake (offline, deterministic)
# ---------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps 'make.com'; '1,250' -> '1', '250'."""
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(feature: str) -> int:
    # blake2b, not hash(): Python's hash() is salted per process, which would make ingestion and
    # retrieval disagree.
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


class FakeEmbedder(Embedder):
    """Signed feature hashing of unigrams and bigrams, L2-normalised.

    It captures lexical similarity, so retrieval metrics are meaningful offline, and it is identical
    across processes and machines.
    """

    name = "fake"

    def __init__(self, dim: int) -> None:
        if dim < 8:
            raise ConfigurationError("EMBEDDING_DIM must be >= 8.")
        self.dim = dim

    def _embed(self, text: str) -> Vector:
        tokens = tokenize(text)
        features = [(t, 1.0) for t in tokens] + [(f"{a} {b}", 0.5) for a, b in itertools.pairwise(tokens)]
        if not features:
            features = [("__empty__", 1.0)]
        vec = [0.0] * self.dim
        for feature, weight in features:
            h = _stable_hash(feature)
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:  # all features cancelled out: fall back to the empty-text direction
            vec = [0.0] * self.dim
            vec[_stable_hash("__empty__") % self.dim] = 1.0
            return vec
        return [x / norm for x in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._embed(text)


# ---------------------------------------------------------------------------------------------
# HTTP providers (no SDK dependency; tested with respx)
# ---------------------------------------------------------------------------------------------


class _HttpEmbedder(Embedder):
    default_base_url: str

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dim: int,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError(f"EMBEDDING_API_KEY is required for {self.name}.")
        if not model:
            raise ConfigurationError(f"EMBEDDING_MODEL is required for {self.name}.")
        self.model = model
        self.dim = dim
        self._client = httpx.AsyncClient(
            base_url=(base_url or self.default_base_url).rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @abstractmethod
    def _payload(self, texts: Sequence[str], *, query: bool) -> dict[str, Any]: ...

    async def _post(self, texts: Sequence[str], *, query: bool) -> list[Vector]:
        if not texts:
            return []
        try:
            resp = await self._client.post("/embeddings", json=self._payload(texts, query=query))
        except httpx.TimeoutException:
            raise UpstreamTimeout(f"{self.name} embeddings timed out.") from None
        except httpx.TransportError as exc:
            raise TransientProviderError(f"{self.name} embeddings unreachable ({type(exc).__name__}).") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientProviderError(
                f"{self.name} embeddings returned {resp.status_code}.", details={"status": resp.status_code}
            )
        if resp.status_code in (401, 403):
            raise ProviderError(f"{self.name} rejected the credentials; check EMBEDDING_API_KEY.")
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.name} embeddings returned {resp.status_code}.", details={"status": resp.status_code}
            )
        try:
            data = resp.json()["data"]
            ordered = sorted(data, key=lambda d: int(d["index"]))
            vectors = [[float(x) for x in item["embedding"]] for item in ordered]
        except (ValueError, KeyError, TypeError):
            raise ProviderError(f"{self.name} returned a malformed embeddings response.") from None
        return self._check(vectors, len(texts))

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return await self._post(texts, query=False)

    async def embed_query(self, text: str) -> Vector:
        return (await self._post([text], query=True))[0]


class VoyageEmbedder(_HttpEmbedder):
    name = "voyage"
    default_base_url = "https://api.voyageai.com/v1"

    def _payload(self, texts: Sequence[str], *, query: bool) -> dict[str, Any]:
        return {
            "input": list(texts),
            "model": self.model,
            "input_type": "query" if query else "document",
            "output_dimension": self.dim,
        }


class OpenAIEmbedder(_HttpEmbedder):
    """OpenAI or any OpenAI-compatible endpoint (set EMBEDDING_BASE_URL)."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def _payload(self, texts: Sequence[str], *, query: bool) -> dict[str, Any]:
        return {"input": list(texts), "model": self.model, "dimensions": self.dim, "encoding_format": "float"}


class SentenceTransformersEmbedder(Embedder):
    """Local model. The heavy package is optional: install `sentence-transformers` to use it."""

    name = "sentence_transformers"

    def __init__(self, *, model: str, dim: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=sentence_transformers needs the optional package 'sentence-transformers'."
            ) from None
        self.dim = dim
        self._model = SentenceTransformer(model)

    async def _encode(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        raw = await asyncio.to_thread(self._model.encode, list(texts), normalize_embeddings=True)
        return self._check([[float(x) for x in row] for row in raw], len(texts))

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return await self._encode(texts)

    async def embed_query(self, text: str) -> Vector:
        return (await self._encode([text]))[0]


# ---------------------------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Embedder]] = {
    "fake": lambda **kw: FakeEmbedder(kw["dim"]),
    "voyage": lambda **kw: VoyageEmbedder(**kw),
    "openai": lambda **kw: OpenAIEmbedder(**kw),
    "sentence_transformers": lambda **kw: SentenceTransformersEmbedder(model=kw["model"], dim=kw["dim"]),
}


def build_embedder(
    provider: str,
    *,
    api_key: str = "",
    model: str = "",
    dim: int,
    base_url: str | None = None,
    timeout_s: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Embedder:
    try:
        factory = _REGISTRY[provider]
    except KeyError:
        raise ConfigurationError(f"Unknown EMBEDDING_PROVIDER '{provider}'. Options: {sorted(_REGISTRY)}") from None
    return factory(api_key=api_key, model=model, dim=dim, base_url=base_url, timeout_s=timeout_s, transport=transport)


def providers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Embedder",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformersEmbedder",
    "Vector",
    "VoyageEmbedder",
    "build_embedder",
    "providers",
    "tokenize",
]
