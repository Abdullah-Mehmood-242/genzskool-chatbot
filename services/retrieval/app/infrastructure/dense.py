"""Dense retrievers: Qdrant (default) and in-memory (offline; embeds the corpus itself)."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Collection

from qdrant_client import AsyncQdrantClient, models

from app.domain.models import Candidate, Corpus
from app.domain.ports import DenseRetriever
from app.domain.ranking import ranked
from common.embeddings import Embedder


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryDenseRetriever(DenseRetriever):
    """Embeds the active corpus once per signature. Used with VECTOR_STORE=memory and in tests."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._signature: str | None = None
        self._vectors: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def _ensure(self, corpus: Corpus) -> None:
        async with self._lock:
            if self._signature != corpus.signature:
                vectors = await self.embedder.embed_documents([c.text for c in corpus.chunks])
                self._vectors = {c.id: v for c, v in zip(corpus.chunks, vectors, strict=True)}
                self._signature = corpus.signature

    async def search(self, query_text: str, corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        if not len(corpus) or not allowed_ids:
            return []
        await self._ensure(corpus)
        q = await self.embedder.embed_query(query_text)
        pairs = [(cid, cosine(q, self._vectors[cid])) for cid in allowed_ids if cid in self._vectors]
        return ranked(pairs, min_score=0.0, k=k)

    async def healthy(self) -> bool:
        return True


class QdrantDenseRetriever(DenseRetriever):
    """Searches the collection ingestion fills. Restricting to ``allowed_ids`` (HasId) both applies the
    metadata filters exactly and pins results to the active index versions."""

    def __init__(self, client: AsyncQdrantClient, collection: str, embedder: Embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        api_key: str | None,
        collection: str,
        embedder: Embedder,
        timeout_s: float,
        check_compatibility: bool = True,
    ) -> QdrantDenseRetriever:
        client = AsyncQdrantClient(
            url=url, api_key=api_key or None, timeout=int(max(1, timeout_s)), check_compatibility=check_compatibility
        )
        return cls(client, collection, embedder)

    async def search(self, query_text: str, corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        if not allowed_ids:
            return []
        vector = await self.embedder.embed_query(query_text)
        response = await self.client.query_points(
            self.collection,
            query=vector,
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=list(allowed_ids))]),
            limit=k,
            with_payload=False,
        )
        return ranked([(str(p.id), float(p.score)) for p in response.points], min_score=0.0, k=k)

    async def healthy(self) -> bool:
        try:
            return bool(await self.client.collection_exists(self.collection))
        except Exception:
            return False

    async def close(self) -> None:
        await self.client.close()
