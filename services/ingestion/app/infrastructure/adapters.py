"""Small adapters: embedding bridge, clock, and in-memory registry/artifacts for tests and offline runs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.domain.models import IndexRecord
from app.domain.ports import ChunkArtifactStore, Clock, EmbeddingProvider, IngestionRegistry, Vector
from common.contracts import Chunk
from common.embeddings import Embedder


class EmbeddingAdapter(EmbeddingProvider):
    """Bridges the shared embedding clients (common.embeddings) to this service's port."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    @property
    def dim(self) -> int:
        return self.embedder.dim

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return await self.embedder.embed_documents(texts)

    async def aclose(self) -> None:
        await self.embedder.aclose()


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


class InMemoryRegistry(IngestionRegistry):
    def __init__(self) -> None:
        self.records: dict[str, IndexRecord] = {}

    def get(self, doc_id: str) -> IndexRecord | None:
        return self.records.get(doc_id)

    def put(self, record: IndexRecord) -> None:
        self.records[record.doc_id] = record

    def delete(self, doc_id: str) -> None:
        self.records.pop(doc_id, None)

    def list(self) -> list[IndexRecord]:
        return sorted(self.records.values(), key=lambda r: r.doc_id)


class InMemoryChunkArtifactStore(ChunkArtifactStore):
    def __init__(self) -> None:
        self.files: dict[tuple[str, int], list[Chunk]] = {}

    def write(self, doc_id: str, index_version: int, chunks: Sequence[Chunk]) -> None:
        self.files[(doc_id, index_version)] = list(chunks)

    def read(self, doc_id: str, index_version: int) -> list[Chunk]:
        return list(self.files.get((doc_id, index_version), []))

    def delete(self, doc_id: str, index_version: int | None = None) -> None:
        for key in [k for k in self.files if k[0] == doc_id and (index_version is None or k[1] == index_version)]:
            del self.files[key]
