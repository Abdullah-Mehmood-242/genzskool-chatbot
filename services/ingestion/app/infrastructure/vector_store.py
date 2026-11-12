"""Vector store adapters: Qdrant (default) and in-memory. Both pass tests/contract/test_vector_store.py."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.domain.ports import Vector, VectorStore
from common.contracts import Chunk
from common.errors import ConfigurationError

logger = logging.getLogger(__name__)

UPSERT_BATCH = 128

# Payload fields that retrieval filters on (ADR-0002).
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "index_version": models.PayloadSchemaType.INTEGER,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "course_slug": models.PayloadSchemaType.KEYWORD,
    "category": models.PayloadSchemaType.KEYWORD,
    "level": models.PayloadSchemaType.KEYWORD,
    "mentor": models.PayloadSchemaType.KEYWORD,
    "fee_pkr": models.PayloadSchemaType.INTEGER,
}


def to_payload(chunk: Chunk) -> dict[str, Any]:
    return {"text": chunk.text, **chunk.metadata.model_dump(mode="json")}


def _validate(chunks: Sequence[Chunk], vectors: Sequence[Vector], dim: int | None) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if dim is not None and any(len(v) != dim for v in vectors):
        raise ValueError(f"vector dimension must be {dim}")


class InMemoryVectorStore(VectorStore):
    """Process-local store for tests and VECTOR_STORE=memory (data is lost on restart)."""

    def __init__(self) -> None:
        self.dim: int | None = None
        self.points: dict[str, tuple[Vector, dict[str, Any]]] = {}

    async def ensure_collection(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim:
            raise ConfigurationError(f"Collection has dimension {self.dim}, embeddings have {dim}.")
        self.dim = dim

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if self.dim is None:
            raise ConfigurationError("ensure_collection() must be called before upsert().")
        _validate(chunks, vectors, self.dim)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.id] = (list(vector), to_payload(chunk))

    def _match(self, doc_id: str, version: int | None) -> list[str]:
        return [
            pid
            for pid, (_, p) in self.points.items()
            if p["doc_id"] == doc_id and (version is None or p["index_version"] == version)
        ]

    async def count(self, doc_id: str, index_version: int) -> int:
        return len(self._match(doc_id, index_version))

    async def delete_version(self, doc_id: str, index_version: int) -> None:
        for pid in self._match(doc_id, index_version):
            del self.points[pid]

    async def delete_document(self, doc_id: str) -> None:
        for pid in self._match(doc_id, None):
            del self.points[pid]

    async def healthy(self) -> bool:
        return True


def _doc_filter(doc_id: str, index_version: int | None = None) -> models.Filter:
    must: list[models.Condition] = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    if index_version is not None:
        must.append(models.FieldCondition(key="index_version", match=models.MatchValue(value=index_version)))
    return models.Filter(must=must)


class QdrantVectorStore(VectorStore):
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def from_url(
        cls, url: str, *, api_key: str | None, collection: str, timeout_s: float, check_compatibility: bool = True
    ) -> QdrantVectorStore:
        client = AsyncQdrantClient(
            url=url, api_key=api_key or None, timeout=int(max(1, timeout_s)), check_compatibility=check_compatibility
        )
        return cls(client, collection)

    async def _exists(self) -> bool:
        return bool(await self.client.collection_exists(self.collection))

    async def ensure_collection(self, dim: int) -> None:
        if await self._exists():
            info = await self.client.get_collection(self.collection)
            params = info.config.params.vectors
            size = params.size if isinstance(params, models.VectorParams) else None
            if size != dim:
                raise ConfigurationError(
                    f"Qdrant collection '{self.collection}' has dimension {size}, embeddings have {dim}. "
                    "Use a new QDRANT_COLLECTION (or drop the old one) and re-ingest."
                )
            return
        try:
            await self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
        except Exception:
            if not await self._exists():  # a concurrent creator is fine; anything else is a real error
                raise
        for field_name, schema in PAYLOAD_INDEXES.items():
            await self.client.create_payload_index(self.collection, field_name=field_name, field_schema=schema)
        logger.info("qdrant_collection_created", extra={"collection": self.collection, "dim": dim})

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        _validate(chunks, vectors, None)
        points = [
            models.PointStruct(id=c.id, vector=list(v), payload=to_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            await self.client.upsert(self.collection, points=points[i : i + UPSERT_BATCH], wait=True)

    async def count(self, doc_id: str, index_version: int) -> int:
        if not await self._exists():
            return 0
        result = await self.client.count(self.collection, count_filter=_doc_filter(doc_id, index_version), exact=True)
        return int(result.count)

    async def _delete(self, flt: models.Filter) -> None:
        if not await self._exists():
            return
        await self.client.delete(self.collection, points_selector=models.FilterSelector(filter=flt), wait=True)

    async def delete_version(self, doc_id: str, index_version: int) -> None:
        await self._delete(_doc_filter(doc_id, index_version))

    async def delete_document(self, doc_id: str) -> None:
        await self._delete(_doc_filter(doc_id))

    async def healthy(self) -> bool:
        try:
            await self.client.get_collections()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        await self.client.close()
