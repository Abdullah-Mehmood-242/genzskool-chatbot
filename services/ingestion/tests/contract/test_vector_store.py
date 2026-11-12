"""VectorStore port contract: every adapter must behave identically (Liskov substitution)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient

from app.domain.ports import VectorStore
from app.infrastructure.vector_store import InMemoryVectorStore, QdrantVectorStore, to_payload
from common.contracts import Chunk, ChunkMetadata, DocType
from common.errors import ConfigurationError

DIM = 8

# Qdrant's in-process mode (used to run the real adapter without Docker) ignores payload indexes.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant:UserWarning")


def chunk(i: int, *, doc: str = "kb", version: int = 1, fee: int | None = 1000) -> Chunk:
    import uuid

    meta = ChunkMetadata(
        doc_id=doc,
        source_file=f"{doc}.pdf",
        page=1,
        section=f"s{i}",
        doc_type=DocType.COURSE,
        course_slug=f"c{i}",
        fee_pkr=fee,
        content_hash=f"h{i}",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        index_version=version,
    )
    return Chunk(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc}-{version}-{i}")), text=f"text {i}", metadata=meta)


def vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


@pytest.fixture(params=["memory", "qdrant"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[VectorStore]:
    if request.param == "memory":
        yield InMemoryVectorStore()
        return
    client = AsyncQdrantClient(location=":memory:")
    qstore = QdrantVectorStore(client, "contract_test")
    yield qstore
    await qstore.close()


async def test_count_on_missing_collection_is_zero(store: VectorStore) -> None:
    assert await store.count("kb", 1) == 0


async def test_delete_on_missing_collection_is_noop(store: VectorStore) -> None:
    await store.delete_version("kb", 1)
    await store.delete_document("kb")


async def test_upsert_and_count_by_version(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    await store.upsert([chunk(i) for i in range(3)], [vec(i) for i in range(3)])
    await store.upsert([chunk(i, version=2) for i in range(5)], [vec(i) for i in range(5)])
    assert await store.count("kb", 1) == 3
    assert await store.count("kb", 2) == 5
    assert await store.count("other", 1) == 0


async def test_upsert_is_idempotent_for_same_ids(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    for _ in range(3):
        await store.upsert([chunk(0), chunk(1)], [vec(0), vec(1)])
    assert await store.count("kb", 1) == 2


async def test_delete_version_keeps_other_versions_and_docs(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    await store.upsert([chunk(0), chunk(1, version=2), chunk(2, doc="other")], [vec(0), vec(1), vec(2)])
    await store.delete_version("kb", 1)
    assert await store.count("kb", 1) == 0
    assert await store.count("kb", 2) == 1
    assert await store.count("other", 1) == 1


async def test_delete_document_removes_all_versions_only_for_that_doc(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    await store.upsert([chunk(0), chunk(1, version=2), chunk(2, doc="other")], [vec(0), vec(1), vec(2)])
    await store.delete_document("kb")
    assert await store.count("kb", 1) == 0
    assert await store.count("kb", 2) == 0
    assert await store.count("other", 1) == 1


async def test_ensure_collection_is_idempotent(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    await store.ensure_collection(DIM)
    await store.upsert([chunk(0)], [vec(0)])
    await store.ensure_collection(DIM)
    assert await store.count("kb", 1) == 1


async def test_dimension_mismatch_is_a_configuration_error(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    with pytest.raises(ConfigurationError, match="dimension"):
        await store.ensure_collection(DIM * 2)


async def test_mismatched_lengths_rejected(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    with pytest.raises(ValueError, match="vectors"):
        await store.upsert([chunk(0), chunk(1)], [vec(0)])


async def test_empty_upsert_is_noop(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    await store.upsert([], [])
    assert await store.count("kb", 1) == 0


async def test_healthy(store: VectorStore) -> None:
    assert await store.healthy() is True


async def test_large_upsert_is_batched(store: VectorStore) -> None:
    await store.ensure_collection(DIM)
    n = 300  # > UPSERT_BATCH
    await store.upsert([chunk(i) for i in range(n)], [vec(i) for i in range(n)])
    assert await store.count("kb", 1) == n


# ---------- adapter specifics ----------


def test_payload_contains_text_and_all_metadata() -> None:
    payload = to_payload(chunk(1, fee=None))
    assert payload["text"] == "text 1"
    assert payload["fee_pkr"] is None
    assert payload["doc_type"] == "course"
    assert payload["index_version"] == 1
    assert set(ChunkMetadata.model_fields) <= set(payload)
    assert isinstance(payload["ingested_at"], str)


async def test_memory_store_requires_collection_first() -> None:
    with pytest.raises(ConfigurationError):
        await InMemoryVectorStore().upsert([chunk(0)], [vec(0)])


async def test_memory_store_rejects_wrong_dimension() -> None:
    s = InMemoryVectorStore()
    await s.ensure_collection(DIM)
    with pytest.raises(ValueError, match="dimension"):
        await s.upsert([chunk(0)], [[1.0] * (DIM + 1)])


async def test_qdrant_payload_is_filterable_and_text_retrievable() -> None:
    client = AsyncQdrantClient(location=":memory:")
    s = QdrantVectorStore(client, "payload_test")
    await s.ensure_collection(DIM)
    await s.upsert([chunk(0)], [vec(0)])
    points, _ = await client.scroll("payload_test", with_payload=True, limit=10)
    assert points[0].payload is not None
    assert points[0].payload["text"] == "text 0"
    assert points[0].payload["course_slug"] == "c0"
    await s.close()


async def test_qdrant_unreachable_is_unhealthy() -> None:
    s = QdrantVectorStore.from_url(
        "http://127.0.0.1:1", api_key=None, collection="x", timeout_s=1, check_compatibility=False
    )
    assert await s.healthy() is False
    await s.close()
