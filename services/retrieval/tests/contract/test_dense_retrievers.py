"""DenseRetriever port contract: in-memory and Qdrant adapters must behave the same (Liskov)."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import section_id

from app.domain.models import Corpus
from app.domain.ports import DenseRetriever
from app.infrastructure.dense import InMemoryDenseRetriever, QdrantDenseRetriever, cosine
from common.embeddings import FakeEmbedder

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
EMB = FakeEmbedder(256)


async def _qdrant(corpus: Corpus) -> QdrantDenseRetriever:
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection("kb", vectors_config=models.VectorParams(size=256, distance=models.Distance.COSINE))
    vectors = await EMB.embed_documents([c.text for c in corpus.chunks])
    await client.upsert(
        "kb",
        points=[
            models.PointStruct(id=c.id, vector=v, payload={"doc_id": c.metadata.doc_id})
            for c, v in zip(corpus.chunks, vectors, strict=True)
        ],
        wait=True,
    )
    return QdrantDenseRetriever(client, "kb", EMB)


@pytest.fixture(params=["memory", "qdrant"])
async def dense(request: pytest.FixtureRequest, kb_corpus: Corpus) -> AsyncIterator[DenseRetriever]:
    if request.param == "memory":
        yield InMemoryDenseRetriever(EMB)
        return
    retriever = await _qdrant(kb_corpus)
    yield retriever
    await retriever.close()


async def test_contract_finds_most_similar_chunk(dense: DenseRetriever, kb_corpus: Corpus) -> None:
    target = next(c for c in kb_corpus.chunks if c.metadata.course_slug == "ai-automation")
    results = await dense.search(target.text, kb_corpus, {c.id for c in kb_corpus.chunks}, 5)
    assert results[0].chunk_id == target.id
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


async def test_contract_ranks_and_orders(dense: DenseRetriever, kb_corpus: Corpus) -> None:
    results = await dense.search("Shopify store dropshipping", kb_corpus, {c.id for c in kb_corpus.chunks}, 10)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(r.score > 0 for r in results)
    assert len(results) <= 10


async def test_contract_respects_allowed_ids(dense: DenseRetriever, kb_corpus: Corpus) -> None:
    allowed = {section_id(kb_corpus, "Mentor: "), section_id(kb_corpus, "Contact")}
    results = await dense.search("Shopify store", kb_corpus, allowed, 10)
    assert {r.chunk_id for r in results} <= allowed


async def test_contract_empty_allowed_set(dense: DenseRetriever, kb_corpus: Corpus) -> None:
    assert await dense.search("anything", kb_corpus, set(), 5) == []


async def test_contract_k_limits(dense: DenseRetriever, kb_corpus: Corpus) -> None:
    results = await dense.search("course fee mentor", kb_corpus, {c.id for c in kb_corpus.chunks}, 2)
    assert len(results) <= 2


async def test_contract_healthy(dense: DenseRetriever) -> None:
    assert await dense.healthy() is True


async def test_memory_retriever_reembeds_only_when_corpus_changes(kb_corpus: Corpus) -> None:
    calls: list[int] = []

    class Counting(FakeEmbedder):
        async def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            calls.append(len(texts))
            return await super().embed_documents(texts)

    retriever = InMemoryDenseRetriever(Counting(64))
    allowed = {c.id for c in kb_corpus.chunks}
    await retriever.search("a", kb_corpus, allowed, 3)
    await retriever.search("b", kb_corpus, allowed, 3)
    assert calls == [22]
    smaller = Corpus(chunks=kb_corpus.chunks[:5], versions=kb_corpus.versions, signature="other")
    await retriever.search("c", smaller, {c.id for c in smaller.chunks}, 3)
    assert calls == [22, 5]


async def test_memory_retriever_empty_corpus() -> None:
    assert await InMemoryDenseRetriever(EMB).search("q", Corpus(), {"x"}, 3) == []


async def test_qdrant_missing_collection_raises_for_graceful_degradation(kb_corpus: Corpus) -> None:
    retriever = QdrantDenseRetriever(AsyncQdrantClient(location=":memory:"), "missing", EMB)
    assert await retriever.healthy() is False
    with pytest.raises(Exception):  # noqa: B017, PT011 - any client error; the service degrades to BM25
        await retriever.search("q", kb_corpus, {kb_corpus.chunks[0].id}, 3)
    await retriever.close()


async def test_qdrant_unreachable_is_unhealthy() -> None:
    r = QdrantDenseRetriever.from_url(
        "http://127.0.0.1:1", api_key=None, collection="x", embedder=EMB, timeout_s=1, check_compatibility=False
    )
    assert await r.healthy() is False
    await r.close()


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([0.0, 0.0], [1.0, 0.0], 0.0),
        ([3.0, 4.0], [6.0, 8.0], 1.0),
    ],
)
def test_cosine(a: list[float], b: list[float], expected: float) -> None:
    assert cosine(a, b) == pytest.approx(expected)
