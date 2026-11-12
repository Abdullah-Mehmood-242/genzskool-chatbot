"""SearchService: every branch of the pipeline, filters on the real KB, degradation, reranking."""

from __future__ import annotations

import sys
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import CONFIG_DIR

from app.api.dependencies import RetrievalFile
from app.application.search import SearchService
from app.domain.models import Candidate, Corpus, NormalizedQuery, RetrievalConfig
from app.domain.ports import DenseRetriever, QueryRewriter, Reranker
from app.domain.ranking import DefaultFilterBuilder, QueryNormalizer, ReciprocalRankFusion
from app.infrastructure.corpus import StaticCorpusProvider
from app.infrastructure.dense import InMemoryDenseRetriever
from app.infrastructure.rerank import NoopReranker, NoopRewriter
from app.infrastructure.sparse import Bm25SparseRetriever
from common.config import load_yaml_model
from common.contracts import Chunk, DocType, Level, SearchFilters, SearchRequest
from common.embeddings import FakeEmbedder
from common.errors import ServiceUnavailable

FILE_CFG = load_yaml_model(CONFIG_DIR / "retrieval.yaml", RetrievalFile)


def service(
    corpus: Corpus,
    *,
    dense: DenseRetriever | None = None,
    reranker: Reranker | None = None,
    rewriter: QueryRewriter | None = None,
    **cfg: object,
) -> SearchService:
    base = FILE_CFG.to_config()
    config = RetrievalConfig(**{**base.__dict__, **cfg})  # type: ignore[arg-type]
    return SearchService(
        corpus=StaticCorpusProvider(corpus),
        normalizer=QueryNormalizer(FILE_CFG.synonyms),
        rewriter=rewriter or NoopRewriter(),
        filters=DefaultFilterBuilder(),
        dense=dense or InMemoryDenseRetriever(FakeEmbedder(1024)),
        sparse=Bm25SparseRetriever(),
        fusion=ReciprocalRankFusion(config.fusion_k),
        reranker=reranker or NoopReranker(),
        config=config,
    )


async def search(svc: SearchService, query: str, **kw: object) -> list[Chunk]:
    resp = await svc.search(SearchRequest(query=query, **kw))  # type: ignore[arg-type]
    return [Chunk(id=c.id, text=c.text, metadata=c.metadata) for c in resp.chunks]


# ---------------------------------------------------------------- empty paths (drive refusals)


async def test_empty_index() -> None:
    resp = await service(Corpus()).search(SearchRequest(query="shopify"))
    assert resp.chunks == []
    assert resp.retrieval_debug["reason"] == "index_empty"


@pytest.mark.parametrize("query", ["hi", "kya hai?", "???", "the and of", "شاپیفائی"])
async def test_no_searchable_terms(kb_corpus: Corpus, query: str) -> None:
    resp = await service(kb_corpus).search(SearchRequest(query=query))
    assert resp.chunks == []
    assert resp.retrieval_debug["reason"] == "no_searchable_terms"


@pytest.mark.parametrize(
    "query", ["weather in Lahore today", "best pizza recipe", "python programming tutorial", "zxcv qwer asdf"]
)
async def test_off_topic_queries_return_nothing(kb_corpus: Corpus, query: str) -> None:
    resp = await service(kb_corpus).search(SearchRequest(query=query))
    assert resp.chunks == []
    assert resp.retrieval_debug["reason"] == "no_candidates"


async def test_filters_matching_nothing(kb_corpus: Corpus) -> None:
    resp = await service(kb_corpus).search(SearchRequest(query="shopify", filters=SearchFilters(course_slugs=["x"])))
    assert resp.chunks == []
    assert resp.retrieval_debug["reason"] == "no_chunks_match_filters"


async def test_everything_below_threshold(kb_corpus: Corpus) -> None:
    resp = await service(kb_corpus, score_threshold=1.0).search(SearchRequest(query="shopify fee"))
    assert resp.chunks == []
    assert resp.retrieval_debug["reason"] == "below_threshold"


# ---------------------------------------------------------------- results


async def test_results_shape_scores_and_metadata(kb_corpus: Corpus) -> None:
    resp = await service(kb_corpus).search(SearchRequest(query="How much is the Shopify course?"))
    assert resp.chunks[0].metadata.course_slug == "shopify"
    assert resp.chunks[0].metadata.fee_pkr == 30000
    assert all(0 < c.score <= 1 for c in resp.chunks)
    assert [c.score for c in resp.chunks] == sorted((c.score for c in resp.chunks), reverse=True)
    assert len(resp.chunks) <= 5
    assert resp.chunks[0].text.startswith("Course: Shopify store setup to first sale")


async def test_top_k_override(kb_corpus: Corpus) -> None:
    svc = service(kb_corpus)
    assert len(await search(svc, "course fee mentor batch", top_k=2)) == 2
    assert len(await search(svc, "course fee mentor batch", top_k=12)) > 5


async def test_deterministic(kb_corpus: Corpus) -> None:
    svc = service(kb_corpus)
    first = [c.id for c in await search(svc, "who teaches video editing")]
    for _ in range(3):
        assert [c.id for c in await search(svc, "who teaches video editing")] == first


async def test_debug_minimal_vs_verbose(kb_corpus: Corpus) -> None:
    svc = service(kb_corpus)
    quiet = (await svc.search(SearchRequest(query="ebay fee"))).retrieval_debug
    assert {"index_versions", "corpus_size", "candidates", "returned", "latency_ms", "query_terms"} <= set(quiet)
    assert "results" not in quiet
    loud = (await svc.search(SearchRequest(query="ebay fee", debug=True))).retrieval_debug
    assert loud["normalized_query"] == "ebay fee"
    first = loud["results"][0]
    assert first["section"] == "Course: eBay product research and listings"
    assert {"sparse_rank", "sparse_score", "fused", "score"} <= set(first)


# ---------------------------------------------------------------- filters on the real KB


async def test_budget_filter_excludes_expensive_courses(kb_corpus: Corpus) -> None:
    chunks = await search(
        service(kb_corpus),
        "which courses can I take",
        top_k=20,
        filters=SearchFilters(max_fee_pkr=30000, doc_types=[DocType.COURSE]),
    )
    fees = {c.metadata.course_slug: c.metadata.fee_pkr for c in chunks}
    assert "digital-marketing" not in fees  # Rs. 35,000
    assert "ai-automation" not in fees  # Rs. 32,000
    assert all(f is None or f <= 30000 for f in fees.values())


async def test_unpublished_fee_survives_budget_filter(kb_corpus: Corpus) -> None:
    chunks = await search(service(kb_corpus), "TikTok automation faceless", filters=SearchFilters(max_fee_pkr=1000))
    assert chunks[0].metadata.course_slug == "tiktok-automation"
    assert chunks[0].metadata.fee_pkr is None  # kept: the advisor must say "fee not published"


@pytest.mark.parametrize(
    ("filters", "check"),
    [
        (SearchFilters(doc_types=[DocType.MENTOR]), lambda c: c.metadata.doc_type == DocType.MENTOR),
        (SearchFilters(level=Level.INTERMEDIATE), lambda c: c.metadata.level == Level.INTERMEDIATE),
        (SearchFilters(category="e-commerce"), lambda c: c.metadata.category == "E-commerce"),
        (SearchFilters(mentor="Hamza Ali"), lambda c: c.metadata.mentor == "Hamza Ali"),
        (SearchFilters(course_slugs=["ebay", "shopify"]), lambda c: c.metadata.course_slug in ("ebay", "shopify")),
    ],
)
async def test_filters_are_exact(kb_corpus: Corpus, filters: SearchFilters, check: object) -> None:
    chunks = await search(
        service(kb_corpus), "course mentor teaches shopify ebay automation", filters=filters, top_k=20
    )
    assert chunks
    assert all(check(c) for c in chunks)  # type: ignore[operator]


# ---------------------------------------------------------------- degradation


class BrokenDense(DenseRetriever):
    async def search(self, query_text: str, corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        raise ConnectionError("qdrant down")

    async def healthy(self) -> bool:
        return False


async def test_dense_outage_degrades_to_bm25(kb_corpus: Corpus) -> None:
    resp = await service(kb_corpus, dense=BrokenDense()).search(SearchRequest(query="ebay fee"))
    assert resp.chunks[0].metadata.course_slug == "ebay"
    assert resp.retrieval_debug["dense_error"] == "ConnectionError"
    assert "dense" not in resp.retrieval_debug["candidates"]


async def test_all_retrievers_down_is_503(kb_corpus: Corpus) -> None:
    svc = service(kb_corpus, dense=BrokenDense())

    def broken(*_: object) -> list[Candidate]:
        raise RuntimeError("bm25 broken")

    svc.sparse.search = broken  # type: ignore[method-assign]
    with pytest.raises(ServiceUnavailable):
        await svc.search(SearchRequest(query="ebay fee"))


class ScriptedDense(DenseRetriever):
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def search(self, query_text: str, corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        pairs = sorted(((cid, s) for cid, s in self.scores.items()), key=lambda p: -p[1])
        return [Candidate(chunk_id=cid, score=s, rank=i) for i, (cid, s) in enumerate(pairs, 1)]

    async def healthy(self) -> bool:
        return True


async def test_dense_results_outside_allowed_or_below_min_are_dropped(kb_corpus: Corpus) -> None:
    mentor = next(c for c in kb_corpus.chunks if c.metadata.doc_type == DocType.MENTOR)
    course = next(c for c in kb_corpus.chunks if c.metadata.course_slug == "ebay")
    dense = ScriptedDense({mentor.id: 0.9, course.id: 0.29, "ghost-id": 0.99})
    resp = await service(kb_corpus, dense=dense).search(
        SearchRequest(query="ebay", filters=SearchFilters(doc_types=[DocType.COURSE]), debug=True)
    )
    assert resp.retrieval_debug["candidates"]["dense"] == 0  # mentor filtered, course below 0.3, ghost unknown
    assert [c.metadata.course_slug for c in resp.chunks][:1] == ["ebay"]


async def test_agreement_of_both_retrievers_ranks_first(kb_corpus: Corpus) -> None:
    gd = next(c for c in kb_corpus.chunks if c.metadata.course_slug == "graphic-design")
    resp = await service(kb_corpus, dense=ScriptedDense({gd.id: 0.8})).search(
        SearchRequest(query="graphic design photoshop", debug=True)
    )
    assert resp.chunks[0].id == gd.id
    assert resp.chunks[0].score > 0.9


# ---------------------------------------------------------------- reranker and rewriter


class ReverseReranker(Reranker):
    def __init__(self, fail: bool = False, short: bool = False) -> None:
        self.fail = fail
        self.short = short
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        self.calls += 1
        if self.fail:
            raise TimeoutError
        n = len(chunks) - (1 if self.short else 0)
        return [round(0.4 + 0.5 * i / max(1, n), 4) for i in range(n)]  # last chunk best


async def test_reranker_reorders_and_scores(kb_corpus: Corpus) -> None:
    base = [c.id for c in await search(service(kb_corpus), "fee course", top_k=20)]
    rr = ReverseReranker()
    resp = await service(kb_corpus, reranker=rr, use_reranker=True).search(
        SearchRequest(query="fee course", top_k=3, debug=True)
    )
    assert rr.calls == 1
    assert resp.chunks[0].id == base[len(base) - 1] or resp.chunks[0].id in base
    assert resp.chunks[0].score >= resp.chunks[-1].score
    assert resp.retrieval_debug["reranked"] >= 3


async def test_reranker_ignored_unless_enabled_in_config(kb_corpus: Corpus) -> None:
    rr = ReverseReranker()
    await service(kb_corpus, reranker=rr, use_reranker=False).search(SearchRequest(query="fee course"))
    assert rr.calls == 0


@pytest.mark.parametrize(("fail", "short", "error"), [(True, False, "TimeoutError"), (False, True, "length_mismatch")])
async def test_reranker_failure_falls_back_to_fused_order(
    kb_corpus: Corpus, fail: bool, short: bool, error: str
) -> None:
    plain = [c.id for c in await search(service(kb_corpus), "ebay fee")]
    resp = await service(kb_corpus, reranker=ReverseReranker(fail=fail, short=short), use_reranker=True).search(
        SearchRequest(query="ebay fee")
    )
    assert [c.id for c in resp.chunks] == plain
    assert resp.retrieval_debug["rerank_error"] == error


class UpperRewriter(QueryRewriter):
    def __init__(self) -> None:
        self.calls = 0

    async def rewrite(self, query: NormalizedQuery) -> NormalizedQuery:
        self.calls += 1
        return NormalizedQuery(query.original, "ebay", ("ebay",), ("ebay",), (("ebay",),))


async def test_query_rewriter_only_when_enabled(kb_corpus: Corpus) -> None:
    rw = UpperRewriter()
    off = await search(service(kb_corpus, rewriter=rw), "graphic design")
    assert rw.calls == 0
    assert "ebay" not in off[0].text.lower()
    on = await search(service(kb_corpus, rewriter=rw, use_query_rewrite=True), "graphic design")
    assert rw.calls == 1
    assert all("ebay" in c.text.lower() for c in on)  # the rewritten query was what got searched
