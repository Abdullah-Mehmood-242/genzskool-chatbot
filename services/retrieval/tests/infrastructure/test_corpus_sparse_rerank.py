"""Corpus provider (reload + failure modes), BM25 retriever, rerankers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import DOC, load_fixture_chunks, make_chunk, write_index

from app.domain.models import Corpus
from app.infrastructure import rerank as rerank_mod
from app.infrastructure.corpus import FileCorpusProvider, StaticCorpusProvider
from app.infrastructure.rerank import HttpReranker, NoopReranker, NoopRewriter, build_reranker
from app.infrastructure.sparse import Bm25SparseRetriever
from common.errors import ConfigurationError, ProviderError, TransientProviderError, UpstreamTimeout

# ---------------------------------------------------------------- corpus provider


async def test_loads_fixture_index(processed_dir: Path) -> None:
    corpus = await FileCorpusProvider(processed_dir).current()
    assert len(corpus) == 22
    assert corpus.versions == ((DOC, 1),)
    assert corpus.warnings == ()
    assert corpus.signature == f"22:{DOC}@1"


async def test_missing_registry_is_empty_not_an_error(tmp_path: Path) -> None:
    corpus = await FileCorpusProvider(tmp_path / "nothing").current()
    assert len(corpus) == 0
    assert "nothing has been ingested" in corpus.warnings[0]


@pytest.mark.parametrize("content", ["{oops", "[]", '{"documents": {"x": {}}}', '{"documents": 3}'])
async def test_unreadable_registry_serves_empty(tmp_path: Path, content: str) -> None:
    tmp_path.joinpath("registry.json").write_text(content, encoding="utf-8")
    corpus = await FileCorpusProvider(tmp_path).current()
    assert len(corpus) == 0
    assert "unreadable" in corpus.warnings[0]


async def test_missing_and_corrupt_chunk_files_skip_only_that_document(tmp_path: Path) -> None:
    good = [make_chunk("g1", "good text", doc="good")]
    write_index(
        tmp_path,
        {
            "good": (1, good),
            "missing": (1, [make_chunk("m1", "x", doc="missing")]),
            "corrupt": (1, [make_chunk("c1", "x", doc="corrupt")]),
        },
    )
    (tmp_path / "missing.v1.chunks.jsonl").unlink()
    (tmp_path / "corrupt.v1.chunks.jsonl").write_text("{not json}\n", encoding="utf-8")
    corpus = await FileCorpusProvider(tmp_path).current()
    assert [c.id for c in corpus.chunks] == ["g1"]
    assert corpus.versions == (("good", 1),)
    assert any("missing.v1" in w for w in corpus.warnings)
    assert any("corrupt.v1" in w for w in corpus.warnings)


async def test_chunks_with_wrong_doc_or_version_are_dropped(tmp_path: Path) -> None:
    write_index(
        tmp_path,
        {
            "d": (
                2,
                [
                    make_chunk("ok", "x", doc="d", version=2),
                    make_chunk("old", "x", doc="d", version=1),
                    make_chunk("alien", "x", doc="other", version=2),
                ],
            )
        },
    )
    corpus = await FileCorpusProvider(tmp_path).current()
    assert [c.id for c in corpus.chunks] == ["ok"]
    assert any("wrong doc/version" in w for w in corpus.warnings)


async def test_unsafe_doc_ids_in_registry_are_ignored(tmp_path: Path) -> None:
    tmp_path.joinpath("registry.json").write_text(
        json.dumps({"documents": {"../etc": {"active_index_version": 1}}}), encoding="utf-8"
    )
    corpus = await FileCorpusProvider(tmp_path).current()
    assert len(corpus) == 0
    assert any("unsafe" in w for w in corpus.warnings)


async def test_reloads_when_registry_is_replaced(tmp_path: Path) -> None:
    write_index(tmp_path, {"d": (1, [make_chunk("v1a", "alpha", doc="d", version=1)])})
    provider = FileCorpusProvider(tmp_path)
    first = await provider.current()
    assert [c.id for c in first.chunks] == ["v1a"]
    # Same doc, new version, same sizes: only the atomic replace (new inode) reveals the change.
    write_index(tmp_path, {"d": (2, [make_chunk("v2a", "alpha", doc="d", version=2)])})
    second = await provider.current()
    assert [c.id for c in second.chunks] == ["v2a"]
    assert second.signature != first.signature


async def test_does_not_reload_when_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_index(tmp_path, {"d": (1, [make_chunk("a", "x", doc="d")])})
    provider = FileCorpusProvider(tmp_path)
    loads = {"n": 0}
    real = provider._load

    def counting() -> Corpus:
        loads["n"] += 1
        return real()

    monkeypatch.setattr(provider, "_load", counting)
    for _ in range(5):
        await provider.current()
    assert loads["n"] == 1


async def test_registry_removed_after_load_empties_corpus(tmp_path: Path) -> None:
    write_index(tmp_path, {"d": (1, [make_chunk("a", "x", doc="d")])})
    provider = FileCorpusProvider(tmp_path)
    assert len(await provider.current()) == 1
    (tmp_path / "registry.json").unlink()
    assert len(await provider.current()) == 0


async def test_concurrent_first_loads_are_safe(processed_dir: Path) -> None:
    provider = FileCorpusProvider(processed_dir)
    results = await asyncio.gather(*(provider.current() for _ in range(10)))
    assert {len(c) for c in results} == {22}


async def test_static_provider(kb_corpus: Corpus) -> None:
    assert await StaticCorpusProvider(kb_corpus).current() is kb_corpus


# ---------------------------------------------------------------- BM25


def _corpus(*chunks: tuple[str, str], sig: str = "s") -> Corpus:
    return Corpus(chunks=tuple(make_chunk(cid, text) for cid, text in chunks), signature=sig)


def test_bm25_ranks_by_keyword_relevance() -> None:
    # equal lengths, so only term frequency differs (BM25 also normalises by document length)
    corpus = _corpus(
        ("a", "shopify shopify store"),
        ("b", "shopify store items"),
        ("c", "graphic design work"),
        ("d", "filler words here"),
    )
    results = Bm25SparseRetriever().search(["shopify"], corpus, {"a", "b", "c", "d"}, 10)
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert all(r.score > 0 for r in results)


def test_bm25_respects_allowed_ids_and_k() -> None:
    corpus = _corpus(*((str(i), "fee course") for i in range(10)), ("x", "unrelated"))
    results = Bm25SparseRetriever().search(["fee"], corpus, {"1", "2", "3"}, 2)
    assert len(results) == 2
    assert {r.chunk_id for r in results} <= {"1", "2", "3"}


def test_bm25_no_terms_or_empty_corpus() -> None:
    corpus = _corpus(("a", "text"))
    assert Bm25SparseRetriever().search([], corpus, {"a"}, 5) == []
    assert Bm25SparseRetriever().search(["text"], Corpus(), {"a"}, 5) == []


def test_bm25_corpus_of_only_stopwords() -> None:
    corpus = _corpus(("a", "the and of"), ("b", "is it"))
    assert Bm25SparseRetriever().search(["shopify"], corpus, {"a", "b"}, 5) == []


def test_bm25_rebuilds_index_on_signature_change() -> None:
    bm25 = Bm25SparseRetriever()
    first = _corpus(("a", "shopify"), ("z", "other"), sig="1")
    assert [r.chunk_id for r in bm25.search(["shopify"], first, {"a", "z"}, 5)] == ["a"]
    second = _corpus(("b", "shopify"), ("z", "other"), sig="2")
    assert [r.chunk_id for r in bm25.search(["shopify"], second, {"b", "z"}, 5)] == ["b"]


@pytest.mark.parametrize("n_docs", [2, 4, 10, 22])
def test_bm25_term_in_exactly_half_or_more_of_docs_still_matches(n_docs: int) -> None:
    """Regression: rank_bm25's default IDF is 0 at n = N/2 and negative above, dropping words like 'fee'."""
    half = [(f"fee{i}", "fee details") for i in range(n_docs // 2)]
    rest = [(f"other{i}", "unrelated text") for i in range(n_docs - n_docs // 2)]
    results = Bm25SparseRetriever().search(["fee"], _corpus(*half, *rest), {c for c, _ in half + rest}, 50)
    assert {r.chunk_id for r in results} == {c for c, _ in half}
    everywhere = _corpus(*((f"d{i}", "fee") for i in range(n_docs)))
    assert len(Bm25SparseRetriever().search(["fee"], everywhere, {f"d{i}" for i in range(n_docs)}, 50)) == n_docs


def test_bm25_rarer_terms_weigh_more() -> None:
    corpus = _corpus(("rare", "shopify fee"), ("c1", "fee"), ("c2", "fee"), ("c3", "fee"))
    results = Bm25SparseRetriever().search(["shopify", "fee"], corpus, {"rare", "c1", "c2", "c3"}, 5)
    assert results[0].chunk_id == "rare"


def test_bm25_uses_same_analysis_as_queries() -> None:
    corpus = _corpus(("a", "Fees: Rs. 30,000"), ("b", "other text"), ("c", "more text"))
    assert [r.chunk_id for r in Bm25SparseRetriever().search(["fee", "30000"], corpus, {"a", "b", "c"}, 5)] == ["a"]


# ---------------------------------------------------------------- rerankers


async def test_noop_reranker_and_rewriter(kb_corpus: Corpus) -> None:
    r = NoopReranker()
    assert r.enabled is False
    assert await r.rerank("q", list(kb_corpus.chunks[:3])) == [1.0, 1.0, 1.0]
    from app.domain.models import NormalizedQuery

    q = NormalizedQuery("a", "a", ("a",), ("a",))
    assert await NoopRewriter().rewrite(q) is q


KEY = "-".join(["rerank", "test", "key"])


def _reranker(provider: str, handler: object) -> HttpReranker:
    base, _ = HttpReranker.ENDPOINTS[provider]
    router = respx.Router(base_url=base)
    router.post("/rerank").mock(side_effect=handler)
    return HttpReranker(provider, api_key=KEY, model="rerank-x", transport=httpx.MockTransport(router.handler))


@pytest.mark.parametrize(("provider", "key"), [("cohere", "results"), ("voyage", "data")])
async def test_http_reranker_maps_scores_by_index(provider: str, key: str) -> None:
    chunks = load_fixture_chunks()[:3]
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return httpx.Response(
            200,
            json={
                key: [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 1.7},
                    {"index": 1, "relevance_score": -0.2},
                ]
            },
        )

    scores = await _reranker(provider, handler).rerank("query", chunks)
    assert scores == [1.0, 0.0, 0.9]  # clamped to [0, 1], in input order
    assert seen[0]["documents"] == [c.text for c in chunks]
    assert seen[0]["model"] == "rerank-x"
    assert seen[0]["query"] == "query"


@pytest.mark.parametrize(
    ("status", "exc"),
    [(429, TransientProviderError), (503, TransientProviderError), (401, ProviderError), (400, ProviderError)],
)
async def test_http_reranker_errors(status: int, exc: type[ProviderError]) -> None:
    r = _reranker("cohere", lambda req: httpx.Response(status, json={"error": KEY}))
    with pytest.raises(exc) as info:
        await r.rerank("q", load_fixture_chunks()[:1])
    assert type(info.value) is exc
    assert KEY not in str(info.value)


async def test_http_reranker_timeout_network_malformed() -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("x")

    def refused(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("x")

    chunk = load_fixture_chunks()[:1]
    with pytest.raises(UpstreamTimeout):
        await _reranker("voyage", timeout).rerank("q", chunk)
    with pytest.raises(TransientProviderError):
        await _reranker("voyage", refused).rerank("q", chunk)
    for body in ({"nope": 1}, {"data": [{"index": 5, "relevance_score": 1}]}, {"data": [{"index": 0}]}):
        with pytest.raises(ProviderError, match="malformed"):
            await _reranker("voyage", lambda req, b=body: httpx.Response(200, json=b)).rerank("q", chunk)


async def test_http_reranker_empty_input_makes_no_call() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no call expected")

    assert await _reranker("cohere", fail).rerank("q", []) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"provider": "jina", "api_key": "k", "model": "m"}, "Unknown"),
        ({"provider": "cohere", "api_key": "", "model": "m"}, "RERANKER_API_KEY"),
        ({"provider": "cohere", "api_key": "k", "model": ""}, "RERANKER_MODEL"),
    ],
)
def test_http_reranker_config_errors(kwargs: dict[str, str], match: str) -> None:
    provider = kwargs.pop("provider")
    with pytest.raises(ConfigurationError, match=match):
        HttpReranker(provider, **kwargs)


def test_build_reranker() -> None:
    assert isinstance(build_reranker("none", api_key="", model=""), NoopReranker)
    assert isinstance(build_reranker("voyage", api_key="k", model="m"), HttpReranker)
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigurationError, match="sentence-transformers"):
            build_reranker("cross_encoder", api_key="", model="cross-x")


async def test_cross_encoder_with_stub_model(monkeypatch: pytest.MonkeyPatch, kb_corpus: Corpus) -> None:
    """The cross-encoder adapter logic (sigmoid of logits) without the heavy optional package."""

    class StubModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.0, 10.0, -10.0][: len(pairs)]

    r = object.__new__(rerank_mod.CrossEncoderReranker)
    r._model = StubModel()  # type: ignore[attr-defined]
    scores = await r.rerank("q", list(kb_corpus.chunks[:3]))
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] > 0.99
    assert scores[2] < 0.01
    assert r.enabled is True
    assert await r.rerank("q", []) == []
