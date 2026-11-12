"""Search use case: normalise -> (rewrite) -> filter -> dense + sparse -> RRF -> (rerank) -> threshold."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.domain.models import Candidate, Corpus, FusedCandidate, NormalizedQuery, RetrievalConfig
from app.domain.ports import (
    CorpusProvider,
    DenseRetriever,
    FusionStrategy,
    MetadataFilterBuilder,
    QueryRewriter,
    Reranker,
    SparseRetriever,
)
from app.domain.ranking import QueryNormalizer
from app.domain.text import analyze
from common.contracts import RetrievedChunk, SearchRequest, SearchResponse
from common.errors import ServiceUnavailable

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        *,
        corpus: CorpusProvider,
        normalizer: QueryNormalizer,
        rewriter: QueryRewriter,
        filters: MetadataFilterBuilder,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        fusion: FusionStrategy,
        reranker: Reranker,
        config: RetrievalConfig,
    ) -> None:
        self.corpus = corpus
        self.normalizer = normalizer
        self.rewriter = rewriter
        self.filters = filters
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
        self.reranker = reranker
        self.config = config

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        cfg = self.config
        top_k = request.top_k or cfg.top_k
        corpus = await self.corpus.current()
        debug: dict[str, Any] = {
            "index_versions": dict(corpus.versions),
            "corpus_size": len(corpus),
            "top_k": top_k,
        }
        if not len(corpus):
            return self._empty(debug, "index_empty", started)

        query = self.normalizer.normalize(request.query)
        if cfg.use_query_rewrite:
            query = await self.rewriter.rewrite(query)
        debug["query_terms"] = list(query.expanded_terms)
        if not query.terms:
            return self._empty(debug, "no_searchable_terms", started)

        keep = self.filters.build(request.filters)
        allowed = {c.id for c in corpus.chunks if keep(c.metadata)}
        debug["allowed_chunks"] = len(allowed)
        if not allowed:
            return self._empty(debug, "no_chunks_match_filters", started)

        lists = await self._retrieve(query, corpus, allowed, debug)
        fused = self.fusion.fuse(lists)
        scored = await self._maybe_rerank(request.query, fused, corpus, debug)
        results = [(f, s) for f, s in scored if s >= cfg.score_threshold][:top_k]

        debug["candidates"] = {name: len(c) for name, c in lists.items()}
        debug["fused"] = len(fused)
        debug["returned"] = len(results)
        if not results:
            debug["reason"] = "below_threshold" if fused else "no_candidates"
        if request.debug:
            debug["normalized_query"] = query.text
            debug["results"] = [self._explain(f, s, corpus) for f, s in results]
        debug["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        chunks = [
            RetrievedChunk(
                id=f.chunk_id,
                text=corpus.by_id[f.chunk_id].text,
                score=round(s, 4),
                metadata=corpus.by_id[f.chunk_id].metadata,
            )
            for f, s in results
        ]
        logger.info(
            "search",
            extra={
                "returned": len(chunks),
                "candidates": debug["candidates"],
                "reason": debug.get("reason"),
                "latency_ms": debug["latency_ms"],
            },
        )
        return SearchResponse(chunks=chunks, retrieval_debug=debug)

    # ------------------------------------------------------------------ internals

    async def _retrieve(
        self, query: NormalizedQuery, corpus: Corpus, allowed: set[str], debug: dict[str, Any]
    ) -> dict[str, list[Candidate]]:
        cfg = self.config
        lists: dict[str, list[Candidate]] = {}
        dense_result, sparse_result = await asyncio.gather(
            self.dense.search(query.text, corpus, allowed, cfg.candidate_k),
            asyncio.to_thread(self.sparse.search, query.expanded_terms, corpus, allowed, cfg.candidate_k),
            return_exceptions=True,
        )
        for name, result, min_score in (
            ("dense", dense_result, cfg.dense_min_score),
            ("sparse", sparse_result, cfg.sparse_min_score),
        ):
            if isinstance(result, BaseException):
                # Graceful degradation: one retriever down still yields results from the other.
                debug[f"{name}_error"] = type(result).__name__
                logger.warning("retriever_failed", extra={"retriever": name, "error_type": type(result).__name__})
                continue
            kept = [c for c in result if c.chunk_id in allowed and c.score > 0 and c.score >= min_score]
            if name == "sparse":
                kept = [c for c in kept if self._covers(query, corpus, c.chunk_id)]
            lists[name] = [Candidate(chunk_id=c.chunk_id, score=c.score, rank=i) for i, c in enumerate(kept, start=1)]
        if not lists:
            raise ServiceUnavailable("All retrievers failed; search is temporarily unavailable.")
        return lists

    def _covers(self, query: NormalizedQuery, corpus: Corpus, chunk_id: str) -> bool:
        matched, coverage = query.coverage(set(analyze(corpus.by_id[chunk_id].text)))
        return self.config.passes_coverage(matched, coverage)

    async def _maybe_rerank(
        self, query: str, fused: list[FusedCandidate], corpus: Corpus, debug: dict[str, Any]
    ) -> list[tuple[FusedCandidate, float]]:
        scored = [(f, f.score) for f in fused]
        if not (self.config.use_reranker and self.reranker.enabled and fused):
            return scored
        head = fused[: self.config.candidate_k]
        try:
            scores = await self.reranker.rerank(query, [corpus.by_id[f.chunk_id] for f in head])
        except Exception as exc:
            debug["rerank_error"] = type(exc).__name__
            logger.warning("rerank_failed", extra={"error_type": type(exc).__name__})
            return scored
        if len(scores) != len(head):
            debug["rerank_error"] = "length_mismatch"
            return scored
        debug["reranked"] = len(head)
        return sorted(zip(head, scores, strict=True), key=lambda p: (-p[1], p[0].chunk_id))

    def _explain(self, fused: FusedCandidate, score: float, corpus: Corpus) -> dict[str, Any]:
        chunk = corpus.by_id[fused.chunk_id]
        out: dict[str, Any] = {
            "id": fused.chunk_id,
            "section": chunk.metadata.section,
            "score": round(score, 4),
            "fused": round(fused.score, 4),
        }
        for name, cand in fused.contributions.items():
            out[f"{name}_rank"] = cand.rank
            out[f"{name}_score"] = round(cand.score, 4)
        return out

    def _empty(self, debug: dict[str, Any], reason: str, started: float) -> SearchResponse:
        debug.update(reason=reason, returned=0, latency_ms=round((time.perf_counter() - started) * 1000, 2))
        logger.info("search", extra={"returned": 0, "reason": reason})
        return SearchResponse(chunks=[], retrieval_debug=debug)
