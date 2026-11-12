"""Composition root: the ONLY place where concrete adapters are chosen and wired to ports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field, field_validator

from app.api.routes import router
from app.application.search import SearchService
from app.domain.models import RetrievalConfig
from app.domain.ports import CorpusProvider, DenseRetriever, Reranker
from app.domain.ranking import DefaultFilterBuilder, QueryNormalizer, ReciprocalRankFusion
from app.infrastructure.corpus import FileCorpusProvider
from app.infrastructure.dense import InMemoryDenseRetriever, QdrantDenseRetriever
from app.infrastructure.rerank import NoopRewriter, build_reranker
from app.infrastructure.sparse import Bm25SparseRetriever
from app.settings import Settings
from common.config import load_yaml_model
from common.embeddings import Embedder, build_embedder
from common.observability import ReadinessCheck


class RetrievalFile(BaseModel):
    """Schema of config/retrieval.yaml (validated at startup)."""

    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)
    fusion_k: int = Field(default=60, ge=1)
    score_threshold: float = Field(default=0.3, ge=0, le=1)
    dense_min_score: float = Field(default=0.15, ge=0, le=1)
    sparse_min_score: float = Field(default=0.0, ge=0)
    min_term_matches: int = Field(default=2, ge=1)
    min_term_coverage: float = Field(default=0.5, ge=0, le=1)
    use_reranker: bool = False
    use_query_rewrite: bool = False
    synonyms: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("synonyms")
    @classmethod
    def _lower(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        return {k.lower(): [s.lower() for s in vals] for k, vals in v.items()}

    def to_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            fusion_k=self.fusion_k,
            score_threshold=self.score_threshold,
            dense_min_score=self.dense_min_score,
            sparse_min_score=self.sparse_min_score,
            min_term_matches=self.min_term_matches,
            min_term_coverage=self.min_term_coverage,
            use_reranker=self.use_reranker,
            use_query_rewrite=self.use_query_rewrite,
        )


@dataclass
class Container:
    settings: Settings
    corpus: CorpusProvider
    dense: DenseRetriever
    reranker: Reranker
    embedder: Embedder
    service: SearchService

    def readiness_checks(self) -> dict[str, ReadinessCheck]:
        async def index_loaded() -> bool:
            return len(await self.corpus.current()) > 0

        async def vector_store() -> bool:
            return await self.dense.healthy()

        return {"index": index_loaded, "vector_store": vector_store}

    def routers(self) -> list[APIRouter]:
        return [router]

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        await self.corpus.current()  # warm the corpus (and fail loudly in logs if unreadable)
        try:
            yield
        finally:
            for resource in (self.dense, self.reranker, self.embedder):
                close: Any = getattr(resource, "aclose", None) or getattr(resource, "close", None)
                if close is not None:
                    await close()


def build_container(
    settings: Settings,
    *,
    corpus: CorpusProvider | None = None,
    dense: DenseRetriever | None = None,
    reranker: Reranker | None = None,
    embedder: Embedder | None = None,
) -> Container:
    """Wire adapters from settings. Keyword overrides exist for tests only."""
    file_cfg = load_yaml_model(settings.config_dir / "retrieval.yaml", RetrievalFile)
    embedder = embedder or build_embedder(
        settings.embedding_provider,
        api_key=settings.embedding_api_key.get_secret_value(),
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        base_url=settings.embedding_base_url,
        timeout_s=settings.embed_timeout_s,
    )
    corpus = corpus or FileCorpusProvider(settings.processed_dir)
    if dense is None:
        dense = (
            InMemoryDenseRetriever(embedder)
            if settings.vector_store == "memory"
            else QdrantDenseRetriever.from_url(
                settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value(),
                collection=settings.qdrant_collection,
                embedder=embedder,
                timeout_s=settings.qdrant_timeout_s,
            )
        )
    reranker = reranker or build_reranker(
        settings.reranker_provider,
        api_key=settings.reranker_api_key.get_secret_value(),
        model=settings.reranker_model,
        timeout_s=settings.reranker_timeout_s,
    )
    config = file_cfg.to_config()
    service = SearchService(
        corpus=corpus,
        normalizer=QueryNormalizer(file_cfg.synonyms),
        rewriter=NoopRewriter(),
        filters=DefaultFilterBuilder(),
        dense=dense,
        sparse=Bm25SparseRetriever(),
        fusion=ReciprocalRankFusion(config.fusion_k),
        reranker=reranker,
        config=config,
    )
    return Container(
        settings=settings, corpus=corpus, dense=dense, reranker=reranker, embedder=embedder, service=service
    )
