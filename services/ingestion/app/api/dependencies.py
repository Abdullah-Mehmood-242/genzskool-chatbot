"""Composition root: the ONLY place where concrete adapters are chosen and wired to ports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from app.api.routes import router
from app.application.auto_ingest import AutoIngestor
from app.application.ingest import IngestionPolicy, IngestionService
from app.domain.chunking import StructureAwareChunker
from app.domain.cleaning import StandardTextCleaner
from app.domain.models import ChunkingConfig
from app.domain.ports import (
    ChunkArtifactStore,
    Clock,
    DocumentLoader,
    DocumentSource,
    EmbeddingProvider,
    IngestionRegistry,
    VectorStore,
)
from app.infrastructure.adapters import EmbeddingAdapter, SystemClock
from app.infrastructure.files import JsonFileRegistry, JsonlChunkArtifactStore
from app.infrastructure.pdf_loader import PdfPlumberLoader
from app.infrastructure.source import LocalDocumentSource
from app.infrastructure.vector_store import InMemoryVectorStore, QdrantVectorStore
from app.settings import Settings
from common.config import load_yaml_model
from common.embeddings import build_embedder
from common.errors import TransientProviderError, UpstreamTimeout
from common.observability import ReadinessCheck
from common.resilience import RetryPolicy


class ChunkingFile(BaseModel):
    """Schema of config/chunking.yaml (validated at startup)."""

    max_tokens: int = Field(default=400, ge=20)
    overlap_tokens: int = Field(default=40, ge=0)
    min_chunk_tokens: int = Field(default=15, ge=0)
    context_header: bool = True
    strip_patterns: list[str] = Field(default_factory=list)

    def to_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            min_chunk_tokens=self.min_chunk_tokens,
            context_header=self.context_header,
            strip_patterns=tuple(self.strip_patterns),
        )


@dataclass
class Container:
    settings: Settings
    source: DocumentSource
    embedder: EmbeddingProvider
    store: VectorStore
    registry: IngestionRegistry
    artifacts: ChunkArtifactStore
    service: IngestionService
    auto_ingestor: AutoIngestor

    def readiness_checks(self) -> dict[str, ReadinessCheck]:
        async def vector_store() -> bool:
            return await self.store.healthy()

        async def pdf_dir() -> bool:
            exists = getattr(self.source, "exists", None)
            return bool(exists()) if callable(exists) else True

        async def startup_ingest() -> bool:
            return self.auto_ingestor.startup_done.is_set()

        return {"vector_store": vector_store, "pdf_dir": pdf_dir, "startup_ingest": startup_ingest}

    def routers(self) -> list[APIRouter]:
        return [router]

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        self.auto_ingestor.start()
        try:
            yield
        finally:
            await self.auto_ingestor.stop()
            for resource in (self.embedder, self.store):
                close: Any = getattr(resource, "aclose", None) or getattr(resource, "close", None)
                if close is not None:
                    await close()


def build_container(
    settings: Settings,
    *,
    store: VectorStore | None = None,
    registry: IngestionRegistry | None = None,
    artifacts: ChunkArtifactStore | None = None,
    source: DocumentSource | None = None,
    loader: DocumentLoader | None = None,
    embedder: EmbeddingProvider | None = None,
    clock: Clock | None = None,
) -> Container:
    """Wire adapters from settings. Keyword overrides exist for tests (fakes are injected here only)."""
    chunking = load_yaml_model(settings.config_dir / "chunking.yaml", ChunkingFile).to_config()
    source = source or LocalDocumentSource(settings.pdf_dir, max_bytes=settings.max_pdf_bytes)
    embedder = embedder or EmbeddingAdapter(
        build_embedder(
            settings.embedding_provider,
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            base_url=settings.embedding_base_url,
            timeout_s=settings.embed_timeout_s,
        )
    )
    if store is None:
        store = (
            InMemoryVectorStore()
            if settings.vector_store == "memory"
            else QdrantVectorStore.from_url(
                settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value(),
                collection=settings.qdrant_collection,
                timeout_s=settings.qdrant_timeout_s,
            )
        )
    registry = registry or JsonFileRegistry(settings.processed_dir)
    artifacts = artifacts or JsonlChunkArtifactStore(settings.processed_dir)
    clock = clock or SystemClock()
    policy = IngestionPolicy(
        embed_batch_size=settings.embed_batch_size,
        retry=RetryPolicy(
            attempts=settings.embed_retry_attempts,
            base_delay_s=0.5,
            max_delay_s=4.0,
            retry_on=(TransientProviderError, UpstreamTimeout),
        ),
        scanned_min_chars_per_page=settings.scanned_min_chars_per_page,
        prune_deleted=settings.auto_ingest_prune,
    )
    service = IngestionService(
        source=source,
        loader=loader or PdfPlumberLoader(),
        cleaner=StandardTextCleaner(chunking.strip_patterns),
        chunker=StructureAwareChunker(chunking),
        embedder=embedder,
        store=store,
        registry=registry,
        artifacts=artifacts,
        clock=clock,
        policy=policy,
    )
    auto = AutoIngestor(
        service,
        source,
        clock,
        on_startup=settings.auto_ingest_on_startup,
        interval_s=settings.auto_ingest_interval_s,
    )
    return Container(
        settings=settings,
        source=source,
        embedder=embedder,
        store=store,
        registry=registry,
        artifacts=artifacts,
        service=service,
        auto_ingestor=auto,
    )
