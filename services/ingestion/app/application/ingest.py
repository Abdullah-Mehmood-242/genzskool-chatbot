"""Ingestion use cases: ingest (idempotent, blue/green), delete, status, drift detection."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

from app.domain.errors import DocumentError, IndexWriteFailed
from app.domain.models import IndexRecord, SourceFile, is_scanned
from app.domain.ports import (
    ChunkArtifactStore,
    Chunker,
    Clock,
    DocumentLoader,
    DocumentSource,
    EmbeddingProvider,
    IngestionRegistry,
    TextCleaner,
    Vector,
    VectorStore,
)
from app.domain.text import sha256_hex
from common.contracts import Chunk, FileOutcome, FileReport, IngestReport
from common.errors import AppError, ConfigurationError, NotFound, ProviderError, TransientProviderError, UpstreamTimeout
from common.resilience import RetryPolicy, retry_async

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionPolicy:
    embed_batch_size: int = 64
    retry: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            attempts=3, base_delay_s=0.5, max_delay_s=4.0, retry_on=(TransientProviderError, UpstreamTimeout)
        )
    )
    scanned_min_chars_per_page: int = 20
    prune_deleted: bool = True

    def __post_init__(self) -> None:
        if self.embed_batch_size < 1:
            raise ValueError("embed_batch_size must be >= 1")


class _FileFailed(Exception):
    def __init__(self, message: str, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.warnings = warnings or []


class IngestionService:
    def __init__(
        self,
        *,
        source: DocumentSource,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        store: VectorStore,
        registry: IngestionRegistry,
        artifacts: ChunkArtifactStore,
        clock: Clock,
        policy: IngestionPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.source = source
        self.loader = loader
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.registry = registry
        self.artifacts = artifacts
        self.clock = clock
        self.policy = policy or IngestionPolicy()
        self._sleep = sleep
        self._lock = asyncio.Lock()  # one ingestion at a time (API and auto-ingest share it)

    # ------------------------------------------------------------------ queries

    def status(self) -> list[IndexRecord]:
        return self.registry.list()

    async def has_drift(self) -> bool:
        """True if any active index version is missing points (e.g. the Qdrant volume was wiped)."""
        for record in self.registry.list():
            try:
                if await self.store.count(record.doc_id, record.active_index_version) != record.chunks:
                    return True
            except Exception:
                logger.warning("drift_check_failed", extra={"doc_id": record.doc_id})
                return False  # cannot heal while the store is unreachable; try again next tick
        return False

    # ------------------------------------------------------------------ commands

    async def ingest(self, file: str | None = None, *, force: bool = False) -> IngestReport:
        async with self._lock:
            started = self.clock.now()
            sources = [self.source.resolve(file)] if file else self.source.list_files()
            reports = [await self._ingest_one(s, force=force) for s in sources]
            pruned = await self._prune(sources) if file is None and self.policy.prune_deleted else []
            report = IngestReport(files=reports, pruned=pruned, started_at=started, finished_at=self.clock.now())
        logger.info(
            "ingest_finished",
            extra={
                "files": len(reports),
                "ok": report.ok,
                "outcomes": [r.outcome.value for r in reports],
                "pruned": pruned,
            },
        )
        return report

    async def delete_document(self, doc_id: str) -> None:
        async with self._lock:
            if self.registry.get(doc_id) is None:
                raise NotFound(f"Document '{doc_id}' is not indexed.")
            await self._remove(doc_id)

    # ------------------------------------------------------------------ internals

    async def _remove(self, doc_id: str) -> None:
        await self.store.delete_document(doc_id)
        self.artifacts.delete(doc_id)
        self.registry.delete(doc_id)

    async def _prune(self, sources: list[SourceFile]) -> list[str]:
        present = {s.doc_id for s in sources}
        stale = [r.doc_id for r in self.registry.list() if r.doc_id not in present]
        if stale and not sources:
            # An empty PDF dir with indexed docs is far more likely a broken mount than intent.
            logger.warning("prune_skipped_empty_source", extra={"would_prune": stale})
            return []
        for doc_id in stale:
            await self._remove(doc_id)
            logger.info("document_pruned", extra={"doc_id": doc_id})
        return stale

    async def _ingest_one(self, src: SourceFile, *, force: bool) -> FileReport:
        existing = self.registry.get(src.doc_id)
        new_version = (existing.active_index_version + 1) if existing else 1
        try:
            data = await asyncio.to_thread(self.source.read, src)
            content_hash = sha256_hex(data)
            if existing and not force and existing.content_hash == content_hash:
                if await self.store.count(src.doc_id, existing.active_index_version) == existing.chunks:
                    return FileReport(
                        file=src.file,
                        doc_id=src.doc_id,
                        outcome=FileOutcome.UNCHANGED,
                        chunks=existing.chunks,
                        index_version=existing.active_index_version,
                    )
                logger.warning("index_drift_reingest", extra={"doc_id": src.doc_id})

            doc = await asyncio.to_thread(self.loader.load, data, src, content_hash)
            warnings = list(doc.warnings)
            if is_scanned(doc, self.policy.scanned_min_chars_per_page):
                return FileReport(
                    file=src.file,
                    doc_id=src.doc_id,
                    outcome=FileOutcome.SKIPPED_SCANNED,
                    warnings=[*warnings, "No extractable text: looks like a scanned PDF (OCR is out of scope)."],
                )
            cleaned = self.cleaner.clean(doc)
            result = self.chunker.chunk(cleaned, index_version=new_version, ingested_at=self.clock.now())
            warnings.extend(result.warnings)
            if not result.chunks:
                raise _FileFailed("No chunks could be produced from the document.", warnings)

            vectors = await self._embed([c.text for c in result.chunks])
            await self._write_version(src, existing, new_version, content_hash, result.chunks, vectors)
            return FileReport(
                file=src.file,
                doc_id=src.doc_id,
                outcome=FileOutcome.INGESTED,
                chunks=len(result.chunks),
                index_version=new_version,
                warnings=warnings,
            )
        except _FileFailed as exc:
            return self._failed(src, exc.message, exc.warnings)
        except (DocumentError, ProviderError, ConfigurationError) as exc:
            return self._failed(src, exc.message)
        except AppError as exc:
            return self._failed(src, exc.message)
        except Exception as exc:  # never let one file break the batch
            logger.exception("ingest_unexpected_error", extra={"doc_id": src.doc_id})
            return self._failed(src, f"Unexpected error ({type(exc).__name__}).")

    def _failed(self, src: SourceFile, message: str, warnings: list[str] | None = None) -> FileReport:
        existing = self.registry.get(src.doc_id)
        if existing:
            message += f" The previous index version {existing.active_index_version} is still being served."
        logger.warning("ingest_file_failed", extra={"doc_id": src.doc_id, "reason": message})
        return FileReport(
            file=src.file,
            doc_id=src.doc_id,
            outcome=FileOutcome.FAILED,
            error=message,
            warnings=warnings or [],
            index_version=existing.active_index_version if existing else None,
        )

    async def _embed(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        size = self.policy.embed_batch_size
        for i in range(0, len(texts), size):
            batch = texts[i : i + size]
            try:
                vectors.extend(
                    await retry_async(
                        partial(self.embedder.embed_documents, batch), self.policy.retry, sleep=self._sleep
                    )
                )
            except ProviderError as exc:
                raise _FileFailed(
                    f"Embedding failed after {self.policy.retry.attempts} attempt(s) at batch {i // size + 1}: "
                    f"{exc.message}"
                ) from None
        return vectors

    async def _write_version(
        self,
        src: SourceFile,
        existing: IndexRecord | None,
        version: int,
        content_hash: str,
        chunks: list[Chunk],
        vectors: list[Vector],
    ) -> None:
        """Blue/green: write version N+1 fully, verify, flip the registry, then delete version N."""
        await self.store.ensure_collection(self.embedder.dim)
        try:
            await self.store.delete_version(src.doc_id, version)  # leftovers from an earlier crashed attempt
            await self.store.upsert(chunks, vectors)
            stored = await self.store.count(src.doc_id, version)
            if stored != len(chunks):
                raise IndexWriteFailed(f"Vector store holds {stored} of {len(chunks)} chunks for version {version}.")
            self.artifacts.write(src.doc_id, version, chunks)
        except BaseException:
            await self._rollback(src.doc_id, version)
            raise
        self.registry.put(
            IndexRecord(
                doc_id=src.doc_id,
                source_file=src.file,
                content_hash=content_hash,
                active_index_version=version,
                chunks=len(chunks),
                ingested_at=self._ingested_at(chunks),
            )
        )
        if existing and existing.active_index_version != version:
            try:
                await self.store.delete_version(src.doc_id, existing.active_index_version)
                self.artifacts.delete(src.doc_id, existing.active_index_version)
            except Exception:  # the new version is live; a leftover old version is only garbage
                logger.warning("old_version_cleanup_failed", extra={"doc_id": src.doc_id})

    async def _rollback(self, doc_id: str, version: int) -> None:
        try:
            await self.store.delete_version(doc_id, version)
            self.artifacts.delete(doc_id, version)
        except Exception:
            logger.error("rollback_failed", extra={"doc_id": doc_id, "index_version": version})

    def _ingested_at(self, chunks: list[Chunk]) -> datetime:
        return chunks[0].metadata.ingested_at
