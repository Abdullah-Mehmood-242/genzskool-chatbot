"""IngestionService: idempotency, blue/green, rollback, retries, partial failure, pruning, concurrency."""

from __future__ import annotations

import asyncio
import io
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pypdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import KB_LIKE_PAGES, REAL_PDF, ScriptedEmbedder, build_service, make_pdf

from app.application.ingest import IngestionPolicy
from app.domain.ports import Vector
from app.infrastructure.adapters import InMemoryChunkArtifactStore, InMemoryRegistry
from app.infrastructure.vector_store import InMemoryVectorStore
from common.contracts import Chunk, FileOutcome
from common.errors import NotFound, ProviderError, TransientProviderError, UpstreamTimeout
from common.resilience import RetryPolicy

DOC = "genzskool_knowledge_base"


class Env:
    def __init__(self, pdf_dir: Path, **kw: object) -> None:
        self.pdf_dir = pdf_dir
        self.store: InMemoryVectorStore = kw.pop("store", None) or InMemoryVectorStore()  # type: ignore[assignment]
        self.registry = InMemoryRegistry()
        self.artifacts = InMemoryChunkArtifactStore()
        self.embedder: ScriptedEmbedder = kw.pop("embedder", None) or ScriptedEmbedder()  # type: ignore[assignment]
        self.service = build_service(
            pdf_dir, embedder=self.embedder, store=self.store, registry=self.registry, artifacts=self.artifacts, **kw
        )  # type: ignore[arg-type]

    async def points(self, version: int, doc: str = DOC) -> int:
        return await self.store.count(doc, version)


@pytest.fixture
def env(real_pdf_dir: Path) -> Env:
    return Env(real_pdf_dir)


# ---------- happy path and idempotency ----------


async def test_first_ingest_creates_version_1(env: Env) -> None:
    report = await env.service.ingest()
    assert report.ok
    (f,) = report.files
    assert (f.outcome, f.chunks, f.index_version, f.doc_id) == (FileOutcome.INGESTED, 22, 1, DOC)
    assert f.warnings == []
    assert await env.points(1) == 22
    record = env.registry.get(DOC)
    assert record is not None
    assert record.active_index_version == 1
    assert record.chunks == 22
    assert len(env.artifacts.read(DOC, 1)) == 22
    assert report.started_at <= report.finished_at


async def test_second_ingest_is_a_noop(env: Env) -> None:
    await env.service.ingest()
    calls = len(env.embedder.calls)
    report = await env.service.ingest()
    (f,) = report.files
    assert (f.outcome, f.chunks, f.index_version) == (FileOutcome.UNCHANGED, 22, 1)
    assert len(env.embedder.calls) == calls, "unchanged files must not be re-embedded"


async def test_force_creates_new_version_and_removes_old(env: Env) -> None:
    await env.service.ingest()
    report = await env.service.ingest(force=True)
    assert report.files[0].outcome == FileOutcome.INGESTED
    assert report.files[0].index_version == 2
    assert await env.points(1) == 0
    assert await env.points(2) == 22
    assert env.artifacts.read(DOC, 1) == []
    assert len(env.artifacts.read(DOC, 2)) == 22
    assert env.registry.get(DOC).active_index_version == 2  # type: ignore[union-attr]


async def test_changed_file_is_reingested(env: Env) -> None:
    await env.service.ingest()
    (env.pdf_dir / REAL_PDF.name).write_bytes(make_pdf(KB_LIKE_PAGES))
    report = await env.service.ingest()
    f = report.files[0]
    assert f.outcome == FileOutcome.INGESTED
    assert f.index_version == 2
    assert env.registry.get(DOC).content_hash != "h"  # type: ignore[union-attr]
    assert await env.points(1) == 0


async def test_self_heals_when_vector_store_lost_points(env: Env) -> None:
    await env.service.ingest()
    env.store.points.clear()  # e.g. the Qdrant volume was wiped
    assert await env.service.has_drift()
    report = await env.service.ingest()
    assert report.files[0].outcome == FileOutcome.INGESTED
    assert report.files[0].index_version == 2
    assert await env.points(2) == 22
    assert not await env.service.has_drift()


async def test_single_file_ingest(env: Env) -> None:
    report = await env.service.ingest(REAL_PDF.name)
    assert [f.file for f in report.files] == [REAL_PDF.name]


async def test_single_file_errors(env: Env) -> None:
    with pytest.raises(NotFound):
        await env.service.ingest("missing.pdf")


async def test_empty_directory_ingest_is_ok_and_empty(pdf_dir: Path) -> None:
    report = await Env(pdf_dir).service.ingest()
    assert report.files == []
    assert report.ok


# ---------- multi-file, partial failure ----------


async def test_bad_files_do_not_stop_good_ones(env: Env) -> None:
    (env.pdf_dir / "a_corrupt.pdf").write_bytes(b"not a pdf")
    (env.pdf_dir / "b_empty.pdf").write_bytes(b"")
    (env.pdf_dir / "c_scanned.pdf").write_bytes(make_pdf([[], [], []]))
    (env.pdf_dir / "d_small.pdf").write_bytes(make_pdf(KB_LIKE_PAGES))
    report = await env.service.ingest()
    outcomes = {f.file: f.outcome for f in report.files}
    assert outcomes == {
        "a_corrupt.pdf": FileOutcome.FAILED,
        "b_empty.pdf": FileOutcome.FAILED,
        "c_scanned.pdf": FileOutcome.SKIPPED_SCANNED,
        "d_small.pdf": FileOutcome.INGESTED,
        REAL_PDF.name: FileOutcome.INGESTED,
    }
    assert not report.ok
    by_file = {f.file: f for f in report.files}
    assert "not a PDF" in (by_file["a_corrupt.pdf"].error or "")
    assert "0 bytes" in (by_file["b_empty.pdf"].error or "")
    assert any("scanned" in w for w in by_file["c_scanned.pdf"].warnings)
    assert env.registry.get("a_corrupt") is None
    assert env.registry.get("c_scanned") is None


async def test_scanned_only_is_not_a_failure(pdf_dir: Path) -> None:
    (pdf_dir / "scan.pdf").write_bytes(make_pdf([[]]))
    report = await Env(pdf_dir).service.ingest()
    assert report.ok
    assert report.files[0].outcome == FileOutcome.SKIPPED_SCANNED


async def test_encrypted_pdf_fails_cleanly(pdf_dir: Path) -> None:
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(make_pdf(KB_LIKE_PAGES)))
    writer.encrypt(user_password="pw", owner_password="o", algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    (pdf_dir / "locked.pdf").write_bytes(buf.getvalue())
    report = await Env(pdf_dir).service.ingest()
    assert report.files[0].outcome == FileOutcome.FAILED


async def test_oversized_pdf_fails_cleanly(pdf_dir: Path) -> None:
    from app.infrastructure.source import LocalDocumentSource

    (pdf_dir / "big.pdf").write_bytes(make_pdf(KB_LIKE_PAGES) + b"\n%" + b"0" * 5000)
    e = Env(pdf_dir)
    e.service.source = LocalDocumentSource(pdf_dir, max_bytes=1024)
    report = await e.service.ingest()
    assert report.files[0].outcome == FileOutcome.FAILED
    assert "limit" in (report.files[0].error or "")


async def test_text_without_structure_still_ingests_with_warning(pdf_dir: Path) -> None:
    (pdf_dir / "notes.pdf").write_bytes(make_pdf([["Just a paragraph of plain notes without any headings."]]))
    report = await Env(pdf_dir).service.ingest()
    f = report.files[0]
    assert f.outcome == FileOutcome.INGESTED
    assert any("fell back" in w for w in f.warnings)


# ---------- embedding failures and retries ----------


async def test_transient_embedding_errors_are_retried(real_pdf_dir: Path) -> None:
    emb = ScriptedEmbedder(failures=[TransientProviderError("429"), UpstreamTimeout("slow")])
    e = Env(real_pdf_dir, embedder=emb)
    report = await e.service.ingest()
    assert report.files[0].outcome == FileOutcome.INGESTED
    assert len(emb.calls) == 3


async def test_retries_exhausted_fails_file_with_partial_failure_report(real_pdf_dir: Path) -> None:
    emb = ScriptedEmbedder(failures=[TransientProviderError("busy")] * 3)
    e = Env(real_pdf_dir, embedder=emb)
    report = await e.service.ingest()
    f = report.files[0]
    assert f.outcome == FileOutcome.FAILED
    assert "after 3 attempt(s) at batch 1" in (f.error or "")
    assert e.registry.get(DOC) is None
    assert e.store.points == {}


async def test_non_transient_errors_are_not_retried(real_pdf_dir: Path) -> None:
    policy = IngestionPolicy(
        retry=RetryPolicy(attempts=5, base_delay_s=0, max_delay_s=0, retry_on=(TransientProviderError, UpstreamTimeout))
    )
    emb = ScriptedEmbedder(failures=[ProviderError("bad key")])
    e = Env(real_pdf_dir, embedder=emb, policy=policy)
    report = await e.service.ingest()
    assert report.files[0].outcome == FileOutcome.FAILED
    assert len(emb.calls) == 1


async def test_failed_reingest_keeps_previous_version_serving(env: Env) -> None:
    await env.service.ingest()
    env.embedder.failures = [ProviderError("down")] * 5
    report = await env.service.ingest(force=True)
    f = report.files[0]
    assert f.outcome == FileOutcome.FAILED
    assert "previous index version 1 is still being served" in (f.error or "")
    assert f.index_version == 1
    assert await env.points(1) == 22
    assert await env.points(2) == 0
    assert env.registry.get(DOC).active_index_version == 1  # type: ignore[union-attr]


async def test_batching(real_pdf_dir: Path) -> None:
    emb = ScriptedEmbedder()
    policy = IngestionPolicy(embed_batch_size=5, retry=RetryPolicy(attempts=1))
    e = Env(real_pdf_dir, embedder=emb, policy=policy)
    await e.service.ingest()
    assert emb.calls == [5, 5, 5, 5, 2]


async def test_failure_in_later_batch_reports_batch_number(real_pdf_dir: Path) -> None:
    class FailThird(ScriptedEmbedder):
        async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
            if len(self.calls) == 2:
                self.calls.append(len(texts))
                raise ProviderError("boom")
            return await super().embed_documents(texts)

    policy = IngestionPolicy(embed_batch_size=5, retry=RetryPolicy(attempts=1))
    e = Env(real_pdf_dir, embedder=FailThird(), policy=policy)
    report = await e.service.ingest()
    assert "at batch 3" in (report.files[0].error or "")
    assert e.store.points == {}


async def test_wrong_vector_dimension_fails_and_rolls_back(real_pdf_dir: Path) -> None:
    class WrongDim(ScriptedEmbedder):
        async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
            return [[1.0] * (self.dim + 1) for _ in texts]

    e = Env(real_pdf_dir, embedder=WrongDim())
    report = await e.service.ingest()
    assert report.files[0].outcome == FileOutcome.FAILED
    assert e.store.points == {}


# ---------- vector store failures ----------


class FlakyStore(InMemoryVectorStore):
    def __init__(self, *, drop: int = 0, fail_upsert: bool = False, fail_cleanup: bool = False) -> None:
        super().__init__()
        self.drop = drop
        self.fail_upsert = fail_upsert
        self.fail_cleanup = fail_cleanup
        self.deleted: list[tuple[str, int]] = []

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if self.fail_upsert:
            raise ConnectionError("qdrant down")
        await super().upsert(chunks[self.drop :], vectors[self.drop :])

    async def delete_version(self, doc_id: str, index_version: int) -> None:
        self.deleted.append((doc_id, index_version))
        if self.fail_cleanup and index_version == 1 and len(self.deleted) > 2:
            raise ConnectionError("cleanup failed")
        await super().delete_version(doc_id, index_version)


async def test_partial_write_is_detected_and_rolled_back(real_pdf_dir: Path) -> None:
    store = FlakyStore(drop=3)
    e = Env(real_pdf_dir, store=store)
    report = await e.service.ingest()
    f = report.files[0]
    assert f.outcome == FileOutcome.FAILED
    assert "holds 19 of 22" in (f.error or "")
    assert store.points == {}
    assert e.registry.get(DOC) is None
    assert e.artifacts.files == {}


async def test_store_outage_fails_file_without_crash(real_pdf_dir: Path) -> None:
    e = Env(real_pdf_dir, store=FlakyStore(fail_upsert=True))
    report = await e.service.ingest()
    assert report.files[0].outcome == FileOutcome.FAILED
    assert "Unexpected error (ConnectionError)" in (report.files[0].error or "")


async def test_old_version_cleanup_failure_does_not_fail_ingest(real_pdf_dir: Path) -> None:
    store = FlakyStore(fail_cleanup=True)
    e = Env(real_pdf_dir, store=store)
    await e.service.ingest()
    report = await e.service.ingest(force=True)
    assert report.files[0].outcome == FileOutcome.INGESTED
    assert e.registry.get(DOC).active_index_version == 2  # type: ignore[union-attr]


async def test_leftovers_from_crashed_attempt_are_cleared_before_writing(env: Env) -> None:
    await env.service.ingest()
    # Simulate a crash that left half of version 2 behind.
    stale = env.artifacts.read(DOC, 1)[0].model_copy(update={"id": "stale-v2"})
    meta = stale.metadata.model_copy(update={"index_version": 2})
    await env.store.upsert([stale.model_copy(update={"metadata": meta})], [[0.0] * 31 + [1.0]])
    report = await env.service.ingest(force=True)
    assert report.files[0].outcome == FileOutcome.INGESTED
    assert await env.points(2) == 22
    assert "stale-v2" not in env.store.points


async def test_drift_check_tolerates_store_errors(env: Env) -> None:
    await env.service.ingest()

    async def broken(*_: object) -> int:
        raise ConnectionError

    env.store.count = broken  # type: ignore[method-assign]
    assert await env.service.has_drift() is False


# ---------- delete and prune ----------


async def test_delete_document(env: Env) -> None:
    await env.service.ingest()
    await env.service.delete_document(DOC)
    assert env.registry.get(DOC) is None
    assert env.store.points == {}
    assert env.artifacts.files == {}
    with pytest.raises(NotFound):
        await env.service.delete_document(DOC)


async def test_prune_removes_documents_whose_pdf_was_deleted(env: Env) -> None:
    (env.pdf_dir / "extra.pdf").write_bytes(make_pdf(KB_LIKE_PAGES))
    await env.service.ingest()
    (env.pdf_dir / "extra.pdf").unlink()
    report = await env.service.ingest()
    assert report.pruned == ["extra"]
    assert env.registry.get("extra") is None
    assert await env.points(1, "extra") == 0
    assert env.registry.get(DOC) is not None


async def test_prune_disabled(real_pdf_dir: Path) -> None:
    e = Env(real_pdf_dir, policy=IngestionPolicy(prune_deleted=False))
    (real_pdf_dir / "extra.pdf").write_bytes(make_pdf(KB_LIKE_PAGES))
    await e.service.ingest()
    (real_pdf_dir / "extra.pdf").unlink()
    report = await e.service.ingest()
    assert report.pruned == []
    assert e.registry.get("extra") is not None


async def test_prune_refuses_to_empty_index_when_source_vanishes(env: Env) -> None:
    await env.service.ingest()
    shutil.rmtree(env.pdf_dir)  # e.g. the volume was not mounted
    report = await env.service.ingest()
    assert report.pruned == []
    assert env.registry.get(DOC) is not None
    assert await env.points(1) == 22


async def test_single_file_ingest_never_prunes(env: Env) -> None:
    (env.pdf_dir / "extra.pdf").write_bytes(make_pdf(KB_LIKE_PAGES))
    await env.service.ingest()
    (env.pdf_dir / "extra.pdf").unlink()
    report = await env.service.ingest(REAL_PDF.name)
    assert report.pruned == []
    assert env.registry.get("extra") is not None


# ---------- concurrency ----------


async def test_concurrent_ingests_are_serialised(env: Env) -> None:
    first, second = await asyncio.gather(env.service.ingest(), env.service.ingest())
    outcomes = sorted([first.files[0].outcome, second.files[0].outcome])
    assert outcomes == [FileOutcome.INGESTED, FileOutcome.UNCHANGED]
    assert await env.points(1) == 22


async def test_status_lists_records(env: Env) -> None:
    assert env.service.status() == []
    await env.service.ingest()
    assert [r.doc_id for r in env.service.status()] == [DOC]


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="embed_batch_size"):
        IngestionPolicy(embed_batch_size=0)
