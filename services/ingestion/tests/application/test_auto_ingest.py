"""AutoIngestor: startup run, folder watcher, drift self-heal, backoff, lifecycle."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import KB_LIKE_PAGES, REAL_PDF, ScriptedEmbedder, StepClock, build_service, make_pdf

from app.application.auto_ingest import AutoIngestor
from app.application.ingest import IngestionPolicy, IngestionService
from app.infrastructure.vector_store import InMemoryVectorStore
from common.contracts import AutoIngestTrigger, FileOutcome
from common.errors import ProviderError
from common.resilience import RetryPolicy

DOC = "genzskool_knowledge_base"


def _auto(service: IngestionService, **kw: object) -> AutoIngestor:
    return AutoIngestor(service, service.source, StepClock(), **kw)  # type: ignore[arg-type]


def _touch(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    st = path.stat()
    os.utime(path, ns=(st.st_mtime_ns + 5 * 10**9, st.st_mtime_ns + 5 * 10**9))


async def test_startup_ingests_everything(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service)
    await auto.run_startup()
    assert auto.startup_done.is_set()
    assert service.registry.get(DOC) is not None
    s = auto.status
    assert (s.runs, s.last_trigger, s.last_ok, s.last_error, s.consecutive_failures) == (
        1,
        AutoIngestTrigger.STARTUP,
        True,
        None,
        0,
    )
    assert s.last_run_at is not None


async def test_startup_disabled_only_snapshots(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service, on_startup=False, interval_s=10)
    await auto.run_startup()
    assert service.registry.get(DOC) is None
    assert auto.status.runs == 0
    assert await auto.tick() is False  # unchanged since the baseline -> nothing to do


async def test_tick_noop_when_nothing_changed(real_pdf_dir: Path) -> None:
    emb = ScriptedEmbedder()
    service = build_service(real_pdf_dir, embedder=emb)
    auto = _auto(service)
    await auto.run_startup()
    calls = len(emb.calls)
    for _ in range(3):
        assert await auto.tick() is False
    assert len(emb.calls) == calls
    assert auto.status.runs == 1


async def test_new_file_is_picked_up(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service)
    await auto.run_startup()
    _touch(real_pdf_dir / "new_course.pdf", make_pdf(KB_LIKE_PAGES))
    assert await auto.tick() is True
    assert service.registry.get("new_course") is not None
    assert auto.status.last_trigger == AutoIngestTrigger.FILE_CHANGE


async def test_modified_file_is_reingested(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service)
    await auto.run_startup()
    _touch(real_pdf_dir / REAL_PDF.name, make_pdf(KB_LIKE_PAGES))
    assert await auto.tick() is True
    assert service.registry.get(DOC).active_index_version == 2  # type: ignore[union-attr]


async def test_touched_but_identical_file_is_cheap(real_pdf_dir: Path) -> None:
    emb = ScriptedEmbedder()
    service = build_service(real_pdf_dir, embedder=emb)
    auto = _auto(service)
    await auto.run_startup()
    calls = len(emb.calls)
    _touch(real_pdf_dir / REAL_PDF.name, REAL_PDF.read_bytes())  # mtime changes, content does not
    assert await auto.tick() is True
    assert len(emb.calls) == calls  # content hash says unchanged: no re-embedding
    assert service.registry.get(DOC).active_index_version == 1  # type: ignore[union-attr]


async def test_deleted_file_is_pruned(real_pdf_dir: Path) -> None:
    _touch(real_pdf_dir / "extra.pdf", make_pdf(KB_LIKE_PAGES))
    service = build_service(real_pdf_dir)
    auto = _auto(service)
    await auto.run_startup()
    assert service.registry.get("extra") is not None
    (real_pdf_dir / "extra.pdf").unlink()
    assert await auto.tick() is True
    assert service.registry.get("extra") is None
    assert service.registry.get(DOC) is not None


async def test_index_drift_triggers_self_heal(real_pdf_dir: Path) -> None:
    store = InMemoryVectorStore()
    service = build_service(real_pdf_dir, store=store)
    auto = _auto(service)
    await auto.run_startup()
    store.points.clear()
    assert await auto.tick() is True
    assert auto.status.last_trigger == AutoIngestTrigger.INDEX_DRIFT
    assert await store.count(DOC, 2) == 22


async def test_failures_are_recorded_retried_and_backed_off(real_pdf_dir: Path) -> None:
    emb = ScriptedEmbedder(failures=[ProviderError("down")] * 2)
    service = build_service(real_pdf_dir, embedder=emb, policy=IngestionPolicy(retry=RetryPolicy(attempts=1)))
    auto = _auto(service, interval_s=10, max_backoff_factor=8)
    await auto.run_startup()
    assert auto.status.last_ok is False
    assert "down" in (auto.status.last_error or "")
    assert auto.status.consecutive_failures == 1
    assert auto.next_delay() == 20
    # A failed run leaves a retry pending -> the next tick retries even though no file changed.
    assert await auto.tick() is True
    assert auto.status.last_trigger == AutoIngestTrigger.RETRY
    assert auto.status.consecutive_failures == 2
    assert auto.next_delay() == 40
    assert await auto.tick() is True
    assert auto.status.last_ok is True
    assert auto.status.consecutive_failures == 0
    assert auto.next_delay() == 10
    assert service.registry.get(DOC) is not None


def test_backoff_is_capped(real_pdf_dir: Path) -> None:
    auto = _auto(build_service(real_pdf_dir), interval_s=5, max_backoff_factor=8)
    auto.status = auto.status.model_copy(update={"consecutive_failures": 50})
    assert auto.next_delay() == 40


async def test_unexpected_exception_is_recorded_not_raised(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)

    async def explode(*_: object, **__: object) -> None:
        raise RuntimeError("kaput")

    service.ingest = explode  # type: ignore[method-assign]
    auto = _auto(service)
    await auto.run_startup()
    assert auto.status.last_ok is False
    assert "RuntimeError: kaput" in (auto.status.last_error or "")
    assert auto.startup_done.is_set()


async def test_corrupt_file_reported_in_status(real_pdf_dir: Path) -> None:
    (real_pdf_dir / "broken.pdf").write_bytes(b"junk")
    auto = _auto(build_service(real_pdf_dir))
    await auto.run_startup()
    assert auto.status.last_ok is False
    assert "broken.pdf" in (auto.status.last_error or "")


async def test_fingerprint_errors_are_tolerated(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service)
    await auto.run_startup()

    def broken() -> list[object]:
        raise OSError("io error")

    auto.source.fingerprints = broken  # type: ignore[method-assign]
    await auto.tick()  # must not raise


def test_negative_interval_rejected(real_pdf_dir: Path) -> None:
    with pytest.raises(ValueError, match="interval_s"):
        _auto(build_service(real_pdf_dir), interval_s=-1)


# ---------- lifecycle ----------


async def test_loop_runs_startup_then_ticks_until_stopped(real_pdf_dir: Path) -> None:
    sleeps: list[float] = []
    gate = asyncio.Event()

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)
        if len(sleeps) >= 3:
            gate.set()
            await asyncio.Event().wait()  # park until cancelled
        await asyncio.sleep(0)

    service = build_service(real_pdf_dir)
    auto = AutoIngestor(service, service.source, StepClock(), interval_s=7, sleep=fake_sleep)
    auto.start()
    assert auto.running
    await asyncio.wait_for(gate.wait(), timeout=10)
    assert auto.startup_done.is_set()
    assert sleeps == [7, 7, 7]
    await auto.stop()
    assert not auto.running
    await auto.stop()  # idempotent


async def test_watcher_disabled_runs_only_startup(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service, interval_s=0)
    auto.start()
    await asyncio.wait_for(auto.startup_done.wait(), timeout=10)
    await asyncio.sleep(0)
    assert not auto.running  # loop exits after the startup run
    assert service.registry.get(DOC) is not None
    await auto.stop()


async def test_fully_disabled_never_starts_but_is_ready(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service, on_startup=False, interval_s=0)
    assert not auto.enabled
    auto.start()
    assert not auto.running
    assert auto.startup_done.is_set()
    assert service.registry.get(DOC) is None


async def test_start_twice_does_not_spawn_two_loops(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service, interval_s=3600)
    auto.start()
    task = auto._task
    auto.start()
    assert auto._task is task
    await auto.stop()


async def test_status_reflects_policy(real_pdf_dir: Path) -> None:
    auto = _auto(build_service(real_pdf_dir), on_startup=False, interval_s=12.5)
    assert auto.status.watch_interval_s == 12.5
    assert auto.status.on_startup is False
    assert auto.status.prune_deleted is True


async def test_manual_and_auto_ingest_do_not_interleave(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service)
    manual, _ = await asyncio.gather(service.ingest(), auto.run_startup())
    assert service.registry.get(DOC).active_index_version == 1  # type: ignore[union-attr]
    assert manual.files[0].outcome in (FileOutcome.INGESTED, FileOutcome.UNCHANGED)


async def test_recovers_when_failing_file_is_removed_after_a_good_baseline(real_pdf_dir: Path) -> None:
    """Regression (found by e2e): good run -> corrupt file added -> fails -> corrupt file removed.

    The folder then matches the last *successful* fingerprint again; the watcher must still run once
    more, so status recovers instead of reporting the stale failure forever.
    """
    service = build_service(real_pdf_dir)
    auto = _auto(service, interval_s=10)
    await auto.run_startup()
    assert auto.status.last_ok is True
    _touch(real_pdf_dir / "broken.pdf", b"junk")
    assert await auto.tick() is True
    assert auto.status.last_ok is False
    (real_pdf_dir / "broken.pdf").unlink()
    assert await auto.tick() is True, "removing the bad file must trigger a run that clears the failure"
    assert auto.status.last_ok is True
    assert auto.status.last_trigger == AutoIngestTrigger.FILE_CHANGE
    assert auto.status.consecutive_failures == 0
    assert await auto.tick() is False  # and then it settles


async def test_persistent_failure_keeps_retrying_with_backoff(real_pdf_dir: Path) -> None:
    service = build_service(real_pdf_dir)
    auto = _auto(service, interval_s=10, max_backoff_factor=4)
    await auto.run_startup()
    _touch(real_pdf_dir / "broken.pdf", b"junk")
    for expected_failures in (1, 2, 3):
        assert await auto.tick() is True
        assert auto.status.consecutive_failures == expected_failures
    assert auto.next_delay() == 40  # capped at 4x
