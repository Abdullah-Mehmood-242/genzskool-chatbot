"""Auto-ingest (ADR-0011): ingest on startup, watch PDF_DIR, self-heal index drift.

- Startup: one ``ingest()`` of every PDF (a no-op when nothing changed, thanks to content hashes).
- Watcher: every ``interval_s`` it compares cheap file fingerprints (name, size, mtime). A new,
  changed or deleted PDF triggers an ingest (deleted ones are pruned when enabled).
- Self-heal: if the vector store lost points (e.g. a wiped Qdrant volume) it re-ingests.
- Failures are retried on every tick until a run succeeds, backing off exponentially
  (interval * 2^n, capped), and never stop the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from app.application.ingest import IngestionService
from app.domain.models import SourceFingerprint
from app.domain.ports import Clock, DocumentSource
from common.contracts import AutoIngestStatus, AutoIngestTrigger

logger = logging.getLogger(__name__)


class AutoIngestor:
    def __init__(
        self,
        service: IngestionService,
        source: DocumentSource,
        clock: Clock,
        *,
        on_startup: bool = True,
        interval_s: float = 30.0,
        max_backoff_factor: int = 8,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if interval_s < 0:
            raise ValueError("interval_s must be >= 0")
        self.service = service
        self.source = source
        self.clock = clock
        self.on_startup = on_startup
        self.interval_s = interval_s
        self.max_backoff_factor = max_backoff_factor
        self._sleep = sleep
        self._fingerprint: list[SourceFingerprint] | None = None
        self._retry_pending = False
        self._task: asyncio.Task[None] | None = None
        self.startup_done = asyncio.Event()
        self.status = AutoIngestStatus(
            on_startup=on_startup, watch_interval_s=interval_s, prune_deleted=service.policy.prune_deleted
        )

    # ---------------------------------------------------------------- lifecycle

    @property
    def enabled(self) -> bool:
        return self.on_startup or self.interval_s > 0

    def start(self) -> None:
        if self._task is None and self.enabled:
            self._task = asyncio.create_task(self._loop(), name="auto-ingest")
        elif not self.enabled:
            self.startup_done.set()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ---------------------------------------------------------------- behaviour

    def next_delay(self) -> float:
        # int ** int is typed Any (negative exponents give floats); a shift is exact and typed int.
        factor = min(1 << min(self.status.consecutive_failures, 30), self.max_backoff_factor)
        return self.interval_s * factor

    async def run_startup(self) -> None:
        try:
            if self.on_startup:
                await self._run(AutoIngestTrigger.STARTUP)
            else:
                self._fingerprint = self._safe_fingerprints()  # baseline: only later changes trigger
        finally:
            self.startup_done.set()

    async def tick(self) -> bool:
        """One watcher check. Returns True if an ingest ran."""
        current = self._safe_fingerprints()
        if current != self._fingerprint:
            await self._run(AutoIngestTrigger.FILE_CHANGE, fingerprint=current)
            return True
        if self._retry_pending:
            # The last run failed. The folder may now equal an older, good state (e.g. the bad file was
            # removed), so a fingerprint comparison alone would never re-run and status would stay failed.
            await self._run(AutoIngestTrigger.RETRY, fingerprint=current)
            return True
        if await self.service.has_drift():
            await self._run(AutoIngestTrigger.INDEX_DRIFT, fingerprint=current)
            return True
        return False

    async def _loop(self) -> None:
        await self.run_startup()
        if self.interval_s <= 0:
            return
        while True:
            await self._sleep(self.next_delay())
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - _run already records failures; belt and braces
                logger.exception("auto_ingest_tick_crashed")

    def _safe_fingerprints(self) -> list[SourceFingerprint]:
        try:
            return self.source.fingerprints()
        except OSError:
            logger.warning("auto_ingest_fingerprint_failed")
            return []

    async def _run(self, trigger: AutoIngestTrigger, fingerprint: list[SourceFingerprint] | None = None) -> None:
        fingerprint = fingerprint if fingerprint is not None else self._safe_fingerprints()
        error: str | None = None
        ok = False
        try:
            report = await self.service.ingest()
            ok = report.ok
            if not ok:
                failed = [f"{f.file}: {f.error}" for f in report.files if f.error]
                error = "; ".join(failed)[:500] or "ingest reported failures"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            logger.exception("auto_ingest_failed", extra={"trigger": trigger.value})
        self._fingerprint = fingerprint
        self._retry_pending = not ok  # failed runs are retried on the next (backed-off) tick
        self.status = self.status.model_copy(
            update={
                "runs": self.status.runs + 1,
                "last_run_at": self.clock.now(),
                "last_trigger": trigger,
                "last_ok": ok,
                "last_error": error,
                "consecutive_failures": 0 if ok else self.status.consecutive_failures + 1,
            }
        )
        logger.info("auto_ingest_run", extra={"trigger": trigger.value, "ok": ok})
