"""File-backed registry (registry.json) and chunk artifacts (<doc_id>.v<N>.chunks.jsonl) in PROCESSED_DIR.

Both are written atomically (temp file + os.replace) so a crash never leaves a half-written file.
Retrieval reads the same two artifacts: the registry tells it which index version is active.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from app.domain.models import IndexRecord
from app.domain.ports import ChunkArtifactStore, IngestionRegistry
from common.contracts import Chunk

logger = logging.getLogger(__name__)

REGISTRY_FILE = "registry.json"
REGISTRY_SCHEMA = 1
_SAFE_ID = re.compile(r"^[a-z0-9_]{1,100}$")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _safe(doc_id: str) -> str:
    if not _SAFE_ID.match(doc_id):
        raise ValueError(f"unsafe doc_id for a file name: {doc_id!r}")
    return doc_id


class JsonFileRegistry(IngestionRegistry):
    def __init__(self, processed_dir: Path) -> None:
        self.path = processed_dir / REGISTRY_FILE
        self._lock = threading.Lock()

    def _load(self) -> dict[str, IndexRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            docs = raw["documents"]
            return {
                doc_id: IndexRecord(
                    doc_id=doc_id,
                    source_file=str(d["source_file"]),
                    content_hash=str(d["content_hash"]),
                    active_index_version=int(d["active_index_version"]),
                    chunks=int(d["chunks"]),
                    ingested_at=datetime.fromisoformat(d["ingested_at"]),
                )
                for doc_id, d in docs.items()
            }
        except (ValueError, KeyError, TypeError, AttributeError):
            # Self-heal: keep the bad file for inspection and start empty -> everything is re-ingested.
            backup = self.path.with_name(f"{REGISTRY_FILE}.corrupt-{os.getpid()}")
            os.replace(self.path, backup)
            logger.warning("registry_corrupt_reset", extra={"backup": backup.name})
            return {}

    def _save(self, records: dict[str, IndexRecord]) -> None:
        body = {
            "schema": REGISTRY_SCHEMA,
            "documents": {
                r.doc_id: {
                    "source_file": r.source_file,
                    "content_hash": r.content_hash,
                    "active_index_version": r.active_index_version,
                    "chunks": r.chunks,
                    "ingested_at": r.ingested_at.isoformat(),
                }
                for r in sorted(records.values(), key=lambda r: r.doc_id)
            },
        }
        atomic_write(self.path, json.dumps(body, indent=2, ensure_ascii=False) + "\n")

    def get(self, doc_id: str) -> IndexRecord | None:
        with self._lock:
            return self._load().get(doc_id)

    def put(self, record: IndexRecord) -> None:
        with self._lock:
            records = self._load()
            records[record.doc_id] = record
            self._save(records)

    def delete(self, doc_id: str) -> None:
        with self._lock:
            records = self._load()
            if records.pop(doc_id, None) is not None:
                self._save(records)

    def list(self) -> list[IndexRecord]:
        with self._lock:
            return sorted(self._load().values(), key=lambda r: r.doc_id)


class JsonlChunkArtifactStore(ChunkArtifactStore):
    def __init__(self, processed_dir: Path) -> None:
        self.dir = processed_dir

    def path(self, doc_id: str, index_version: int) -> Path:
        return self.dir / f"{_safe(doc_id)}.v{int(index_version)}.chunks.jsonl"

    def write(self, doc_id: str, index_version: int, chunks: Sequence[Chunk]) -> None:
        body = "".join(c.model_dump_json() + "\n" for c in chunks)
        atomic_write(self.path(doc_id, index_version), body)

    def read(self, doc_id: str, index_version: int) -> list[Chunk]:
        path = self.path(doc_id, index_version)
        if not path.exists():
            return []
        return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def delete(self, doc_id: str, index_version: int | None = None) -> None:
        if index_version is not None:
            self.path(doc_id, index_version).unlink(missing_ok=True)
            return
        for path in self.dir.glob(f"{_safe(doc_id)}.v*.chunks.jsonl"):
            path.unlink(missing_ok=True)
