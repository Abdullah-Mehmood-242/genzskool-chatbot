"""Corpus providers. The file provider reads the artifacts ingestion writes in PROCESSED_DIR:

- ``registry.json``: the active ``index_version`` per document (ingestion flips it atomically);
- ``<doc_id>.v<N>.chunks.jsonl``: the chunks of that version.

It re-reads them only when the registry changes (a cheap stat per search), so a new index
version becomes searchable on the next request without any service-to-service call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from app.domain.models import Corpus
from app.domain.ports import CorpusProvider
from common.contracts import Chunk

logger = logging.getLogger(__name__)
_SAFE_ID = re.compile(r"^[a-z0-9_]{1,100}$")


class FileCorpusProvider(CorpusProvider):
    def __init__(self, processed_dir: Path) -> None:
        self.dir = processed_dir
        self.registry = processed_dir / "registry.json"
        self._corpus = Corpus()
        self._stat: tuple[int, int, int] | None = None
        self._loaded = False
        self._lock = asyncio.Lock()

    def _registry_stat(self) -> tuple[int, int, int] | None:
        """(inode, mtime, size). Ingestion replaces the file atomically, so the inode changes on every
        write; that catches v1 -> v2 switches where size and (coarse) mtime may be identical."""
        try:
            st = self.registry.stat()
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    async def current(self) -> Corpus:
        stat = self._registry_stat()
        if self._loaded and stat == self._stat:
            return self._corpus
        async with self._lock:
            stat = self._registry_stat()
            if not self._loaded or stat != self._stat:
                self._corpus = await asyncio.to_thread(self._load)
                self._stat = stat
                self._loaded = True
                logger.info(
                    "corpus_loaded",
                    extra={
                        "chunks": len(self._corpus),
                        "versions": dict(self._corpus.versions),
                        "warnings": list(self._corpus.warnings),
                    },
                )
        return self._corpus

    def _load(self) -> Corpus:
        if not self.registry.exists():
            return Corpus(warnings=("registry.json not found: nothing has been ingested yet",))
        try:
            raw = self.registry.read_bytes()
            docs = json.loads(raw)["documents"]
            active = {str(doc_id): int(d["active_index_version"]) for doc_id, d in docs.items()}
        except (ValueError, KeyError, TypeError, AttributeError):
            return Corpus(warnings=("registry.json is unreadable; serving an empty index",))
        chunks: list[Chunk] = []
        versions: list[tuple[str, int]] = []
        warnings: list[str] = []
        for doc_id, version in sorted(active.items()):
            if not _SAFE_ID.match(doc_id):
                warnings.append(f"skipped unsafe doc_id {doc_id!r}")
                continue
            path = self.dir / f"{doc_id}.v{version}.chunks.jsonl"
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                doc_chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
            except FileNotFoundError:
                warnings.append(f"{path.name} missing (version switch in progress?)")
                continue
            except (ValidationError, ValueError):
                warnings.append(f"{path.name} is corrupt; document skipped")
                continue
            consistent = [c for c in doc_chunks if c.metadata.doc_id == doc_id and c.metadata.index_version == version]
            if len(consistent) != len(doc_chunks):
                warnings.append(f"{path.name}: {len(doc_chunks) - len(consistent)} chunk(s) with wrong doc/version")
            chunks.extend(consistent)
            versions.append((doc_id, version))
        signature = f"{len(chunks)}:" + ",".join(f"{d}@{v}" for d, v in versions)
        return Corpus(chunks=tuple(chunks), versions=tuple(versions), signature=signature, warnings=tuple(warnings))


class StaticCorpusProvider(CorpusProvider):
    """Fixed corpus (tests, in-process evaluation)."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    async def current(self) -> Corpus:
        return self.corpus
