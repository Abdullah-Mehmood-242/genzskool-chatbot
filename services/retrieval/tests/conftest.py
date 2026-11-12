"""Shared fixtures for retrieval tests. The corpus is the real KB fixture produced by ingestion."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.domain.models import Corpus
from common.contracts import Chunk, ChunkMetadata, DocType, Level

REPO = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "kb"
CONFIG_DIR = REPO / "config"
EVAL_CASES = REPO / "data" / "eval" / "retrieval_cases.jsonl"
DOC = "genzskool_knowledge_base"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    shutil.copytree(FIXTURE_DIR, processed)
    monkeypatch.setenv("CONFIG_DIR", str(CONFIG_DIR))
    monkeypatch.setenv("PROCESSED_DIR", str(processed))
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    return tmp_path / "processed"


def load_fixture_chunks() -> list[Chunk]:
    path = FIXTURE_DIR / f"{DOC}.v1.chunks.jsonl"
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="session")
def kb_corpus() -> Corpus:
    chunks = load_fixture_chunks()
    return Corpus(chunks=tuple(chunks), versions=((DOC, 1),), signature="fixture")


def section_id(corpus: Corpus, prefix: str) -> str:
    return next(c.id for c in corpus.chunks if c.metadata.section.startswith(prefix))


def make_chunk(
    cid: str,
    text: str,
    *,
    doc: str = "d",
    version: int = 1,
    doc_type: DocType = DocType.COURSE,
    fee: int | None = None,
    slug: str | None = None,
    category: str | None = None,
    level: Level | None = None,
    mentor: str | None = None,
    section: str | None = None,
) -> Chunk:
    meta = ChunkMetadata(
        doc_id=doc,
        source_file=f"{doc}.pdf",
        page=1,
        section=section or f"Section {cid}",
        doc_type=doc_type,
        course_slug=slug,
        category=category,
        level=level,
        fee_pkr=fee,
        mentor=mentor,
        content_hash="h",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        index_version=version,
    )
    return Chunk(id=cid, text=text, metadata=meta)


def write_index(processed: Path, docs: dict[str, tuple[int, list[Chunk]]]) -> None:
    """Write registry.json + chunk files exactly as ingestion does."""
    processed.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {"schema": 1, "documents": {}}
    for doc_id, (version, chunks) in docs.items():
        registry["documents"][doc_id] = {
            "source_file": f"{doc_id}.pdf",
            "content_hash": "h",
            "active_index_version": version,
            "chunks": len(chunks),
            "ingested_at": "2026-01-01T00:00:00+00:00",
        }
        (processed / f"{doc_id}.v{version}.chunks.jsonl").write_text(
            "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8"
        )
    tmp = processed / ".registry.tmp"
    tmp.write_text(json.dumps(registry), encoding="utf-8")
    tmp.replace(processed / "registry.json")  # atomic replace like ingestion (new inode)
