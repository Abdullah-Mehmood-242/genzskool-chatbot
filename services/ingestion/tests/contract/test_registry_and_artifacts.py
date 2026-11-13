"""Registry and ChunkArtifactStore port contracts, plus file-adapter durability guarantees."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.domain.models import IndexRecord
from app.domain.ports import ChunkArtifactStore, IngestionRegistry
from app.infrastructure.adapters import InMemoryChunkArtifactStore, InMemoryRegistry
from app.infrastructure.files import REGISTRY_FILE, JsonFileRegistry, JsonlChunkArtifactStore, atomic_write
from common.contracts import Chunk, ChunkMetadata, DocType

NOW = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)


def record(doc_id: str = "kb", version: int = 1, chunks: int = 22) -> IndexRecord:
    return IndexRecord(
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        content_hash="h" * 64,
        active_index_version=version,
        chunks=chunks,
        ingested_at=NOW,
    )


def chunk(i: int, version: int = 1) -> Chunk:
    meta = ChunkMetadata(
        doc_id="kb",
        source_file="kb.pdf",
        page=1,
        section=f"s{i}",
        doc_type=DocType.POLICY,
        content_hash="h",
        ingested_at=NOW,
        index_version=version,
    )
    return Chunk(id=f"id-{i}-{version}", text=f"text {i} شاپیفائی", metadata=meta)


RegistryFactory = Callable[[Path], IngestionRegistry]
REGISTRIES: dict[str, RegistryFactory] = {"memory": lambda _: InMemoryRegistry(), "json": JsonFileRegistry}
ArtifactFactory = Callable[[Path], ChunkArtifactStore]
ARTIFACTS: dict[str, ArtifactFactory] = {
    "memory": lambda _: InMemoryChunkArtifactStore(),
    "jsonl": JsonlChunkArtifactStore,
}


@pytest.fixture(params=sorted(REGISTRIES))
def registry(request: pytest.FixtureRequest, tmp_path: Path) -> IngestionRegistry:
    return REGISTRIES[request.param](tmp_path)


@pytest.fixture(params=sorted(ARTIFACTS))
def artifacts(request: pytest.FixtureRequest, tmp_path: Path) -> ChunkArtifactStore:
    return ARTIFACTS[request.param](tmp_path)


# ---------- registry contract ----------


def test_registry_empty(registry: IngestionRegistry) -> None:
    assert registry.get("kb") is None
    assert registry.list() == []


def test_registry_put_get_overwrite(registry: IngestionRegistry) -> None:
    registry.put(record(version=1))
    registry.put(record(version=2, chunks=23))
    assert registry.get("kb") == record(version=2, chunks=23)


def test_registry_list_sorted(registry: IngestionRegistry) -> None:
    for d in ("zeta", "alpha", "mid"):
        registry.put(record(d))
    assert [r.doc_id for r in registry.list()] == ["alpha", "mid", "zeta"]


def test_registry_delete_and_delete_missing(registry: IngestionRegistry) -> None:
    registry.put(record("a"))
    registry.put(record("b"))
    registry.delete("a")
    registry.delete("never-existed")
    assert [r.doc_id for r in registry.list()] == ["b"]


def test_registry_preserves_timezone(registry: IngestionRegistry) -> None:
    registry.put(record())
    got = registry.get("kb")
    assert got is not None
    assert got.ingested_at == NOW
    assert got.ingested_at.tzinfo is not None


# ---------- JSON registry specifics ----------


def test_json_registry_persists_across_instances(tmp_path: Path) -> None:
    JsonFileRegistry(tmp_path).put(record())
    assert JsonFileRegistry(tmp_path).get("kb") == record()


def test_json_registry_file_is_human_readable(tmp_path: Path) -> None:
    JsonFileRegistry(tmp_path).put(record())
    data = json.loads((tmp_path / REGISTRY_FILE).read_text(encoding="utf-8"))
    assert data["schema"] == 1
    assert data["documents"]["kb"]["active_index_version"] == 1
    assert data["documents"]["kb"]["ingested_at"] == "2026-11-12T10:00:00+00:00"


@pytest.mark.parametrize(
    "content", ["{not json", '{"documents": []}', '{"documents": {"kb": {"chunks": "x"}}}', "[]", '{"other": 1}']
)
def test_json_registry_self_heals_from_corruption(tmp_path: Path, content: str) -> None:
    (tmp_path / REGISTRY_FILE).write_text(content, encoding="utf-8")
    reg = JsonFileRegistry(tmp_path)
    assert reg.list() == []  # treated as empty -> everything gets re-ingested
    backups = list(tmp_path.glob(f"{REGISTRY_FILE}.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == content  # kept for inspection
    reg.put(record())
    assert reg.get("kb") == record()


def test_json_registry_creates_missing_directory(tmp_path: Path) -> None:
    reg = JsonFileRegistry(tmp_path / "nested" / "dir")
    reg.put(record())
    assert reg.get("kb") == record()


def test_atomic_write_leaves_no_temp_files_and_replaces(tmp_path: Path) -> None:
    tmp_path = tmp_path / "isolated"
    tmp_path.mkdir()
    target = tmp_path / "f.json"
    atomic_write(target, "one")
    atomic_write(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_atomic_write_failure_keeps_old_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = tmp_path / "isolated"
    tmp_path.mkdir()
    import os

    target = tmp_path / "f.json"
    atomic_write(target, "original")

    def boom(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write(target, "new")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


# ---------- artifacts contract ----------


def test_artifacts_round_trip(artifacts: ChunkArtifactStore) -> None:
    chunks = [chunk(0), chunk(1)]
    artifacts.write("kb", 1, chunks)
    assert artifacts.read("kb", 1) == chunks


def test_artifacts_missing_reads_empty(artifacts: ChunkArtifactStore) -> None:
    assert artifacts.read("kb", 9) == []


def test_artifacts_versions_are_independent(artifacts: ChunkArtifactStore) -> None:
    artifacts.write("kb", 1, [chunk(0, 1)])
    artifacts.write("kb", 2, [chunk(0, 2), chunk(1, 2)])
    artifacts.delete("kb", 1)
    assert artifacts.read("kb", 1) == []
    assert len(artifacts.read("kb", 2)) == 2


def test_artifacts_delete_all_versions(artifacts: ChunkArtifactStore) -> None:
    artifacts.write("kb", 1, [chunk(0)])
    artifacts.write("kb", 2, [chunk(0, 2)])
    artifacts.delete("kb")
    assert artifacts.read("kb", 1) == []
    assert artifacts.read("kb", 2) == []


def test_artifacts_delete_missing_is_noop(artifacts: ChunkArtifactStore) -> None:
    artifacts.delete("kb", 3)
    artifacts.delete("kb")


def test_artifacts_overwrite(artifacts: ChunkArtifactStore) -> None:
    artifacts.write("kb", 1, [chunk(0)])
    artifacts.write("kb", 1, [chunk(5)])
    assert [c.id for c in artifacts.read("kb", 1)] == ["id-5-1"]


# ---------- JSONL specifics ----------


def test_jsonl_file_layout_is_inspectable(tmp_path: Path) -> None:
    store = JsonlChunkArtifactStore(tmp_path)
    store.write("kb", 3, [chunk(0, 3), chunk(1, 3)])
    path = tmp_path / "kb.v3.chunks.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(Chunk.model_validate_json(line) for line in lines)
    assert "شاپیفائی" in lines[0]  # Unicode stays readable


def test_jsonl_delete_all_does_not_touch_other_docs(tmp_path: Path) -> None:
    tmp_path = tmp_path / "isolated"
    tmp_path.mkdir()
    store = JsonlChunkArtifactStore(tmp_path)
    store.write("kb", 1, [chunk(0)])
    store.write("kb_extra", 1, [chunk(0)])
    store.delete("kb")
    assert [p.name for p in tmp_path.iterdir()] == ["kb_extra.v1.chunks.jsonl"]


@pytest.mark.parametrize("doc_id", ["../escape", "a/b", "A-B", "", "x" * 101, "a.b"])
def test_jsonl_rejects_unsafe_doc_ids(tmp_path: Path, doc_id: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        JsonlChunkArtifactStore(tmp_path).write(doc_id, 1, [chunk(0)])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_artifacts_are_world_readable(tmp_path: Path) -> None:
    """Regression (Linux CI): mkstemp's 0600 made data/processed unreadable outside the container."""
    JsonFileRegistry(tmp_path).put(record())
    JsonlChunkArtifactStore(tmp_path).write("kb", 1, [chunk(0)])
    for name in (REGISTRY_FILE, "kb.v1.chunks.jsonl"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o644, name
