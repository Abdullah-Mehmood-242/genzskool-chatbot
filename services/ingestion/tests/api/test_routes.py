"""HTTP API through the real composition root (fake embeddings, in-memory vector store, temp dirs)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import KB_LIKE_PAGES, REAL_PDF, make_pdf

from app.main import create_app

DOC = "genzskool_knowledge_base"


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    pdfs, processed = tmp_path / "pdfs", tmp_path / "processed"
    (pdfs / REAL_PDF.name).write_bytes(REAL_PDF.read_bytes())
    return pdfs, processed


@pytest.fixture
async def app(dirs: tuple[Path, Path]) -> AsyncIterator[FastAPI]:
    application = create_app()
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_ingest_all_then_unchanged(client: httpx.AsyncClient, dirs: tuple[Path, Path]) -> None:
    first = await client.post("/v1/ingest")
    assert first.status_code == 200
    body = first.json()
    assert body["ok"] is True
    assert body["pruned"] == []
    (f,) = body["files"]
    assert (f["file"], f["doc_id"], f["outcome"], f["chunks"], f["index_version"]) == (
        REAL_PDF.name,
        DOC,
        "ingested",
        22,
        1,
    )
    second = (await client.post("/v1/ingest", json={})).json()
    assert second["files"][0]["outcome"] == "unchanged"
    # Inspectable artifacts on disk (spec: chunks inspectable in data/processed/)
    processed = dirs[1]
    assert sorted(p.name for p in processed.iterdir()) == [f"{DOC}.v1.chunks.jsonl", "registry.json"]
    registry = json.loads((processed / "registry.json").read_text(encoding="utf-8"))
    assert registry["documents"][DOC]["active_index_version"] == 1


async def test_force_and_single_file(client: httpx.AsyncClient, dirs: tuple[Path, Path]) -> None:
    await client.post("/v1/ingest")
    r = (await client.post("/v1/ingest", json={"file": REAL_PDF.name, "force": True})).json()
    assert [(f["outcome"], f["index_version"]) for f in r["files"]] == [("ingested", 2)]
    assert sorted(p.name for p in dirs[1].iterdir()) == [f"{DOC}.v2.chunks.jsonl", "registry.json"]


async def test_status(client: httpx.AsyncClient) -> None:
    empty = (await client.get("/v1/ingest/status")).json()
    assert empty["documents"] == []
    assert empty["auto_ingest"]["on_startup"] is False
    await client.post("/v1/ingest")
    body = (await client.get("/v1/ingest/status")).json()
    (doc,) = body["documents"]
    assert doc["doc_id"] == DOC
    assert doc["chunks"] == 22
    assert len(doc["content_hash"]) == 64


async def test_unknown_file_is_404(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/ingest", json={"file": "nope.pdf"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("name", ["../secret.pdf", "a/b.pdf", "x.txt", ".env"])
async def test_unsafe_file_names_rejected(client: httpx.AsyncClient, name: str) -> None:
    r = await client.post("/v1/ingest", json={"file": name})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_file_name"


@pytest.mark.parametrize("body", [{"force": "maybe"}, {"file": 3}, [1, 2]])
async def test_malformed_body(client: httpx.AsyncClient, body: object) -> None:
    r = await client.post("/v1/ingest", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


async def test_partial_failure_is_200_with_ok_false(client: httpx.AsyncClient, dirs: tuple[Path, Path]) -> None:
    (dirs[0] / "broken.pdf").write_bytes(b"junk")
    body = (await client.post("/v1/ingest")).json()
    assert body["ok"] is False
    outcomes = {f["file"]: f["outcome"] for f in body["files"]}
    assert outcomes == {"broken.pdf": "failed", REAL_PDF.name: "ingested"}


async def test_delete_document(client: httpx.AsyncClient, dirs: tuple[Path, Path]) -> None:
    await client.post("/v1/ingest")
    r = await client.delete(f"/v1/documents/{DOC}")
    assert r.status_code == 204
    assert r.content == b""
    assert (await client.get("/v1/ingest/status")).json()["documents"] == []
    assert sorted(p.name for p in dirs[1].iterdir()) == ["registry.json"]
    again = await client.delete(f"/v1/documents/{DOC}")
    assert again.status_code == 404


@pytest.mark.parametrize("doc_id", ["UPPER", "a-b", "x" * 101])
async def test_delete_validates_doc_id(client: httpx.AsyncClient, doc_id: str) -> None:
    assert (await client.delete(f"/v1/documents/{doc_id}")).status_code == 422


async def test_body_size_limit(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/ingest", content=b"{" + b" " * 10_000 + b"}", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 413


async def test_readiness_reports_each_dependency(client: httpx.AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"] == {"vector_store": True, "pdf_dir": True, "startup_ingest": True}


async def test_readiness_fails_without_pdf_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PDF_DIR", str(tmp_path / "does-not-exist"))
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["pdf_dir"] is False


async def test_openapi_documents_endpoints(client: httpx.AsyncClient) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/v1/ingest", "/v1/ingest/status", "/v1/documents/{doc_id}"} <= set(paths)


# ---------- auto-ingest through the real app lifecycle ----------


async def test_auto_ingest_on_startup(monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path]) -> None:
    monkeypatch.setenv("AUTO_INGEST_ON_STARTUP", "true")
    app = create_app()
    async with app.router.lifespan_context(app):
        await app.state.container.auto_ingestor.startup_done.wait()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            body = (await c.get("/v1/ingest/status")).json()
    assert [d["doc_id"] for d in body["documents"]] == [DOC]
    assert body["auto_ingest"]["last_trigger"] == "startup"
    assert body["auto_ingest"]["last_ok"] is True


async def test_auto_ingest_watcher_picks_up_new_pdf(monkeypatch: pytest.MonkeyPatch, dirs: tuple[Path, Path]) -> None:
    import asyncio

    monkeypatch.setenv("AUTO_INGEST_ON_STARTUP", "true")
    monkeypatch.setenv("AUTO_INGEST_INTERVAL_S", "0.05")
    app = create_app()
    async with app.router.lifespan_context(app):
        container = app.state.container
        await container.auto_ingestor.startup_done.wait()
        new = dirs[0] / "late.pdf"
        new.write_bytes(make_pdf(KB_LIKE_PAGES))
        st = new.stat()
        os.utime(new, ns=(st.st_mtime_ns + 10**9, st.st_mtime_ns + 10**9))
        for _ in range(200):
            if container.registry.get("late") is not None:
                break
            await asyncio.sleep(0.05)
        assert container.registry.get("late") is not None
        new.unlink()
        for _ in range(200):
            if container.registry.get("late") is None:
                break
            await asyncio.sleep(0.05)
        assert container.registry.get("late") is None
    assert not container.auto_ingestor.running  # stopped on shutdown
