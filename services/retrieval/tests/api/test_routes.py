"""HTTP API through the real composition root (fixture index on disk, fake embeddings, memory dense)."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import DOC, load_fixture_chunks, make_chunk, write_index

from app.main import create_app


@pytest.fixture
async def app() -> AsyncIterator[FastAPI]:
    application = create_app()
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_search(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/search", json={"query": "How much is the eBay course?"})
    assert r.status_code == 200
    body = r.json()
    top = body["chunks"][0]
    assert set(top) == {"id", "text", "score", "metadata"}
    assert top["metadata"]["course_slug"] == "ebay"
    assert top["metadata"]["fee_pkr"] == 28000
    assert body["retrieval_debug"]["index_versions"] == {DOC: 1}


async def test_search_with_filters_and_debug(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/search",
        json={
            "query": "courses for beginners",
            "top_k": 10,
            "debug": True,
            "filters": {"max_fee_pkr": 30000, "doc_types": ["course"], "level": "beginner"},
        },
    )
    body = r.json()
    assert body["chunks"]
    for c in body["chunks"]:
        assert c["metadata"]["doc_type"] == "course"
        assert c["metadata"]["level"] == "beginner"
        assert c["metadata"]["fee_pkr"] is None or c["metadata"]["fee_pkr"] <= 30000
    assert "results" in body["retrieval_debug"]


async def test_off_topic_returns_empty_list_not_error(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/search", json={"query": "weather in Lahore"})
    assert r.status_code == 200
    assert r.json()["chunks"] == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "x" * 1001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 51},
        {"query": "ok", "filters": {"max_fee_pkr": -1}},
        {"query": "ok", "filters": {"doc_types": ["blog"]}},
        {"query": "ok", "filters": {"level": "expert"}},
        {"query": 5},
    ],
)
async def test_validation_errors(client: httpx.AsyncClient, body: dict[str, object]) -> None:
    r = await client.post("/v1/search", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


async def test_body_size_limit(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/search", content=b'{"query": "' + b"a" * 9000 + b'"}', headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 413


async def test_index_endpoint(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/index")).json()
    assert body == {
        "documents": [{"doc_id": DOC, "index_version": 1}],
        "chunks": 22,
        "signature": f"22:{DOC}@1",
        "warnings": [],
    }


async def test_readiness(client: httpx.AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"] == {"index": True, "vector_store": True}


async def test_not_ready_before_anything_is_ingested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "empty"))
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        ready = await c.get("/readyz")
        search = await c.post("/v1/search", json={"query": "shopify"})
    assert ready.status_code == 503
    assert ready.json()["checks"]["index"] is False
    assert search.status_code == 200
    assert search.json()["chunks"] == []
    assert search.json()["retrieval_debug"]["reason"] == "index_empty"


async def test_picks_up_new_index_version_without_restart(client: httpx.AsyncClient, processed_dir: Path) -> None:
    before = (await client.post("/v1/search", json={"query": "ebay fee"})).json()["chunks"][0]["id"]
    # Simulate ingestion switching to version 2 (new ids), then deleting version 1.
    v2 = [
        c.model_copy(update={"id": f"v2-{c.id}", "metadata": c.metadata.model_copy(update={"index_version": 2})})
        for c in load_fixture_chunks()
    ]
    write_index(processed_dir, {DOC: (2, v2)})
    (processed_dir / f"{DOC}.v1.chunks.jsonl").unlink()
    after = (await client.post("/v1/search", json={"query": "ebay fee"})).json()
    assert after["chunks"][0]["id"] == f"v2-{before}"
    assert after["retrieval_debug"]["index_versions"] == {DOC: 2}


async def test_new_document_becomes_searchable(client: httpx.AsyncClient, processed_dir: Path) -> None:
    extra = [
        make_chunk(
            "extra-1",
            "Course: Python bootcamp | Fee: Rs. 50,000\nLearn python programming basics",
            doc="extra_doc",
            slug="python",
            fee=50000,
        )
    ]
    write_index(processed_dir, {DOC: (1, load_fixture_chunks()), "extra_doc": (1, extra)})
    body = (await client.post("/v1/search", json={"query": "python programming bootcamp"})).json()
    assert body["chunks"][0]["id"] == "extra-1"


async def test_openapi_has_search(client: httpx.AsyncClient) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/v1/search", "/v1/index"} <= set(paths)
