"""Phase 3 e2e: retrieval against real Qdrant + the artifacts ingestion writes, in the running stack.

Includes the retrieval evaluation over HTTP (dense path = real Qdrant) and a Qdrant outage drill.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ROOT, compose, url

DOC = "genzskool_knowledge_base"
PDF_DIR = ROOT / "data" / "pdfs"
CASES = [
    json.loads(line) for line in (ROOT / "data/eval/retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines()
]


def search(query: str, **body: Any) -> dict[str, Any]:
    r = httpx.post(url("retrieval", "/v1/search"), json={"query": query, **body}, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def ingestion_docs() -> dict[str, int]:
    status = httpx.get(url("ingestion", "/v1/ingest/status"), timeout=10).json()
    return {d["doc_id"]: d["active_index_version"] for d in status["documents"]}


def retrieval_docs() -> dict[str, int]:
    return {
        d["doc_id"]: d["index_version"]
        for d in httpx.get(url("retrieval", "/v1/index"), timeout=10).json()["documents"]
    }


def wait_for(condition: Callable[[], bool], what: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if condition():
                return
        except (httpx.HTTPError, KeyError, ValueError, AssertionError):
            pass
        time.sleep(1)
    pytest.fail(f"timed out waiting for: {what}")


@pytest.fixture(scope="module", autouse=True)
def indexed() -> None:
    wait_for(lambda: DOC in ingestion_docs() and retrieval_docs() == ingestion_docs(), "retrieval sees the index")


def test_serves_exactly_what_ingestion_activated() -> None:
    assert retrieval_docs() == ingestion_docs()
    index = httpx.get(url("retrieval", "/v1/index")).json()
    assert index["chunks"] == 22
    assert index["warnings"] == []


def test_hybrid_search_uses_real_qdrant() -> None:
    body = search("AI automation with ChatGPT and Make.com", debug=True)
    debug = body["retrieval_debug"]
    assert "dense_error" not in debug
    assert debug["candidates"]["dense"] >= 1, "dense candidates must come from Qdrant"
    top = body["chunks"][0]
    assert top["metadata"]["course_slug"] == "ai-automation"
    assert top["metadata"]["fee_pkr"] == 32000
    assert "dense_rank" in debug["results"][0]


def test_fee_lookup_and_budget_filter() -> None:
    assert search("How much is the eBay course?")["chunks"][0]["metadata"]["fee_pkr"] == 28000
    body = search("which courses can I take", top_k=20, filters={"max_fee_pkr": 30000, "doc_types": ["course"]})
    fees = {c["metadata"]["course_slug"]: c["metadata"]["fee_pkr"] for c in body["chunks"]}
    assert fees
    assert "digital-marketing" not in fees
    assert "ai-automation" not in fees


def test_off_topic_returns_empty() -> None:
    for q in ("weather in Lahore today", "best pizza recipe", "hi"):
        assert search(q)["chunks"] == [], q


def test_retrieval_eval_over_http_meets_gate() -> None:
    positives = [c for c in CASES if not c["expect_empty"]]
    hits = 0
    rr = 0.0
    for case in positives:
        sections = [c["metadata"]["section"] for c in search(case["query"], top_k=5)["chunks"]]
        ranks = [i for i, s in enumerate(sections, 1) if s in case["expected_sections"]]
        hits += bool(ranks)
        rr += 1 / ranks[0] if ranks else 0.0
    assert hits / len(positives) >= 0.9
    assert rr / len(positives) >= 0.8
    for case in (c for c in CASES if c["expect_empty"]):
        assert search(case["query"])["chunks"] == [], case["id"]


def test_follows_a_forced_reingest_without_restart() -> None:
    before_id = search("ebay fee")["chunks"][0]["id"]
    r = httpx.post(url("ingestion", "/v1/ingest"), json={"force": True}, timeout=120).json()
    new_version = r["files"][0]["index_version"]
    after = search("ebay fee")
    assert after["retrieval_debug"]["index_versions"] == {DOC: new_version}
    assert after["chunks"][0]["id"] != before_id, "ids are version-scoped: results must come from the new version"
    assert after["chunks"][0]["metadata"]["index_version"] == new_version
    assert after["chunks"][0]["metadata"]["course_slug"] == "ebay"


def test_new_pdf_becomes_searchable_and_disappears_when_deleted() -> None:
    probe = PDF_DIR / "zz_e2e_retrieval_probe.pdf"
    try:
        shutil.copyfile(PDF_DIR / f"{DOC}.pdf", probe)
        wait_for(lambda: "zz_e2e_retrieval_probe" in retrieval_docs(), "probe document searchable")
        docs = {c["metadata"]["doc_id"] for c in search("ebay product research", top_k=10)["chunks"]}
        assert docs == {DOC, "zz_e2e_retrieval_probe"}
    finally:
        probe.unlink(missing_ok=True)
    wait_for(lambda: "zz_e2e_retrieval_probe" not in retrieval_docs(), "probe document gone after deletion")
    assert {c["metadata"]["doc_id"] for c in search("ebay product research", top_k=10)["chunks"]} == {DOC}


def test_qdrant_outage_degrades_to_keyword_search_and_recovers() -> None:
    compose("stop", "qdrant")
    try:
        wait_for(lambda: httpx.get(url("retrieval", "/readyz")).status_code == 503, "readiness reports the outage")
        ready = httpx.get(url("retrieval", "/readyz")).json()
        assert ready["checks"] == {"index": True, "vector_store": False}
        body = search("How much is the eBay course?", debug=True)
        assert body["retrieval_debug"]["dense_error"]
        assert body["chunks"][0]["metadata"]["course_slug"] == "ebay", "BM25 alone must still answer"
        assert httpx.get(url("retrieval", "/healthz")).status_code == 200  # alive, just degraded
    finally:
        compose("start", "qdrant")
    wait_for(lambda: httpx.get(url("retrieval", "/readyz")).status_code == 200, "recovered after Qdrant restart")
    assert "dense_error" not in search("AI automation", debug=True)["retrieval_debug"]


def test_search_logs_are_structured_and_never_contain_the_query() -> None:
    marker = "zebracorn"  # unique word: if it shows up in the logs, raw queries are being logged
    search(f"shopify fee {marker} 0300-1234567")
    time.sleep(0.5)
    logs = compose("logs", "--no-log-prefix", "--since", "1m", "retrieval").stdout
    events = [json.loads(ln) for ln in logs.splitlines() if '"msg": "search"' in ln]
    assert events
    assert {"returned", "candidates", "latency_ms"} <= set(events[-1])
    assert marker not in logs, "queries may contain PII and must not be logged"
    assert "1234567" not in logs
