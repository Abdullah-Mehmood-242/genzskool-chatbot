"""Phase 2 e2e: ingestion + auto-ingest against real Qdrant, the real bind-mounted data/pdfs, real containers.

These tests write probe PDFs into ./data/pdfs (cleaned up afterwards) and wait for the watcher, so
they take a couple of minutes. They assume the compose defaults (watch interval 15 s).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ROOT, compose, url

DOC = "genzskool_knowledge_base"
PDF_DIR = ROOT / "data" / "pdfs"
PROCESSED = ROOT / "data" / "processed"
REAL_PDF = PDF_DIR / f"{DOC}.pdf"
QDRANT = "http://127.0.0.1:6333"
COLLECTION = "genzskool_kb"
WAIT_S = 120


def status() -> dict[str, Any]:
    return httpx.get(url("ingestion", "/v1/ingest/status"), timeout=10).json()  # type: ignore[no-any-return]


def docs() -> dict[str, dict[str, Any]]:
    return {d["doc_id"]: d for d in status()["documents"]}


def qdrant_count(doc_id: str | None = None) -> int:
    body: dict[str, Any] = {"exact": True}
    if doc_id:
        body["filter"] = {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}
    r = httpx.post(f"{QDRANT}/collections/{COLLECTION}/points/count", json=body, timeout=10)
    if r.status_code == 404:
        return 0
    return int(r.json()["result"]["count"])


def wait_for(condition: Callable[[], bool], what: str, timeout: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if condition():
                return
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            last_error = exc
        time.sleep(1)
    pytest.fail(f"timed out after {timeout}s waiting for: {what} (last error: {last_error}); status={status()}")


@pytest.fixture(scope="module", autouse=True)
def initial_index() -> None:
    wait_for(lambda: DOC in docs() and status()["auto_ingest"]["runs"] >= 1, "startup auto-ingest")


@pytest.fixture
def probe_pdf() -> Iterator[Path]:
    """A copy of the KB under a new name: a second document the watcher must pick up."""
    path = PDF_DIR / "zz_e2e_probe.pdf"
    yield path
    path.unlink(missing_ok=True)
    wait_for(lambda: "zz_e2e_probe" not in docs(), "probe document pruned during cleanup")


# ---------------------------------------------------------------------------- startup state


def test_startup_auto_ingest_populated_the_index() -> None:
    s = status()
    assert s["auto_ingest"]["on_startup"] is True
    assert s["auto_ingest"]["watch_interval_s"] == 15
    assert s["auto_ingest"]["last_ok"] is True
    doc = docs()[DOC]
    assert doc["chunks"] == 22
    assert qdrant_count(DOC) == 22


def test_readyz_includes_startup_ingest() -> None:
    r = httpx.get(url("ingestion", "/readyz"))
    assert r.status_code == 200
    assert r.json()["checks"] == {"vector_store": True, "pdf_dir": True, "startup_ingest": True}


def test_chunks_are_inspectable_on_the_host() -> None:
    version = docs()[DOC]["active_index_version"]
    registry = json.loads((PROCESSED / "registry.json").read_text(encoding="utf-8"))
    assert registry["documents"][DOC]["active_index_version"] == version
    lines = (PROCESSED / f"{DOC}.v{version}.chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 22
    chunks = [json.loads(line) for line in lines]
    fees = {c["metadata"]["course_slug"]: c["metadata"]["fee_pkr"] for c in chunks if c["metadata"]["course_slug"]}
    assert fees == {
        "shopify": 30000,
        "tiktok-shop": 30000,
        "ebay": 28000,
        "video-editing": 30000,
        "digital-marketing": 35000,
        "graphic-design": 28000,
        "ai-automation": 32000,
        "tiktok-automation": None,
    }
    old = [p.name for p in PROCESSED.glob(f"{DOC}.v*.chunks.jsonl")]
    assert old == [f"{DOC}.v{version}.chunks.jsonl"], "only the active version's file may remain"


def test_processed_files_are_world_readable_inside_container() -> None:
    """Regression (Linux CI): artifacts were 0600, unreadable on the host. Checked in the container so
    it also fails on Docker Desktop, where bind mounts do not enforce Linux permissions."""
    listing = compose("exec", "-T", "ingestion", "sh", "-c", "stat -c '%a %n' /app/data/processed/*").stdout
    modes = dict(line.split(" ", 1)[::-1] for line in listing.splitlines() if line.strip())
    assert modes, "no artifacts found"
    assert all(mode == "644" for mode in modes.values()), modes


def test_qdrant_collection_shape_and_payload_indexes() -> None:
    info = httpx.get(f"{QDRANT}/collections/{COLLECTION}").json()["result"]
    assert info["config"]["params"]["vectors"] == {"size": 1024, "distance": "Cosine"} | {
        k: v for k, v in info["config"]["params"]["vectors"].items() if k not in ("size", "distance")
    }
    assert set(info["payload_schema"]) >= {
        "doc_id",
        "index_version",
        "doc_type",
        "course_slug",
        "category",
        "level",
        "mentor",
        "fee_pkr",
    }


def test_qdrant_payload_carries_text_and_metadata() -> None:
    r = httpx.post(
        f"{QDRANT}/collections/{COLLECTION}/points/scroll",
        json={
            "filter": {"must": [{"key": "course_slug", "match": {"value": "ebay"}}]},
            "with_payload": True,
            "limit": 5,
        },
    )
    (point,) = r.json()["result"]["points"]
    payload = point["payload"]
    assert payload["mentor"] == "Hamza Ali"
    assert payload["fee_pkr"] == 28000
    assert payload["text"].startswith("Course: eBay product research and listings")


# ---------------------------------------------------------------------------- manual triggers


def test_ingest_script_reports_unchanged() -> None:
    sh = shutil.which("sh") or shutil.which("bash")
    if sh is None:
        pytest.skip("no POSIX shell")
    out = subprocess.run(  # noqa: S603 - fixed argv
        [sh, "scripts/ingest_all.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "unchanged" in out.stdout
    assert out.stdout.strip().endswith("OK")


def test_dev_runner_ingest_reports_unchanged() -> None:
    out = subprocess.run(
        [sys.executable, "scripts/dev.py", "ingest"], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "unchanged" in out.stdout


def test_force_reingest_swaps_versions_atomically() -> None:
    before = docs()[DOC]["active_index_version"]
    r = httpx.post(url("ingestion", "/v1/ingest"), json={"force": True}, timeout=120).json()
    after = r["files"][0]["index_version"]
    assert after == before + 1
    assert qdrant_count(DOC) == 22, "old version must be deleted after the switch"
    assert not (PROCESSED / f"{DOC}.v{before}.chunks.jsonl").exists()
    assert (PROCESSED / f"{DOC}.v{after}.chunks.jsonl").exists()


@pytest.mark.parametrize(
    ("body", "status_code", "code"),
    [
        ({"file": "missing.pdf"}, 404, "not_found"),
        ({"file": "../../etc/passwd.pdf"}, 422, "invalid_file_name"),
        ({"force": "maybe"}, 422, "validation_error"),  # note: pydantic accepts "yes"/"no" as booleans
    ],
)
def test_api_errors_in_container(body: dict[str, Any], status_code: int, code: str) -> None:
    r = httpx.post(url("ingestion", "/v1/ingest"), json=body, timeout=30)
    assert r.status_code == status_code
    assert r.json()["error"]["code"] == code


# ---------------------------------------------------------------------------- watcher


def test_watcher_ingests_new_pdf_and_prunes_it_when_deleted(probe_pdf: Path) -> None:
    shutil.copyfile(REAL_PDF, probe_pdf)
    wait_for(lambda: "zz_e2e_probe" in docs(), "new PDF auto-ingested")
    assert docs()["zz_e2e_probe"]["chunks"] == 22
    assert qdrant_count("zz_e2e_probe") == 22
    assert status()["auto_ingest"]["last_trigger"] == "file_change"
    probe_pdf.unlink()
    wait_for(lambda: "zz_e2e_probe" not in docs(), "deleted PDF pruned")
    assert qdrant_count("zz_e2e_probe") == 0
    assert not list(PROCESSED.glob("zz_e2e_probe.v*.chunks.jsonl"))
    assert DOC in docs(), "pruning must not touch other documents"


def test_corrupt_pdf_is_reported_and_does_not_disturb_serving(probe_pdf: Path) -> None:
    probe_pdf.write_bytes(b"this is not a pdf")
    wait_for(lambda: status()["auto_ingest"]["last_ok"] is False, "corrupt file reported")
    s = status()
    assert "zz_e2e_probe.pdf" in s["auto_ingest"]["last_error"]
    assert "zz_e2e_probe" not in docs()
    assert qdrant_count(DOC) == 22
    probe_pdf.unlink()
    wait_for(lambda: status()["auto_ingest"]["last_ok"] is True, "recovered after removing the corrupt file")


def test_self_heals_after_qdrant_collection_is_lost() -> None:
    before = docs()[DOC]["active_index_version"]
    assert httpx.delete(f"{QDRANT}/collections/{COLLECTION}").json()["result"] is True
    wait_for(lambda: qdrant_count(DOC) == 22, "vectors re-created after data loss")
    assert docs()[DOC]["active_index_version"] == before + 1
    assert status()["auto_ingest"]["last_trigger"] == "index_drift"


# ---------------------------------------------------------------------------- restarts


def test_restart_does_not_reembed_unchanged_documents() -> None:
    version = docs()[DOC]["active_index_version"]
    compose("restart", "ingestion")
    wait_for(lambda: status()["auto_ingest"]["runs"] >= 1, "startup run after restart")
    s = status()
    assert s["auto_ingest"]["last_trigger"] == "startup"
    assert s["auto_ingest"]["last_ok"] is True
    assert docs()[DOC]["active_index_version"] == version, "unchanged PDF must not create a new version"
    logs = compose("logs", "--no-log-prefix", "--since", "2m", "ingestion").stdout
    finished = [json.loads(ln) for ln in logs.splitlines() if '"ingest_finished"' in ln]
    assert finished
    assert finished[-1]["outcomes"] == ["unchanged"]
