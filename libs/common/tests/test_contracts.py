"""Contract tests: pin every inter-service JSON schema.

A schema change fails this test. If the change is intentional and backward compatible, regenerate
the snapshots with ``UPDATE_SNAPSHOTS=1 make test-common`` and review the diff in code review.
Breaking changes need a new API version (/v2).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from common import contracts as c

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
PINNED: list[type[BaseModel]] = [
    c.Chunk,
    c.IngestRequest,
    c.IngestReport,
    c.IngestStatus,
    c.SearchRequest,
    c.SearchResponse,
    c.AdviseRequest,
    c.AdviseResponse,
    c.SessionView,
    c.ErrorResponse,
    c.HealthStatus,
]


@pytest.mark.parametrize("model", PINNED, ids=lambda m: m.__name__)
def test_schema_is_pinned(model: type[BaseModel]) -> None:
    path = SNAPSHOT_DIR / f"{model.__name__}.json"
    current = json.dumps(model.model_json_schema(), indent=2, sort_keys=True)
    if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not path.exists():
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path.write_text(current + "\n", encoding="utf-8", newline="\n")  # LF on every OS (no git churn)
    assert path.read_text(encoding="utf-8").strip() == current, f"{model.__name__} schema changed"


def _meta(**kw: object) -> c.ChunkMetadata:
    base: dict[str, object] = {
        "doc_id": "kb",
        "source_file": "kb.pdf",
        "page": 1,
        "section": "Shopify",
        "doc_type": "course",
        "content_hash": "h",
        "ingested_at": datetime(2026, 11, 12, tzinfo=UTC),
        "index_version": 1,
    }
    base.update(kw)
    return c.ChunkMetadata.model_validate(base)


def test_chunk_metadata_round_trip() -> None:
    m = _meta(fee_pkr=30000, level="beginner", mentor="Hamza Ali")
    again = c.ChunkMetadata.model_validate_json(m.model_dump_json())
    assert again == m
    assert again.level is c.Level.BEGINNER


def test_chunk_metadata_rejects_negative_fee() -> None:
    with pytest.raises(ValidationError):
        _meta(fee_pkr=-1)


@pytest.mark.parametrize("sid", ["", "a b", "x" * 65, "../etc"])
def test_advise_request_rejects_bad_session_ids(sid: str) -> None:
    with pytest.raises(ValidationError):
        c.AdviseRequest(session_id=sid, message="hi")


def test_advise_request_message_limit() -> None:
    with pytest.raises(ValidationError):
        c.AdviseRequest(session_id="s1", message="x" * (c.MAX_MESSAGE_CHARS + 1))


def test_advise_response_caps_clarifying_questions() -> None:
    with pytest.raises(ValidationError):
        c.AdviseResponse(session_id="s", answer="a", intent=c.Intent.CHIT_CHAT, clarifying_questions=["1", "2", "3"])


def test_ingest_report_ok_flag() -> None:
    now = datetime.now(UTC)
    ok = c.IngestReport(
        files=[c.FileReport(file="a.pdf", doc_id="a", outcome=c.FileOutcome.UNCHANGED)], started_at=now, finished_at=now
    )
    bad = c.IngestReport(
        files=[c.FileReport(file="a.pdf", doc_id="a", outcome=c.FileOutcome.FAILED)], started_at=now, finished_at=now
    )
    assert ok.ok
    assert not bad.ok
