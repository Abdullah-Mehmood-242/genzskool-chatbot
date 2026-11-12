"""Field-level validation of every shared DTO (boundaries, enums, immutability, defaults)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from common import contracts as c


def _meta(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "doc_id": "kb",
        "source_file": "kb.pdf",
        "page": 1,
        "section": "s",
        "doc_type": "course",
        "content_hash": "h",
        "ingested_at": datetime(2026, 11, 12, tzinfo=UTC),
        "index_version": 1,
    }
    base.update(kw)
    return base


def test_intents_match_spec_exactly() -> None:
    assert {i.value for i in c.Intent} == {
        "course_recommendation",
        "course_info",
        "fees_and_payment",
        "enrollment",
        "certificate",
        "mentor_info",
        "contact",
        "out_of_scope",
        "chit_chat",
        "unsafe",
    }


def test_doc_types_match_spec_exactly() -> None:
    assert {d.value for d in c.DocType} == {"course", "mentor", "policy", "contact", "overview"}
    assert {lv.value for lv in c.Level} == {"beginner", "intermediate"}


# ---------- ChunkMetadata / Chunk ----------


@pytest.mark.parametrize(
    "override",
    [
        {"page": 0},
        {"index_version": 0},
        {"fee_pkr": -1},
        {"doc_type": "blog"},
        {"level": "Beginner"},  # enum values are exact lower-case; ingestion must normalise
        {"level": "advanced"},
        {"ingested_at": "not-a-date"},
    ],
    ids=lambda o: next(iter(o)) + "=" + str(next(iter(o.values()))),
)
def test_chunk_metadata_rejects(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        c.ChunkMetadata.model_validate(_meta(**override))


@pytest.mark.parametrize(
    "field", ["doc_id", "source_file", "page", "section", "doc_type", "content_hash", "ingested_at", "index_version"]
)
def test_chunk_metadata_required_fields(field: str) -> None:
    data = _meta()
    del data[field]
    with pytest.raises(ValidationError):
        c.ChunkMetadata.model_validate(data)


def test_chunk_metadata_optional_fields_default_to_none() -> None:
    m = c.ChunkMetadata.model_validate(_meta())
    assert (m.course_slug, m.category, m.level, m.fee_pkr, m.mentor) == (None, None, None, None, None)


def test_fee_zero_is_valid_and_distinct_from_unpublished() -> None:
    assert c.ChunkMetadata.model_validate(_meta(fee_pkr=0)).fee_pkr == 0
    assert c.ChunkMetadata.model_validate(_meta(fee_pkr=None)).fee_pkr is None


def test_chunk_is_immutable() -> None:
    chunk = c.Chunk(id="1", text="t", metadata=c.ChunkMetadata.model_validate(_meta()))
    with pytest.raises(ValidationError):
        chunk.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        chunk.metadata.fee_pkr = 1  # type: ignore[misc]


def test_chunk_is_hashable_for_dedup() -> None:
    m = c.ChunkMetadata.model_validate(_meta())
    assert len({c.Chunk(id="1", text="t", metadata=m), c.Chunk(id="1", text="t", metadata=m)}) == 1


# ---------- Retrieval ----------


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x" * 1001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 51},
        {"query": "ok", "filters": {"max_fee_pkr": -5}},
        {"query": "ok", "filters": {"doc_types": ["blog"]}},
        {"query": "ok", "filters": {"level": "expert"}},
    ],
)
def test_search_request_rejects(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        c.SearchRequest.model_validate(payload)


def test_search_request_accepts_boundaries() -> None:
    assert c.SearchRequest(query="x" * 1000, top_k=50).top_k == 50
    req = c.SearchRequest.model_validate(
        {"query": "shopify", "top_k": 1, "filters": {"max_fee_pkr": 0, "doc_types": ["course"], "level": "beginner"}}
    )
    assert req.filters is not None
    assert req.filters.doc_types == [c.DocType.COURSE]
    assert c.SearchRequest(query="q").debug is False


def test_search_response_empty_is_valid() -> None:
    r = c.SearchResponse(chunks=[])
    assert r.chunks == []
    assert r.retrieval_debug == {}


# ---------- Advisor ----------


@pytest.mark.parametrize("sid", ["a", "user_1", "abc-DEF-123", "x" * 64])
def test_advise_request_accepts_valid_session_ids(sid: str) -> None:
    assert c.AdviseRequest(session_id=sid, message="hi").session_id == sid


@pytest.mark.parametrize("message", ["", "x" * (c.MAX_MESSAGE_CHARS + 1)])
def test_advise_request_rejects_messages(message: str) -> None:
    with pytest.raises(ValidationError):
        c.AdviseRequest(session_id="s", message=message)


def test_advise_request_accepts_urdu_and_max_length() -> None:
    assert c.AdviseRequest(session_id="s", message="مجھے شاپیفائی سیکھنا ہے").message
    assert len(c.AdviseRequest(session_id="s", message="x" * c.MAX_MESSAGE_CHARS).message) == c.MAX_MESSAGE_CHARS


def test_advise_response_defaults() -> None:
    r = c.AdviseResponse(session_id="s", answer="a", intent=c.Intent.CONTACT)
    assert r.grounded is True
    assert r.fallback_used is False
    assert r.citations == []
    assert r.recommendations == []
    assert r.total_fee_pkr is None
    assert c.AdviseResponse(session_id="s", answer="a", intent="contact", clarifying_questions=["a", "b"])


def test_recommended_course_order_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        c.RecommendedCourse(course_slug="s", title="t", fee_pkr=1, mentor=None, order=0, reason="r")


def test_learner_profile_defaults() -> None:
    p = c.LearnerProfileDTO()
    assert p.language == "en"
    assert p.interests == []
    assert p.budget_pkr is None


# ---------- Ingestion / envelope / health ----------


def test_ingest_request_defaults() -> None:
    r = c.IngestRequest()
    assert r.file is None
    assert r.force is False


def test_file_report_defaults() -> None:
    r = c.FileReport(file="a.pdf", doc_id="a", outcome="ingested")
    assert (r.chunks, r.index_version, r.warnings, r.error) == (0, None, [], None)


def test_file_outcomes() -> None:
    assert {o.value for o in c.FileOutcome} == {"ingested", "unchanged", "failed", "skipped_scanned"}


def test_envelope_is_generic() -> None:
    env = c.Envelope[c.HealthStatus](data=c.HealthStatus(status="ok", service="s", version="1"), request_id="r")
    dumped = env.model_dump()
    assert dumped == {"data": {"status": "ok", "service": "s", "version": "1", "checks": {}}, "request_id": "r"}
    with pytest.raises(ValidationError):
        c.Envelope[c.HealthStatus].model_validate({"data": {"nope": 1}})


def test_error_response_round_trip() -> None:
    raw = '{"error": {"code": "x", "message": "m", "request_id": null, "details": {"a": [1]}}}'
    parsed = c.ErrorResponse.model_validate_json(raw)
    assert parsed.error.details == {"a": [1]}
    assert c.ErrorResponse.model_validate_json(parsed.model_dump_json()) == parsed


def test_contracts_package_exports_everything_it_declares() -> None:
    for name in c.__all__:
        assert hasattr(c, name), name
