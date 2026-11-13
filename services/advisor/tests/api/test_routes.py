"""HTTP API through the real composition root and the real HTTP retrieval client (retrieval mocked by respx)."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import StubRetrieval

from app.main import create_app
from common.contracts import SearchRequest

STUB = StubRetrieval()


def retrieval_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json=STUB.respond(SearchRequest.model_validate_json(request.content)).model_dump(mode="json")
    )


@pytest.fixture
def retrieval() -> respx.MockRouter:
    with respx.mock(base_url="http://retrieval.test", assert_all_called=False) as router:
        router.post("/v1/search").mock(side_effect=retrieval_handler)
        yield router


@pytest.fixture
async def client(retrieval: respx.MockRouter) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def advise(client: httpx.AsyncClient, message: str, sid: str = "s1") -> httpx.Response:
    return await client.post("/v1/advise", json={"session_id": sid, "message": message})


async def test_advise_recommendation(client: httpx.AsyncClient) -> None:
    r = await advise(client, "Shopify or TikTok Shop? Budget is Rs. 30,000.")
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "course_recommendation"
    assert [x["course_slug"] for x in body["recommendations"]] == ["shopify", "tiktok-shop"]
    assert body["total_fee_pkr"] == 30000
    assert body["grounded"] is True
    assert body["citations"][0]["doc"] == "genzskool_knowledge_base"
    assert "0309-4190245" in body["answer"]
    assert body["usage"]["model"] == "fake"


async def test_advise_out_of_scope_is_polite_refusal(client: httpx.AsyncClient, retrieval: respx.MockRouter) -> None:
    body = (await advise(client, "I want a job as a data scientist")).json()
    assert body["intent"] == "out_of_scope"
    assert "doesn't teach that" in body["answer"]
    assert retrieval.calls.call_count == 0


async def test_retrieval_outage_still_answers_with_handoff(
    client: httpx.AsyncClient, retrieval: respx.MockRouter
) -> None:
    retrieval.post("/v1/search").mock(return_value=httpx.Response(503))
    r = await advise(client, "How much is the eBay course?")
    assert r.status_code == 200
    assert "can't look up course details right now" in r.json()["answer"]
    assert "info@genzskool.com" in r.json()["answer"]


async def test_request_id_is_propagated_to_retrieval(client: httpx.AsyncClient, retrieval: respx.MockRouter) -> None:
    await client.post(
        "/v1/advise",
        json={"session_id": "s1", "message": "How much is the eBay course?"},
        headers={"X-Request-ID": "trace-advisor-0001"},
    )
    assert retrieval.calls.last.request.headers["X-Request-ID"] == "trace-advisor-0001"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"session_id": "s1"},
        {"message": "hi"},
        {"session_id": "", "message": "hi"},
        {"session_id": "bad id!", "message": "hi"},
        {"session_id": "s1", "message": ""},
        {"session_id": "s1", "message": "x" * 2001},
    ],
)
async def test_validation(client: httpx.AsyncClient, body: dict[str, str]) -> None:
    r = await client.post("/v1/advise", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


async def test_body_size_limit(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/advise", content=b"x" * 20_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


async def test_sse_stream(client: httpx.AsyncClient) -> None:
    async with client.stream(
        "POST", "/v1/advise/stream", json={"session_id": "s2", "message": "How much is the eBay course?"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join([chunk async for chunk in r.aiter_text()])
    events = [block for block in raw.split("\n\n") if block.strip()]
    parsed = [
        (b.split("\n")[0].removeprefix("event: "), json.loads(b.split("\n")[1].removeprefix("data: "))) for b in events
    ]
    kinds = [k for k, _ in parsed]
    assert kinds[0] == "status"
    assert kinds[-1] == "done"
    assert "answer" in kinds
    answer = "".join(d["delta"] for k, d in parsed if k == "answer")
    assert answer == parsed[-1][1]["answer"]
    assert "Rs. 28,000" in answer


async def test_sse_error_event(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.application.advise import AdviseService

    async def broken(self: AdviseService, request: object) -> AsyncIterator[tuple[str, dict[str, object]]]:
        raise RuntimeError("boom")
        yield ("never", {})  # pragma: no cover

    monkeypatch.setattr(AdviseService, "stream", broken)
    async with client.stream("POST", "/v1/advise/stream", json={"session_id": "s3", "message": "hi"}) as r:
        raw = "".join([chunk async for chunk in r.aiter_text()])
    assert raw.startswith("event: error")
    assert '"code": "internal_error"' in raw
    assert "boom" not in raw


async def test_session_endpoints(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/sessions/s9")).status_code == 404
    await advise(client, "budget 30k, I like shopify", sid="s9")
    view = (await client.get("/v1/sessions/s9")).json()
    assert view == {
        "session_id": "s9",
        "profile": {
            "goal": None,
            "experience_level": None,
            "budget_pkr": 30000,
            "weekly_hours": None,
            "device": None,
            "preferred_batch_time": None,
            "interests": ["shopify"],
            "language": "en",
        },
        "turns": 2,
    }
    assert (await client.delete("/v1/sessions/s9")).status_code == 204
    assert (await client.delete("/v1/sessions/s9")).status_code == 404
    assert (await client.get("/v1/sessions/bad%20id")).status_code == 422


async def test_readiness_does_not_depend_on_retrieval(client: httpx.AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"] == {"prompts": True}


async def test_openapi(client: httpx.AsyncClient) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/v1/advise", "/v1/advise/stream", "/v1/sessions/{session_id}"} <= set(paths)


async def test_logs_never_contain_the_learners_message(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        await advise(client, "my number 0300-7654321, zebracorn, how much is ebay?")
    text = " ".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert "zebracorn" not in text
    assert "7654321" not in text
