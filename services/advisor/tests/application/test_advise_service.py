"""AdviseService: multi-turn sessions, PII never stored, response mapping, streaming."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import PROMPTS_DIR, ManualClock, ScriptedLLM, StubRetrieval, load_rules

from app.application.advise import AdviseService, chunk_text
from app.application.prompting import PromptBuilder
from app.application.workflow import AdvisorWorkflow
from app.domain.compose import AnswerComposer
from app.domain.grounding import GroundingVerifier
from app.domain.ports import LLMClient
from app.infrastructure.llm import FakeLLM
from app.infrastructure.prompts import load_prompt_set
from app.infrastructure.sessions import InMemorySessionRepository
from common.contracts import AdviseRequest, Intent
from common.errors import NotFound

PROMPTS = load_prompt_set(PROMPTS_DIR, "v1")
RULES = load_rules()


def service(llm: LLMClient | None = None) -> tuple[AdviseService, InMemorySessionRepository]:
    clock = ManualClock()
    llm = llm or FakeLLM(AnswerComposer(PROMPTS.messages))
    sessions = InMemorySessionRepository(ttl_s=3600, max_sessions=100, clock=clock)
    wf = AdvisorWorkflow(
        rules=RULES,
        messages=PROMPTS.messages,
        prompts=PromptBuilder(PROMPTS.templates, whatsapp=RULES.whatsapp),
        llm=llm,
        retrieval=StubRetrieval(),
        verifier=GroundingVerifier([PROMPTS.templates.system]),
    )
    return AdviseService(
        workflow=wf, sessions=sessions, llm=llm, rules=RULES, messages=PROMPTS.messages, clock=clock
    ), sessions


def req(message: str, session_id: str = "s1") -> AdviseRequest:
    return AdviseRequest(session_id=session_id, message=message)


async def test_response_mapping_for_a_recommendation() -> None:
    svc, _ = service()
    r = await svc.advise(req("Shopify or TikTok Shop? Budget is Rs. 30,000."))
    assert r.intent == Intent.COURSE_RECOMMENDATION
    assert [(x.course_slug, x.order, x.fee_pkr, x.mentor) for x in r.recommendations] == [
        ("shopify", 1, 30000, "Hamza Ali"),
        ("tiktok-shop", 2, 30000, "Sana Iqbal"),
    ]
    assert r.recommendations[1].reason.endswith("later, when budget allows")
    assert r.total_fee_pkr == 30000  # with a budget: what fits now
    assert r.grounded
    assert not r.fallback_used
    assert r.language == "en"
    assert r.usage is not None
    assert r.usage.model == "fake"
    assert r.usage.llm_calls == 1
    assert r.citations


async def test_total_without_budget_is_the_full_path() -> None:
    svc, _ = service()
    r = await svc.advise(req("Should I take shopify and ebay?"))
    assert r.total_fee_pkr == 58000


async def test_refusal_has_no_usage_or_recommendations() -> None:
    svc, _ = service()
    r = await svc.advise(req("I want to become a data scientist"))
    assert r.intent == Intent.OUT_OF_SCOPE
    assert r.recommendations == []
    assert r.usage is None
    assert r.citations == []


async def test_profile_builds_up_over_turns() -> None:
    svc, _ = service()
    first = await svc.advise(req("I want to learn something new"))
    assert len(first.clarifying_questions) == 2
    second = await svc.advise(req("Graphic design"))
    assert second.intent == Intent.COURSE_INFO
    third = await svc.advise(req("what would you recommend? my budget is 30k"))
    assert [x.course_slug for x in third.recommendations] == ["graphic-design"]
    view = await svc.get_session("s1")
    assert view.profile.interests == ["graphic-design"]
    assert view.profile.budget_pkr == 30000
    assert view.turns == 6


async def test_clarifying_questions_are_not_repeated_in_the_next_turn() -> None:
    svc, sessions = service()
    await svc.advise(req("I want to learn something new"))
    stored = await sessions.get("s1")
    assert stored is not None
    assert stored.asked == {"clarify_interest", "clarify_budget"}
    again = await svc.advise(req("I want to learn something new"))
    assert again.clarifying_questions == []


async def test_learner_pii_is_never_stored_but_official_contacts_are() -> None:
    svc, sessions = service()
    await svc.advise(req("my number is 0300-1234567 and email ali@example.com, what is your whatsapp?"))
    stored = await sessions.get("s1")
    assert stored is not None
    user_turn, assistant_turn = stored.turns
    assert "1234567" not in user_turn.content
    assert "ali@example.com" not in user_turn.content
    assert "[phone removed]" in user_turn.content
    assert "0309-4190245" in assistant_turn.content


async def test_history_is_bounded() -> None:
    svc, sessions = service()
    for i in range(5):
        await svc.advise(req(f"How much is the eBay course? ({i})"))
    stored = await sessions.get("s1")
    assert stored is not None
    assert len(stored.turns) == RULES.max_history_turns


async def test_sessions_are_isolated() -> None:
    svc, _ = service()
    await svc.advise(req("budget 30k shopify", "alice"))
    await svc.advise(req("hi", "bob"))
    assert (await svc.get_session("alice")).profile.budget_pkr == 30000
    assert (await svc.get_session("bob")).profile.budget_pkr is None


async def test_get_and_delete_session() -> None:
    svc, _ = service()
    with pytest.raises(NotFound):
        await svc.get_session("nope")
    await svc.advise(req("hi"))
    await svc.delete_session("s1")
    with pytest.raises(NotFound):
        await svc.get_session("s1")
    with pytest.raises(NotFound):
        await svc.delete_session("s1")


async def test_stream_events_and_answer_reassembly() -> None:
    svc, _ = service()
    events = [e async for e in svc.stream(req("How much is the eBay course?"))]
    kinds = [k for k, _ in events]
    assert kinds[0] == "status"
    assert kinds[-1] == "done"
    steps = [d["step"] for k, d in events if k == "status"]
    assert steps == ["understanding", "searching", "checking", "writing", "verifying"]
    first_answer = kinds.index("answer")
    assert all(k == "status" for k in kinds[:first_answer])  # nothing is shown before verification
    streamed = "".join(d["delta"] for k, d in events if k == "answer")
    done = events[-1][1]
    assert streamed == done["answer"]
    assert done["intent"] == "fees_and_payment"


async def test_stream_and_plain_give_the_same_answer() -> None:
    svc_a, _ = service()
    svc_b, _ = service()
    plain = await svc_a.advise(req("Who teaches graphic design?"))
    events = [e async for e in svc_b.stream(req("Who teaches graphic design?"))]
    assert events[-1][1]["answer"] == plain.answer


async def test_stream_saves_the_session() -> None:
    svc, sessions = service()
    _ = [e async for e in svc.stream(req("budget 30k for ebay"))]
    stored = await sessions.get("s1")
    assert stored is not None
    assert stored.profile.budget_pkr == 30000


async def test_llm_usage_accumulates_over_regeneration() -> None:
    llm = ScriptedLLM("It costs Rs. 25,000.", "The eBay course costs Rs. 28,000 (instalments).")
    svc, _ = service(llm)
    r = await svc.advise(req("How much is the eBay course?"))
    assert r.usage is not None
    assert (r.usage.llm_calls, r.usage.input_tokens, r.usage.output_tokens) == (2, 200, 40)
    assert r.usage.model == "scripted"


@pytest.mark.parametrize("text", ["", "short", "a " * 200, "word" * 50, "x y\nz " * 40])
def test_chunk_text_reassembles_exactly(text: str) -> None:
    pieces = chunk_text(text, size=30)
    assert "".join(pieces) == text
    assert all(len(p) <= 30 or " " not in p for p in pieces)
