"""The advisor as a LangGraph state machine (spec 6.3):

classify_intent -> extract_profile -> (need_more_info | retrieve) -> retrieve -> grade_evidence
-> generate_answer -> verify_grounding -> format_response

Deterministic code decides intent, profile, eligibility, order and cost; the LLM only words the
answer, and every draft is verified before anyone sees it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.application.prompting import PromptBuilder
from app.domain.compose import AnswerComposer, ComposedAnswer, GenerationContext, Messages, citation
from app.domain.grounding import GroundingVerifier
from app.domain.models import AdvisorRules, CourseOffer, Language, LearnerProfile, RecommendationPlan, Session
from app.domain.ports import LLMClient, LLMRequest, RetrievalClient
from app.domain.recommend import Recommender
from app.domain.safety import redact_pii
from app.domain.understanding import IntentClassifier, ProfileExtractor
from common.contracts import Citation, DocType, Intent, RetrievedChunk, SearchFilters, SearchRequest
from common.errors import AppError, ProviderError

logger = logging.getLogger(__name__)

TERMINAL_INTENTS = (Intent.UNSAFE, Intent.OUT_OF_SCOPE, Intent.CHIT_CHAT)
# Goals that already imply a sensible path (beginner-friendly courses), so no question is needed.
ACTIONABLE_GOALS = frozenset({"earn_online", "business"})


class AdvisorState(TypedDict, total=False):
    # input
    message: str
    session: Session
    # understanding
    language: Language
    intent: Intent
    profile: LearnerProfile
    mentioned: tuple[str, ...]
    questions: list[str]
    question_keys: list[str]
    # evidence and decisions
    evidence: list[RetrievedChunk]
    retrieval_error: str | None
    evidence_ok: bool
    plan: RecommendationPlan | None
    # generation
    draft: str
    draft_used: tuple[str, ...]
    attempts: int
    violations: list[str]
    grounded: bool
    fallback_used: bool
    llm_calls: int
    input_tokens: int
    output_tokens: int
    # output
    answer: str
    citations: list[Citation]


class AdvisorWorkflow:
    def __init__(
        self,
        *,
        rules: AdvisorRules,
        messages: Messages,
        prompts: PromptBuilder,
        llm: LLMClient,
        retrieval: RetrievalClient,
        verifier: GroundingVerifier,
    ) -> None:
        self.rules = rules
        self.m = messages
        self.prompts = prompts
        self.llm = llm
        self.retrieval = retrieval
        self.verifier = verifier
        self.extractor = ProfileExtractor(rules)
        self.classifier = IntentClassifier(rules, self.extractor)
        self.recommender = Recommender(rules)
        self.composer = AnswerComposer(messages)
        self.graph = self._build()

    # ------------------------------------------------------------------ graph

    def _build(self) -> Any:
        g = StateGraph(AdvisorState)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("extract_profile", self.extract_profile)
        g.add_node("need_more_info", self.need_more_info)
        g.add_node("retrieve", self.retrieve)
        g.add_node("grade_evidence", self.grade_evidence)
        g.add_node("generate_answer", self.generate_answer)
        g.add_node("verify_grounding", self.verify_grounding)
        g.add_node("format_response", self.format_response)
        g.add_edge(START, "classify_intent")
        g.add_edge("classify_intent", "extract_profile")
        g.add_conditional_edges(
            "extract_profile", self.route_after_profile, ["need_more_info", "retrieve", "format_response"]
        )
        g.add_edge("need_more_info", "format_response")
        g.add_edge("retrieve", "grade_evidence")
        g.add_conditional_edges("grade_evidence", self.route_after_grading, ["generate_answer", "format_response"])
        g.add_edge("generate_answer", "verify_grounding")
        g.add_conditional_edges(
            "verify_grounding", self.route_after_verification, ["generate_answer", "format_response"]
        )
        g.add_edge("format_response", END)
        return g.compile()

    async def run(self, message: str, session: Session) -> AdvisorState:
        result: AdvisorState = await self.graph.ainvoke({"message": message, "session": session})
        return result

    async def stream(self, message: str, session: Session) -> AsyncIterator[tuple[str, AdvisorState]]:
        """Yields (node, accumulated state) after every node; the last item is format_response."""
        state: AdvisorState = {"message": message, "session": session}
        async for update in self.graph.astream(state, stream_mode="updates"):
            for node, delta in update.items():
                state = {**state, **(delta or {})}
                yield node, state

    # ------------------------------------------------------------------ nodes

    async def classify_intent(self, state: AdvisorState) -> AdvisorState:
        message = state["message"]
        return {"intent": self.classifier.classify(message), "language": self.extractor.extract(message).language}

    async def extract_profile(self, state: AdvisorState) -> AdvisorState:
        update = self.extractor.extract(state["message"])
        profile = state["session"].profile.merge(update)
        return {"profile": profile, "mentioned": update.interests}

    def _questions(self, state: AdvisorState) -> tuple[list[str], list[str]]:
        """At most 2 clarifying questions, never one already asked (spec: never interrogate)."""
        profile = state["profile"]
        if state["intent"] != Intent.COURSE_RECOMMENDATION or profile.interests or profile.goal in ACTIONABLE_GOALS:
            return [], []
        asked = state["session"].asked
        keys = [
            k
            for k in ("clarify_interest", "clarify_budget")
            if k not in asked and not (k == "clarify_budget" and profile.budget_pkr is not None)
        ]
        keys = keys[: self.rules.max_clarifying_questions]
        return keys, [self.m.get(k, state["language"]) for k in keys]

    def route_after_profile(self, state: AdvisorState) -> str:
        if state["intent"] in TERMINAL_INTENTS:
            return "format_response"
        keys, _ = self._questions(state)
        return "need_more_info" if keys else "retrieve"

    async def need_more_info(self, state: AdvisorState) -> AdvisorState:
        keys, questions = self._questions(state)
        return {"question_keys": keys, "questions": questions}

    async def retrieve(self, state: AdvisorState) -> AdvisorState:
        profile = state["profile"]
        intent = state["intent"]
        requests: list[SearchRequest] = []
        if intent == Intent.COURSE_RECOMMENDATION:
            course_filter = SearchFilters(doc_types=[DocType.COURSE])
            if profile.interests:
                requests.append(
                    SearchRequest(
                        query=f"{state['message']} {' '.join(s.replace('-', ' ') for s in profile.interests)} course",
                        top_k=10,
                        filters=SearchFilters(doc_types=[DocType.COURSE], course_slugs=list(profile.interests)),
                    )
                )
            requests.append(SearchRequest(query="course fee level mentor beginner", top_k=10, filters=course_filter))
            requests.append(
                SearchRequest(
                    query="course catalogue overview fees", top_k=1, filters=SearchFilters(doc_types=[DocType.OVERVIEW])
                )
            )
        else:
            requests.append(SearchRequest(query=state["message"], top_k=5))
            if state["mentioned"]:
                requests.append(
                    SearchRequest(
                        query=f"{state['message']} course",
                        top_k=3,
                        filters=SearchFilters(course_slugs=list(state["mentioned"])),
                    )
                )
        evidence: dict[str, RetrievedChunk] = {}
        try:
            for request in requests:
                for chunk in (await self.retrieval.search(request)).chunks:
                    evidence.setdefault(chunk.id, chunk)
        except AppError as exc:
            # Graceful degradation (spec 7): no evidence -> honest hand-off, never a guess.
            logger.warning("retrieval_unavailable", extra={"error_code": exc.code})
            return {"evidence": [], "retrieval_error": exc.code}
        return {"evidence": list(evidence.values()), "retrieval_error": None}

    async def grade_evidence(self, state: AdvisorState) -> AdvisorState:
        evidence = state.get("evidence", [])
        plan = None
        if state["intent"] == Intent.COURSE_RECOMMENDATION:
            offers = [o for c in evidence if (o := CourseOffer.from_chunk(c))]
            plan = self.recommender.plan(state["profile"], offers)
            ok = bool(plan.items or plan.over_budget or evidence)
        else:
            ok = bool(evidence)
        return {
            "evidence_ok": ok,
            "plan": plan,
            "attempts": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "violations": [],
            "fallback_used": False,
        }

    def route_after_grading(self, state: AdvisorState) -> str:
        return "generate_answer" if state.get("evidence_ok") else "format_response"

    def _context(self, state: AdvisorState) -> GenerationContext:
        return GenerationContext(
            intent=state["intent"],
            language=state["language"],
            user_message=state["message"],
            evidence=tuple(state.get("evidence", [])),
            plan=state.get("plan"),
            profile=state["profile"],
            mentioned=state.get("mentioned", ()),
        )

    def _fallback(self, state: AdvisorState) -> ComposedAnswer:
        return self.composer.compose(self._context(state))

    async def generate_answer(self, state: AdvisorState) -> AdvisorState:
        attempts = state.get("attempts", 0) + 1
        user = self.prompts.user(
            message=state["message"],
            intent=state["intent"],
            evidence=state.get("evidence", []),
            plan=state.get("plan"),
            history=state["session"].turns,
        )
        messages: list[tuple[str, str]] = [("user", user)]
        if attempts > 1 and state.get("draft"):
            messages += [("assistant", state["draft"]), ("user", self.prompts.regenerate(state.get("violations", [])))]
        request = LLMRequest(
            system=self.prompts.system(state["language"]), messages=tuple(messages), context=self._context(state)
        )
        base: AdvisorState = {"attempts": attempts, "llm_calls": state.get("llm_calls", 0) + 1}
        try:
            response = await self.llm.generate(request)
        except ProviderError as exc:
            logger.warning("llm_failed_using_fallback", extra={"error_code": exc.code})
            composed = self._fallback(state)
            return {
                **base,
                "draft": composed.text,
                "draft_used": composed.used,
                "fallback_used": True,
                "attempts": self.rules.regenerate_on_ungrounded + 1,
            }
        tokens: AdvisorState = {
            "input_tokens": state.get("input_tokens", 0) + response.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + response.output_tokens,
        }
        if response.refused or not response.text.strip():
            composed = self._fallback(state)
            return {
                **base,
                **tokens,
                "draft": composed.text,
                "draft_used": composed.used,
                "fallback_used": True,
                "attempts": self.rules.regenerate_on_ungrounded + 1,
            }
        return {**base, **tokens, "draft": response.text, "draft_used": response.cited}

    async def verify_grounding(self, state: AdvisorState) -> AdvisorState:
        plan = state.get("plan")
        report = self.verifier.verify(
            state.get("draft", ""),
            [c.text for c in state.get("evidence", [])],
            allowed_numbers=plan.computed_numbers if plan else (),
            user_message=state["message"],
        )
        if report.ok:
            return {"grounded": True, "violations": []}
        logger.warning(
            "ungrounded_draft", extra={"violations": list(report.violations), "attempt": state.get("attempts", 1)}
        )
        if state.get("attempts", 1) <= self.rules.regenerate_on_ungrounded:
            return {"grounded": False, "violations": list(report.violations)}
        # Out of retries: the deterministic composer is grounded by construction.
        composed = self._fallback(state)
        return {
            "grounded": True,
            "violations": list(report.violations),
            "fallback_used": True,
            "draft": composed.text,
            "draft_used": composed.used,
        }

    def route_after_verification(self, state: AdvisorState) -> str:
        return "format_response" if state.get("grounded") else "generate_answer"

    async def format_response(self, state: AdvisorState) -> AdvisorState:
        lang = state.get("language", Language.EN)
        intent = state["intent"]
        handoff = self.m.get(
            "handoff",
            lang,
            whatsapp=self.rules.whatsapp,
            whatsapp_link=self.rules.whatsapp_link,
            email=self.rules.email,
            hours=self.rules.hours,
        )
        reminder = self.m.get("confirm_reminder", lang, whatsapp=self.rules.whatsapp)
        citations: list[Citation] = []
        if intent == Intent.UNSAFE:
            parts = [self.m.get("refusal_unsafe", lang)]
        elif intent == Intent.OUT_OF_SCOPE:
            advice = self.classifier.is_advice_request(state["message"])
            parts = [self.m.get("refusal_advice" if advice else "refusal_out_of_scope", lang), handoff]
        elif intent == Intent.CHIT_CHAT:
            parts = [self.m.get("greeting", lang)]
        elif state.get("questions"):
            parts = state["questions"]
        elif not state.get("evidence_ok"):
            key = "retrieval_down" if state.get("retrieval_error") else "insufficient_evidence"
            parts = [self.m.get(key, lang), handoff]
        else:
            draft = state.get("draft", "")
            citations = self._citations(state, draft)
            parts = [draft]
            if citations and "[source:" not in draft:
                parts.append(
                    self.m.get("sources_label", lang)
                    + " "
                    + " ".join(f"[source: {c.doc}, {c.section}]" for c in citations)
                )
            parts.append(reminder)
        allowed = [
            self.rules.whatsapp,
            self.rules.whatsapp_link,
            self.rules.email,
            *(c.text for c in state.get("evidence", [])),
        ]
        answer = redact_pii("\n\n".join(p for p in parts if p), allowed=allowed)
        return {"answer": answer, "citations": citations}

    @staticmethod
    def _citations(state: AdvisorState, draft: str) -> list[Citation]:
        evidence = state.get("evidence", [])
        cited = [c for c in evidence if citation(c) in draft]
        if not cited:
            used = set(state.get("draft_used", ()))
            cited = [c for c in evidence if c.id in used] or evidence[:2]
        return [Citation(chunk_id=c.id, doc=c.metadata.doc_id, section=c.metadata.section) for c in cited]
