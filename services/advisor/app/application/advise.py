"""Advise use cases: one conversational turn (plain or streamed) and session management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from app.application.workflow import AdvisorState, AdvisorWorkflow
from app.domain.compose import Messages
from app.domain.models import AdvisorRules, Session, Turn
from app.domain.ports import Clock, LLMClient, SessionRepository
from app.domain.safety import redact_pii
from common.contracts import (
    AdviseRequest,
    AdviseResponse,
    LearnerProfileDTO,
    RecommendedCourse,
    SessionView,
    Usage,
)
from common.errors import NotFound

logger = logging.getLogger(__name__)

# Human-readable progress for streaming clients (keys are graph node names).
PROGRESS = {
    "classify_intent": "understanding",
    "extract_profile": "understanding",
    "need_more_info": "clarifying",
    "retrieve": "searching",
    "grade_evidence": "checking",
    "generate_answer": "writing",
    "verify_grounding": "verifying",
    "format_response": "done",
}


def chunk_text(text: str, size: int = 80) -> list[str]:
    """Split on whitespace into pieces of about ``size`` chars (concatenation == original)."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start, end)
            end = space + 1 if space > start else end
        pieces.append(text[start:end])
        start = end
    return pieces


class AdviseService:
    def __init__(
        self,
        *,
        workflow: AdvisorWorkflow,
        sessions: SessionRepository,
        llm: LLMClient,
        rules: AdvisorRules,
        messages: Messages,
        clock: Clock,
    ) -> None:
        self.workflow = workflow
        self.sessions = sessions
        self.llm = llm
        self.rules = rules
        self.m = messages
        self.clock = clock

    async def _load(self, session_id: str) -> Session:
        now = self.clock.now()
        return await self.sessions.get(session_id) or Session(session_id=session_id, created_at=now, updated_at=now)

    async def _save(self, session: Session, request: AdviseRequest, state: AdvisorState) -> None:
        official = [self.rules.whatsapp, self.rules.whatsapp_link, self.rules.email]
        turns = (
            Turn("user", redact_pii(request.message)),  # learners' own numbers/emails are never stored
            Turn("assistant", redact_pii(state.get("answer", ""), allowed=official)),
        )
        updated = replace(
            session.with_turns(*turns, keep=self.rules.max_history_turns),
            profile=state.get("profile", session.profile),
            asked=session.asked | frozenset(state.get("question_keys", [])),
            updated_at=self.clock.now(),
        )
        await self.sessions.save(updated)

    def _response(self, request: AdviseRequest, state: AdvisorState) -> AdviseResponse:
        plan = state.get("plan")
        lang = state.get("language")
        recommendations = []
        total = None
        if plan is not None:
            recommendations = [
                RecommendedCourse(
                    course_slug=i.offer.slug,
                    title=i.offer.title,
                    fee_pkr=i.offer.fee_pkr,
                    mentor=i.offer.mentor,
                    order=i.order,
                    reason=", ".join(self.m.get(f"reason_{r.value}", lang or "en") for r in i.reasons),
                )
                for i in plan.items
            ]
            total = plan.total_now_pkr if plan.budget_pkr is not None else plan.total_all_pkr
        calls = state.get("llm_calls", 0)
        return AdviseResponse(
            session_id=request.session_id,
            answer=state["answer"],
            intent=state["intent"],
            citations=state.get("citations", []),
            recommendations=recommendations,
            total_fee_pkr=total,
            clarifying_questions=state.get("questions", [])[:2],
            grounded=state.get("grounded", True),
            fallback_used=state.get("fallback_used", False),
            language=str(lang.value if lang else "en"),
            usage=Usage(
                model=self.llm.name,
                llm_calls=calls,
                input_tokens=state.get("input_tokens", 0),
                output_tokens=state.get("output_tokens", 0),
            )
            if calls
            else None,
        )

    async def advise(self, request: AdviseRequest) -> AdviseResponse:
        session = await self._load(request.session_id)
        state = await self.workflow.run(request.message, session)
        await self._save(session, request, state)
        response = self._response(request, state)
        logger.info(
            "advise",
            extra={
                "intent": response.intent.value,
                "grounded": response.grounded,
                "fallback": response.fallback_used,
                "citations": len(response.citations),
                "llm_calls": state.get("llm_calls", 0),
            },
        )
        return response

    async def stream(self, request: AdviseRequest) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """(event, data) pairs: progress per node, then the verified answer in pieces, then the full response.

        The answer is only streamed after verify_grounding: users never see an unverified draft.
        """
        session = await self._load(request.session_id)
        state: AdvisorState = {}
        last_step = None
        async for node, latest in self.workflow.stream(request.message, session):
            state = latest  # keep the final, fully accumulated state for saving and the response
            step = PROGRESS.get(node, node)
            if step != last_step and step != "done":
                yield "status", {"step": step, "node": node}
                last_step = step
        await self._save(session, request, state)
        response = self._response(request, state)
        for piece in chunk_text(response.answer):
            yield "answer", {"delta": piece}
        yield "done", response.model_dump(mode="json")

    async def get_session(self, session_id: str) -> SessionView:
        session = await self.sessions.get(session_id)
        if session is None:
            raise NotFound("No such session (it may have expired).")
        p = session.profile
        return SessionView(
            session_id=session_id,
            profile=LearnerProfileDTO(
                goal=p.goal,
                experience_level=p.experience_level.value if p.experience_level else None,
                budget_pkr=p.budget_pkr,
                weekly_hours=p.weekly_hours,
                device=p.device,
                preferred_batch_time=p.preferred_batch_time,
                interests=list(p.interests),
                language=p.language.value,
            ),
            turns=len(session.turns),
        )

    async def delete_session(self, session_id: str) -> None:
        if not await self.sessions.delete(session_id):
            raise NotFound("No such session.")
