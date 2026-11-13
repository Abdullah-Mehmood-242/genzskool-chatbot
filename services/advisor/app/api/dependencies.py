"""Composition root: the ONLY place where concrete adapters are chosen and wired to ports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, FastAPI

from app.api.config import AdvisorFile
from app.api.routes import router
from app.application.advise import AdviseService
from app.application.prompting import PromptBuilder
from app.application.workflow import AdvisorWorkflow
from app.domain.compose import AnswerComposer
from app.domain.grounding import GroundingVerifier
from app.domain.ports import Clock, LLMClient, RetrievalClient, SessionRepository
from app.infrastructure.llm import build_llm
from app.infrastructure.prompts import SystemClock, load_prompt_set
from app.infrastructure.retrieval_client import HttpRetrievalClient
from app.infrastructure.sessions import InMemorySessionRepository, RedisSessionRepository
from app.settings import Settings
from common.config import load_yaml_model
from common.observability import ReadinessCheck


@dataclass
class Container:
    settings: Settings
    llm: LLMClient
    retrieval: RetrievalClient
    sessions: SessionRepository
    service: AdviseService
    prompts_version: str

    def readiness_checks(self) -> dict[str, ReadinessCheck]:
        # Retrieval is deliberately not a readiness gate: when it is down the advisor still answers
        # with the WhatsApp/email hand-off (graceful degradation, spec section 7).
        async def prompts() -> bool:
            return bool(self.prompts_version)

        return {"prompts": prompts}

    def routers(self) -> list[APIRouter]:
        return [router]

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for resource in (self.llm, self.retrieval):
                close: Any = getattr(resource, "aclose", None)
                if close is not None:
                    await close()


def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    retrieval: RetrievalClient | None = None,
    sessions: SessionRepository | None = None,
    clock: Clock | None = None,
) -> Container:
    """Wire adapters from settings. Keyword overrides exist for tests only."""
    advisor_cfg = load_yaml_model(settings.config_dir / "advisor.yaml", AdvisorFile)
    rules = advisor_cfg.to_rules()
    prompt_set = load_prompt_set(settings.prompts_dir, advisor_cfg.prompts_version)
    clock = clock or SystemClock()
    composer = AnswerComposer(prompt_set.messages)
    llm = llm or build_llm(
        settings.llm_provider,
        composer=composer,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        effort=settings.llm_effort,
        refusal_fallback=settings.llm_refusal_fallback,
        base_url=settings.llm_base_url,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    retrieval = retrieval or HttpRetrievalClient(settings.retrieval_url, timeout_s=settings.retrieval_timeout_s)
    if sessions is None:
        sessions = (
            RedisSessionRepository.from_url(settings.redis_url, ttl_s=settings.session_ttl_s)
            if settings.session_store == "redis"
            else InMemorySessionRepository(ttl_s=settings.session_ttl_s, max_sessions=settings.session_max, clock=clock)
        )
    workflow = AdvisorWorkflow(
        rules=rules,
        messages=prompt_set.messages,
        prompts=PromptBuilder(prompt_set.templates, whatsapp=rules.whatsapp),
        llm=llm,
        retrieval=retrieval,
        verifier=GroundingVerifier(protected_fragments=[prompt_set.templates.system]),
    )
    service = AdviseService(
        workflow=workflow, sessions=sessions, llm=llm, rules=rules, messages=prompt_set.messages, clock=clock
    )
    return Container(
        settings=settings,
        llm=llm,
        retrieval=retrieval,
        sessions=sessions,
        service=service,
        prompts_version=prompt_set.version,
    )
