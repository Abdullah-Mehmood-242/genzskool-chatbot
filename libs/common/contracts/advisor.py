"""Advisor service API contracts (v1). The gateway forwards these unchanged."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

MAX_MESSAGE_CHARS = 2000


class Intent(StrEnum):
    COURSE_RECOMMENDATION = "course_recommendation"
    COURSE_INFO = "course_info"
    FEES_AND_PAYMENT = "fees_and_payment"
    ENROLLMENT = "enrollment"
    CERTIFICATE = "certificate"
    MENTOR_INFO = "mentor_info"
    CONTACT = "contact"
    OUT_OF_SCOPE = "out_of_scope"
    CHIT_CHAT = "chit_chat"
    UNSAFE = "unsafe"


class AdviseRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class Citation(BaseModel):
    chunk_id: str
    doc: str
    section: str


class RecommendedCourse(BaseModel):
    course_slug: str
    title: str
    fee_pkr: int | None
    mentor: str | None
    order: int = Field(ge=1)
    reason: str


class LearnerProfileDTO(BaseModel):
    goal: str | None = None
    experience_level: str | None = None
    budget_pkr: int | None = None
    weekly_hours: int | None = None
    device: str | None = None
    preferred_batch_time: str | None = None
    interests: list[str] = Field(default_factory=list)
    language: str = "en"


class Usage(BaseModel):
    """LLM usage for one advise call (cost estimation and evaluation)."""

    model: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class AdviseResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent
    citations: list[Citation] = Field(default_factory=list)
    recommendations: list[RecommendedCourse] = Field(default_factory=list)
    total_fee_pkr: int | None = None
    clarifying_questions: list[str] = Field(default_factory=list, max_length=2)
    grounded: bool = True
    fallback_used: bool = False
    language: str = "en"
    usage: Usage | None = None


class SessionView(BaseModel):
    session_id: str
    profile: LearnerProfileDTO
    turns: int
