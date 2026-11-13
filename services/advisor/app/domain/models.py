"""Advisor domain models. Pure Python (pydantic contracts only)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from common.contracts import ChunkMetadata, DocType, Level, RetrievedChunk


class Language(StrEnum):
    EN = "en"
    ROMAN_URDU = "roman_ur"
    URDU = "ur"  # Urdu script


class Experience(StrEnum):
    NONE = "none"  # "no experience", "fresher"
    EXPERIENCED = "experienced"


@dataclass(frozen=True)
class LearnerProfile:
    goal: str | None = None
    experience_level: Experience | None = None
    budget_pkr: int | None = None
    weekly_hours: int | None = None
    device: str | None = None
    preferred_batch_time: str | None = None
    interests: tuple[str, ...] = ()  # course slugs, in the order the learner mentioned them
    language: Language = Language.EN

    def merge(self, update: LearnerProfile) -> LearnerProfile:
        """Newer non-empty values win; interests accumulate (first mention keeps its place)."""
        interests = tuple(dict.fromkeys((*self.interests, *update.interests)))
        return LearnerProfile(
            goal=update.goal or self.goal,
            experience_level=update.experience_level or self.experience_level,
            budget_pkr=update.budget_pkr if update.budget_pkr is not None else self.budget_pkr,
            weekly_hours=update.weekly_hours if update.weekly_hours is not None else self.weekly_hours,
            device=update.device or self.device,
            preferred_batch_time=update.preferred_batch_time or self.preferred_batch_time,
            interests=interests,
            language=update.language,
        )

    @property
    def is_beginner(self) -> bool:
        return self.experience_level in (None, Experience.NONE)


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Session:
    session_id: str
    profile: LearnerProfile = field(default_factory=LearnerProfile)
    turns: tuple[Turn, ...] = ()
    asked: frozenset[str] = frozenset()  # clarifying-question keys already asked (never re-ask)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def with_turns(self, *turns: Turn, keep: int) -> Session:
        return replace(self, turns=(*self.turns, *turns)[-keep:] if keep else ())


@dataclass(frozen=True)
class CourseOffer:
    """A course as described by retrieved evidence (never invented)."""

    slug: str
    title: str
    fee_pkr: int | None
    level: Level | None
    mentor: str | None
    category: str | None
    chunk_id: str
    doc_id: str
    section: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> CourseOffer | None:
        m: ChunkMetadata = chunk.metadata
        if m.doc_type != DocType.COURSE or not m.course_slug:
            return None
        title = m.section.removeprefix("Course: ").split(" (part ")[0]
        return cls(
            slug=m.course_slug,
            title=title,
            fee_pkr=m.fee_pkr,
            level=m.level,
            mentor=m.mentor,
            category=m.category,
            chunk_id=chunk.id,
            doc_id=m.doc_id,
            section=m.section,
        )


class Reason(StrEnum):
    MATCHES_INTEREST = "matches_interest"
    BEGINNER_FRIENDLY = "beginner_friendly"
    FITS_BUDGET = "fits_budget"
    FEE_UNPUBLISHED = "fee_unpublished"
    TAKE_FIRST = "take_first"  # prerequisite-style sequencing: comes before another pick
    INTERMEDIATE = "intermediate_level"
    LATER_BUDGET = "later_when_budget_allows"


@dataclass(frozen=True)
class RecommendationItem:
    offer: CourseOffer
    order: int
    reasons: tuple[Reason, ...]
    now: bool  # False: recommended for later because the budget is used up


@dataclass(frozen=True)
class RecommendationPlan:
    items: tuple[RecommendationItem, ...] = ()
    budget_pkr: int | None = None
    total_now_pkr: int | None = None  # sum of known fees of the courses to take now
    total_all_pkr: int | None = None  # sum of known fees of every recommended course
    over_budget: tuple[CourseOffer, ...] = ()
    mentor_bundles: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (mentor, slugs) teaching 2+ picks
    next_steps: tuple[CourseOffer, ...] = ()  # sequencing successors worth knowing about

    @property
    def unknown_fee(self) -> tuple[CourseOffer, ...]:
        return tuple(i.offer for i in self.items if i.offer.fee_pkr is None)

    @property
    def computed_numbers(self) -> frozenset[int]:
        """Numbers the plan derives (totals). The grounding check accepts these besides evidence."""
        return frozenset(n for n in (self.total_now_pkr, self.total_all_pkr, self.budget_pkr) if n is not None)


@dataclass(frozen=True)
class AdvisorRules:
    """Business configuration (config/advisor.yaml), passed in as data."""

    max_clarifying_questions: int = 2
    max_history_turns: int = 6
    max_recommendations: int = 3
    regenerate_on_ungrounded: int = 1
    sequencing: tuple[tuple[str, str], ...] = ()
    interests: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (slug, keywords)
    ambiguous_interests: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (word, slugs)
    goals: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (goal, keywords)
    out_of_scope_topics: tuple[str, ...] = ()
    whatsapp: str = "0309-4190245"
    whatsapp_link: str = "https://wa.me/923094190245"
    email: str = "info@genzskool.com"
    hours: str = "Mon-Sat, 10am-8pm"

    def __post_init__(self) -> None:
        if not 0 <= self.max_clarifying_questions <= 2:
            raise ValueError("max_clarifying_questions must be 0..2 (spec: never interrogate)")
        if self.max_recommendations < 1:
            raise ValueError("max_recommendations must be >= 1")
        if self.regenerate_on_ungrounded < 0:
            raise ValueError("regenerate_on_ungrounded must be >= 0")
