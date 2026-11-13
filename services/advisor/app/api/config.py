"""Schema of config/advisor.yaml (validated at startup) and its conversion to domain rules."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.models import AdvisorRules


class Handoff(BaseModel):
    whatsapp: str
    whatsapp_link: str
    email: str
    hours: str


class AdvisorFile(BaseModel):
    max_clarifying_questions: int = Field(default=2, ge=0, le=2)
    max_history_turns: int = Field(default=6, ge=0, le=50)
    max_recommendations: int = Field(default=3, ge=1, le=8)
    regenerate_on_ungrounded: int = Field(default=1, ge=0, le=3)
    prompts_version: str = Field(default="v1", pattern=r"^v\d+$")
    handoff: Handoff
    sequencing: list[tuple[str, str]] = Field(default_factory=list)
    interests: dict[str, list[str]] = Field(default_factory=dict)
    ambiguous_interests: dict[str, list[str]] = Field(default_factory=dict)
    goals: dict[str, list[str]] = Field(default_factory=dict)
    out_of_scope_topics: list[str] = Field(default_factory=list)

    def to_rules(self) -> AdvisorRules:
        def low(values: list[str]) -> tuple[str, ...]:
            return tuple(v.lower() for v in values)

        return AdvisorRules(
            max_clarifying_questions=self.max_clarifying_questions,
            max_history_turns=self.max_history_turns,
            max_recommendations=self.max_recommendations,
            regenerate_on_ungrounded=self.regenerate_on_ungrounded,
            sequencing=tuple((a, b) for a, b in self.sequencing),
            interests=tuple((slug, low(kw)) for slug, kw in self.interests.items()),
            ambiguous_interests=tuple((w.lower(), tuple(s)) for w, s in self.ambiguous_interests.items()),
            goals=tuple((goal, low(kw)) for goal, kw in self.goals.items()),
            out_of_scope_topics=low(self.out_of_scope_topics),
            whatsapp=self.handoff.whatsapp,
            whatsapp_link=self.handoff.whatsapp_link,
            email=self.handoff.email,
            hours=self.handoff.hours,
        )
