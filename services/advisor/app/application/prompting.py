"""Render prompts from versioned templates. Untrusted text is delimited and escaped (spec section 9)."""

from __future__ import annotations

import html
from collections.abc import Sequence
from string import Template

from app.domain.compose import PromptTemplates, money
from app.domain.models import Language, RecommendationPlan, Turn
from common.contracts import Intent, RetrievedChunk

LANGUAGE_NAMES = {
    Language.EN: "English",
    Language.ROMAN_URDU: "Roman Urdu (Urdu written in English letters)",
    Language.URDU: "Urdu",
}


def escape(text: str) -> str:
    """Neutralise markup in untrusted text so it cannot close or forge our delimiter tags."""
    return html.escape(text, quote=True)


def render_evidence(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "<evidence>none</evidence>"
    return "\n".join(
        f'<evidence id="{escape(c.id)}" doc="{escape(c.metadata.doc_id)}" section="{escape(c.metadata.section)}">\n'
        f"{escape(c.text)}\n</evidence>"
        for c in chunks
    )


def render_plan(plan: RecommendationPlan | None) -> str:
    if plan is None:
        return "none"
    lines: list[str] = []
    if plan.budget_pkr is not None:
        lines.append(f"budget: {money(plan.budget_pkr)}")
    for item in plan.items:
        fee = money(item.offer.fee_pkr) if item.offer.fee_pkr is not None else "not published"
        when = "now" if item.now else "later (budget)"
        reasons = ", ".join(r.value for r in item.reasons)
        lines.append(
            f"{item.order}. {item.offer.title} | fee: {fee} | mentor: {item.offer.mentor} | {when} | {reasons}"
        )
    if plan.total_now_pkr is not None:
        lines.append(f"total for courses taken now: {money(plan.total_now_pkr)}")
    if plan.total_all_pkr is not None:
        lines.append(f"total for all recommended courses: {money(plan.total_all_pkr)}")
    for offer in plan.over_budget:
        lines.append(f"over budget: {offer.title} ({money(offer.fee_pkr) if offer.fee_pkr else 'fee not published'})")
    for mentor, slugs in plan.mentor_bundles:
        lines.append(f"same mentor: {mentor} teaches {', '.join(slugs)}")
    return "\n".join(lines) or "no course matched"


def render_history(turns: Sequence[Turn]) -> str:
    return "\n".join(f"{t.role}: {escape(t.content)}" for t in turns) or "(first message)"


class PromptBuilder:
    def __init__(self, templates: PromptTemplates, *, whatsapp: str) -> None:
        self.t = templates
        self.whatsapp = whatsapp

    def system(self, language: Language) -> str:
        return Template(self.t.system).safe_substitute(whatsapp=self.whatsapp, language_name=LANGUAGE_NAMES[language])

    def user(
        self,
        *,
        message: str,
        intent: Intent,
        evidence: Sequence[RetrievedChunk],
        plan: RecommendationPlan | None,
        history: Sequence[Turn],
    ) -> str:
        return Template(self.t.answer).safe_substitute(
            history=render_history(history),
            evidence=render_evidence(evidence),
            plan=escape(render_plan(plan)),
            message=escape(message),
            intent=intent.value,
        )

    def regenerate(self, violations: Sequence[str]) -> str:
        return Template(self.t.regenerate).safe_substitute(violations="\n".join(f"- {v}" for v in violations))
