"""Deterministic answers built only from evidence lines and the plan's computed numbers.

Used as the offline ``fake`` LLM and as the safe fallback when a real LLM's draft fails the grounding
check. Wording (refusals, hand-off, framing) comes from versioned prompt files as ``Messages``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from app.domain.models import CourseOffer, Language, LearnerProfile, RecommendationPlan
from common.contracts import DocType, Intent, RetrievedChunk

_FEE_IN_HEADER = re.compile(r"\|\s*Fee:\s*([^|]+?)\s*(?:\||$)")

# Which evidence best answers each intent (checked in order of preference by the evidence composer).
_INTENT_SECTIONS: dict[Intent, Callable[[RetrievedChunk], bool]] = {
    Intent.ENROLLMENT: lambda c: c.metadata.section.startswith("Enrollment"),
    Intent.CERTIFICATE: lambda c: c.metadata.section.startswith("FAQ and certificate"),
    Intent.CONTACT: lambda c: c.metadata.doc_type == DocType.CONTACT,
    Intent.COURSE_INFO: lambda c: c.metadata.doc_type == DocType.COURSE,
}


@dataclass(frozen=True)
class Messages:
    """Localised templates: {key: {language: template}}. Unknown languages fall back to English."""

    templates: Mapping[str, Mapping[str, str]]

    def get(self, key: str, language: Language | str = Language.EN, **values: object) -> str:
        variants = self.templates.get(key)
        if variants is None:
            raise KeyError(f"missing message template '{key}'")
        lang = str(language)
        template = variants.get(lang) or variants.get("roman_ur" if lang == "ur" else "en") or variants["en"]
        return template.format(**values)


@dataclass(frozen=True)
class PromptTemplates:
    """Versioned prompt texts (prompts/<version>/*.md), loaded by infrastructure."""

    system: str
    answer: str
    regenerate: str


@dataclass(frozen=True)
class GenerationContext:
    intent: Intent
    language: Language
    user_message: str
    evidence: tuple[RetrievedChunk, ...] = ()
    plan: RecommendationPlan | None = None
    profile: LearnerProfile = field(default_factory=LearnerProfile)
    mentioned: tuple[str, ...] = ()  # course slugs the message names


@dataclass(frozen=True)
class ComposedAnswer:
    text: str
    used: tuple[str, ...]  # chunk ids the answer draws on (for citations)


def citation(chunk: RetrievedChunk) -> str:
    return f"[source: {chunk.metadata.doc_id}, {chunk.metadata.section}]"


def body_lines(chunk: RetrievedChunk) -> list[str]:
    """Chunk text without its context header line."""
    lines = [ln.strip() for ln in chunk.text.splitlines() if ln.strip()]
    return lines[1:] if len(lines) > 1 else lines


def fee_text(chunk: RetrievedChunk) -> str | None:
    match = _FEE_IN_HEADER.search(chunk.text.splitlines()[0] if chunk.text else "")
    return match.group(1) if match else None


def money(amount: int) -> str:
    return f"Rs. {amount:,}"


class AnswerComposer:
    def __init__(self, messages: Messages) -> None:
        self.m = messages

    def compose(self, ctx: GenerationContext) -> ComposedAnswer:
        if ctx.intent == Intent.COURSE_RECOMMENDATION and ctx.plan is not None:
            return self._recommendation(ctx, ctx.plan)
        if ctx.intent == Intent.FEES_AND_PAYMENT:
            fees = self._fees(ctx)
            if fees.text:
                return fees
        if ctx.intent == Intent.MENTOR_INFO:
            mentors = self._mentors(ctx)
            if mentors.text:
                return mentors
        return self._from_evidence(ctx)

    # ------------------------------------------------------------------ intents

    def _chunk_for(self, ctx: GenerationContext, offer: CourseOffer) -> RetrievedChunk | None:
        return next((c for c in ctx.evidence if c.id == offer.chunk_id), None)

    def _offer_line(self, ctx: GenerationContext, offer: CourseOffer) -> str:
        chunk = self._chunk_for(ctx, offer)
        fee = (fee_text(chunk) if chunk else None) or (money(offer.fee_pkr) if offer.fee_pkr else None)
        fee = fee or self.m.get("fee_unpublished", ctx.language)
        parts = [f"{offer.title}: {fee}"]
        if offer.mentor:
            parts.append(self.m.get("mentor_label", ctx.language, mentor=offer.mentor))
        if offer.level:
            parts.append(offer.level.value.capitalize())
        return "; ".join(parts)

    def _recommendation(self, ctx: GenerationContext, plan: RecommendationPlan) -> ComposedAnswer:
        lang = ctx.language
        lines: list[str] = []
        used: list[str] = []
        if plan.items:
            intro_key = "plan_intro_budget" if plan.budget_pkr else "plan_intro"
            lines.append(self.m.get(intro_key, lang, budget=money(plan.budget_pkr) if plan.budget_pkr else ""))
            for item in plan.items:
                reason_text = ", ".join(self.m.get(f"reason_{r.value}", lang) for r in item.reasons)
                line = f"{item.order}. {self._offer_line(ctx, item.offer)}"
                lines.append(f"{line} ({reason_text})" if reason_text else line)
                used.append(item.offer.chunk_id)
            if plan.budget_pkr is not None and plan.total_now_pkr is not None:
                lines.append(self.m.get("plan_total_now", lang, total=money(plan.total_now_pkr)))
            elif plan.budget_pkr is None and plan.total_all_pkr is not None and len(plan.items) > 1:
                lines.append(self.m.get("plan_total_all", lang, total=money(plan.total_all_pkr)))
            if plan.unknown_fee:
                lines.append(self.m.get("plan_unknown_fee", lang, courses=", ".join(o.title for o in plan.unknown_fee)))
            for mentor, slugs in plan.mentor_bundles:
                titles = [i.offer.title for i in plan.items if i.offer.slug in slugs]
                lines.append(self.m.get("plan_bundle", lang, mentor=mentor, courses=" and ".join(titles)))
            for nxt in plan.next_steps[:2]:
                lines.append(self.m.get("plan_next_step", lang, course=nxt.title))
                used.append(nxt.chunk_id)
        else:
            lines.append(self.m.get("plan_none", lang))
        if plan.over_budget:
            cheapest = min(plan.over_budget, key=lambda o: o.fee_pkr or 0)
            lines.append(
                self.m.get(
                    "plan_over_budget",
                    lang,
                    courses=", ".join(f"{o.title} ({money(o.fee_pkr)})" for o in plan.over_budget if o.fee_pkr),
                    cheapest=cheapest.title,
                )
            )
            used.extend(o.chunk_id for o in plan.over_budget)
        if not plan.items and not plan.over_budget:
            overview = next((c for c in ctx.evidence if c.metadata.doc_type == DocType.OVERVIEW), None)
            if overview:
                lines.extend(ln for ln in body_lines(overview) if ln.startswith("- "))
                used.append(overview.id)
        return ComposedAnswer("\n".join(lines), tuple(dict.fromkeys(used)))

    def _fees(self, ctx: GenerationContext) -> ComposedAnswer:
        offers = [o for c in ctx.evidence if (o := CourseOffer.from_chunk(c))]
        if ctx.mentioned:
            offers = [o for o in offers if o.slug in ctx.mentioned] or offers
        offers = list({o.slug: o for o in offers}.values())
        if not offers:
            overview = next((c for c in ctx.evidence if c.metadata.doc_type == DocType.OVERVIEW), None)
            if overview is None:
                return ComposedAnswer("", ())
            lines = [ln for ln in body_lines(overview) if not ln.startswith("The catalogue lists")]
            return ComposedAnswer("\n".join(lines), (overview.id,))
        lines = [self.m.get("fees_intro", ctx.language)]
        lines.extend(f"- {self._offer_line(ctx, o)}" for o in offers[:4])
        return ComposedAnswer("\n".join(lines), tuple(o.chunk_id for o in offers[:4]))

    def _mentors(self, ctx: GenerationContext) -> ComposedAnswer:
        mentor_chunks = [c for c in ctx.evidence if c.metadata.doc_type == DocType.MENTOR]
        if ctx.mentioned:
            offers = [o for c in ctx.evidence if (o := CourseOffer.from_chunk(c)) and o.slug in ctx.mentioned]
            names = {o.mentor for o in offers if o.mentor}
            mentor_chunks = [c for c in mentor_chunks if c.metadata.mentor in names] or mentor_chunks
        if not mentor_chunks:
            return ComposedAnswer("", ())
        chosen = mentor_chunks[:2]
        lines = [c.text.splitlines()[0].replace(" | ", "; ") for c in chosen]
        lines.extend(ln for ln in body_lines(chosen[0]) if not ln.startswith("- "))
        return ComposedAnswer("\n".join(lines), tuple(c.id for c in chosen))

    def _from_evidence(self, ctx: GenerationContext) -> ComposedAnswer:
        if not ctx.evidence:
            return ComposedAnswer("", ())
        chunks: Sequence[RetrievedChunk] = ctx.evidence
        # Prefer the section that answers this kind of question (retrieval may rank e.g. the FAQ, which
        # merely *lists* "How do I enroll?", above the Enrollment section that answers it).
        wanted = _INTENT_SECTIONS.get(ctx.intent)
        if wanted:
            preferred = [c for c in chunks if wanted(c)]
            chunks = preferred + [c for c in chunks if c not in preferred]
        if ctx.mentioned:
            preferred = [c for c in chunks if c.metadata.course_slug in ctx.mentioned]
            chunks = preferred + [c for c in chunks if c not in preferred]
        top = chunks[0]
        lines = body_lines(top)[:7]
        used = [top.id]
        # Batch questions: surface every evidence line about open batches / late joining.
        if ctx.intent == Intent.ENROLLMENT:
            for chunk in chunks[1:]:
                extra = [ln for ln in body_lines(chunk) if re.search(r"late joining|open batch", ln, re.IGNORECASE)]
                if extra:
                    lines.extend(extra)
                    used.append(chunk.id)
        return ComposedAnswer("\n".join(lines), tuple(dict.fromkeys(used)))
