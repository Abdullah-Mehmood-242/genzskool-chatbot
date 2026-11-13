"""Deterministic answers: grounded by construction (verified), localised, and complete."""

from __future__ import annotations

import string
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import PROMPTS_DIR, by_section

from app.domain.compose import AnswerComposer, GenerationContext, Messages, citation, fee_text, money
from app.domain.grounding import GroundingVerifier
from app.domain.models import AdvisorRules, CourseOffer, Experience, Language, LearnerProfile, Reason
from app.domain.recommend import Recommender
from app.domain.understanding import ProfileExtractor
from common.contracts import Intent, RetrievedChunk

RAW = yaml.safe_load((PROMPTS_DIR / "v1" / "messages.yaml").read_text(encoding="utf-8"))
MESSAGES = Messages(RAW)


@pytest.fixture
def composer() -> AnswerComposer:
    return AnswerComposer(MESSAGES)


# ---------------------------------------------------------------- message templates


def test_every_template_has_english_and_roman_urdu() -> None:
    for key, variants in RAW.items():
        assert set(variants) == {"en", "roman_ur"}, key
        assert all(isinstance(v, str) and v.strip() for v in variants.values()), key


def test_every_reason_has_wording() -> None:
    for reason in Reason:
        assert f"reason_{reason.value}" in RAW


def test_placeholders_match_between_languages() -> None:
    fields = lambda t: {f for _, f, _, _ in string.Formatter().parse(t) if f}  # noqa: E731
    for key, variants in RAW.items():
        assert fields(variants["en"]) == fields(variants["roman_ur"]), key


def test_templates_never_promise_or_spell_numbers() -> None:
    verifier = GroundingVerifier()
    for key, variants in RAW.items():
        for text in variants.values():
            sample = text.format(**{f: "X" for _, f, _, _ in string.Formatter().parse(text) if f})
            violations = [v for v in verifier.verify(sample, []).violations if not v.startswith("number")]
            assert violations == [], (key, violations)


def test_messages_fallbacks() -> None:
    m = Messages({"k": {"en": "hello {name}", "roman_ur": "salam {name}"}})
    assert m.get("k", Language.EN, name="A") == "hello A"
    assert m.get("k", Language.ROMAN_URDU, name="A") == "salam A"
    assert m.get("k", Language.URDU, name="A") == "salam A"  # Urdu script -> Roman Urdu wording
    assert Messages({"k": {"en": "only en"}}).get("k", Language.ROMAN_URDU) == "only en"
    with pytest.raises(KeyError, match="missing"):
        m.get("nope")


# ---------------------------------------------------------------- helpers


def test_citation_and_fee_helpers(kb: list[RetrievedChunk]) -> None:
    shopify = by_section(kb, "Course: Shopify")
    assert citation(shopify) == "[source: genzskool_knowledge_base, Course: Shopify store setup to first sale]"
    assert fee_text(shopify) == "Rs. 30,000 (instalments)"
    assert fee_text(by_section(kb, "Mentor: Hamza")) is None
    assert money(125000) == "Rs. 125,000"


# ---------------------------------------------------------------- recommendation answers


def _ctx(
    kb: list[RetrievedChunk],
    rules: AdvisorRules,
    profile: LearnerProfile,
    intent: Intent,
    message: str = "",
    language: Language = Language.EN,
) -> GenerationContext:
    offers = [o for c in kb if (o := CourseOffer.from_chunk(c))]
    plan = Recommender(rules).plan(profile, offers) if intent == Intent.COURSE_RECOMMENDATION else None
    return GenerationContext(
        intent=intent,
        language=language,
        user_message=message,
        evidence=tuple(kb),
        plan=plan,
        profile=profile,
        mentioned=profile.interests,
    )


def test_recommendation_answer(composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules) -> None:
    profile = LearnerProfile(interests=("shopify", "tiktok-shop"), budget_pkr=30000)
    out = composer.compose(_ctx(kb, rules, profile, Intent.COURSE_RECOMMENDATION))
    assert "budget of Rs. 30,000" in out.text
    assert "1. Shopify store setup to first sale: Rs. 30,000 (instalments); mentor Hamza Ali; Beginner" in out.text
    assert "later, when budget allows" in out.text
    assert "Total for the courses that fit your budget: Rs. 30,000." in out.text
    assert len(out.used) == 2


def test_recommendation_unknown_fee_and_over_budget(
    composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules
) -> None:
    profile = LearnerProfile(interests=("tiktok-automation", "digital-marketing"), budget_pkr=30000)
    out = composer.compose(_ctx(kb, rules, profile, Intent.COURSE_RECOMMENDATION))
    assert "Not published: fees and batch dates on WhatsApp" in out.text
    assert "The fee for TikTok automation: faceless videos that earn is not published" in out.text
    assert "Above your budget: Digital marketing: SEO, social and ads (Rs. 35,000)" in out.text


def test_recommendation_with_nothing_found_lists_catalogue(
    composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules
) -> None:
    overview = [by_section(kb, "Course catalogue overview")]
    ctx = GenerationContext(
        intent=Intent.COURSE_RECOMMENDATION,
        language=Language.EN,
        user_message="",
        evidence=tuple(overview),
        plan=Recommender(rules).plan(LearnerProfile(), []),
    )
    out = composer.compose(ctx)
    assert "I couldn't find a course that matches yet." in out.text
    assert "- Shopify store setup to first sale; Fee: Rs. 30,000 (instalments)" in out.text
    assert out.used == (overview[0].id,)


def test_mentor_bundle_and_next_step(composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules) -> None:
    out = composer.compose(_ctx(kb, rules, LearnerProfile(interests=("shopify", "ebay")), Intent.COURSE_RECOMMENDATION))
    assert "Hamza Ali teaches both" in out.text
    out2 = composer.compose(
        _ctx(kb, rules, LearnerProfile(interests=("digital-marketing",)), Intent.COURSE_RECOMMENDATION)
    )
    assert "Possible next step later:" in out2.text


def test_roman_urdu_framing(composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules) -> None:
    out = composer.compose(
        _ctx(
            kb,
            rules,
            LearnerProfile(interests=("ebay",), budget_pkr=30000),
            Intent.COURSE_RECOMMENDATION,
            language=Language.ROMAN_URDU,
        )
    )
    assert "ke budget ke liye meri suggestion yeh hai" in out.text
    assert "Rs. 28,000" in out.text  # facts stay exact in every language


# ---------------------------------------------------------------- other intents


def test_fee_answer_for_mentioned_course(
    composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules
) -> None:
    out = composer.compose(_ctx(kb, rules, LearnerProfile(interests=("ebay",)), Intent.FEES_AND_PAYMENT))
    assert out.text.splitlines() == [
        "Course fees:",
        "- eBay product research and listings: Rs. 28,000 (instalments); mentor Hamza Ali; Beginner",
    ]


def test_fee_answer_without_course_uses_overview(composer: AnswerComposer, kb: list[RetrievedChunk]) -> None:
    overview = by_section(kb, "Course catalogue overview")
    ctx = GenerationContext(
        intent=Intent.FEES_AND_PAYMENT, language=Language.EN, user_message="fees?", evidence=(overview,)
    )
    out = composer.compose(ctx)
    assert "Prices are in Pakistani Rupees" in out.text
    assert "TikTok automation: faceless videos that earn; Fee: Not published" in out.text


def test_mentor_answer(composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules) -> None:
    out = composer.compose(_ctx(kb, rules, LearnerProfile(interests=("graphic-design",)), Intent.MENTOR_INFO))
    assert out.text.splitlines()[0] == "Mentor: Zara Malik; Role: Brand designer; Teaches: Graphic design"
    assert "reviews student assignments personally" in out.text


@pytest.mark.parametrize(
    ("intent", "section", "fact"),
    [
        (Intent.ENROLLMENT, "Enrollment", "No payment is taken on the website"),
        (Intent.CERTIFICATE, "FAQ and certificate", "Fiverr or Upwork"),
        (Intent.CONTACT, "Contact and location", "0309-4190245"),
        (Intent.COURSE_INFO, "Course: AI automation", "Learn: ChatGPT, Make.com, Workflows, AI tools."),
    ],
)
def test_evidence_answers(
    composer: AnswerComposer, kb: list[RetrievedChunk], intent: Intent, section: str, fact: str
) -> None:
    top = by_section(kb, section)
    ctx = GenerationContext(
        intent=intent, language=Language.EN, user_message="q", evidence=(top, *[c for c in kb if c is not top][:3])
    )
    out = composer.compose(ctx)
    assert fact in out.text
    assert out.used[0] == top.id


def test_late_joining_lines_surface_for_enrollment(composer: AnswerComposer, kb: list[RetrievedChunk]) -> None:
    ctx = GenerationContext(
        intent=Intent.ENROLLMENT,
        language=Language.EN,
        user_message="join started batch?",
        evidence=(
            by_section(kb, "Enrollment"),
            by_section(kb, "Course: Digital marketing"),
            by_section(kb, "Course: Graphic"),
        ),
    )
    out = composer.compose(ctx)
    assert "late joining allowed" in out.text
    assert len(out.used) == 3


def test_no_evidence_gives_empty_answer(composer: AnswerComposer) -> None:
    out = composer.compose(GenerationContext(intent=Intent.COURSE_INFO, language=Language.EN, user_message="x"))
    assert out == type(out)("", ())


# ---------------------------------------------------------------- the key property


PROFILES = [
    LearnerProfile(),
    LearnerProfile(goal="earn_online", experience_level=Experience.NONE),
    LearnerProfile(interests=("shopify", "tiktok-shop"), budget_pkr=30000),
    LearnerProfile(interests=("tiktok-automation", "ai-automation"), budget_pkr=20000),
    LearnerProfile(interests=("digital-marketing", "shopify", "ebay"), budget_pkr=100000),
    LearnerProfile(interests=("video-editing",), budget_pkr=5000),
    LearnerProfile(interests=("graphic-design", "video-editing"), experience_level=Experience.EXPERIENCED),
]


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: ",".join(p.interests) or "none")
@pytest.mark.parametrize(
    "intent",
    [
        Intent.COURSE_RECOMMENDATION,
        Intent.FEES_AND_PAYMENT,
        Intent.MENTOR_INFO,
        Intent.COURSE_INFO,
        Intent.ENROLLMENT,
        Intent.CERTIFICATE,
        Intent.CONTACT,
    ],
)
@pytest.mark.parametrize("language", [Language.EN, Language.ROMAN_URDU])
def test_composed_answers_are_always_grounded(
    composer: AnswerComposer,
    kb: list[RetrievedChunk],
    rules: AdvisorRules,
    profile: LearnerProfile,
    intent: Intent,
    language: Language,
) -> None:
    ctx = _ctx(kb, rules, profile, intent, language=language)
    out = composer.compose(ctx)
    allowed = ctx.plan.computed_numbers if ctx.plan else frozenset()
    report = GroundingVerifier().verify(out.text, [c.text for c in kb], allowed_numbers=allowed)
    assert report.ok, (out.text, report.violations)
    assert set(out.used) <= {c.id for c in kb}


def test_profile_extracted_from_spec_question_flows_into_answer(
    composer: AnswerComposer, kb: list[RetrievedChunk], rules: AdvisorRules
) -> None:
    profile = ProfileExtractor(rules).extract("Shopify or TikTok Shop? Budget is Rs. 30,000.")
    out = composer.compose(_ctx(kb, rules, profile, Intent.COURSE_RECOMMENDATION))
    assert out.text.splitlines()[1].startswith("1. Shopify store setup to first sale")


@pytest.mark.parametrize(
    ("intent", "ranked_first", "expected_section", "fact"),
    [
        (Intent.ENROLLMENT, "FAQ and certificate", "Enrollment", "No payment is taken on the website"),
        (Intent.CONTACT, "Enrollment", "Contact and location", "Farhan Plaza"),
        (Intent.CERTIFICATE, "How it works", "FAQ and certificate", "Will I get a certificate? Yes."),
    ],
)
def test_intent_section_is_preferred_over_retrieval_rank(
    composer: AnswerComposer,
    kb: list[RetrievedChunk],
    intent: Intent,
    ranked_first: str,
    expected_section: str,
    fact: str,
) -> None:
    """Regression (found in e2e with real retrieval): 'How do I enroll?' ranked the FAQ list first."""
    evidence = (by_section(kb, ranked_first), by_section(kb, expected_section))
    out = composer.compose(GenerationContext(intent=intent, language=Language.EN, user_message="q", evidence=evidence))
    assert fact in out.text
    assert out.used[0] == by_section(kb, expected_section).id
