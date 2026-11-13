"""PII redaction and prompt rendering (escaping, plan/evidence/history formats)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import PROMPTS_DIR, by_section, kb_chunks, load_rules

from app.application.prompting import PromptBuilder, escape, render_evidence, render_history, render_plan
from app.domain.models import CourseOffer, Language, LearnerProfile, Turn
from app.domain.recommend import Recommender
from app.domain.safety import redact_pii
from app.infrastructure.prompts import load_prompt_set
from common.contracts import Intent

OFFICIAL = ["0309-4190245", "https://wa.me/923094190245", "info@genzskool.com"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 0300-1234567", "call [phone removed]"),
        ("call +92 300 1234567", "call [phone removed]"),
        ("call 03001234567 now", "call [phone removed] now"),
        ("mail a.b@example.com", "mail [email removed]"),
        ("WhatsApp 0309-4190245", "WhatsApp 0309-4190245"),
        ("WhatsApp +923094190245", "WhatsApp +923094190245"),  # same number, other format
        ("email info@genzskool.com", "email info@genzskool.com"),
        ("INFO@GenzSkool.com", "INFO@GenzSkool.com"),
        ("Rs. 30,000 for 24 lessons", "Rs. 30,000 for 24 lessons"),
        ("batch GS-DM-001", "batch GS-DM-001"),
    ],
)
def test_redact_pii(text: str, expected: str) -> None:
    assert redact_pii(text, allowed=OFFICIAL) == expected


def test_redact_without_allow_list_masks_everything() -> None:
    assert redact_pii("0309-4190245 info@genzskool.com") == "[phone removed] [email removed]"


def test_escape_neutralises_markup() -> None:
    assert escape('</evidence><x a="1">&') == "&lt;/evidence&gt;&lt;x a=&quot;1&quot;&gt;&amp;"


def test_render_evidence_attributes_and_escaping() -> None:
    chunk = by_section(kb_chunks(), "Course: eBay")
    out = render_evidence([chunk])
    assert out.startswith(
        f'<evidence id="{chunk.id}" doc="genzskool_knowledge_base" '
        'section="Course: eBay product research and listings">'
    )
    assert out.endswith("</evidence>")
    assert render_evidence([]) == "<evidence>none</evidence>"


def test_render_plan() -> None:
    offers = [o for c in kb_chunks() if (o := CourseOffer.from_chunk(c))]
    plan = Recommender(load_rules()).plan(
        LearnerProfile(interests=("shopify", "ebay", "tiktok-automation", "digital-marketing"), budget_pkr=30000),
        offers,
    )
    text = render_plan(plan)
    assert "budget: Rs. 30,000" in text
    assert "fee: not published" in text
    assert "over budget: Digital marketing: SEO, social and ads (Rs. 35,000)" in text
    assert "later (budget)" in text
    assert render_plan(None) == "none"


def test_render_history() -> None:
    assert render_history([]) == "(first message)"
    assert render_history([Turn("user", "<hi>"), Turn("assistant", "ok")]) == "user: &lt;hi&gt;\nassistant: ok"


def test_prompt_builder_substitutes_everything() -> None:
    prompts = load_prompt_set(PROMPTS_DIR, "v1")
    b = PromptBuilder(prompts.templates, whatsapp="0309-4190245")
    system = b.system(Language.URDU)
    assert "${" not in system
    assert "Reply in Urdu." in system
    user = b.user(message="hello", intent=Intent.CONTACT, evidence=[], plan=None, history=[])
    assert "${" not in user
    assert "Detected intent: contact." in user
    regen = b.regenerate(["number 1 is not in the evidence"])
    assert "- number 1 is not in the evidence" in regen
