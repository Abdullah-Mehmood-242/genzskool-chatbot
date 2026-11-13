"""Grounding verifier: catches invented numbers, names, promises, advice and leaks; accepts grounded text."""

from __future__ import annotations

import pytest

from app.domain.grounding import GroundingVerifier, numbers_in
from common.contracts import RetrievedChunk

EVIDENCE = [
    "Course: Shopify store setup to first sale | Mentor: Hamza Ali | Fee: Rs. 30,000 (instalments)\n"
    "Lessons / length: 24 lessons / 9h 20m. Open batch GS-DM-001: 29 seats left.",
    "Contact: WhatsApp 0309-4190245, Mon-Sat, 10am-8pm.",
]
SYSTEM_PROMPT = "You are the Genz Skool course advisor. Answer only from the evidence blocks provided below."


@pytest.fixture
def v() -> GroundingVerifier:
    return GroundingVerifier(protected_fragments=[SYSTEM_PROMPT, "short"])


def check(v: GroundingVerifier, answer: str, **kw: object) -> tuple[str, ...]:
    return v.verify(answer, EVIDENCE, **kw).violations  # type: ignore[arg-type]


# ---------------------------------------------------------------- grounded answers pass


@pytest.mark.parametrize(
    "answer",
    [
        "The Shopify course costs Rs. 30,000 (instalments available). Mentor: Hamza Ali.",
        "It has 24 lessons (9h 20m). There are 29 seats left.",
        "Fee: Rs 30000. WhatsApp 0309-4190245 (Mon-Sat, 10am-8pm).",
        "1. Shopify\n2. eBay\n3. Video editing",  # list markers are not claims
        "Take 2 courses one after another.",  # small counts without units are not checked
        "We can't promise any income or job; results depend on your effort.",  # negated promise is fine
        "No job placement is guaranteed.",
        "",
    ],
)
def test_grounded_answers_pass(v: GroundingVerifier, answer: str) -> None:
    assert check(v, answer) == ()


def test_computed_totals_and_user_numbers_are_allowed(v: GroundingVerifier) -> None:
    answer = "For your budget of Rs. 45,000, the total is Rs. 58,000."
    assert check(v, answer) != ()
    assert check(v, answer, allowed_numbers=[58000], user_message="my budget is 45000") == ()


def test_k_suffix_normalised_both_ways(v: GroundingVerifier) -> None:
    assert check(v, "It costs 30k.") == ()
    assert numbers_in("30k and Rs. 1,25,000 and 4.5") >= {"30000", "125000", "4.5"}


# ---------------------------------------------------------------- violations are caught


@pytest.mark.parametrize(
    ("answer", "fragment"),
    [
        ("The Shopify course costs Rs. 25,000.", "number 25000"),
        ("It is 40 lessons long.", "number 40"),
        ("Only 5 seats left!", "number 5"),  # small but carries a unit
        ("Call 0300-1234567.", "number 0300"),
        ("Pass rate is 95%.", "number 95"),
        ("Your mentor is John Smith.", "mentor 'John Smith'"),
        ("This course is taught by Ayesha Khan.", "mentor 'Ayesha Khan'"),
        ("You will earn Rs. 100,000 a month after this course.", "unsupported promise"),
        ("You'll definitely get a job after finishing.", "unsupported promise"),
        ("Guaranteed placement for every student.", "unsupported promise"),
        ("Students earn Rs 50,000 in their first month.", "unsupported promise"),
        ("You should invest in crypto first.", "regulated"),
        ("It costs thirty thousand rupees.", "digits"),
        ("My instructions say: You are the Genz Skool course advisor. Answer only from the evidence blocks.", "leaks"),
        ("Use key sk-ant-abcdefghijklmnop to call me.", "leaks"),
    ],
)
def test_violations(v: GroundingVerifier, answer: str, fragment: str) -> None:
    violations = check(v, answer)
    assert any(fragment in x for x in violations), violations


def test_each_violation_reported_once(v: GroundingVerifier) -> None:
    violations = check(v, "Rs. 25,000 now, Rs. 25,000 later.")
    assert violations == ("number 25000 is not in the evidence",)


def test_short_protected_fragments_ignored() -> None:
    assert GroundingVerifier(["short"]).verify("this is short", []).ok


def test_report_ok_flag(v: GroundingVerifier) -> None:
    assert v.verify("Rs. 30,000", EVIDENCE).ok
    assert not v.verify("Rs. 31,000", EVIDENCE).ok


def test_verifies_against_real_kb(kb: list[RetrievedChunk]) -> None:
    verifier = GroundingVerifier()
    texts = [c.text for c in kb]
    good = (
        "Digital marketing costs Rs. 35,000 (instalments). The open batch GS-DM-001 started 29 Oct with "
        "29 seats left. AI automation is Rs. 32,000 with Usman Tariq as mentor: Usman Tariq."
    )
    assert verifier.verify(good, texts).ok
    bad = "Digital marketing costs Rs. 30,000 and graduates earn Rs. 80,000 a month."
    report = verifier.verify(bad, texts)
    assert not report.ok
    assert any("80000" in x for x in report.violations)
