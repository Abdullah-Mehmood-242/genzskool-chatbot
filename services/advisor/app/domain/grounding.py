"""verify_grounding (ADR-0006): every number, mentor name and claim in an answer must be backed by the
retrieved evidence, the deterministic plan, or the learner's own words. Pure and deterministic."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# A number token: 30,000 / 30000 / 9h / 20m / 0309-4190245 parts / 30k / 4.5
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(k\b)?", re.IGNORECASE)
# Units that make a small number a factual claim. Course/step counts are excluded: the advisor computes
# those ("take 2 courses"); they are not KB facts.
_UNIT_AFTER = re.compile(
    r"^\s*(%|h\b|m\b|hours?|hrs?|mins?|minutes?|lessons?|modules?|seats?|weeks?|months?|days?|"
    r"percent)",
    re.IGNORECASE,
)
_CURRENCY_BEFORE = re.compile(r"(rs\.?|pkr|rupees?|\$)\s*$", re.IGNORECASE)
_LIST_MARKER = re.compile(r"(?m)^\s*\d{1,2}[.)]\s")
_WORD_NUMBERS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|hundred|thousand|lakh|"
    r"hazar|hazaar)\b[\s-]*(thousand|hundred|lakh|hazar|hazaar|rupees|rs\b|pkr)",
    re.IGNORECASE,
)
_MENTOR_CUE = re.compile(
    r"(?:mentor(?:ed)?(?:\s+is)?|taught by|teacher(?:\s+is)?|instructor(?:\s+is)?|trainer(?:\s+is)?)[:\s]+"
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)"
)
# Promises the KB cannot back (spec: no income/job promises). Negated statements are allowed.
_PROMISE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(you('| wi)ll|you can|guarantee[ds]?|guaranteed|definitely|surely|certainly|assured)\b[^.!?\n]{0,60}"
        r"\b(earn|income|salary|job|placement|hired|money|clients?|profit|kama)\w*",
        r"\b(earn|make|income of|salary of)\b[^.!?\n]{0,25}(rs\.?|pkr|\$|rupees)\s*\d",
        r"\b(100%|guaranteed)\s+(job|placement|income|success)",
    )
]
_NEGATION = re.compile(
    r"\b(no|not|never|cannot|can't|can not|don't|doesn't|won't|isn't|aren't|nahi|nahin)\b", re.IGNORECASE
)
_ADVICE = re.compile(
    r"\b(legal advice|medical advice|financial advice|invest in|you should invest|tax advice)\b", re.IGNORECASE
)
_SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{10,}|api[_-]?key\s*[:=])", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    """Canonical number strings (commas removed, '30k' -> '30000')."""
    out: set[str] = set()
    for match in _NUMBER.finditer(text):
        raw = match.group(1).replace(",", "").rstrip(".")
        if match.group(2):
            try:
                out.add(str(int(float(raw) * 1000)))
            except ValueError:
                continue
        out.add(raw)
    return out


def _checked_numbers(answer: str) -> list[str]:
    """Numbers that make factual claims: >= 10, or carrying a unit/currency. List markers are ignored."""
    text = _LIST_MARKER.sub(" ", answer)
    claims: list[str] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1).replace(",", "").rstrip(".")
        value = raw
        if match.group(2):
            value = str(int(float(raw) * 1000))
        before = text[max(0, match.start() - 6) : match.start()]
        after = text[match.end() : match.end() + 12]
        try:
            magnitude = float(raw)
        except ValueError:
            continue
        if magnitude >= 10 or _UNIT_AFTER.match(after) or _CURRENCY_BEFORE.search(before):
            claims.append(value)
    return claims


@dataclass(frozen=True)
class GroundingReport:
    violations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.violations


class GroundingVerifier:
    def __init__(self, protected_fragments: Iterable[str] = ()) -> None:
        # Distinctive sentences of the system prompt: if an answer echoes one, the prompt is leaking.
        sentences = (
            part.strip().lower().rstrip(".!?:")
            for fragment in protected_fragments
            for part in re.split(r"(?<=[.!?:])\s+|\n", fragment)
        )
        self.protected = tuple(dict.fromkeys(s for s in sentences if len(s) >= 25))

    def verify(
        self,
        answer: str,
        evidence: Iterable[str],
        *,
        allowed_numbers: Iterable[int | str] = (),
        user_message: str = "",
    ) -> GroundingReport:
        evidence_text = "\n".join(evidence)
        known = numbers_in(evidence_text) | numbers_in(user_message) | {str(n) for n in allowed_numbers}
        violations: list[str] = []

        for number in dict.fromkeys(_checked_numbers(answer)):
            if number not in known:
                violations.append(f"number {number} is not in the evidence")

        for name in dict.fromkeys(m.group(1) for m in _MENTOR_CUE.finditer(answer)):
            if name not in evidence_text:
                violations.append(f"mentor '{name}' is not in the evidence")

        for sentence in re.split(r"(?<=[.!?\n])\s+", answer):
            if any(p.search(sentence) for p in _PROMISE) and not _NEGATION.search(sentence):
                violations.append(f"unsupported promise: '{sentence.strip()[:80]}'")
            if _ADVICE.search(sentence) and not _NEGATION.search(sentence):
                violations.append("gives regulated (legal/medical/financial) advice")

        if _WORD_NUMBERS.search(answer):
            violations.append("amounts must be written in digits so they can be verified")
        lowered = answer.lower()
        if _SECRET.search(answer) or any(fragment in lowered for fragment in self.protected):
            violations.append("answer leaks configuration or instructions")
        return GroundingReport(violations=tuple(dict.fromkeys(violations)))
