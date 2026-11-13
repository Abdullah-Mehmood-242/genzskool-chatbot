"""Understanding the learner: language, profile facts and intent. Deterministic, explainable rules."""

from __future__ import annotations

import re
import unicodedata

from app.domain.models import AdvisorRules, Experience, Language, LearnerProfile
from common.contracts import Intent

# ------------------------------------------------------------------------------------- language

_ROMAN_URDU = frozenset(
    [
        "mujhe",
        "mujh",
        "mera",
        "meri",
        "mere",
        "hai",
        "hain",
        "kya",
        "kia",
        "kaise",
        "kese",
        "kitne",
        "kitni",
        "kitna",
        "paise",
        "chahta",
        "chahti",
        "chahiye",
        "karna",
        "karni",
        "seekhna",
        "seekhni",
        "sikhna",
        "sikhni",
        "sakta",
        "sakti",
        "kaun",
        "sikhata",
        "sikhati",
        "aap",
        "nahi",
        "nahin",
        "haan",
        "batao",
        "bataen",
        "kab",
        "kahan",
        "kyun",
        "hum",
        "humein",
        "mein",
        "ke",
        "ki",
        "ka",
        "ko",
        "se",
        "aur",
        "ya",
        "bhi",
        "wala",
        "wali",
        "ghar",
        "kaam",
    ]
)
_URDU_SCRIPT = re.compile(r"[؀-ۿ]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", normalize(text))


def detect_language(text: str) -> Language:
    if _URDU_SCRIPT.search(text):
        return Language.URDU
    tokens = words(text)
    hits = sum(1 for t in tokens if t in _ROMAN_URDU)
    # Two markers, or one in a very short message ("kitne?"), is a reliable Roman Urdu signal.
    if hits >= 2 or (hits == 1 and len(tokens) <= 3 and tokens and tokens[0] in _ROMAN_URDU):
        return Language.ROMAN_URDU
    return Language.EN


# ------------------------------------------------------------------------------------- profile

_MULTIPLIERS = {"k": 1_000, "thousand": 1_000, "hazar": 1_000, "hazaar": 1_000, "lakh": 100_000, "lac": 100_000}
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_MULT = r"(k|thousand|hazar|hazaar|lakh|lac)"
_CUR = r"(?:rs\.?|pkr|rupees?)"
# Each pattern captures (number, optional multiplier). A bare number never counts as a budget.
_BUDGET_PATTERNS = [
    re.compile(_CUR + r"\s*" + _NUM + r"\s*" + _MULT + r"?\b"),  # Rs. 30,000 / PKR 30k
    re.compile(_NUM + r"\s*" + _MULT + r"\b"),  # 30k / 30 hazar / 1.5 lakh
    re.compile(r"budget\D{0,20}?" + _NUM + r"\s*" + _MULT + r"?\b"),  # budget is 30000
    re.compile(_NUM + r"\s*" + _MULT + r"?\s*" + _CUR + r"(?![a-z])"),  # 30000 rupees
]
_HOURS = re.compile(r"(\d{1,2})\s*(?:hours?|hrs?|ghante|ghanta)\s*(?:a|per|every|har|in a)?\s*(?:week|hafta|hafte)")
_NO_EXPERIENCE = (
    "no experience",
    "zero experience",
    "without experience",
    "beginner",
    "fresher",
    "new to this",
    "never done",
    "complete beginner",
    "kuch nahi aata",
    "pehli baar",
    "no skills",
    "starting from zero",
    "from scratch",
)
_EXPERIENCED = (
    "some experience",
    "experienced",
    "already know",
    "i know the basics",
    "intermediate",
    "advanced",
    "working as",
    "years of experience",
)
_BATCH = (
    ("morning", "morning"),
    ("subah", "morning"),
    ("evening", "evening"),
    ("shaam", "evening"),
    ("weekend", "weekend"),
    ("weekends", "weekend"),
    ("saturday", "weekend"),
    ("sunday", "weekend"),
)
_DEVICE = (
    ("laptop", "laptop"),
    ("computer", "laptop"),
    ("pc", "laptop"),
    ("mobile only", "mobile"),
    ("only a phone", "mobile"),
    ("only mobile", "mobile"),
    ("smartphone", "mobile"),
)


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def parse_amount(number: str, multiplier: str | None) -> int | None:
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return None
    if multiplier:
        value *= _MULTIPLIERS[multiplier]
    return int(value) if value >= 1 else None


def extract_budget(text: str) -> int | None:
    t = normalize(text)
    for pattern in _BUDGET_PATTERNS:
        for match in pattern.finditer(t):
            amount = parse_amount(match.group(1), match.group(2))
            if amount is not None and amount >= 1_000:  # nothing here costs less than Rs. 1,000
                return amount
    return None


class ProfileExtractor:
    def __init__(self, rules: AdvisorRules) -> None:
        self.rules = rules

    def interests(self, text: str) -> tuple[str, ...]:
        t = normalize(text)
        found: list[tuple[int, str]] = []
        for slug, keywords in self.rules.interests:
            positions = [
                m.start() for k in keywords for m in re.finditer(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", t)
            ]
            if positions:
                found.append((min(positions), slug))
        if not found:
            for word, slugs in self.rules.ambiguous_interests:
                match = re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", t)
                if match:
                    found.extend((match.start(), s) for s in slugs)
        # Order by first mention; ties (an ambiguous word) keep the configured order (stable sort).
        return tuple(dict.fromkeys(slug for _, slug in sorted(found, key=lambda f: f[0])))

    def goal(self, text: str) -> str | None:
        t = normalize(text)
        for goal, keywords in self.rules.goals:
            if any(_has(t, k) for k in keywords):
                return goal
        return None

    def extract(self, text: str) -> LearnerProfile:
        t = normalize(text)
        experience = None
        if any(_has(t, p) for p in _NO_EXPERIENCE):
            experience = Experience.NONE
        elif any(_has(t, p) for p in _EXPERIENCED):
            experience = Experience.EXPERIENCED
        hours = _HOURS.search(t)
        return LearnerProfile(
            goal=self.goal(t),
            experience_level=experience,
            budget_pkr=extract_budget(t),
            weekly_hours=int(hours.group(1)) if hours else None,
            device=next((v for k, v in _DEVICE if _has(t, k)), None),
            preferred_batch_time=next((v for k, v in _BATCH if _has(t, k)), None),
            interests=self.interests(t),
            language=detect_language(text),
        )


# ------------------------------------------------------------------------------------- intent

# Prompt-injection / exfiltration attempts. Checked first; matched messages never reach the LLM.
_INJECTION = [
    re.compile(p)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instruction|instructions|rules|prompt|above|previous|prior)\b",
        r"\b(system|hidden|initial|developer)\s+(prompt|message|instructions?)\b",
        r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,30}"
        r"\b(prompt|instructions|rules|configuration|api key|keys?|secret|password|token)\b",
        r"\b(api[ _-]?key|secret key|access token|password|credentials)\b",
        r"\byou are now\b|\bact as\b|\bpretend (to be|you are)\b|\bdeveloper mode\b|\bjailbreak\b|\bdan mode\b",
        r"<\s*/?\s*(system|evidence|user_message|instructions)\s*>",
    )
]
_HARMFUL = [
    re.compile(p)
    for p in (
        r"\b(hack|hacking|crack|phishing|malware|ddos|carding|stolen (cards?|accounts?))\b",
        r"\b(fake (reviews|certificates?|degrees?))\b",
    )
]
# Regulated advice the advisor must not give (spec: no legal/medical/financial advice).
_ADVICE = [
    re.compile(p)
    for p in (
        r"\b(legal advice|lawyer|lawsuit|sue\b|court case)",
        r"\b(medical advice|diagnos|medicine for|symptoms?|doctor)\b",
        r"\b(invest(ment)?s? advice|should i invest|stock market|crypto(currency)?|forex|tax advice|loan)\b",
    )
]
_GREETING = frozenset(
    [
        "hi",
        "hello",
        "hey",
        "salam",
        "assalam",
        "assalamualaikum",
        "alaikum",
        "walaikum",
        "aoa",
        "thanks",
        "thank",
        "thx",
        "ok",
        "okay",
        "bye",
        "good",
        "morning",
        "evening",
        "shukriya",
        "jazakallah",
        "great",
        "nice",
        "cool",
    ]
)
_GREETING_FILLER = frozenset(
    ["you", "how", "are", "u", "very", "much", "so", "a", "lot", "sir", "bro", "dear", "o", "and"]
)
_RECOMMEND = (
    "recommend",
    "suggest",
    "which course",
    "what course",
    "which one",
    "should i",
    "where do i start",
    "where to start",
    "where should i start",
    "how do i start",
    "best course",
    "best for me",
    "fit",
    "suit",
    "konsa",
    "kaunsa",
    "kon sa",
    "behtar",
    "start kahan",
    "what should",
    "help me choose",
    "career",
    "earn",
    "kamana",
)
_CONTACT = (
    "whatsapp",
    "phone",
    "contact",
    "address",
    "location",
    "office",
    "email",
    "call you",
    "number",
    "located",
    "visit",
    "advisor hours",
    "timings of office",
    "kahan hai",
    "kahan ho",
)
_CERTIFICATE = ("certificate", "certification", "certified", "sanad")
_ENROLL = (
    "enroll",
    "enrol",
    "admission",
    "register",
    "sign up",
    "signup",
    "join",
    "seat",
    "batch",
    "already started",
    "late joining",
    "dakhla",
)
_MENTOR = (
    "who teaches",
    "who will teach",
    "mentor",
    "teacher",
    "instructor",
    "trainer",
    "kaun sikhata",
    "kaun sikhati",
    "kon sikhata",
    "who is ",
)
_FEES = (
    "fee",
    "fees",
    "how much",
    "price",
    "cost",
    "charges",
    "kitne",
    "kitni",
    "paise",
    "instalment",
    "installment",
    "payment",
    "pay",
    "afford",
    "rupees",
    "qist",
)


class IntentClassifier:
    def __init__(self, rules: AdvisorRules, extractor: ProfileExtractor) -> None:
        self.rules = rules
        self.extractor = extractor

    @staticmethod
    def is_injection(text: str) -> bool:
        t = normalize(text)
        return any(p.search(t) for p in _INJECTION)

    @staticmethod
    def is_advice_request(text: str) -> bool:
        """Legal, medical or financial advice: refused with its own wording."""
        t = normalize(text)
        return any(p.search(t) for p in _ADVICE)

    def is_out_of_scope_topic(self, text: str) -> bool:
        t = normalize(text)
        return any(_has(t, topic) for topic in self.rules.out_of_scope_topics) and not self.extractor.interests(t)

    def classify(self, text: str) -> Intent:
        t = normalize(text)
        tokens = words(t)
        if self.is_injection(t) or any(p.search(t) for p in _HARMFUL):
            return Intent.UNSAFE
        if self.is_advice_request(t) or self.is_out_of_scope_topic(t):
            return Intent.OUT_OF_SCOPE
        if (
            tokens
            and len(tokens) <= 6
            and any(w in _GREETING for w in tokens)
            and all(w in _GREETING or w in _GREETING_FILLER for w in tokens)
        ):
            return Intent.CHIT_CHAT
        interests = self.extractor.interests(t)

        def mentions(options: tuple[str, ...]) -> bool:
            # Plural forms count too ("instalments", "mentors", "certificates").
            return any(_has(t, o) or _has(t, o + "s") for o in options)

        choosing_between = len(interests) >= 2 and (_has(t, "or") or _has(t, "ya") or _has(t, "vs"))
        if choosing_between or mentions(_RECOMMEND):
            return Intent.COURSE_RECOMMENDATION
        if mentions(_CONTACT):
            return Intent.CONTACT
        if mentions(_CERTIFICATE):
            return Intent.CERTIFICATE
        if mentions(_MENTOR):
            return Intent.MENTOR_INFO
        if mentions(_ENROLL):
            return Intent.ENROLLMENT
        if mentions(_FEES):
            return Intent.FEES_AND_PAYMENT
        if interests and extract_budget(t):
            return Intent.COURSE_RECOMMENDATION  # a course plus a budget asks for a decision
        if interests:
            return Intent.COURSE_INFO
        if self.extractor.goal(t) or extract_budget(t):
            return Intent.COURSE_RECOMMENDATION
        return Intent.COURSE_INFO
