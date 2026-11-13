"""Shared fixtures for advisor tests: real config, real KB chunks (as retrieval results), fakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.config import AdvisorFile
from app.domain.models import AdvisorRules, CourseOffer
from common.config import load_yaml_model
from common.contracts import Chunk, RetrievedChunk

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "config"
PROMPTS_DIR = REPO / "prompts"
FIXTURE = REPO / "tests" / "fixtures" / "kb" / "genzskool_knowledge_base.v1.chunks.jsonl"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIG_DIR", str(CONFIG_DIR))
    monkeypatch.setenv("PROMPTS_DIR", str(PROMPTS_DIR))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("SESSION_STORE", "memory")
    monkeypatch.setenv("RETRIEVAL_URL", "http://retrieval.test")


def load_rules() -> AdvisorRules:
    return load_yaml_model(CONFIG_DIR / "advisor.yaml", AdvisorFile).to_rules()


@pytest.fixture(scope="session")
def rules() -> AdvisorRules:
    return load_rules()


def kb_chunks() -> list[RetrievedChunk]:
    chunks = [Chunk.model_validate_json(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    return [RetrievedChunk(id=c.id, text=c.text, score=0.9, metadata=c.metadata) for c in chunks]


@pytest.fixture(scope="session")
def kb() -> list[RetrievedChunk]:
    return kb_chunks()


@pytest.fixture(scope="session")
def offers(kb: list[RetrievedChunk]) -> dict[str, CourseOffer]:
    return {o.slug: o for c in kb if (o := CourseOffer.from_chunk(c))}


def by_section(kb: list[RetrievedChunk], prefix: str) -> RetrievedChunk:
    return next(c for c in kb if c.metadata.section.startswith(prefix))


# ------------------------------------------------------------------------------------- doubles

import re  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402

from app.domain.ports import Clock, LLMClient, LLMRequest, LLMResponse, RetrievalClient  # noqa: E402
from common.contracts import SearchRequest, SearchResponse  # noqa: E402
from common.errors import AppError, UpstreamError  # noqa: E402


class StubRetrieval(RetrievalClient):
    """Filters and ranks the real KB fixture the way retrieval does (metadata filters + word overlap)."""

    def __init__(
        self, chunks: Sequence[RetrievedChunk] | None = None, *, fail: AppError | None = None, empty: bool = False
    ) -> None:
        self.chunks = list(chunks) if chunks is not None else kb_chunks()
        self.fail = fail
        self.empty = empty
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        if self.fail:
            raise self.fail
        return self.respond(request)

    def respond(self, request: SearchRequest) -> SearchResponse:
        if self.empty:
            return SearchResponse(chunks=[])
        f = request.filters
        words = set(re.findall(r"[a-z0-9]+", request.query.lower())) - {"the", "a", "is", "i", "do", "how", "what"}
        scored = []
        for c in self.chunks:
            m = c.metadata
            if f and f.doc_types is not None and m.doc_type not in f.doc_types:
                continue
            if f and f.course_slugs is not None and m.course_slug not in f.course_slugs:
                continue
            overlap = len(words & set(re.findall(r"[a-z0-9]+", c.text.lower())))
            if overlap or f:
                scored.append((overlap, c))
        scored.sort(key=lambda p: -p[0])
        return SearchResponse(chunks=[c for _, c in scored[: request.top_k or 5]])


class ScriptedLLM(LLMClient):
    """Returns scripted drafts in order (str = answer, Exception = raise, None = refusal); records requests."""

    def __init__(self, *script: str | Exception | None, name: str = "scripted") -> None:
        self.script = list(script)
        self.requests: list[LLMRequest] = []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        item = self.script.pop(0) if self.script else "No further script."
        if isinstance(item, Exception):
            raise item
        if item is None:
            return LLMResponse(text="", model="scripted", stop_reason="refusal", refused=True)
        return LLMResponse(text=item, model="scripted", input_tokens=100, output_tokens=20, stop_reason="end_turn")


class ManualClock(Clock):
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


RETRIEVAL_DOWN = UpstreamError("retrieval down")
