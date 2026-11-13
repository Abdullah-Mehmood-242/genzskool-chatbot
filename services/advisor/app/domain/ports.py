"""Advisor ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from app.domain.compose import GenerationContext
from app.domain.models import Session
from common.contracts import SearchRequest, SearchResponse


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: tuple[tuple[str, str], ...]  # (role, content), alternating, ending with "user"
    # Structured view of the same request. Real providers ignore it; the offline fake LLM composes
    # its answer from it (so tests and CI need no keys and stay deterministic).
    context: GenerationContext | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    refused: bool = False  # the provider declined (e.g. stop_reason == "refusal")
    cited: tuple[str, ...] = ()  # chunk ids the answer used, when the provider knows (the fake does)
    extras: dict[str, str] = field(default_factory=dict)


class LLMClient(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Raises ProviderError / TransientProviderError / UpstreamTimeout on failure."""


class RetrievalClient(ABC):
    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Raises UpstreamError / UpstreamTimeout / CircuitOpen when retrieval is unavailable."""


class SessionRepository(ABC):
    @abstractmethod
    async def get(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def save(self, session: Session) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """True if a session existed."""


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime: ...
