"""Session stores (ADR-0007): in-memory (default, TTL + LRU bound) and Redis (optional, lazy import)."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from app.domain.models import Experience, Language, LearnerProfile, Session, Turn
from app.domain.ports import Clock, SessionRepository
from common.errors import ConfigurationError


def to_json(session: Session) -> str:
    p = session.profile
    return json.dumps(
        {
            "session_id": session.session_id,
            "profile": {
                "goal": p.goal,
                "experience_level": p.experience_level.value if p.experience_level else None,
                "budget_pkr": p.budget_pkr,
                "weekly_hours": p.weekly_hours,
                "device": p.device,
                "preferred_batch_time": p.preferred_batch_time,
                "interests": list(p.interests),
                "language": p.language.value,
            },
            "turns": [[t.role, t.content] for t in session.turns],
            "asked": sorted(session.asked),
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        },
        ensure_ascii=False,
    )


def from_json(raw: str | bytes) -> Session:
    d = json.loads(raw)
    p = d["profile"]
    return Session(
        session_id=d["session_id"],
        profile=LearnerProfile(
            goal=p["goal"],
            experience_level=Experience(p["experience_level"]) if p["experience_level"] else None,
            budget_pkr=p["budget_pkr"],
            weekly_hours=p["weekly_hours"],
            device=p["device"],
            preferred_batch_time=p["preferred_batch_time"],
            interests=tuple(p["interests"]),
            language=Language(p["language"]),
        ),
        turns=tuple(Turn(role, content) for role, content in d["turns"]),
        asked=frozenset(d["asked"]),
        created_at=datetime.fromisoformat(d["created_at"]) if d["created_at"] else None,
        updated_at=datetime.fromisoformat(d["updated_at"]) if d["updated_at"] else None,
    )


class InMemorySessionRepository(SessionRepository):
    """Per-process store. Sessions expire after ``ttl_s`` idle; the oldest are evicted beyond ``max_sessions``."""

    def __init__(self, *, ttl_s: float, max_sessions: int, clock: Clock) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self.clock = clock
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()  # id -> (saved_at, json)
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        return self.clock.now().timestamp()

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            item = self._data.get(session_id)
            if item is None:
                return None
            if self._now() - item[0] > self.ttl_s:
                del self._data[session_id]
                return None
            return from_json(item[1])

    async def save(self, session: Session) -> None:
        async with self._lock:
            self._data[session.session_id] = (self._now(), to_json(session))
            self._data.move_to_end(session.session_id)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._data.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._data)


class RedisLike(Protocol):
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...
    async def delete(self, key: str) -> Any: ...


class RedisSessionRepository(SessionRepository):
    """Shared store for several advisor replicas. Needs the optional `redis` package."""

    PREFIX = "genz:advisor:session:"

    def __init__(self, client: RedisLike, *, ttl_s: int) -> None:
        self.client = client
        self.ttl_s = ttl_s

    @classmethod
    def from_url(
        cls, url: str, *, ttl_s: int, factory: Callable[[str], RedisLike] | None = None
    ) -> RedisSessionRepository:
        if factory is None:
            try:
                import redis.asyncio as redis_asyncio
            except ImportError:
                raise ConfigurationError("SESSION_STORE=redis needs the optional package 'redis'.") from None
            factory = redis_asyncio.from_url
        return cls(factory(url), ttl_s=ttl_s)

    async def get(self, session_id: str) -> Session | None:
        raw = await self.client.get(self.PREFIX + session_id)
        return from_json(raw) if raw else None

    async def save(self, session: Session) -> None:
        await self.client.set(self.PREFIX + session.session_id, to_json(session), ex=self.ttl_s)

    async def delete(self, session_id: str) -> bool:
        return bool(await self.client.delete(self.PREFIX + session_id))
