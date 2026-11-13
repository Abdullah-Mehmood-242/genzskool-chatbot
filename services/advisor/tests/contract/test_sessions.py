"""SessionRepository port contract: in-memory and Redis adapters behave alike."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import ManualClock

from app.domain.models import Experience, Language, LearnerProfile, Session, Turn
from app.domain.ports import SessionRepository
from app.infrastructure.sessions import (
    InMemorySessionRepository,
    RedisSessionRepository,
    from_json,
    to_json,
)
from common.errors import ConfigurationError


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}

    async def get(self, key: str) -> Any:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> Any:
        self.data[key] = value
        self.ttl[key] = ex
        return True

    async def delete(self, key: str) -> Any:
        return 1 if self.data.pop(key, None) is not None else 0


FULL = Session(
    session_id="s1",
    profile=LearnerProfile(
        goal="earn_online",
        experience_level=Experience.NONE,
        budget_pkr=30000,
        weekly_hours=10,
        device="laptop",
        preferred_batch_time="evening",
        interests=("shopify", "ebay"),
        language=Language.ROMAN_URDU,
    ),
    turns=(Turn("user", "kitne paise? شاپیفائی"), Turn("assistant", "Rs. 30,000")),
    asked=frozenset({"clarify_budget"}),
    created_at=ManualClock().now(),
    updated_at=ManualClock().now(),
)

FACTORIES: dict[str, Callable[[], SessionRepository]] = {
    "memory": lambda: InMemorySessionRepository(ttl_s=3600, max_sessions=100, clock=ManualClock()),
    "redis": lambda: RedisSessionRepository(FakeRedis(), ttl_s=3600),
}


@pytest.fixture(params=sorted(FACTORIES))
def repo(request: pytest.FixtureRequest) -> SessionRepository:
    return FACTORIES[request.param]()


async def test_contract_missing_is_none(repo: SessionRepository) -> None:
    assert await repo.get("nope") is None


async def test_contract_round_trip_preserves_everything(repo: SessionRepository) -> None:
    await repo.save(FULL)
    assert await repo.get("s1") == FULL


async def test_contract_overwrite(repo: SessionRepository) -> None:
    await repo.save(FULL)
    await repo.save(Session(session_id="s1"))
    got = await repo.get("s1")
    assert got is not None
    assert got.turns == ()


async def test_contract_delete(repo: SessionRepository) -> None:
    await repo.save(FULL)
    assert await repo.delete("s1") is True
    assert await repo.get("s1") is None
    assert await repo.delete("s1") is False


async def test_contract_sessions_are_independent(repo: SessionRepository) -> None:
    await repo.save(FULL)
    await repo.save(Session(session_id="other"))
    await repo.delete("other")
    assert await repo.get("s1") == FULL


def test_json_round_trip_with_empty_fields() -> None:
    empty = Session(session_id="x")
    assert from_json(to_json(empty)) == empty
    assert from_json(to_json(FULL).encode()) == FULL


async def test_memory_ttl_expiry() -> None:
    clock = ManualClock()
    repo = InMemorySessionRepository(ttl_s=60, max_sessions=10, clock=clock)
    await repo.save(FULL)
    clock.advance(59)
    assert await repo.get("s1") is not None
    clock.advance(2)
    assert await repo.get("s1") is None
    assert len(repo) == 0


async def test_memory_saving_refreshes_ttl() -> None:
    clock = ManualClock()
    repo = InMemorySessionRepository(ttl_s=60, max_sessions=10, clock=clock)
    await repo.save(FULL)
    clock.advance(50)
    await repo.save(FULL)
    clock.advance(50)
    assert await repo.get("s1") is not None


async def test_memory_evicts_least_recently_saved() -> None:
    repo = InMemorySessionRepository(ttl_s=3600, max_sessions=2, clock=ManualClock())
    for sid in ("a", "b", "c"):
        await repo.save(Session(session_id=sid))
    assert await repo.get("a") is None
    assert await repo.get("b") is not None
    assert len(repo) == 2


async def test_redis_sets_ttl_and_prefixes_keys() -> None:
    fake = FakeRedis()
    await RedisSessionRepository(fake, ttl_s=900).save(FULL)
    assert list(fake.data) == ["genz:advisor:session:s1"]
    assert fake.ttl["genz:advisor:session:s1"] == 900


def test_redis_from_url_with_factory_and_without_package() -> None:
    made: list[str] = []
    repo = RedisSessionRepository.from_url(
        "redis://r:6379/0", ttl_s=60, factory=lambda url: made.append(url) or FakeRedis()
    )  # type: ignore[func-returns-value]
    assert made == ["redis://r:6379/0"]
    assert repo.ttl_s == 60
    try:
        import redis  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigurationError, match="redis"):
            RedisSessionRepository.from_url("redis://r", ttl_s=60)
