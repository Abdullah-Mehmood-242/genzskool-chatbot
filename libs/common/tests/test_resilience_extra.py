"""Retry and circuit-breaker edge cases."""

from __future__ import annotations

import asyncio
import random

import pytest

from common.errors import CircuitOpen
from common.resilience import CircuitBreaker, RetryPolicy, retry_async


def test_delay_never_overflows_for_huge_attempts() -> None:
    policy = RetryPolicy(base_delay_s=0.1, max_delay_s=1.0)
    for attempt in (0, 31, 32, 1_000, 100_000):
        assert 0 <= policy.delay(attempt) <= 1.0


def test_first_delay_is_bounded_by_base() -> None:
    policy = RetryPolicy(base_delay_s=0.2, max_delay_s=5.0)
    rng = random.Random(1)
    assert all(policy.delay(0, rng) <= 0.2 for _ in range(200))


def test_delay_cap_grows_exponentially_until_max() -> None:
    policy = RetryPolicy(base_delay_s=0.1, max_delay_s=0.5)

    class MaxRng(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return b

    rng = MaxRng()
    assert [round(policy.delay(a, rng), 3) for a in range(5)] == [0.1, 0.2, 0.4, 0.5, 0.5]


async def test_sleeps_between_attempts_with_policy_delays() -> None:
    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    async def fail() -> None:
        raise TimeoutError

    with pytest.raises(TimeoutError):
        await retry_async(fail, RetryPolicy(attempts=4, base_delay_s=0.1, max_delay_s=0.3), sleep=fake_sleep)
    assert len(slept) == 3  # no sleep after the final attempt
    assert all(0 <= d <= 0.3 for d in slept)


async def test_on_retry_receives_the_exception() -> None:
    seen: list[tuple[int, str]] = []
    calls = {"n": 0}

    async def flaky() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("first")
        return 7

    async def no_sleep(_: float) -> None:
        return None

    result = await retry_async(flaky, sleep=no_sleep, on_retry=lambda a, e: seen.append((a, str(e))))
    assert result == 7
    assert seen == [(1, "first")]


async def test_single_attempt_means_no_retry() -> None:
    calls = {"n": 0}

    async def fail() -> None:
        calls["n"] += 1
        raise TimeoutError

    with pytest.raises(TimeoutError):
        await retry_async(fail, RetryPolicy(attempts=1))
    assert calls["n"] == 1


async def test_cancellation_is_never_retried() -> None:
    calls = {"n": 0}

    async def cancelled() -> None:
        calls["n"] += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_async(cancelled, RetryPolicy(attempts=5))
    assert calls["n"] == 1


async def test_retry_succeeds_first_time_without_sleeping() -> None:
    async def ok() -> str:
        return "ok"

    async def boom_sleep(_: float) -> None:
        raise AssertionError("must not sleep")

    assert await retry_async(ok, sleep=boom_sleep) == "ok"


# ---------- circuit breaker ----------


async def _fail() -> None:
    raise ConnectionError


async def _ok() -> str:
    return "ok"


async def test_success_resets_consecutive_failure_count() -> None:
    b = CircuitBreaker("d", failure_threshold=2)
    with pytest.raises(ConnectionError):
        await b.call(_fail)
    assert await b.call(_ok) == "ok"
    with pytest.raises(ConnectionError):
        await b.call(_fail)
    assert b.state == "closed"


async def test_threshold_of_one_opens_immediately() -> None:
    b = CircuitBreaker("d", failure_threshold=1)
    with pytest.raises(ConnectionError):
        await b.call(_fail)
    assert b.state == "open"


async def test_open_circuit_does_not_call_function_and_names_dependency() -> None:
    called = {"n": 0}

    async def track() -> None:
        called["n"] += 1

    b = CircuitBreaker("retrieval", failure_threshold=1)
    with pytest.raises(ConnectionError):
        await b.call(_fail)
    with pytest.raises(CircuitOpen) as exc:
        await b.call(track)
    assert called["n"] == 0
    assert exc.value.details == {"dependency": "retrieval"}
    assert exc.value.status_code == 503


async def test_half_open_success_closes_and_resets() -> None:
    now = {"t": 0.0}
    b = CircuitBreaker("d", failure_threshold=1, reset_after_s=5, clock=lambda: now["t"])
    with pytest.raises(ConnectionError):
        await b.call(_fail)
    now["t"] = 4.99
    assert b.state == "open"
    now["t"] = 5.0
    assert b.state == "half_open"
    assert await b.call(_ok) == "ok"
    assert b.state == "closed"
    # after recovery it takes a full threshold of failures again
    b2 = CircuitBreaker("d2", failure_threshold=2, reset_after_s=5, clock=lambda: now["t"])
    with pytest.raises(ConnectionError):
        await b2.call(_fail)
    assert b2.state == "closed"


async def test_breakers_are_independent() -> None:
    a, b = CircuitBreaker("a", failure_threshold=1), CircuitBreaker("b", failure_threshold=1)
    with pytest.raises(ConnectionError):
        await a.call(_fail)
    assert a.state == "open"
    assert b.state == "closed"


async def test_retry_inside_breaker_counts_one_failure_per_call() -> None:
    b = CircuitBreaker("d", failure_threshold=2)

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(ConnectionError):
        await b.call(lambda: retry_async(_fail, RetryPolicy(attempts=3), sleep=no_sleep))
    assert b.state == "closed"
