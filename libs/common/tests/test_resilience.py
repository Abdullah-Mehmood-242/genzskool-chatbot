from __future__ import annotations

import random

import pytest

from common.errors import CircuitOpen
from common.resilience import CircuitBreaker, RetryPolicy, retry_async


async def _no_sleep(_: float) -> None:
    return None


async def test_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError
        return "ok"

    retried: list[int] = []
    result = await retry_async(
        flaky, RetryPolicy(attempts=3), sleep=_no_sleep, on_retry=lambda a, _e: retried.append(a)
    )
    assert result == "ok"
    assert calls["n"] == 3
    assert retried == [1, 2]


async def test_retry_gives_up_and_reraises() -> None:
    async def always() -> None:
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        await retry_async(always, RetryPolicy(attempts=2), sleep=_no_sleep)


async def test_retry_does_not_retry_unlisted_errors() -> None:
    calls = {"n": 0}

    async def bad() -> None:
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await retry_async(bad, RetryPolicy(attempts=5, retry_on=(TimeoutError,)), sleep=_no_sleep)
    assert calls["n"] == 1


async def test_retry_rejects_zero_attempts() -> None:
    async def ok() -> int:
        return 1

    with pytest.raises(ValueError, match="attempts"):
        await retry_async(ok, RetryPolicy(attempts=0))


def test_jitter_is_bounded() -> None:
    policy = RetryPolicy(base_delay_s=0.1, max_delay_s=0.5)
    rng = random.Random(42)
    delays = [policy.delay(a, rng) for a in range(10) for _ in range(20)]
    assert all(0 <= d <= 0.5 for d in delays)
    assert len(set(delays)) > 50  # jittered, not constant


async def test_circuit_breaker_opens_and_recovers() -> None:
    now = {"t": 0.0}
    breaker = CircuitBreaker("dep", failure_threshold=2, reset_after_s=10, clock=lambda: now["t"])

    async def fail() -> None:
        raise ConnectionError

    async def ok() -> str:
        return "ok"

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(fail)
    assert breaker.state == "open"
    with pytest.raises(CircuitOpen):
        await breaker.call(ok)

    now["t"] = 11
    assert breaker.state == "half_open"
    with pytest.raises(ConnectionError):
        await breaker.call(fail)  # a failed probe re-opens immediately
    assert breaker.state == "open"

    now["t"] = 22
    assert await breaker.call(ok) == "ok"
    assert breaker.state == "closed"
