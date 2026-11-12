"""Retry with exponential backoff + full jitter, and a small async circuit breaker."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from common.errors import CircuitOpen

logger = logging.getLogger("common.resilience")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full jitter: uniform(0, min(max, base * 2**attempt))."""
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** min(attempt, 32)))  # cap exponent: no overflow
        return (rng or random).uniform(0, cap)


_DEFAULT_POLICY = RetryPolicy()


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy = _DEFAULT_POLICY,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn`` up to ``policy.attempts`` times, retrying only on ``policy.retry_on``."""
    if policy.attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(policy.attempts):
        try:
            return await fn()
        except policy.retry_on as exc:
            if attempt == policy.attempts - 1:
                raise
            if on_retry:
                on_retry(attempt + 1, exc)
            logger.info("retrying", extra={"attempt": attempt + 1, "error_type": type(exc).__name__})
            await sleep(policy.delay(attempt))
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class CircuitBreaker:
    """Closed -> open after ``failure_threshold`` consecutive failures; half-open after ``reset_after_s``.

    Not shared across processes; one breaker per downstream per process.
    """

    name: str
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self.clock() - self._opened_at >= self.reset_after_s:
            return "half_open"
        return "open"

    async def call[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        if self.state == "open":
            raise CircuitOpen(details={"dependency": self.name})
        try:
            result = await fn()
        except Exception:
            self._record_failure()
            raise
        self._failures = 0
        self._opened_at = None
        return result

    def _record_failure(self) -> None:
        self._failures += 1
        if self.state == "half_open" or self._failures >= self.failure_threshold:
            self._opened_at = self.clock()
            logger.warning("circuit_opened", extra={"dependency": self.name})


__all__ = ["CircuitBreaker", "RetryPolicy", "retry_async"]
