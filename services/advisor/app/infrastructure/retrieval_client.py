"""HTTP client for the retrieval service: timeouts, retry with jitter, circuit breaker, request-id propagation."""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from app.domain.ports import RetrievalClient
from common.contracts import SearchRequest, SearchResponse
from common.errors import UpstreamError, UpstreamTimeout
from common.logging import outbound_headers
from common.resilience import CircuitBreaker, RetryPolicy, retry_async


class _Transient(UpstreamError):
    """A failure worth retrying (network, 5xx)."""


class HttpRetrievalClient(RetrievalClient):
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 8.0,
        retry: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport)
        self.retry = retry or RetryPolicy(
            attempts=2, base_delay_s=0.2, max_delay_s=1.0, retry_on=(_Transient, UpstreamTimeout)
        )
        self.breaker = breaker or CircuitBreaker("retrieval", failure_threshold=5, reset_after_s=15)

    async def _once(self, request: SearchRequest) -> SearchResponse:
        try:
            resp = await self.client.post(
                "/v1/search", json=request.model_dump(mode="json", exclude_none=True), headers=outbound_headers()
            )
        except httpx.TimeoutException:
            raise UpstreamTimeout("Retrieval timed out.") from None
        except httpx.TransportError as exc:
            raise _Transient(f"Retrieval unreachable ({type(exc).__name__}).") from None
        if resp.status_code >= 500:
            raise _Transient(f"Retrieval returned {resp.status_code}.")
        if resp.status_code >= 400:
            raise UpstreamError(f"Retrieval rejected the request ({resp.status_code}).")
        try:
            return SearchResponse.model_validate_json(resp.content)
        except ValidationError:
            raise UpstreamError("Retrieval returned an invalid response.") from None

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self.breaker.call(lambda: retry_async(lambda: self._once(request), self.retry))

    async def healthy(self) -> bool:
        try:
            return (await self.client.get("/healthz", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self.client.aclose()
