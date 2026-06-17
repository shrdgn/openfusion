"""Concurrency cap and per-key rate limiting."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import LimitsConfig, OpenFusionConfig
from openfusion.errors import OverloadedError, RateLimitError
from openfusion.limits import RequestLimiter
from openfusion.server import create_app


def test_rate_limit_blocks_after_budget() -> None:
    limiter = RequestLimiter(LimitsConfig(rate_limit_per_minute=2))
    limiter.check_rate("key")
    limiter.check_rate("key")
    with pytest.raises(RateLimitError):
        limiter.check_rate("key")
    # A different key has its own budget.
    limiter.check_rate("other")


def test_rate_unlimited_by_default() -> None:
    limiter = RequestLimiter(LimitsConfig())
    for _ in range(100):
        limiter.check_rate("key")


def test_concurrency_cap_rejects_when_full() -> None:
    limiter = RequestLimiter(LimitsConfig(max_in_flight=1))
    first = limiter.acquire()
    assert first is True
    with pytest.raises(OverloadedError):
        limiter.acquire()
    limiter.release(first)
    second = limiter.acquire()
    assert second is True
    limiter.release(second)


def test_concurrency_unlimited_by_default() -> None:
    limiter = RequestLimiter(LimitsConfig())
    # Nothing reserved when unlimited, so release is a no-op.
    assert limiter.acquire() is False
    limiter.release(False)


@pytest.mark.asyncio
async def test_server_rate_limit_returns_429(
    test_config: OpenFusionConfig, mock_router
) -> None:
    test_config.limits = LimitsConfig(rate_limit_per_minute=1)
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "solo",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]}
        first = await http_client.post("/v1/chat/completions", json=payload)
        second = await http_client.post("/v1/chat/completions", json=payload)
    await app.state.upstream_client.aclose()

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limit_exceeded"
