"""Concurrency cap and per-key rate limiting."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.responses import StreamingResponse

from openfusion.config import GatewayAuthConfig, LimitsConfig, OpenFusionConfig
from openfusion.errors import OverloadedError, RateLimitError
from openfusion.limits import RequestLimiter
from openfusion.server import _attach_release, create_app


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


def test_rate_window_prunes_stale_entries() -> None:
    limiter = RequestLimiter(LimitsConfig(rate_limit_per_minute=100))
    for i in range(50):
        limiter.check_rate(f"key-{i}")
    assert len(limiter._window) == 50
    # Backdate all entries past the sliding window's two-window carry-over
    # horizon so they appear fully expired.
    limiter._window = {
        k: (v[0], v[1], v[2] - 130.0) for k, v in limiter._window.items()
    }
    # Re-checking an expired key triggers pruning of all other expired entries.
    limiter.check_rate("key-0")
    assert len(limiter._window) == 1


def test_rate_limit_sliding_window_blends_previous_window() -> None:
    """A naive fixed window resets to 0 at the boundary, letting a client
    burst another full rpm worth of requests immediately after using its
    budget just before the boundary (~2x rpm in a couple of seconds). The
    sliding-window blend must instead carry the previous window's usage
    forward so a 60s span stays close to rpm, not 2x rpm."""
    limiter = RequestLimiter(LimitsConfig(rate_limit_per_minute=4))
    key = "burst"
    now = time.monotonic()
    # The previous window used its full budget; we're now just past the
    # window boundary.
    limiter._window[key] = (0, 4, now - 60.01)
    limiter.check_rate(key)  # still allowed: previous window is barely decayed
    with pytest.raises(RateLimitError):
        # A naive fixed window would allow 3 more requests here; the blended
        # estimate is already back at the budget.
        limiter.check_rate(key)


def test_rate_window_bounded_when_keys_never_repeat() -> None:
    """Distinct, never-reused keys (e.g. rotating Bearer tokens) must not grow
    _window without bound: the expiry-based prune never fires for a key that's
    only ever seen once, so a hard LRU cap is the actual backstop."""
    from openfusion.limits import _WINDOW_MAX_KEYS

    limiter = RequestLimiter(LimitsConfig(rate_limit_per_minute=100))
    for i in range(_WINDOW_MAX_KEYS + 500):
        limiter.check_rate(f"key-{i}")
    assert len(limiter._window) == _WINDOW_MAX_KEYS
    # The most recently used key must survive eviction, not an arbitrary one.
    assert f"key-{_WINDOW_MAX_KEYS + 499}" in limiter._window


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


@pytest.mark.asyncio
async def test_server_rate_limit_not_bypassable_by_rotating_bearer_tokens(
    test_config: OpenFusionConfig, mock_router
) -> None:
    """Without a configured gateway.api_keys allowlist, any Bearer value is
    unauthenticated -- a client could otherwise mint a fresh rate-limit bucket
    per request just by sending a new token each time. All such traffic must
    share one 'anonymous' bucket instead, so rate_limit_per_minute is an actual
    cap rather than one an attacker can defeat by rotating headers."""
    test_config.limits = LimitsConfig(rate_limit_per_minute=1)
    assert not test_config.gateway.api_keys
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]}
        first = await http_client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer attacker-1"}
        )
        second = await http_client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer attacker-2"}
        )
    await app.state.upstream_client.aclose()

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_server_rate_limit_still_per_key_when_gateway_allowlist_set(
    test_config: OpenFusionConfig, mock_router
) -> None:
    """A validated gateway key still gets its own independent rate-limit budget."""
    test_config.limits = LimitsConfig(rate_limit_per_minute=1)
    test_config.gateway = GatewayAuthConfig(api_keys=["key-a", "key-b"])
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]}
        first = await http_client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer key-a"}
        )
        second_same_key = await http_client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer key-a"}
        )
        second_other_key = await http_client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer key-b"}
        )
    await app.state.upstream_client.aclose()

    assert first.status_code == 200
    assert second_same_key.status_code == 429
    assert second_other_key.status_code == 200


@pytest.mark.asyncio
async def test_server_rejects_oversized_body_with_413(
    test_config: OpenFusionConfig, mock_router
) -> None:
    test_config.limits = LimitsConfig(max_request_bytes=1024)
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {
            "model": "pass-model",
            "messages": [{"role": "user", "content": "x" * 2048}],
        }
        response = await http_client.post("/v1/chat/completions", json=payload)
    await app.state.upstream_client.aclose()

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    # The oversized body must never reach the upstream mock.
    assert mock_router.calls.call_count == 0


@pytest.mark.asyncio
async def test_max_body_size_middleware_ignores_malformed_content_length() -> None:
    """A non-integer Content-Length can't be compared to the limit, so the
    request is let through rather than raising -- it's still bounded by
    whatever actually reads the body downstream."""
    from openfusion.server import MaxBodySizeMiddleware

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = MaxBodySizeMiddleware(app, max_bytes=10)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(b"content-length", b"not-a-number")],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 200


@pytest.mark.asyncio
async def test_server_allows_body_within_limit(
    test_config: OpenFusionConfig, mock_router
) -> None:
    test_config.limits = LimitsConfig(max_request_bytes=1_000_000)
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]}
        response = await http_client.post("/v1/chat/completions", json=payload)
    await app.state.upstream_client.aclose()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_server_body_size_unlimited_by_default(
    test_config: OpenFusionConfig, mock_router
) -> None:
    assert test_config.limits.max_request_bytes == 0
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {
            "model": "pass-model",
            "messages": [{"role": "user", "content": "x" * 100_000}],
        }
        response = await http_client.post("/v1/chat/completions", json=payload)
    await app.state.upstream_client.aclose()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_server_releases_concurrency_slot_after_response(
    test_config: OpenFusionConfig, mock_router
) -> None:
    """With a concurrency cap set, a completed request frees its slot for the next one."""
    test_config.limits = LimitsConfig(max_in_flight=1)
    app = create_app(test_config)
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        payload = {"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]}
        first = await http_client.post("/v1/chat/completions", json=payload)
        # If the slot wasn't released after `first` finished, this would 503.
        second = await http_client.post("/v1/chat/completions", json=payload)
    await app.state.upstream_client.aclose()

    assert first.status_code == 200
    assert second.status_code == 200


@pytest.mark.asyncio
async def test_streaming_slot_released_even_if_response_background_never_runs() -> None:
    """A disconnected client can make Starlette skip response.background entirely --
    it raises writing the stream's final empty chunk to an already-closed socket
    *before* reaching the background-task call (see StreamingResponse.__call__). The
    concurrency slot must still come back, from the body iterator's own `finally`
    rather than from `response.background`, or `limits.max_in_flight` leaks a slot per
    disconnect until the server 503s every request.
    """
    limiter = RequestLimiter(LimitsConfig(max_in_flight=1))
    acquired = limiter.acquire()
    assert acquired is True

    async def body() -> AsyncIterator[bytes]:
        yield b"chunk-1"
        yield b"chunk-2"

    response = StreamingResponse(body())
    _attach_release(response, limiter, acquired)

    # Drain the body iterator the way Starlette's stream_response() does, but never
    # invoke response.background -- this simulates the disconnect path where
    # Starlette's own send() raises before background ever runs.
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == [b"chunk-1", b"chunk-2"]
    assert limiter._in_flight == 0
    # A slot is available again for the next request.
    assert limiter.acquire() is True
