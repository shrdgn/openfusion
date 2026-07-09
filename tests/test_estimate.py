"""Pre-run cost estimate: unit + endpoint + pricing fetch."""

from __future__ import annotations

import time

import httpx
import pytest

from openfusion import pricing
from openfusion.config import (
    Aggregator,
    CostControlsConfig,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
)
from openfusion.estimate import build_estimate
from openfusion.server import create_app


@pytest.fixture(autouse=True)
def _clear_price_cache() -> None:
    pricing._cache.clear()
    pricing._inflight.clear()


def _config() -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.JUDGE,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        cost_controls=CostControlsConfig(panel_max_tokens=100, judge_max_tokens=200),
    )


def test_estimate_input_tokens_string_content() -> None:
    from openfusion.estimate import _estimate_input_tokens

    messages = [{"role": "user", "content": "x" * 40}]
    assert _estimate_input_tokens(messages) == 10  # 40 chars // 4


def test_estimate_input_tokens_multimodal_text_blocks() -> None:
    from openfusion.estimate import _estimate_input_tokens

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x" * 40},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]
    # Only the text block contributes; the image block is ignored.
    assert _estimate_input_tokens(messages) == 10  # 40 chars // 4


def test_estimate_input_tokens_mixed_messages() -> None:
    from openfusion.estimate import _estimate_input_tokens

    messages = [
        {"role": "system", "content": "You are helpful."},  # 16 chars
        {"role": "user", "content": [{"type": "text", "text": "a" * 20}]},  # 20 chars
    ]
    # 36 chars // 4 = 9, but max(1, ...) so 9
    assert _estimate_input_tokens(messages) == 9


def test_estimate_input_tokens_non_list_returns_zero() -> None:
    from openfusion.estimate import _estimate_input_tokens

    assert _estimate_input_tokens(None) == 0
    assert _estimate_input_tokens("bad") == 0


def test_estimate_input_tokens_empty_messages() -> None:
    from openfusion.estimate import _estimate_input_tokens

    # max(1, 0 // 4) == 1
    assert _estimate_input_tokens([]) == 1


def test_build_estimate_counts_calls_and_prices() -> None:
    prices = {
        "m1": {"prompt": 0.0, "completion": 0.0},
        "m2": {"prompt": 0.0, "completion": 0.0},
        "j": {"prompt": 0.0, "completion": 0.0},
    }
    est = build_estimate(
        {"messages": [{"role": "user", "content": "x" * 40}]}, _config(), prices
    )
    assert est["calls"] == 3  # 2 panel + judge
    assert est["input_tokens"] == 10  # 40 chars // 4
    assert est["cost_usd"] == 0.0  # all priced (at zero)


def test_build_estimate_cost_none_when_unpriced() -> None:
    est = build_estimate(
        {"messages": [{"role": "user", "content": "hi"}]}, _config(), prices={}
    )
    assert est["cost_usd"] is None  # no pricing for these models


async def test_estimate_endpoint_returns_dollars(mock_router) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "m1", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                    {"id": "m2", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                    {"id": "j", "pricing": {"prompt": "0.000003", "completion": "0.000006"}},
                ]
            },
        )
    )
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/estimate", json={"messages": [{"role": "user", "content": "hello there"}]}
        )
    await app.state.upstream_client.aclose()
    body = res.json()
    assert res.status_code == 200
    assert body["calls"] == 3
    assert body["cost_usd"] is not None and body["cost_usd"] > 0


async def test_pricing_falls_back_on_error(mock_router) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(500, json={})
    )
    prices = await pricing.get_prices("https://mock.upstream/v1")
    assert prices == {}  # error → empty, no exception


async def test_concurrent_pricing_calls_only_fetch_once(mock_router) -> None:
    """Two concurrent get_prices calls for the same URL must not both hit the upstream."""
    import asyncio

    call_count = 0

    async def _handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={
                "data": [{"id": "m1", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]
            },
        )

    mock_router.get("https://mock.upstream/v1/models").mock(side_effect=_handler)
    results = await asyncio.gather(
        pricing.get_prices("https://mock.upstream/v1"),
        pricing.get_prices("https://mock.upstream/v1"),
    )
    # Both calls must return valid (non-empty) price data.
    assert all(r is not None for r in results)
    # The second caller must have reused the in-flight state, not started a new fetch.
    assert call_count == 1, f"expected 1 upstream fetch, got {call_count}"


async def test_estimate_endpoint_requires_gateway_auth(mock_router) -> None:
    """Gateway auth is enforced on /v1/estimate when api_keys are configured."""
    from openfusion.config import GatewayAuthConfig

    cfg = _config()
    cfg.gateway = GatewayAuthConfig(api_keys=["secret"])
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # No auth header → 401
        no_auth = await client.post(
            "/v1/estimate", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        # Wrong key → 401
        wrong = await client.post(
            "/v1/estimate",
            headers={"Authorization": "Bearer wrong"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        # Mock pricing so valid auth can proceed to a 200
        mock_router.get("https://mock.upstream/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        ok = await client.post(
            "/v1/estimate",
            headers={"Authorization": "Bearer secret"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()
    assert no_auth.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200


async def test_estimate_endpoint_rejects_invalid_json() -> None:
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/estimate",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 400
    assert res.json()["error"]["type"] == "invalid_request_error"


async def test_estimate_endpoint_rejects_non_object_body() -> None:
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/estimate", json=["not", "an", "object"])
    await app.state.upstream_client.aclose()
    assert res.status_code == 400


async def test_estimate_endpoint_applies_request_override(mock_router) -> None:
    """An `openfusion` override in the body changes which models get priced."""
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "m3", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
                ]
            },
        )
    )
    cfg = _config()
    cfg.allow_request_overrides = True
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/estimate",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "openfusion": {"panel": ["m3"]},
            },
        )
    await app.state.upstream_client.aclose()
    body = res.json()
    assert res.status_code == 200
    assert "m3" in body["models"]
    assert "m1" not in body["models"] and "m2" not in body["models"]


async def test_pricing_cache_hit_skips_fetch(mock_router) -> None:
    """A valid cached entry is returned without making a second HTTP call."""
    call_count = 0

    async def _handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={"data": [{"id": "m", "pricing": {"prompt": "0.001", "completion": "0.002"}}]},
        )

    mock_router.get("https://mock.upstream/v1/models").mock(side_effect=_handler)

    # Prime the cache with one real fetch.
    await pricing.get_prices("https://mock.upstream/v1")
    assert call_count == 1

    # Second call hits the cache — no new HTTP request.
    result = await pricing.get_prices("https://mock.upstream/v1")
    assert call_count == 1
    assert "m" in result


async def test_pricing_stale_cache_triggers_refetch(mock_router) -> None:
    """An expired cache entry causes a fresh HTTP fetch."""
    call_count = 0

    async def _handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={"data": [{"id": "m", "pricing": {"prompt": "0.001", "completion": "0.002"}}]},
        )

    mock_router.get("https://mock.upstream/v1/models").mock(side_effect=_handler)

    # Inject an artificially expired entry.
    pricing._cache["https://mock.upstream/v1"] = (
        time.monotonic() - pricing._TTL_SECONDS - 1,
        {},
    )

    result = await pricing.get_prices("https://mock.upstream/v1")
    assert call_count == 1  # expired → re-fetched
    assert "m" in result


async def test_pricing_stale_cache_returned_on_error(mock_router) -> None:
    """When the refresh fails, stale cached data is returned instead of empty."""
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(503, json={})
    )

    stale = {"m": {"prompt": 0.001, "completion": 0.002}}
    pricing._cache["https://mock.upstream/v1"] = (
        time.monotonic() - pricing._TTL_SECONDS - 1,
        stale,
    )

    result = await pricing.get_prices("https://mock.upstream/v1")
    assert result == stale  # stale data preserved on error
