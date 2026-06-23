"""Pre-run cost estimate: unit + endpoint + pricing fetch."""

from __future__ import annotations

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
            json={"data": [{"id": "m1", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]},
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
