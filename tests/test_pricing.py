"""Unit coverage for pricing.py's defensive/fallback branches.

test_estimate.py already covers the happy path, cache hit/expiry, and the
error-with-no-cache fallback through the /v1/estimate endpoint. This file
targets pricing.py in isolation: malformed upstream data and the
already-in-flight-fetch branch, neither of which test_estimate.py exercises.
"""

from __future__ import annotations

import time

import httpx
import pytest

from openfusion import pricing


@pytest.fixture(autouse=True)
def _clear_price_cache() -> None:
    pricing._cache.clear()
    pricing._inflight.clear()


async def test_fetch_skips_non_string_model_id(mock_router) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": 123, "pricing": {"prompt": "0.001", "completion": "0.002"}},
                    {"id": "good", "pricing": {"prompt": "0.001", "completion": "0.002"}},
                ]
            },
        )
    )
    prices = await pricing.get_prices("https://mock.upstream/v1")
    assert "good" in prices
    assert len(prices) == 1  # the non-string id entry was skipped


async def test_fetch_skips_malformed_pricing_value(mock_router) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "bad", "pricing": {"prompt": "not-a-number", "completion": "0.002"}},
                    {"id": "good", "pricing": {"prompt": "0.001", "completion": "0.002"}},
                ]
            },
        )
    )
    prices = await pricing.get_prices("https://mock.upstream/v1")
    assert "good" in prices
    assert "bad" not in prices  # ValueError on float() conversion skips the entry


async def test_fetch_defaults_missing_pricing_to_zero(mock_router) -> None:
    mock_router.get("https://mock.upstream/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "no-pricing"}]})
    )
    prices = await pricing.get_prices("https://mock.upstream/v1")
    assert prices == {"no-pricing": {"prompt": 0.0, "completion": 0.0}}


async def test_inflight_fetch_returns_stale_cache_without_new_call(mock_router) -> None:
    """A caller arriving while another fetch is in flight gets the stale cache, not a new fetch."""
    call_count = 0

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"data": []})

    mock_router.get("https://mock.upstream/v1/models").mock(side_effect=_handler)

    stale = {"m": {"prompt": 0.001, "completion": 0.002}}
    pricing._cache["https://mock.upstream/v1"] = (
        time.monotonic() - pricing._TTL_SECONDS - 1,
        stale,
    )
    pricing._inflight.add("https://mock.upstream/v1")
    try:
        result = await pricing.get_prices("https://mock.upstream/v1")
    finally:
        pricing._inflight.discard("https://mock.upstream/v1")

    assert result == stale
    assert call_count == 0  # no HTTP call made; stale cache returned as-is


async def test_cache_evicts_least_recently_used_entry_past_the_cap(mock_router) -> None:
    """_cache must stay bounded -- a config_resolver-based multi-tenant deployment
    can see many distinct base_urls (one per tenant), and without eviction the
    module-global cache would grow for the life of the process."""
    for i in range(4):
        mock_router.get(f"https://host{i}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
    pricing._CACHE_MAX_KEYS = 3
    try:
        for i in range(3):
            await pricing.get_prices(f"https://host{i}/v1")
        # Re-fetch host0 (a cache hit) so it's the most-recently-used, leaving
        # host1 as the least-recently-used entry once a 4th URL is added.
        await pricing.get_prices("https://host0/v1")
        await pricing.get_prices("https://host3/v1")

        assert "https://host1/v1" not in pricing._cache  # evicted: least recently used
        assert "https://host0/v1" in pricing._cache
        assert "https://host2/v1" in pricing._cache
        assert "https://host3/v1" in pricing._cache
        assert len(pricing._cache) == 3
    finally:
        pricing._CACHE_MAX_KEYS = 1024


async def test_inflight_fetch_returns_empty_when_no_cache(mock_router) -> None:
    """A caller arriving while another fetch is in flight with no prior cache gets {}."""
    call_count = 0

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"data": []})

    mock_router.get("https://mock.upstream/v1/models").mock(side_effect=_handler)

    pricing._inflight.add("https://mock.upstream/v1")
    try:
        result = await pricing.get_prices("https://mock.upstream/v1")
    finally:
        pricing._inflight.discard("https://mock.upstream/v1")

    assert result == {}
    assert call_count == 0
