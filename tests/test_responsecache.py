"""Response cache: unit behavior + end-to-end serving from cache."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    ResponseCacheConfig,
    Strategy,
    TimeoutsConfig,
)
from openfusion.responsecache import ResponseCache, cache_key
from openfusion.server import create_app


def test_get_put_roundtrip() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=10)
    assert cache.get("k") is None
    cache.put("k", {"content": "hi"})
    assert cache.get("k") == {"content": "hi"}


def test_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr("openfusion.responsecache.time.monotonic", lambda: clock["t"])
    cache = ResponseCache(ttl_seconds=60, max_entries=10)
    cache.put("k", {"content": "hi"})
    clock["t"] += 61
    assert cache.get("k") is None


def test_lru_eviction() -> None:
    cache = ResponseCache(ttl_seconds=100, max_entries=2)
    cache.put("a", {"v": 1})
    cache.put("b", {"v": 2})
    cache.get("a")  # touch a so b is least-recently-used
    cache.put("c", {"v": 3})  # evicts b
    assert cache.get("b") is None
    assert cache.get("a") is not None and cache.get("c") is not None


def test_cache_key_depends_on_prompt_and_recipe() -> None:
    cfg = OpenFusionConfig(
        panel=[PanelMember(base_url="u", api_key="k", model="m")],
        judge=JudgeConfig(base_url="u", api_key="k", model="j"),
    )
    body1 = {"messages": [{"role": "user", "content": "hi"}]}
    body2 = {"messages": [{"role": "user", "content": "bye"}]}
    assert cache_key(body1, cfg) == cache_key(body1, cfg)
    assert cache_key(body1, cfg) != cache_key(body2, cfg)


@pytest.mark.asyncio
async def test_identical_request_served_from_cache(mock_router) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "42"}}]}
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        response_cache=ResponseCacheConfig(enabled=True),
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {"model": "openfusion", "messages": [{"role": "user", "content": "q"}]}
        first = await client.post("/v1/chat/completions", json=body)
        after_first = calls["n"]
        second = await client.post("/v1/chat/completions", json=body)
    await app.state.upstream_client.aclose()

    assert first.json()["choices"][0]["message"]["content"] == "42"
    assert second.json()["choices"][0]["message"]["content"] == "42"
    assert second.json().get("cached") is True
    # The second identical request makes no new upstream calls.
    assert calls["n"] == after_first
