"""Debate strategy: members revise after seeing each other's answers."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import (
    DebateConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
    TimeoutsConfig,
)
from openfusion.panel import gather_panel
from openfusion.upstream import UpstreamClient


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _two_member_config() -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.DEBATE,
        debate=DebateConfig(rounds=1),
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )


@pytest.mark.asyncio
async def test_debate_runs_a_revision_round(mock_router) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        # The revision prompt embeds peers' answers via the debate instruction.
        if "other independent experts" in text:
            return httpx.Response(200, json=_completion("revised"))
        return httpx.Response(200, json=_completion("first"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        _two_member_config(),
        client,
    )

    assert len(result.responses) == 2
    assert all(r.content == "revised" for r in result.responses)
    # Usage accumulates across both rounds (1 + 1 prompt tokens).
    assert all(r.usage and r.usage["prompt_tokens"] == 2 for r in result.responses)
    await client.aclose()


@pytest.mark.asyncio
async def test_debate_preserves_round1_usage_when_revision_has_none(mock_router) -> None:
    """When revision call returns no usage, round-1 usage is carried forward."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        if "other independent experts" in text:
            # Revision succeeds but returns no usage data.
            return httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "revised"}}]}
            )
        return httpx.Response(200, json=_completion("first"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        _two_member_config(),
        client,
    )

    assert len(result.responses) == 2
    assert all(r.content == "revised" for r in result.responses)
    # Round-1 usage must not be silently discarded when the revision has no usage.
    assert all(r.usage is not None for r in result.responses)
    assert all(r.usage.get("prompt_tokens", 0) >= 1 for r in result.responses)
    await client.aclose()


@pytest.mark.asyncio
async def test_debate_noop_with_single_member(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("only"))
    )
    config = _two_member_config()
    config.panel = config.panel[:1]
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        client,
    )

    # No peers to debate against → first-round answer is kept as-is.
    assert len(result.responses) == 1
    assert result.responses[0].content == "only"
    await client.aclose()
