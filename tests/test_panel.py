"""Panel gather tests."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import (
    OpenFusionConfig,
    PanelMember,
    SelfFusionConfig,
    Strategy,
    TimeoutsConfig,
)
from openfusion.errors import UpstreamError
from openfusion.panel import gather_panel
from openfusion.upstream import UpstreamClient


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.asyncio
async def test_gather_panel_all_succeed(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("ok"))
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="k",
                model="m1",
                label="a",
            ),
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="k",
                model="m2",
                label="b",
            ),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        client,
    )

    assert len(result.responses) == 2
    assert not result.failures
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_degrades_on_member_failure(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_completion("good")),
            httpx.Response(500, json={"error": {"message": "boom"}}),
        ]
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        client,
    )

    assert len(result.responses) == 1
    assert len(result.failures) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_all_fail_raises(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "down"}})
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
        ],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()

    with pytest.raises(UpstreamError, match="All panel members failed"):
        await gather_panel(
            {"messages": [{"role": "user", "content": "hi"}]},
            config,
            client,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_self_fusion_expands_members() -> None:
    from openfusion.panel import expand_panel_members

    config = OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        self_fusion=SelfFusionConfig(n=3),
        panel=[
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="k",
                model="solo",
                label="solo",
            ),
        ],
    )
    members = expand_panel_members(config)
    assert len(members) == 3
    assert members[0][1]["temperature"] == 0.3
    assert members[1][1]["temperature"] == 0.7
