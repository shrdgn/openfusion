"""429 retry behavior for panel members."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import OpenFusionConfig, PanelMember, Strategy, TimeoutsConfig
from openfusion.panel import gather_panel
from openfusion.upstream import UpstreamClient


@pytest.mark.asyncio
async def test_panel_retries_429_then_succeeds(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            ),
        ]
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
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
    await client.aclose()
