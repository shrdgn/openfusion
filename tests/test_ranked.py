"""Ranked-choice aggregation: judge picks the best panel answer."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
    TimeoutsConfig,
)
from openfusion.panel import MemberResponse, PanelResult
from openfusion.ranked import _parse_choice, pick_best
from openfusion.server import create_app
from openfusion.upstream import UpstreamClient


def test_parse_choice() -> None:
    assert _parse_choice("2", 3) == 1
    assert _parse_choice("The best is 3.", 3) == 2
    assert _parse_choice("garbage", 3) == 0  # unparseable defaults to first
    assert _parse_choice("9", 3) == 0  # out of range defaults to first


def _config() -> OpenFusionConfig:
    return OpenFusionConfig(
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
    )


@pytest.mark.asyncio
async def test_pick_best_returns_selected_answer(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "2"}}]},
        )
    )
    panel = PanelResult(
        responses=[
            MemberResponse(label="a", content="answer one", model="m"),
            MemberResponse(label="b", content="answer two", model="m"),
        ]
    )
    client = UpstreamClient()

    content, meta = await pick_best(
        {"messages": [{"role": "user", "content": "q"}]}, panel, _config(), client
    )

    assert content == "answer two"
    assert meta["winner"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_pick_best_single_response_skips_judge(mock_router) -> None:
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    panel = PanelResult(responses=[MemberResponse(label="a", content="only", model="m")])
    client = UpstreamClient()

    content, _ = await pick_best(
        {"messages": [{"role": "user", "content": "q"}]}, panel, _config(), client
    )

    assert content == "only"
    assert not route.called
    await client.aclose()


@pytest.mark.asyncio
async def test_ranked_aggregator_end_to_end(mock_router) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        if "Candidate answers:" in text:  # the ranking call
            return httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "2"}}]}
            )
        # a panel member call; content encodes which member via model name
        answer = f"ans-{payload['model']}"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": answer}}]},
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.RANKED,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "q"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    # The judge picked candidate 2 → the second panel member's answer.
    assert response.json()["choices"][0]["message"]["content"] == "ans-m2"
