"""Playground endpoints: /v1/config, static page, and per-request overrides."""

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
from openfusion.server import create_app


async def test_v1_config_reports_active_config(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["panel"] == ["test-model"]
    assert body["judge"] == "judge-model"
    assert body["allow_request_overrides"] is False
    assert "quality" in body["presets"] and "budget" in body["presets"]


async def test_playground_page_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/playground/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Model Fusion" in response.text


async def test_override_rejected_when_disabled(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "hi"}],
            "openfusion": {"panel": ["x"]},
        },
    )
    assert response.status_code == 400
    assert "overrides are disabled" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_override_applies_panel_models(mock_router) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload["model"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "42"}}]},
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        allow_request_overrides=True,
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="base")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "q"}],
                "openfusion": {"panel": ["model-a", "model-b"]},
            },
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    assert set(seen) == {"model-a", "model-b"}
