"""End-to-end tests for `timeouts.total_seconds` enforcement.

`total_seconds` is documented (every `examples/*.yaml.example`) as an aggregate
wall-clock cap on a fused/pipeline request, but nothing enforced it (issue #140)
-- only the per-call `member_seconds`/`judge_seconds` budgets did. These tests
drive a real request through `/v1/chat/completions` against an upstream that
never returns in time, with `total_seconds` set far below `member_seconds` /
`judge_seconds`, so only the new aggregate enforcement -- not the existing
per-call timeouts -- can be what cuts the request off.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    PipelineConfig,
    PipelineStepConfig,
    PipelineStepUse,
    Strategy,
    TimeoutsConfig,
)
from openfusion.server import create_app


async def _slow_handler(request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(10)
    return httpx.Response(200, json={"choices": [{"message": {"content": "too late"}}]})


def _fused_config(total_seconds: float) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        aggregator=Aggregator.JUDGE,
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(
            base_url="https://mock.upstream/v1", api_key="k", model="p"
        ),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=total_seconds),
    )


def _pipeline_config(total_seconds: float) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(
            steps=[PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)]
        ),
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(
            base_url="https://mock.upstream/v1", api_key="k", model="solo-model"
        ),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=total_seconds),
    )


@pytest.mark.asyncio
async def test_non_streaming_fusion_times_out_with_clean_504(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=_slow_handler
    )
    app = create_app(_fused_config(total_seconds=0.05))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 504
    payload = response.json()
    assert payload["error"]["code"] == "total_timeout"
    assert payload["error"]["type"] == "timeout_error"


@pytest.mark.asyncio
async def test_streaming_fusion_emits_timeout_sse_chunk(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=_slow_handler
    )
    app = create_app(_fused_config(total_seconds=0.05))
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response,
    ):
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])
    await app.state.upstream_client.aclose()

    assert '"code": "total_timeout"' in body
    assert '"type": "timeout_error"' in body
    assert body.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_non_streaming_pipeline_times_out_with_clean_504(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=_slow_handler
    )
    app = create_app(_pipeline_config(total_seconds=0.05))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 504
    payload = response.json()
    assert payload["error"]["code"] == "total_timeout"


@pytest.mark.asyncio
async def test_streaming_pipeline_emits_timeout_sse_chunk(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=_slow_handler
    )
    app = create_app(_pipeline_config(total_seconds=0.05))
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response,
    ):
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])
    await app.state.upstream_client.aclose()

    assert '"code": "total_timeout"' in body
    assert body.rstrip().endswith("data: [DONE]")
