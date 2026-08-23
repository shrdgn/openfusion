"""Embeddable engine: per-request config resolver."""

from __future__ import annotations

import httpx
from fastapi import Request

from openfusion.config import JudgeConfig, OpenFusionConfig, PanelMember
from openfusion.server import create_app


def _config_for(user: str) -> OpenFusionConfig:
    return OpenFusionConfig(
        panel=[PanelMember(base_url="https://u/v1", api_key=f"key-{user}", model=f"model-{user}")],
        judge=JudgeConfig(base_url="https://u/v1", api_key=f"key-{user}", model=f"judge-{user}"),
    )


async def _resolver(request: Request) -> OpenFusionConfig:
    return _config_for(request.headers.get("x-user", "default"))


async def test_config_resolver_returns_per_request_config() -> None:
    app = create_app(config_resolver=_resolver)  # no static config needed
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        a = (await client.get("/v1/config", headers={"x-user": "alice"})).json()
        b = (await client.get("/v1/config", headers={"x-user": "bob"})).json()
    await app.state.upstream_client.aclose()

    assert a["panel"] == ["model-alice"] and a["judge"] == "judge-alice"
    assert b["panel"] == ["model-bob"] and b["judge"] == "judge-bob"


async def test_resolver_app_builds_without_static_config() -> None:
    app = create_app(config_resolver=_resolver)
    assert app.state.config is None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/healthz")
        cfg = await client.get("/v1/config")
    await app.state.upstream_client.aclose()
    assert health.status_code == 200
    assert cfg.status_code == 200


async def test_usage_callback_fires_on_fusion(mock_router) -> None:
    from openfusion.config import Aggregator, Strategy, TimeoutsConfig

    seen: list = []

    async def usage_cb(request, usage):
        seen.append((request.url.path, usage))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config, usage_callback=usage_cb)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "q"}]},
        )
    await app.state.upstream_client.aclose()

    assert len(seen) == 1
    assert seen[0][0] == "/v1/chat/completions"
    assert seen[0][1] is not None  # usage dict was passed


async def _pipeline_config(**kwargs: object) -> OpenFusionConfig:
    from openfusion.config import PipelineConfig, PipelineStepConfig, PipelineStepUse, Strategy

    return OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(
            steps=[PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)]
        ),
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        **kwargs,
    )


async def test_usage_callback_fires_on_pipeline_non_streaming(mock_router) -> None:
    seen: list = []

    async def usage_cb(request, usage):
        seen.append(usage)

    sse_body = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"ok"}'
        ',"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, text=sse_body, headers={"content-type": "text/event-stream"}
        )
    )
    config = await _pipeline_config()
    app = create_app(config, usage_callback=usage_cb)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "q"}]},
        )
    await app.state.upstream_client.aclose()

    assert len(seen) == 1
    assert seen[0] is not None


async def test_usage_callback_fires_on_pipeline_streaming(mock_router) -> None:
    seen: list = []

    async def usage_cb(request, usage):
        seen.append(usage)

    sse_body = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"hi"}'
        ',"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, text=sse_body, headers={"content-type": "text/event-stream"}
        )
    )
    config = await _pipeline_config()
    app = create_app(config, usage_callback=usage_cb)

    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
            },
        ) as response,
    ):
        async for _ in response.aiter_lines():
            pass
    await app.state.upstream_client.aclose()

    assert len(seen) == 1
    assert seen[0] is not None


async def test_pipeline_response_cache_hits_on_repeat_request(mock_router) -> None:
    from openfusion.config import ResponseCacheConfig

    sse_body = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"ok"}'
        ',"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, text=sse_body, headers={"content-type": "text/event-stream"}
        )
    )
    config = await _pipeline_config(response_cache=ResponseCacheConfig(enabled=True))
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {"model": "openfusion", "messages": [{"role": "user", "content": "q"}]}
        first = await client.post("/v1/chat/completions", json=body)
        second = await client.post("/v1/chat/completions", json=body)
    await app.state.upstream_client.aclose()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert route.call_count == 1
