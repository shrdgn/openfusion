"""End-to-end tests for the pipeline strategy's server-facing streaming path.

openfusion/pipeline.py's own unit tests (tests/test_pipeline.py) cover
run_pipeline's internals, but nothing exercised the request path that wires it
up: server.py's /v1/chat/completions -> _pipeline_stream -> stream.py's
pipeline_and_stream/buffer_pipeline. That's the code responsible for SSE
framing, client-disconnect cancellation, and error-outcome metrics recording
for every pipeline-strategy request.
"""

from __future__ import annotations

import time

import httpx

from openfusion.config import (
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
from openfusion.server import _pipeline_stream, create_app
from openfusion.upstream import UpstreamClient


class _NeverDisconnectsRequest:
    """Minimal server.Request stand-in for driving _pipeline_stream directly."""

    async def is_disconnected(self) -> bool:
        return False


def _pipeline_config(steps: list[PipelineStepConfig]) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(steps=steps),
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(
            base_url="https://mock.upstream/v1", api_key="k", model="solo-model"
        ),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )


def _sse(*chunks: str) -> httpx.Response:
    body = "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def test_pipeline_stream_emits_sse_and_final_usage(mock_router) -> None:
    """A streaming pipeline request returns SSE deltas, then a usage event, then [DONE]."""
    cfg = _pipeline_config([PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)])
    app = create_app(cfg)

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=_sse(
            '{"choices":[{"delta":{"role":"assistant","content":"hel"},"finish_reason":null}]}',
            '{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],'
            '"usage":{"total_tokens":7}}',
        )
    )

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
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])
    await app.state.upstream_client.aclose()

    assert '"content": "hel"' in body
    assert '"content": "lo"' in body
    assert '"finish_reason": "stop"' in body
    assert '"usage": {"total_tokens": 7}' in body
    assert body.rstrip().endswith("data: [DONE]")


async def test_pipeline_buffer_returns_joined_content_non_streaming(mock_router) -> None:
    """stream=False buffers the full pipeline output into one JSON completion."""
    cfg = _pipeline_config([PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)])
    app = create_app(cfg)

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=_sse(
            '{"choices":[{"delta":{"role":"assistant","content":"foo"},"finish_reason":null}]}',
            '{"choices":[{"delta":{"content":"bar"},"finish_reason":"stop"}]}',
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "foobar"


async def test_pipeline_stream_chains_solo_steps_and_injects_prior_output(mock_router) -> None:
    """Two SOLO steps: the second step's system prompt receives the first step's output."""
    cfg = _pipeline_config(
        [
            PipelineStepConfig(name="draft", use=PipelineStepUse.SOLO),
            PipelineStepConfig(
                name="polish", use=PipelineStepUse.SOLO, system="Polish this: {draft}"
            ),
        ]
    )
    app = create_app(cfg)

    seen_systems: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        payload = _json.loads(request.content)
        system_msg = next(
            (m["content"] for m in payload["messages"] if m["role"] == "system"), None
        )
        seen_systems.append(system_msg)
        if len(seen_systems) == 1:
            return _sse(
                '{"choices":[{"delta":{"role":"assistant","content":"draft-text"},'
                '"finish_reason":"stop"}]}'
            )
        return _sse(
            '{"choices":[{"delta":{"role":"assistant","content":"final-text"},'
            '"finish_reason":"stop"}]}'
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "final-text"
    assert seen_systems[1] == "Polish this: draft-text"


async def test_pipeline_stream_upstream_failure_degrades_to_sse_error_chunk(
    mock_router, monkeypatch
) -> None:
    """An upstream failure mid-pipeline-stream ends the stream gracefully.

    Matches vote_and_stream/ranked_and_stream/synthesize_and_stream: an
    UpstreamError raised while running the pipeline is caught by
    pipeline_and_stream and turned into an SSE error chunk followed by
    [DONE], instead of crashing the response after headers are already
    sent. Exercises _pipeline_stream's finally branch in server.py too: the
    disconnect-watch task is cancelled and awaited even though the client
    never actually disconnected.
    """
    cfg = _pipeline_config([PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)])
    upstream_client = UpstreamClient()

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "openfusion.server._record_request",
        lambda route, outcome, started: recorded.append((route, outcome)),
    )

    response = await _pipeline_stream(
        _NeverDisconnectsRequest(),
        {"messages": [{"role": "user", "content": "hi"}], "stream": True},
        cfg,
        upstream_client,
        started=time.perf_counter(),
    )
    body = "".join([line async for line in response.body_iterator])
    await upstream_client.aclose()

    assert '"code": "pipeline_stream_error"' in body
    assert body.rstrip().endswith("data: [DONE]")
    # No exception escaped event_stream(), so the outcome is "success" -- the
    # same (imperfect but established) semantics as _fusion_stream, where an
    # SSE-level error chunk from a caught panel/judge failure isn't visible to
    # _record_request either.
    assert recorded == [("pipeline", "success")]
