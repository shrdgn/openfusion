"""Stream framing tests."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    SelfFusionConfig,
    Strategy,
    TimeoutsConfig,
)
from openfusion.stream import (
    capture_stream,
    ranked_and_stream,
    synthesize_and_stream,
    vote_and_stream,
)
from openfusion.upstream import UpstreamClient


def _parse_sse(payload: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    for block in payload.split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        events.append((event_name, data))
    return events


@pytest.mark.asyncio
async def test_stream_emits_progress_and_done(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "a"}}]},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "b"}}]},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "c"}}]},
            ),
            httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"final"},"finish_reason":null}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            ),
        ]
    )

    config = OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        self_fusion=SelfFusionConfig(n=3),
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m", label="p"),
        ],
        judge=JudgeConfig(
            base_url="https://mock.upstream/v1",
            api_key="k",
            model="judge",
        ),
    )
    client = UpstreamClient()
    chunks: list[str] = []
    async for line in synthesize_and_stream(
        {"messages": [{"role": "user", "content": "hello"}]},
        config,
        client,
    ):
        chunks.append(line)

    events = _parse_sse("".join(chunks))
    assert any(event == "progress" for event, _ in events)
    assert events[-1][1] == "[DONE]"
    content_events = [data for event, data in events if event is None and data != "[DONE]"]
    assert any("final" in data for data in content_events)

    # Live per-member breakdown: one panel_member event per member, plus a
    # leading "panel" stage that names the models and the judge.
    progress = [json.loads(data) for event, data in events if event == "progress"]
    panel_stage = next(p for p in progress if p.get("stage") == "panel")
    assert panel_stage["total"] == 3 and len(panel_stage["models"]) == 3
    assert panel_stage["judge"] == "judge"
    member_events = [p for p in progress if p.get("stage") == "panel_member"]
    assert len(member_events) == 3
    assert member_events[-1]["completed"] == 3 and member_events[-1]["total"] == 3
    assert all(p["ok"] for p in member_events)
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_emits_panel_answers_when_exposed(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": "answer A"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "answer B"}}]}),
            httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"fused"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            ),
        ]
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        expose_panel=True,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
    )
    client = UpstreamClient()
    chunks = [line async for line in synthesize_and_stream(
        {"messages": [{"role": "user", "content": "hi"}]}, config, client
    )]
    events = _parse_sse("".join(chunks))
    answers = [json.loads(data) for event, data in events if event == "panel_answer"]
    assert {a["model"] for a in answers} == {"m1", "m2"}
    assert {a["content"] for a in answers} == {"answer A", "answer B"}
    # The fused answer still streams as normal content.
    assert any("fused" in data for event, data in events if event is None)
    await client.aclose()


def _panel_config(aggregator: Aggregator) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=aggregator,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )


@pytest.mark.asyncio
async def test_vote_and_stream_sends_error_chunk_on_panel_failure(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "upstream down"}})
    )
    config = _panel_config(Aggregator.VOTE)
    client = UpstreamClient()
    chunks = [line async for line in vote_and_stream(
        {"messages": [{"role": "user", "content": "hi"}]}, config, client
    )]
    await client.aclose()

    events = _parse_sse("".join(chunks))
    # Stream must end with [DONE] even on panel failure.
    assert events[-1][1] == "[DONE]"
    # There must be an error data chunk before [DONE].
    error_events = [
        json.loads(data)
        for event, data in events
        if event is None and data != "[DONE]"
        if "error" in json.loads(data)
    ]
    assert error_events, "expected an SSE error chunk (vote)"


@pytest.mark.asyncio
async def test_ranked_and_stream_sends_error_chunk_on_panel_failure(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "upstream down"}})
    )
    config = _panel_config(Aggregator.RANKED)
    client = UpstreamClient()
    chunks = [line async for line in ranked_and_stream(
        {"messages": [{"role": "user", "content": "hi"}]}, config, client
    )]
    await client.aclose()

    events = _parse_sse("".join(chunks))
    assert events[-1][1] == "[DONE]"
    error_events = [
        json.loads(data)
        for event, data in events
        if event is None and data != "[DONE]"
        if "error" in json.loads(data)
    ]
    assert error_events, "expected an SSE error chunk (ranked)"


@pytest.mark.asyncio
async def test_vote_and_stream_usage_event_includes_total(mock_router) -> None:
    """vote_and_stream must emit a usage SSE event with a 'total' key.

    capture_stream reads `obj.get("total") or obj` from usage events; without
    a 'total' key the full structured payload (with 'panel'/'panel_total'/'judge'
    keys) would be passed to on_complete instead of the flat token counts —
    inconsistent with what buffer_vote returns for non-streaming callers.
    """
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "answer-a"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "answer-a"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ),
        ]
    )
    config = _panel_config(Aggregator.VOTE)
    client = UpstreamClient()
    chunks = [line async for line in vote_and_stream(
        {"messages": [{"role": "user", "content": "hi"}]}, config, client
    )]
    await client.aclose()

    events = _parse_sse("".join(chunks))
    usage_events = [
        json.loads(data) for event, data in events if event == "usage"
    ]
    assert usage_events, "expected a usage SSE event"
    assert "total" in usage_events[0], (
        "usage event must have 'total' key for capture_stream compatibility"
    )
    total = usage_events[0]["total"]
    assert total["prompt_tokens"] == 20
    assert total["completion_tokens"] == 10
    assert total["total_tokens"] == 30


@pytest.mark.asyncio
async def test_ranked_and_stream_success_emits_winner_and_usage(mock_router) -> None:
    """ranked_and_stream's success path (gather → pick_best → content/usage/[DONE])
    previously had no coverage — only the panel-failure branch was tested.
    """
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "answer-a"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "answer-b"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ),
            # Judge ranking call: 1-indexed reply "2" picks panel member index 1 ("answer-b").
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "2"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
                },
            ),
        ]
    )
    config = _panel_config(Aggregator.RANKED)
    client = UpstreamClient()
    chunks = [line async for line in ranked_and_stream(
        {"messages": [{"role": "user", "content": "hi"}]}, config, client
    )]
    await client.aclose()

    events = _parse_sse("".join(chunks))
    assert events[-1][1] == "[DONE]"

    progress_events = [
        json.loads(data) for event, data in events if event == "progress"
    ]
    ranked_progress = [p for p in progress_events if p.get("stage") == "ranked"]
    assert ranked_progress, "expected a 'ranked' progress event"
    assert ranked_progress[0]["winner"] == 1

    content_chunks = [
        json.loads(data)
        for event, data in events
        if event is None and data != "[DONE]"
    ]
    contents = [
        c["choices"][0]["delta"].get("content")
        for c in content_chunks
        if c["choices"][0]["delta"].get("content")
    ]
    assert contents == ["answer-b"]

    finish_reasons = [c["choices"][0].get("finish_reason") for c in content_chunks]
    assert "stop" in finish_reasons


@pytest.mark.asyncio
async def test_capture_stream_skips_on_complete_on_error_chunk() -> None:
    """capture_stream must NOT call on_complete when the stream contains an error chunk.

    The response cache relies on this: a partial or errored answer must never be
    cached as a valid response for future requests.
    """
    completed: list[tuple[str, object]] = []

    async def on_complete(content: str, usage: object) -> None:
        completed.append((content, usage))

    error_sse = (
        'data: {"error": {"message": "upstream down", "type": "upstream_error"}}\n\n'
        "data: [DONE]\n\n"
    )

    async def _source():
        for block in error_sse.split("\n\n"):
            if block.strip():
                yield block + "\n\n"

    lines = [line async for line in capture_stream(_source(), on_complete)]
    assert any("[DONE]" in line for line in lines)
    assert completed == [], "on_complete should not fire when the stream contains an error"


@pytest.mark.asyncio
async def test_capture_stream_calls_on_complete_on_clean_stream() -> None:
    """capture_stream accumulates content and fires on_complete after a clean stream."""
    completed: list[tuple[str, object]] = []

    async def on_complete(content: str, usage: object) -> None:
        completed.append((content, usage))

    usage_obj = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    clean_sse = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        f'event: usage\ndata: {json.dumps({"total": usage_obj})}\n\n'
        "data: [DONE]\n\n"
    )

    async def _source():
        for block in clean_sse.split("\n\n"):
            if block.strip():
                yield block + "\n\n"

    _ = [line async for line in capture_stream(_source(), on_complete)]
    assert len(completed) == 1
    content, usage = completed[0]
    assert content == "Hello world"
    assert usage == usage_obj
