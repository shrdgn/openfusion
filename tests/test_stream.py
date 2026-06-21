"""Stream framing tests."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import JudgeConfig, OpenFusionConfig, PanelMember, SelfFusionConfig, Strategy
from openfusion.stream import synthesize_and_stream
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
