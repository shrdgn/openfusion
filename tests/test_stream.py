"""Stream framing tests."""

from __future__ import annotations

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
    await client.aclose()
