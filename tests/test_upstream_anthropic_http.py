"""HTTP-level tests for the native Anthropic upstream call path.

test_upstream_anthropic.py covers the pure translation helpers; this file
exercises UpstreamClient.chat_completion end to end against a mocked
Anthropic Messages endpoint (dispatch, headers, non-streaming JSON, SSE
streaming, and error handling) which previously had no coverage at all.
"""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import PanelMember
from openfusion.errors import UpstreamError
from openfusion.upstream import UpstreamClient


def _member(**overrides: object) -> PanelMember:
    defaults: dict[str, object] = {
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "sk-ant-test",
        "model": "claude-3-haiku-20240307",
        "label": "claude",
    }
    defaults.update(overrides)
    return PanelMember(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_chat_completion_dispatches_to_anthropic_for_anthropic_provider(
    mock_router,
) -> None:
    route = mock_router.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-haiku-20240307",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hi there"}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )
    )
    client = UpstreamClient()
    member = _member()

    result = await client.chat_completion(
        member, {"messages": [{"role": "user", "content": "hi"}]}, stream=False
    )

    assert route.called
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "sk-ant-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in request.headers
    assert result["choices"][0]["message"]["content"] == "hi there"
    assert result["usage"]["prompt_tokens"] == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_non_streaming_error_raises_upstream_error(mock_router) -> None:
    mock_router.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            429, json={"error": {"message": "rate limited"}}
        )
    )
    client = UpstreamClient()
    member = _member()

    with pytest.raises(UpstreamError) as exc_info:
        await client.chat_completion(
            member, {"messages": [{"role": "user", "content": "hi"}]}, stream=False
        )
    assert exc_info.value.status_code == 429
    assert "rate limited" in str(exc_info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_streaming_yields_converted_chunks(mock_router) -> None:
    sse_body = (
        'event: message_start\n'
        'data: {"type": "message_start", "message": {"id": "msg_1", '
        '"model": "claude-3-haiku-20240307", "usage": {"input_tokens": 5, "output_tokens": 0}}}\n\n'
        'event: content_block_delta\n'
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}\n\n'
        'event: message_delta\n'
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
        '"usage": {"output_tokens": 1}}\n\n'
        "data: [DONE]\n\n"
    )
    mock_router.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )
    )
    client = UpstreamClient()
    member = _member()

    stream = await client.chat_completion(
        member,
        {"messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 3
    assert chunks[0]["usage"]["prompt_tokens"] == 5
    assert chunks[1]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_streaming_error_raises_upstream_error(mock_router) -> None:
    mock_router.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(500, content=b"internal error")
    )
    client = UpstreamClient()
    member = _member()

    stream = await client.chat_completion(
        member,
        {"messages": [{"role": "user", "content": "hi"}]},
        stream=True,
    )
    with pytest.raises(UpstreamError) as exc_info:
        async for _ in stream:
            pass
    assert exc_info.value.status_code == 500
    await client.aclose()


@pytest.mark.asyncio
async def test_explicit_provider_overrides_inferred_openai(mock_router) -> None:
    """A non-Anthropic base_url with provider="anthropic" still hits /messages."""
    route = mock_router.post("https://my-proxy.example/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_2",
                "model": "claude-3-haiku-20240307",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    client = UpstreamClient()
    member = _member(base_url="https://my-proxy.example/v1", provider="anthropic")

    result = await client.chat_completion(
        member, {"messages": [{"role": "user", "content": "hi"}]}, stream=False
    )

    assert route.called
    assert result["choices"][0]["message"]["content"] == "ok"
    await client.aclose()
