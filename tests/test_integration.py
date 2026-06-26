"""End-to-end integration tests."""

from __future__ import annotations

import httpx
import pytest
import respx


def _panel_response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


def _stream_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(
            f'data: {{"choices":[{{"delta":{{"content":"{content}"}}, "finish_reason":null}}]}}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )


def _mock_self_fusion(mock_router: respx.MockRouter, *, judge_content: str) -> None:
    """Mock a self-fusion run: 3 panel calls + 1 streaming judge call."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=[
            _panel_response("a"),
            _panel_response("b"),
            _panel_response("c"),
            _stream_response(judge_content),
        ]
    )


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_models_list_includes_openfusion(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/models")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert "openfusion" in ids


async def test_config_endpoint_returns_strategy(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert "strategy" in body
    assert "panel" in body


@pytest.mark.asyncio
async def test_chat_completions_fusion_non_streaming(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    """Full round-trip: request → panel → judge → fused JSON response."""
    # self_fusion runs N panel calls then a judge synthesis
    _mock_self_fusion(mock_router, judge_content="fused answer")
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "choices" in body
    assert body["choices"][0]["message"]["content"] == "fused answer"


@pytest.mark.asyncio
async def test_chat_completions_fusion_streaming(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    """Streaming fusion: SSE chunks contain content and end with [DONE]."""
    _mock_self_fusion(mock_router, judge_content="streamed")
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    text = response.text
    assert "streamed" in text
    assert "[DONE]" in text


@pytest.mark.asyncio
async def test_chat_completions_pass_through(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    """Requests for the pass-through model bypass fusion entirely."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "direct"}}]},
        )
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "direct"
