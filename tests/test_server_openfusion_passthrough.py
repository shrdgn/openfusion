"""Openfusion non-streaming integration tests."""

from __future__ import annotations

import json

import httpx


def _panel_response(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def test_openfusion_non_streaming(client: httpx.AsyncClient, mock_router) -> None:
    panel_calls = {"count": 0}

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            usage_chunk = (
                '{"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}'
            )
            body = (
                'data: {"choices":[{"delta":{"content":"fused"},"finish_reason":null}]}\n\n'
                f"data: {usage_chunk}\n\n"
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        panel_calls["count"] += 1
        return httpx.Response(200, json=_panel_response(f"answer {panel_calls['count']}"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=upstream_handler)

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "question"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openfusion"
    assert body["choices"][0]["message"]["content"] == "fused"
    assert panel_calls["count"] == 3
