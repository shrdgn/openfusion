"""Server pass-through tests."""

from __future__ import annotations

import httpx


async def test_non_fusion_model_passes_through(client: httpx.AsyncClient, mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "solo",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "solo answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "pass-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "solo answer"


async def test_tool_calls_pass_through(client: httpx.AsyncClient, mock_router) -> None:
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "tool",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
    )

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )

    assert response.status_code == 200
    assert route.called
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
