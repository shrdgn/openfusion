"""Server pass-through tests."""

from __future__ import annotations

import json
import logging

import httpx

from openfusion.config import CostControlsConfig, OpenFusionConfig
from openfusion.server import create_app


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


async def test_pass_through_injects_token_limit_and_logs_usage(
    test_config: OpenFusionConfig,
    mock_router,
    caplog,
) -> None:
    test_config.cost_controls = CostControlsConfig(pass_through_max_tokens=5)
    app = create_app(test_config)

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 5
        assert payload["messages"][0]["content"] == "secret prompt"
        return httpx.Response(
            200,
            json={
                "id": "solo",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=upstream_handler
    )
    caplog.set_level(logging.INFO, logger="openfusion.upstream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "pass-model",
                "messages": [{"role": "user", "content": "secret prompt"}],
            },
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert '"phase": "pass_through"' in log_output
    assert '"total_tokens": 3' in log_output
    assert "secret prompt" not in log_output


async def test_pass_through_rejects_over_limit(
    test_config: OpenFusionConfig,
    mock_router,
) -> None:
    test_config.cost_controls = CostControlsConfig(pass_through_max_tokens=5)
    app = create_app(test_config)
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "pass-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 6,
            },
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "max_tokens_exceeds_limit"
    assert not route.called


async def test_rejects_invalid_max_tokens(
    client: httpx.AsyncClient,
    mock_router,
) -> None:
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "pass-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_max_tokens"
    assert not route.called


async def test_rejects_missing_messages(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "openfusion"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert "messages" in data["error"]["message"]


async def test_rejects_empty_messages(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "openfusion", "messages": []},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert "messages" in data["error"]["message"]


async def test_rejects_non_list_messages(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "openfusion", "messages": "hello"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert "messages" in data["error"]["message"]


async def test_openfusion_rejects_judge_over_limit(
    test_config: OpenFusionConfig,
    mock_router,
) -> None:
    test_config.cost_controls = CostControlsConfig(judge_max_tokens=5)
    app = create_app(test_config)
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 6,
            },
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "max_tokens_exceeds_limit"
    assert not route.called
