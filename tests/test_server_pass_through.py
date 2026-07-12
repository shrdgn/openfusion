"""Server pass-through tests."""

from __future__ import annotations

import json
import logging
import time

import httpx
import pytest

from openfusion.config import CostControlsConfig, OpenFusionConfig
from openfusion.server import _pass_through, create_app
from openfusion.upstream import UpstreamClient


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


async def test_non_fusion_model_streams(client: httpx.AsyncClient, mock_router) -> None:
    """A streaming request for a non-fusion model is forwarded as SSE (`_pass_through`)."""
    body = (
        'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"solo "},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "pass-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "solo " in response.text
    assert "answer" in response.text
    assert "[DONE]" in response.text
    assert json.loads(route.calls.last.request.content)["stream"] is True


async def test_pass_through_stream_raises_when_upstream_not_streaming(app) -> None:
    """`_pass_through` raises an UpstreamError if a streaming call doesn't return a stream.

    This defensive branch (server.py's `if not hasattr(result, "__aiter__")`) isn't
    reachable via the real HTTP client, which always returns an async iterator for
    `stream=True` -- so it's exercised here via monkeypatching `chat_completion`.
    """

    async def _mock_chat_completion(*_args, **_kwargs):
        return {"choices": [{"message": {"content": "not a stream"}}]}

    app.state.upstream_client.chat_completion = _mock_chat_completion

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "pass-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Expected streaming upstream response"


async def test_pass_through_raises_when_upstream_returns_non_dict(app) -> None:
    """`_pass_through` raises an UpstreamError if a non-streaming call doesn't return JSON.

    Like the "not streaming" defensive branch above, the real HTTP client always
    returns a dict for a non-streaming JSON response (or raises earlier while
    parsing it) -- so this is exercised via monkeypatching `chat_completion`.
    """

    async def _mock_chat_completion(*_args, **_kwargs):
        return "not-a-dict"

    app.state.upstream_client.chat_completion = _mock_chat_completion

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={"model": "pass-model", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Expected JSON upstream response"


async def test_pass_through_stream_records_error_outcome_on_upstream_failure(
    test_config: OpenFusionConfig,
) -> None:
    """A mid-stream upstream failure records an error outcome and re-raises.

    _pass_through's event_stream() catches any exception from the upstream
    iterator only to flip its outcome flag before re-raising -- called directly
    since a real upstream failure surfaces as an UpstreamError from the SSE
    parser, not an arbitrary exception type.
    """
    client = UpstreamClient()

    async def _mock_chat_completion(*_args, **_kwargs):
        async def _gen():
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("upstream dropped connection")

        return _gen()

    client.chat_completion = _mock_chat_completion

    response = await _pass_through(
        {"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]},
        test_config,
        client,
        stream=True,
        started=time.perf_counter(),
    )

    received = []
    with pytest.raises(RuntimeError, match="upstream dropped connection"):
        async for chunk in response.body_iterator:
            received.append(chunk)
    assert received  # the first chunk made it out before the failure
    await client.aclose()
