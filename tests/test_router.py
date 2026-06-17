"""Router gate: unit decisions, tool handling, and end-to-end SOLO routing."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import OpenFusionConfig, PanelMember, RouterConfig, RouterMode
from openfusion.router import RouteDecision, route, route_async
from openfusion.server import _requires_pass_through_tools, create_app
from openfusion.tools import WEB_FETCH_TYPE, WEB_SEARCH_TYPE
from openfusion.upstream import UpstreamClient


def _body(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_short_prompt_routes_solo() -> None:
    assert route(_body("hi there"), RouterConfig(enabled=True)) == RouteDecision.SOLO


def test_keyword_routes_fuse() -> None:
    assert route(_body("compare A and B"), RouterConfig(enabled=True)) == RouteDecision.FUSE


def test_long_prompt_routes_fuse() -> None:
    assert route(_body("x " * 200), RouterConfig(enabled=True)) == RouteDecision.FUSE


def test_code_block_routes_fuse() -> None:
    decision = route(_body("fix ```py\nprint(1)\n```"), RouterConfig(enabled=True))
    assert decision == RouteDecision.FUSE


def test_mode_always_overrides_simple_prompt() -> None:
    assert route(_body("hi"), RouterConfig(mode=RouterMode.ALWAYS)) == RouteDecision.FUSE


def test_mode_never_overrides_hard_prompt() -> None:
    assert route(_body("compare " + "x " * 200), RouterConfig(mode=RouterMode.NEVER)) == (
        RouteDecision.SOLO
    )


def test_server_executable_tools_do_not_force_pass_through() -> None:
    body = {
        **_body("research this"),
        "tools": [{"type": WEB_SEARCH_TYPE}, {"type": WEB_FETCH_TYPE}],
    }
    assert _requires_pass_through_tools(body) is False


def test_function_tools_force_pass_through() -> None:
    body = {**_body("hi"), "tools": [{"type": "function", "function": {"name": "f"}}]}
    assert _requires_pass_through_tools(body) is True


def test_mixed_tools_force_pass_through() -> None:
    body = {**_body("hi"), "tools": [{"type": WEB_SEARCH_TYPE}, {"type": "function"}]}
    assert _requires_pass_through_tools(body) is True


def test_tool_role_message_forces_pass_through() -> None:
    assert _requires_pass_through_tools({"messages": [{"role": "tool", "content": "x"}]}) is True


def _model_router() -> RouterConfig:
    return RouterConfig(
        enabled=True,
        mode=RouterMode.MODEL,
        classifier=PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="cls"),
    )


@pytest.mark.asyncio
async def test_route_async_model_mode_respects_classifier(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "SOLO"}}]}
        )
    )
    client = UpstreamClient()
    decision = await route_async(_body("anything"), _model_router(), client)
    assert decision == RouteDecision.SOLO
    await client.aclose()


@pytest.mark.asyncio
async def test_route_async_falls_back_to_heuristic_on_error(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "down"}})
    )
    client = UpstreamClient()
    # Classifier errored, so the heuristic decides: a keyword prompt fuses.
    decision = await route_async(_body("compare these options"), _model_router(), client)
    assert decision == RouteDecision.FUSE
    await client.aclose()


@pytest.mark.asyncio
async def test_route_async_heuristic_mode_makes_no_call(mock_router) -> None:
    route_mock = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    client = UpstreamClient()
    decision = await route_async(_body("hi"), RouterConfig(enabled=True), client)
    assert decision == RouteDecision.SOLO
    assert not route_mock.called
    await client.aclose()


async def test_router_solo_answers_with_single_call(
    test_config: OpenFusionConfig, mock_router
) -> None:
    test_config.router = RouterConfig(enabled=True, mode=RouterMode.NEVER)
    app = create_app(test_config)
    upstream = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "solo",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "single answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "single answer"
    # SOLO routing makes exactly one upstream call, not a panel fan-out.
    assert upstream.call_count == 1
