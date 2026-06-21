"""Router gate: unit decisions, tool handling, and end-to-end SOLO routing."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import (
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    RouteModel,
    RouterConfig,
    RouterMode,
    Strategy,
    Tier,
    TimeoutsConfig,
)
from openfusion.router import (
    RouteDecision,
    prompt_tier,
    route,
    route_async,
    route_request,
    select_model,
)
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
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "single answer"}}]},
        )

    upstream = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=handler
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
    # SOLO routing makes exactly one upstream call to the configured single model
    # (not a panel fan-out, and not the literal "openfusion").
    assert upstream.call_count == 1
    assert seen == ["pass-model"]


def test_prompt_tier_buckets() -> None:
    assert prompt_tier("hi") == Tier.FAST
    assert prompt_tier("x " * 120) == Tier.BALANCED  # ~240 chars
    assert prompt_tier("compare Postgres and SQLite") == Tier.STRONG  # keyword
    assert prompt_tier("```py\nprint(1)\n```") == Tier.STRONG  # code


def _routes() -> RouterConfig:
    return RouterConfig(
        enabled=True,
        mode=RouterMode.NEVER,  # pure routing, never fuse
        route_models=[
            RouteModel(model="cheap/fast", tier=Tier.FAST),
            RouteModel(model="mid/balanced", tier=Tier.BALANCED),
            RouteModel(model="frontier/strong", tier=Tier.STRONG),
        ],
    )


def test_select_model_by_difficulty() -> None:
    cfg = _routes()
    assert select_model(_body("hi"), cfg).model == "cheap/fast"
    assert select_model(_body("analyze the trade-offs here"), cfg).model == "frontier/strong"


def test_select_model_falls_back_to_nearest_tier() -> None:
    cfg = RouterConfig(
        enabled=True,
        route_models=[RouteModel(model="only/strong", tier=Tier.STRONG)],
    )
    # Easy prompt wants FAST, but only STRONG exists -> nearest available.
    assert select_model(_body("hi"), cfg).model == "only/strong"


def test_select_model_none_without_candidates() -> None:
    assert select_model(_body("hi"), RouterConfig(enabled=True)) is None


def test_routes_helper_unused_guard() -> None:
    # _routes() documents a full route_models config; ensure it builds.
    assert len(_routes().route_models) == 3


@pytest.mark.asyncio
async def test_router_routes_to_best_model_end_to_end(mock_router) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        router=RouterConfig(
            enabled=True,
            mode=RouterMode.NEVER,
            route_models=[
                RouteModel(model="cheap/fast", tier=Tier.FAST),
                RouteModel(model="frontier/strong", tier=Tier.STRONG),
            ],
        ),
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="base")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        easy = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
        hard = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "analyze the trade-offs in depth"}],
            },
        )
    await app.state.upstream_client.aclose()

    assert easy.status_code == 200 and hard.status_code == 200
    assert seen == ["cheap/fast", "frontier/strong"]


@pytest.mark.asyncio
async def test_route_request_heuristic_picks_model(mock_router) -> None:
    cfg = _routes()  # mode NEVER + fast/balanced/strong
    cfg.mode = RouterMode.HEURISTIC
    client = UpstreamClient()
    decision, rm = await route_request(_body("hi"), cfg, client)
    assert decision == RouteDecision.SOLO and rm.model == "cheap/fast"
    decision, rm = await route_request(_body("analyze the trade-offs in depth"), cfg, client)
    assert decision == RouteDecision.FUSE and rm is None
    await client.aclose()


def _model_routes() -> RouterConfig:
    cfg = _routes()
    cfg.mode = RouterMode.MODEL
    cfg.classifier = PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="cls")
    return cfg


@pytest.mark.asyncio
async def test_route_request_classifier_picks_model(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "frontier/strong"}}]}
        )
    )
    client = UpstreamClient()
    decision, rm = await route_request(_body("anything"), _model_routes(), client)
    assert decision == RouteDecision.SOLO and rm.model == "frontier/strong"
    await client.aclose()


@pytest.mark.asyncio
async def test_route_request_classifier_says_fuse(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "FUSE"}}]})
    )
    client = UpstreamClient()
    decision, rm = await route_request(_body("hard one"), _model_routes(), client)
    assert decision == RouteDecision.FUSE and rm is None
    await client.aclose()


@pytest.mark.asyncio
async def test_route_request_classifier_error_falls_back(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "down"}})
    )
    client = UpstreamClient()
    # Classifier fails -> heuristic; a short prompt routes SOLO to the fast model.
    decision, rm = await route_request(_body("hi"), _model_routes(), client)
    assert decision == RouteDecision.SOLO and rm.model == "cheap/fast"
    await client.aclose()
