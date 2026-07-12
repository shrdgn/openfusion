"""Ranked-choice aggregation: judge picks the best panel answer."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
    TimeoutsConfig,
)
from openfusion.errors import UpstreamError
from openfusion.panel import MemberResponse, PanelResult
from openfusion.ranked import _original_user_text, _parse_choice, build_ranking_messages, pick_best
from openfusion.server import create_app
from openfusion.upstream import UpstreamClient


def test_parse_choice() -> None:
    assert _parse_choice("2", 3) == 1
    assert _parse_choice("The best is 3.", 3) == 2
    assert _parse_choice("garbage", 3) == 0  # unparseable defaults to first
    assert _parse_choice("9", 3) == 0  # out of range defaults to first


def _config() -> OpenFusionConfig:
    return OpenFusionConfig(
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
    )


@pytest.mark.asyncio
async def test_pick_best_returns_selected_answer(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "2"}}]},
        )
    )
    panel = PanelResult(
        responses=[
            MemberResponse(label="a", content="answer one", model="m"),
            MemberResponse(label="b", content="answer two", model="m"),
        ]
    )
    client = UpstreamClient()

    content, meta = await pick_best(
        {"messages": [{"role": "user", "content": "q"}]}, panel, _config(), client
    )

    assert content == "answer two"
    assert meta["winner"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_pick_best_single_response_skips_judge(mock_router) -> None:
    route = mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    panel = PanelResult(responses=[MemberResponse(label="a", content="only", model="m")])
    client = UpstreamClient()

    content, _ = await pick_best(
        {"messages": [{"role": "user", "content": "q"}]}, panel, _config(), client
    )

    assert content == "only"
    assert not route.called
    await client.aclose()


@pytest.mark.asyncio
async def test_pick_best_requires_judge() -> None:
    config = OpenFusionConfig(
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
        judge=None,
    )
    panel = PanelResult(
        responses=[
            MemberResponse(label="a", content="one", model="m"),
            MemberResponse(label="b", content="two", model="m"),
        ]
    )
    client = UpstreamClient()

    with pytest.raises(UpstreamError, match="requires a judge"):
        await pick_best({"messages": [{"role": "user", "content": "q"}]}, panel, config, client)
    await client.aclose()


@pytest.mark.asyncio
async def test_pick_best_rejects_non_list_messages() -> None:
    panel = PanelResult(
        responses=[
            MemberResponse(label="a", content="one", model="m"),
            MemberResponse(label="b", content="two", model="m"),
        ]
    )
    client = UpstreamClient()

    with pytest.raises(UpstreamError, match="must be a list"):
        await pick_best({"messages": "not-a-list"}, panel, _config(), client)
    await client.aclose()


@pytest.mark.asyncio
async def test_pick_best_rejects_non_json_ranking_response(monkeypatch) -> None:
    # client.chat_completion already guards non-JSON HTTP responses (raises
    # UpstreamError before returning), so this defensive branch is only
    # reachable if a client implementation returns something unexpected.
    panel = PanelResult(
        responses=[
            MemberResponse(label="a", content="one", model="m"),
            MemberResponse(label="b", content="two", model="m"),
        ]
    )
    async def _mock_chat_completion(*_args, **_kwargs):
        return "not-a-dict"

    client = UpstreamClient()
    monkeypatch.setattr(client, "chat_completion", _mock_chat_completion)

    with pytest.raises(UpstreamError, match="Expected JSON ranking response"):
        await pick_best({"messages": [{"role": "user", "content": "q"}]}, panel, _config(), client)
    await client.aclose()


@pytest.mark.asyncio
async def test_ranked_aggregator_end_to_end(mock_router) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        if "Candidate answers:" in text:  # the ranking call
            return httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "2"}}]}
            )
        # a panel member call; content encodes which member via model name
        answer = f"ans-{payload['model']}"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": answer}}]},
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.RANKED,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "q"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 200
    # The judge picked candidate 2 → the second panel member's answer.
    assert response.json()["choices"][0]["message"]["content"] == "ans-m2"


@pytest.mark.asyncio
async def test_ranked_aggregator_streams_via_http(mock_router) -> None:
    """A streaming request with aggregator=ranked wires up to ranked_and_stream.

    Exercises server.py's _fusion_stream branch that selects ranked_and_stream
    as the streamer -- every other case (VOTE, JUDGE) already has HTTP-level
    streaming coverage, but RANKED didn't.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        if "Candidate answers:" in text:  # the ranking call
            return httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "1"}}]}
            )
        answer = f"ans-{payload['model']}"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": answer}}]},
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.RANKED,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1", label="a"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2", label="b"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    app = create_app(config)

    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as http_client,
        http_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "openfusion",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
            },
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])
    await app.state.upstream_client.aclose()

    # The judge picked candidate 1 -> the first panel member's answer.
    assert "ans-m1" in body
    assert body.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------------------
# _original_user_text
# ---------------------------------------------------------------------------


def test_original_user_text_strips_system_messages() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    result = _original_user_text(messages)
    assert "Be helpful." not in result
    assert "What is 2+2?" in result


def test_original_user_text_includes_assistant_turns() -> None:
    # Non-system messages (user + assistant) are included to give the ranker full context.
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    result = _original_user_text(messages)
    assert "Question" in result
    assert "Previous answer" in result


def test_original_user_text_empty_messages() -> None:
    assert _original_user_text([]) == ""


def test_original_user_text_only_system_gives_empty() -> None:
    messages = [{"role": "system", "content": "System only."}]
    assert _original_user_text(messages) == ""


# ---------------------------------------------------------------------------
# build_ranking_messages
# ---------------------------------------------------------------------------


def _two_response_panel() -> PanelResult:
    return PanelResult(
        responses=[
            MemberResponse(label="a", content="first answer", model="m"),
            MemberResponse(label="b", content="second answer", model="m"),
        ]
    )


def test_build_ranking_messages_structure() -> None:
    messages = build_ranking_messages(
        [{"role": "user", "content": "What is the capital of France?"}],
        _two_response_panel(),
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_ranking_messages_numbers_candidates() -> None:
    messages = build_ranking_messages(
        [{"role": "user", "content": "q"}],
        _two_response_panel(),
    )
    user_content = messages[1]["content"]
    assert "[1] first answer" in user_content
    assert "[2] second answer" in user_content


def test_build_ranking_messages_includes_original_question() -> None:
    messages = build_ranking_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Which is better?"},
        ],
        _two_response_panel(),
    )
    user_content = messages[1]["content"]
    # System message should not appear in the ranking prompt body.
    assert "You are helpful." not in user_content
    assert "Which is better?" in user_content


def test_build_ranking_messages_three_candidates_numbered_correctly() -> None:
    panel = PanelResult(
        responses=[
            MemberResponse(label="x", content="ans-one", model="m"),
            MemberResponse(label="y", content="ans-two", model="m"),
            MemberResponse(label="z", content="ans-three", model="m"),
        ]
    )
    messages = build_ranking_messages([{"role": "user", "content": "q"}], panel)
    user_content = messages[1]["content"]
    assert "[1] ans-one" in user_content
    assert "[2] ans-two" in user_content
    assert "[3] ans-three" in user_content
