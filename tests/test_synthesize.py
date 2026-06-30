"""Synthesize prompt assembly and streaming tests."""

from __future__ import annotations

import pytest

from openfusion.config import JudgeConfig, OpenFusionConfig, PanelMember
from openfusion.errors import UpstreamError
from openfusion.panel import MemberResponse, PanelResult
from openfusion.synthesize import (
    _extract_delta_content,
    _extract_finish_reason,
    build_judge_messages,
    synthesize,
)
from openfusion.upstream import UpstreamClient


def test_build_judge_messages_includes_panel_blocks() -> None:
    panel = PanelResult(
        responses=[
            MemberResponse(label="a", content="first", model="m"),
            MemberResponse(label="b", content="second", model="m"),
        ]
    )
    judge = JudgeConfig(
        base_url="https://example.com/v1",
        api_key="k",
        model="j",
        max_panel_tokens=100000,
    )
    messages = build_judge_messages(
        [{"role": "user", "content": "question"}],
        panel,
        judge,
    )

    assert messages[0]["role"] == "system"
    assert "Honor the original user's output format" in messages[0]["content"]
    assert "unique insights" in messages[0]["content"]
    assert "### a" in messages[1]["content"]
    assert "second" in messages[1]["content"]
    # System prompt must appear only in the system message, not duplicated in the user turn.
    assert "You are the synthesizer" not in messages[1]["content"]


def test_truncate_panel_responses() -> None:
    from openfusion.synthesize import _truncate_panel_responses

    responses = [
        ("short", "tiny"),
        ("long", "x" * 10000),
    ]
    capped = _truncate_panel_responses(responses, max_tokens=100)
    assert any("truncated" in content for _, content in capped)


def test_truncate_panel_responses_all_short_breaks_gracefully() -> None:
    """When every response is already ≤200 chars, no further truncation is possible.

    The function must return the responses unchanged rather than looping forever or
    dropping entries — callers must tolerate a result that still exceeds the budget.
    """
    from openfusion.synthesize import _truncate_panel_responses

    # Three 100-char responses; token estimate ≈ 75 total, max_tokens=1 forces break.
    responses = [("a", "a" * 100), ("b", "b" * 100), ("c", "c" * 100)]
    result = _truncate_panel_responses(responses, max_tokens=1)
    # All entries must survive — nothing was dropped.
    assert len(result) == 3
    # None were truncated (they were already at the floor).
    assert all(len(content) == 100 for _, content in result)


# ---------------------------------------------------------------------------
# _extract_delta_content
# ---------------------------------------------------------------------------


def test_extract_delta_content_returns_text() -> None:
    chunk = {"choices": [{"delta": {"content": "hello"}}]}
    assert _extract_delta_content(chunk) == "hello"


def test_extract_delta_content_empty_when_no_choices() -> None:
    assert _extract_delta_content({}) == ""
    assert _extract_delta_content({"choices": []}) == ""


def test_extract_delta_content_empty_when_content_is_none() -> None:
    assert _extract_delta_content({"choices": [{"delta": {"content": None}}]}) == ""


def test_extract_delta_content_empty_when_delta_absent() -> None:
    assert _extract_delta_content({"choices": [{}]}) == ""


# ---------------------------------------------------------------------------
# _extract_finish_reason
# ---------------------------------------------------------------------------


def test_extract_finish_reason_stop() -> None:
    chunk = {"choices": [{"finish_reason": "stop"}]}
    assert _extract_finish_reason(chunk) == "stop"


def test_extract_finish_reason_none_when_no_choices() -> None:
    assert _extract_finish_reason({}) is None
    assert _extract_finish_reason({"choices": []}) is None


def test_extract_finish_reason_none_when_null() -> None:
    # OpenAI streams send null finish_reason on non-terminal chunks.
    assert _extract_finish_reason({"choices": [{"finish_reason": None}]}) is None


def test_extract_finish_reason_length() -> None:
    chunk = {"choices": [{"finish_reason": "length"}]}
    assert _extract_finish_reason(chunk) == "length"


# ---------------------------------------------------------------------------
# synthesize() async generator
# ---------------------------------------------------------------------------


def _synth_config() -> OpenFusionConfig:
    return OpenFusionConfig(
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
    )


def _panel_with(*contents: str) -> PanelResult:
    return PanelResult(
        responses=[
            MemberResponse(label=f"m{i}", content=c, model="m")
            for i, c in enumerate(contents)
        ]
    )


@pytest.mark.asyncio
async def test_synthesize_raises_when_no_judge() -> None:
    config = OpenFusionConfig(
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
    )
    client = UpstreamClient()
    with pytest.raises(UpstreamError, match="Judge is not configured"):
        async for _ in synthesize(
            {"messages": [{"role": "user", "content": "q"}]},
            _panel_with("answer"),
            config,
            client,
        ):
            pass
    await client.aclose()


@pytest.mark.asyncio
async def test_synthesize_raises_when_messages_not_list() -> None:
    client = UpstreamClient()
    with pytest.raises(UpstreamError, match="messages must be a list"):
        async for _ in synthesize(
            {"messages": "bad"},
            _panel_with("answer"),
            _synth_config(),
            client,
        ):
            pass
    await client.aclose()


@pytest.mark.asyncio
async def test_synthesize_yields_deltas_from_stream(monkeypatch) -> None:
    stream_chunks = [
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    async def _mock_chat_completion(*_args, **_kwargs):
        async def _gen():
            for chunk in stream_chunks:
                yield chunk

        return _gen()

    client = UpstreamClient()
    monkeypatch.setattr(client, "chat_completion", _mock_chat_completion)

    deltas, finish_reasons = [], []
    async for delta, _usage, finish_reason in synthesize(
        {"messages": [{"role": "user", "content": "q"}]},
        _panel_with("panel answer"),
        _synth_config(),
        client,
    ):
        deltas.append(delta)
        if finish_reason:
            finish_reasons.append(finish_reason)

    assert deltas == ["hello", " world", ""]
    assert finish_reasons == ["stop"]


@pytest.mark.asyncio
async def test_synthesize_raises_when_response_not_streaming(monkeypatch) -> None:
    async def _mock_chat_completion(*_args, **_kwargs):
        return {"choices": [{"message": {"content": "not a stream"}}]}

    client = UpstreamClient()
    monkeypatch.setattr(client, "chat_completion", _mock_chat_completion)

    with pytest.raises(UpstreamError, match="Expected streaming"):
        async for _ in synthesize(
            {"messages": [{"role": "user", "content": "q"}]},
            _panel_with("panel answer"),
            _synth_config(),
            client,
        ):
            pass


@pytest.mark.asyncio
async def test_synthesize_yields_usage_chunk(monkeypatch) -> None:
    usage_data = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    stream_chunks = [
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage_data},
    ]

    async def _mock_chat_completion(*_args, **_kwargs):
        async def _gen():
            for chunk in stream_chunks:
                yield chunk

        return _gen()

    client = UpstreamClient()
    monkeypatch.setattr(client, "chat_completion", _mock_chat_completion)

    usages = []
    async for _delta, usage, _reason in synthesize(
        {"messages": [{"role": "user", "content": "q"}]},
        _panel_with("answer"),
        _synth_config(),
        client,
    ):
        if usage:
            usages.append(usage)

    assert len(usages) == 1
    assert usages[0]["prompt_tokens"] == 10
