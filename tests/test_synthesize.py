"""Synthesize prompt assembly tests."""

from __future__ import annotations

from openfusion.config import JudgeConfig
from openfusion.panel import MemberResponse, PanelResult
from openfusion.synthesize import build_judge_messages


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
