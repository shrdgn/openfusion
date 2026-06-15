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
    assert "### a" in messages[1]["content"]
    assert "second" in messages[1]["content"]


def test_truncate_panel_responses() -> None:
    from openfusion.synthesize import _truncate_panel_responses

    responses = [
        ("short", "tiny"),
        ("long", "x" * 10000),
    ]
    capped = _truncate_panel_responses(responses, max_tokens=100)
    assert any("truncated" in content for _, content in capped)
