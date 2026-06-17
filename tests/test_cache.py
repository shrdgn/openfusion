"""Prompt-cache breakpoint marking."""

from __future__ import annotations

from openfusion.cache import mark_cache_breakpoint


def test_marks_system_message() -> None:
    body = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]}
    mark_cache_breakpoint(body)
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    # The user message is untouched.
    assert body["messages"][1]["content"] == "u"


def test_marks_first_user_when_no_system() -> None:
    body = {"messages": [{"role": "user", "content": "u"}]}
    mark_cache_breakpoint(body)
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_marks_last_block_of_list_content() -> None:
    body = {"messages": [{"role": "system", "content": [{"type": "text", "text": "a"}]}]}
    mark_cache_breakpoint(body)
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_noop_on_empty_messages() -> None:
    assert mark_cache_breakpoint({"messages": []}) == {"messages": []}
