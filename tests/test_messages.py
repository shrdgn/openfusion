"""Tests for the shared message-content text extraction helper."""

from __future__ import annotations

from openfusion.messages import message_text


def test_message_text_plain_string() -> None:
    assert message_text("hello") == "hello"


def test_message_text_multimodal_blocks_collect_text_only() -> None:
    content = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "in detail"},
    ]
    assert message_text(content) == "describe this\nin detail"


def test_message_text_image_only_blocks_yield_empty_string() -> None:
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert message_text(content) == ""


def test_message_text_none_or_unrecognized_type_yields_empty_string() -> None:
    assert message_text(None) == ""
    assert message_text(42) == ""
    assert message_text({"type": "text", "text": "not a list"}) == ""
