"""Shared helper for extracting plain text from OpenAI-style chat messages."""

from __future__ import annotations

from typing import Any


def message_text(content: Any) -> str:
    """Extract plain text from a message's ``content`` field.

    ``content`` is either a string or a list of OpenAI multimodal content
    blocks (``{"type": "text", "text": ...}``, ``{"type": "image_url", ...}``,
    etc). Only text blocks contribute -- non-text blocks (e.g. a base64 image
    data URL) are dropped rather than being stringified wholesale into a
    downstream prompt.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""
