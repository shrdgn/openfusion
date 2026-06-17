"""Prompt-cache breakpoint marking for the self-fusion shared prefix.

Self-fusion sends the same prompt to one model N times. Providers that support
prompt caching (e.g. Anthropic via OpenRouter) can reuse a cached prefix when a
``cache_control`` breakpoint marks where the shared part ends. This adds that
marker to the prompt's stable prefix (the system message, or the first user
message). Providers that don't support it ignore the field, so it's safe.
"""

from __future__ import annotations

from typing import Any

_BREAKPOINT = {"type": "ephemeral"}


def mark_cache_breakpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Mark the prompt's stable prefix with a ``cache_control`` breakpoint.

    Mutates and returns ``body``. Idempotent and a no-op when there is no
    suitable message.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    target = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            target = message  # last system message wins
    if target is None:
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                target = message
                break
    if target is None:
        return body

    content = target.get("content")
    if isinstance(content, str):
        target["content"] = [
            {"type": "text", "text": content, "cache_control": dict(_BREAKPOINT)}
        ]
    elif isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict):
                block.setdefault("cache_control", dict(_BREAKPOINT))
                break
    return body
