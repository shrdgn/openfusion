"""Tests for Anthropic native upstream support."""

from __future__ import annotations

from openfusion.upstream import (
    _anthropic_stream_event_to_openai,
    _anthropic_to_openai,
    _openai_message_to_anthropic,
    _openai_to_anthropic,
)

# ---------------------------------------------------------------------------
# _openai_to_anthropic
# ---------------------------------------------------------------------------


def test_basic_user_message() -> None:
    body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 512}
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=False)
    assert result["model"] == "claude-3-haiku-20240307"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]
    assert result["max_tokens"] == 512
    assert result["stream"] is False
    assert "system" not in result


def test_system_message_extracted() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
    }
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=True)
    assert result["system"] == "You are a helpful assistant."
    assert result["messages"] == [{"role": "user", "content": "Hi"}]
    assert result["stream"] is True


def test_default_max_tokens_applied_when_missing() -> None:
    body = {"messages": [{"role": "user", "content": "x"}]}
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=False)
    assert result["max_tokens"] == 1024


def test_optional_fields_forwarded() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["\n"],
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
    assert result["stop"] == ["\n"]


def test_none_optional_fields_not_forwarded() -> None:
    body = {"messages": [{"role": "user", "content": "x"}], "temperature": None}
    result = _openai_to_anthropic(body, "m", stream=False)
    assert "temperature" not in result


# ---------------------------------------------------------------------------
# _anthropic_to_openai
# ---------------------------------------------------------------------------


def test_response_conversion() -> None:
    raw = {
        "id": "msg_abc",
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Hello there!"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = _anthropic_to_openai(raw)
    assert result["id"] == "msg_abc"
    assert result["object"] == "chat.completion"
    assert result["model"] == "claude-3-haiku-20240307"
    assert result["choices"][0]["message"]["content"] == "Hello there!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15


def test_max_tokens_stop_reason() -> None:
    raw = {
        "id": "x",
        "model": "m",
        "stop_reason": "max_tokens",
        "content": [{"type": "text", "text": "truncated"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    result = _anthropic_to_openai(raw)
    assert result["choices"][0]["finish_reason"] == "length"


def test_multiple_text_blocks_concatenated() -> None:
    raw = {
        "id": "x",
        "model": "m",
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": "Part 1 "},
            {"type": "text", "text": "Part 2"},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }
    result = _anthropic_to_openai(raw)
    assert result["choices"][0]["message"]["content"] == "Part 1 Part 2"


# ---------------------------------------------------------------------------
# _anthropic_stream_event_to_openai
# ---------------------------------------------------------------------------


def test_content_block_delta() -> None:
    event = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Hello"},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["choices"][0]["delta"]["content"] == "Hello"
    assert chunk["choices"][0]["finish_reason"] is None


def test_message_delta_with_stop() -> None:
    event = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 42},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["choices"][0]["finish_reason"] == "stop"
    assert chunk["usage"]["completion_tokens"] == 42


def test_message_start_with_usage() -> None:
    event = {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 100}},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["usage"]["prompt_tokens"] == 100
    assert chunk["usage"]["completion_tokens"] == 0


def test_ping_returns_none() -> None:
    assert _anthropic_stream_event_to_openai({"type": "ping"}) is None


def test_unknown_event_returns_none() -> None:
    assert _anthropic_stream_event_to_openai({"type": "content_block_start"}) is None


def test_content_block_delta_unknown_delta_type_returns_none() -> None:
    """Anthropic can emit delta kinds this proxy doesn't translate (e.g. a
    citations delta); those are skipped rather than surfaced as a bogus chunk."""
    event = {
        "type": "content_block_delta",
        "delta": {"type": "citations_delta", "citation": {}},
    }
    assert _anthropic_stream_event_to_openai(event) is None


# ---------------------------------------------------------------------------
# Round-trip: config provider inference
# ---------------------------------------------------------------------------


def test_panel_member_infers_anthropic_provider() -> None:
    from openfusion.config import PanelMember

    m = PanelMember(
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        model="claude-3-haiku-20240307",
    )
    assert m.provider == "anthropic"


def test_panel_member_infers_openai_provider() -> None:
    from openfusion.config import PanelMember

    m = PanelMember(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="gpt-4o-mini",
    )
    assert m.provider == "openai"


# ---------------------------------------------------------------------------
# Tool call translation
# ---------------------------------------------------------------------------


def test_tools_translated_to_anthropic_format() -> None:
    body = {
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
    }
    result = _openai_to_anthropic(body, "claude-3-haiku-20240307", stream=False)
    assert "tools" in result
    tool = result["tools"][0]
    assert tool["name"] == "get_weather"
    assert tool["description"] == "Get current weather"
    assert tool["input_schema"]["type"] == "object"
    assert "function" not in tool


def test_tool_choice_auto_mapped() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "auto",
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["tool_choice"] == {"type": "auto"}


def test_tool_choice_none_mapped_to_none() -> None:
    """A client that explicitly disabled tool use for this turn must not have that
    intent silently downgraded to "auto" (which invites a tool_use response the
    caller isn't expecting)."""
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "none",
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["tool_choice"] == {"type": "none"}


def test_tool_choice_required_mapped_to_any() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "required",
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["tool_choice"] == {"type": "any"}


def test_tool_choice_specific_function_mapped() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "my_fn", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "my_fn"}},
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["tool_choice"] == {"type": "tool", "name": "my_fn"}


def test_system_message_returns_none_directly() -> None:
    """``_openai_to_anthropic`` filters system messages before calling this, but
    the function is defensive on its own: a system-role message maps to no
    Anthropic message rather than being mistranslated as a user/assistant turn.
    """
    assert _openai_message_to_anthropic({"role": "system", "content": "be nice"}) is None


def test_assistant_message_with_text_and_tool_calls() -> None:
    """A model can emit text before deciding to call a tool; both must survive
    translation as separate content blocks, text first."""
    body = {
        "messages": [
            {"role": "user", "content": "What's 2+2?"},
            {
                "role": "assistant",
                "content": "Let me calculate that.",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                    }
                ],
            },
        ]
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    blocks = result["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "Let me calculate that."}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["input"] == {"expr": "2+2"}


def test_tool_call_invalid_json_arguments_default_to_empty_dict() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "not-json"},
                    }
                ],
            },
        ]
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    assert result["messages"][0]["content"][0]["input"] == {}


def test_tool_result_message_converted() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "What's 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "4"},
        ]
    }
    result = _openai_to_anthropic(body, "m", stream=False)
    msgs = result["messages"]
    # assistant message with tool_use block
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[1]["content"][0]["id"] == "call_abc"
    assert msgs[1]["content"][0]["input"] == {"expr": "2+2"}
    # tool result as user message
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "call_abc"
    assert msgs[2]["content"][0]["content"] == "4"


def test_anthropic_tool_use_response_converted() -> None:
    raw = {
        "id": "msg_xyz",
        "model": "claude-3-haiku-20240307",
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"location": "NYC"}},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }
    result = _anthropic_to_openai(raw)
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    tc = result["choices"][0]["message"]["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["function"]["name"] == "get_weather"
    import json as _json
    assert _json.loads(tc["function"]["arguments"]) == {"location": "NYC"}


def test_content_block_start_tool_use_streamed() -> None:
    event = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_weather"},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    tc = chunk["choices"][0]["delta"]["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == ""


def test_input_json_delta_streamed() -> None:
    event = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"loc'},
    }
    chunk = _anthropic_stream_event_to_openai(event)
    assert chunk is not None
    assert chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"loc'


def test_content_block_start_text_returns_none() -> None:
    event = {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}
    assert _anthropic_stream_event_to_openai(event) is None


def test_panel_member_explicit_provider_wins() -> None:
    from openfusion.config import PanelMember

    m = PanelMember(
        base_url="https://example.com/v1",
        api_key="key",
        model="custom-model",
        provider="anthropic",
    )
    assert m.provider == "anthropic"
