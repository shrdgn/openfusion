"""Tests for OpenRouter server-tool injection."""

from __future__ import annotations

from openfusion.config import ToolsConfig
from openfusion.tools import (
    WEB_FETCH_TYPE,
    WEB_SEARCH_TYPE,
    apply_web_tools,
    build_web_tools,
)


def test_disabled_is_noop() -> None:
    body: dict = {"messages": []}
    assert apply_web_tools(body, ToolsConfig(web_search=False)) == {"messages": []}
    assert "tools" not in body
    assert build_web_tools(ToolsConfig(web_search=False)) == []


def test_enabled_adds_search_and_fetch() -> None:
    body: dict = {"messages": []}
    apply_web_tools(body, ToolsConfig(web_search=True, max_results=7, engine="exa"))
    assert body["tools"] == [
        {"type": WEB_SEARCH_TYPE, "parameters": {"engine": "exa", "max_results": 7}},
        {"type": WEB_FETCH_TYPE},
    ]


def test_web_fetch_can_be_disabled() -> None:
    tools = build_web_tools(ToolsConfig(web_search=True, web_fetch=False))
    assert [t["type"] for t in tools] == [WEB_SEARCH_TYPE]


def test_excluded_domains_threaded_to_both_tools() -> None:
    tools = build_web_tools(
        ToolsConfig(web_search=True, excluded_domains=["rubric.example.com"])
    )
    search, fetch = tools
    assert search["parameters"]["excluded_domains"] == ["rubric.example.com"]
    assert fetch["parameters"]["blocked_domains"] == ["rubric.example.com"]


def test_idempotent_does_not_duplicate() -> None:
    body: dict = {"tools": [{"type": WEB_SEARCH_TYPE, "parameters": {"max_results": 3}}]}
    apply_web_tools(body, ToolsConfig(web_search=True, max_results=7, web_fetch=False))
    # Existing web_search tool is left as-is, not duplicated.
    assert body["tools"] == [{"type": WEB_SEARCH_TYPE, "parameters": {"max_results": 3}}]


def test_preserves_existing_other_tools() -> None:
    body: dict = {"tools": [{"type": "function", "function": {"name": "foo"}}]}
    apply_web_tools(body, ToolsConfig(web_search=True, web_fetch=False))
    assert body["tools"][0] == {"type": "function", "function": {"name": "foo"}}
    assert body["tools"][1]["type"] == WEB_SEARCH_TYPE
