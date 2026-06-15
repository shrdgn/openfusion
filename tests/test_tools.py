"""Tests for server-side tool injection."""

from __future__ import annotations

from openfusion.config import ToolsConfig
from openfusion.tools import apply_web_tools


def test_disabled_is_noop() -> None:
    body: dict = {"messages": []}
    assert apply_web_tools(body, ToolsConfig(web_search=False)) == {"messages": []}
    assert "plugins" not in body


def test_enabled_adds_web_plugin() -> None:
    body: dict = {"messages": []}
    apply_web_tools(body, ToolsConfig(web_search=True, max_results=7))
    assert body["plugins"] == [{"id": "web", "max_results": 7}]


def test_idempotent_does_not_duplicate() -> None:
    body: dict = {"plugins": [{"id": "web", "max_results": 3}]}
    apply_web_tools(body, ToolsConfig(web_search=True, max_results=7))
    # Existing web plugin is left as-is, not duplicated.
    assert body["plugins"] == [{"id": "web", "max_results": 3}]


def test_preserves_existing_other_plugins() -> None:
    body: dict = {"plugins": [{"id": "fusion"}]}
    apply_web_tools(body, ToolsConfig(web_search=True, max_results=5))
    assert body["plugins"] == [{"id": "fusion"}, {"id": "web", "max_results": 5}]
