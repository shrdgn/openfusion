"""Inject OpenRouter agentic server tools (web search/fetch) into request bodies.

We rely on OpenRouter's server-side tools `openrouter:web_search` and
`openrouter:web_fetch`: the model calls them in an agentic loop that OpenRouter
executes upstream, folding the results into a normal content response. That
lets panel members (and the solo baseline) do real multi-step research without
us implementing a client-side tool-call loop.
"""

from __future__ import annotations

from typing import Any

from openfusion.config import ToolsConfig

WEB_SEARCH_TYPE = "openrouter:web_search"
WEB_FETCH_TYPE = "openrouter:web_fetch"


def build_web_tools(tools: ToolsConfig) -> list[dict[str, Any]]:
    """Return the OpenRouter server-tool definitions for the enabled web tools.

    Empty when ``web_search`` is disabled. Shared by the panel/judge injection
    and the solo baseline so both sides get an identical tool set.
    """
    if not tools.web_search:
        return []

    search_params: dict[str, Any] = {"engine": tools.engine, "max_results": tools.max_results}
    if tools.excluded_domains:
        search_params["excluded_domains"] = list(tools.excluded_domains)
    definitions: list[dict[str, Any]] = [{"type": WEB_SEARCH_TYPE, "parameters": search_params}]

    if tools.web_fetch:
        fetch_def: dict[str, Any] = {"type": WEB_FETCH_TYPE}
        if tools.excluded_domains:
            fetch_def["parameters"] = {"blocked_domains": list(tools.excluded_domains)}
        definitions.append(fetch_def)

    return definitions


def apply_web_tools(body: dict[str, Any], tools: ToolsConfig) -> dict[str, Any]:
    """Add the enabled web server tools to ``body['tools']``.

    No-op when ``web_search`` is disabled. Idempotent: a tool type already
    present is left untouched, and any other tools on the body are preserved.
    Mutates and returns ``body`` for convenience.
    """
    definitions = build_web_tools(tools)
    if not definitions:
        return body

    existing = list(body.get("tools") or [])
    existing_types = {tool.get("type") for tool in existing if isinstance(tool, dict)}
    for definition in definitions:
        if definition["type"] not in existing_types:
            existing.append(definition)
    body["tools"] = existing
    return body
