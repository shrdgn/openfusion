"""Inject server-side tools (web search) into upstream request bodies.

We rely on the upstream's server-side web plugin (OpenRouter's Exa-backed
`web` plugin), which runs search upstream and folds the results into a normal
content response. That means panel members can do real research without us
implementing a client-side tool-call loop.
"""

from __future__ import annotations

from typing import Any

from openfusion.config import ToolsConfig

WEB_PLUGIN_ID = "web"


def apply_web_tools(body: dict[str, Any], tools: ToolsConfig) -> dict[str, Any]:
    """Add the upstream web plugin to ``body['plugins']`` when enabled.

    No-op when ``web_search`` is disabled. Idempotent: an existing ``web``
    plugin is left untouched, and any other plugins already on the body are
    preserved. Mutates and returns ``body`` for convenience.
    """
    if not tools.web_search:
        return body

    plugins = list(body.get("plugins") or [])
    if any(isinstance(plugin, dict) and plugin.get("id") == WEB_PLUGIN_ID for plugin in plugins):
        return body

    plugins.append({"id": WEB_PLUGIN_ID, "max_results": tools.max_results})
    body["plugins"] = plugins
    return body
