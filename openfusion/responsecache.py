"""In-process TTL + LRU cache of fused answers, keyed by prompt and recipe."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from openfusion.config import OpenFusionConfig


def _compact_content(content: Any) -> Any:
    """Collapse large multimodal blocks in a message's ``content`` to a hash.

    A multimodal ``image_url`` block carries its image as a base64 ``data:``
    URL, which can be multiple megabytes. Passing it through unchanged would
    have ``cache_key`` re-serialize and re-hash that data on every
    cache-enabled request (hit or miss) as part of the outer JSON blob.
    Hashing each image block's own payload up front keeps the key just as
    stable and collision-resistant while keeping the outer blob's size
    independent of attachment size.
    """
    if not isinstance(content, list):
        return content
    compact = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            url = block.get("image_url")
            digest = hashlib.sha256(str(url).encode("utf-8")).hexdigest()
            compact.append({"type": "image_url", "hash": digest})
        else:
            compact.append(block)
    return compact


def _compact_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    return [
        {**message, "content": _compact_content(message.get("content"))}
        if isinstance(message, dict)
        else message
        for message in messages
    ]


def cache_key(body: dict[str, Any], config: OpenFusionConfig) -> str:
    """A stable key over the request + the recipe that determines the answer.

    Includes a fingerprint of the resolved API key(s) so that per-tenant
    configs (e.g. a ``config_resolver`` handing out BYO keys, see
    docs/EMBEDDING.md) never share cache entries: a shared, process-wide
    ``ResponseCache`` would otherwise serve one tenant's cached answer to
    another tenant who happens to send the same prompt against the same
    panel/judge model names.
    """
    credentials = "|".join(
        [*(member.api_key for member in config.panel), config.judge.api_key if config.judge else ""]
    )
    payload = {
        "messages": _compact_messages(body.get("messages")),
        "strategy": config.strategy.value,
        "aggregator": config.aggregator.value,
        "panel": [member.model for member in config.panel],
        "judge": config.judge.model if config.judge else None,
        "credentials": hashlib.sha256(credentials.encode("utf-8")).hexdigest(),
        "tools": [config.tools.web_search, config.tools.web_fetch],
        "max_tokens": body.get("max_tokens") or config.cost_controls.judge_max_tokens,
        # Sampling parameters affect the response; omitting them would serve a
        # cached answer from a run with different generation settings.
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "seed": body.get("seed"),
        "n": body.get("n"),
        "presence_penalty": body.get("presence_penalty"),
        "frequency_penalty": body.get("frequency_penalty"),
        # stop sequences, logit bias, and response format change the generated output;
        # omitting them would serve a cached answer from a run with different constraints.
        "stop": body.get("stop"),
        "logit_bias": body.get("logit_bias"),
        "response_format": body.get("response_format"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Bounded TTL cache; least-recently-used entries are evicted past capacity."""

    def __init__(self, ttl_seconds: float, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._store.get(key)
        if item is None:
            return None
        ts, value = item
        if time.monotonic() - ts > self._ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)
