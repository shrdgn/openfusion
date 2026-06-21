"""In-process TTL + LRU cache of fused answers, keyed by prompt and recipe."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from openfusion.config import OpenFusionConfig


def cache_key(body: dict[str, Any], config: OpenFusionConfig) -> str:
    """A stable key over the request + the recipe that determines the answer."""
    payload = {
        "messages": body.get("messages"),
        "strategy": config.strategy.value,
        "aggregator": config.aggregator.value,
        "panel": [member.model for member in config.panel],
        "judge": config.judge.model if config.judge else None,
        "tools": [config.tools.web_search, config.tools.web_fetch],
        "max_tokens": body.get("max_tokens") or config.cost_controls.judge_max_tokens,
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
