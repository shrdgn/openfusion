"""Best-effort per-model pricing from an OpenRouter-compatible /models endpoint.

Used only for the pre-run cost estimate. Cached for an hour; failures fall back
to whatever is cached (or empty), so an estimate is always returned — just
without a dollar figure when pricing is unavailable.
"""

from __future__ import annotations

import time

import httpx

_TTL_SECONDS = 3600.0
# TTL applied when a fetch fails (or nothing is cached yet): short enough that a
# transient upstream outage doesn't poison estimates with no/stale pricing for a
# full hour, but still long enough to avoid hammering a downed upstream.
_FAILURE_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, dict[str, dict[str, float]], float]] = {}
# URLs whose pricing fetch is currently in-flight; prevents redundant concurrent calls.
_inflight: set[str] = set()


async def _fetch(base_url: str) -> dict[str, dict[str, float]]:
    url = base_url.rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
    prices: dict[str, dict[str, float]] = {}
    for model in data:
        model_id = model.get("id")
        pricing = model.get("pricing") or {}
        if not isinstance(model_id, str):
            continue
        try:
            prices[model_id] = {
                "prompt": float(pricing.get("prompt", 0) or 0),
                "completion": float(pricing.get("completion", 0) or 0),
            }
        except (TypeError, ValueError):
            continue
    return prices


async def get_prices(base_url: str) -> dict[str, dict[str, float]]:
    """Return {model_id: {prompt, completion}} in USD per token (cached, best-effort).

    A successful fetch is cached for _TTL_SECONDS; a failed fetch is cached for the
    much shorter _FAILURE_TTL_SECONDS so a transient upstream outage doesn't lock in
    empty/stale pricing for a full hour once the upstream recovers.
    """
    now = time.monotonic()
    cached = _cache.get(base_url)
    if cached is not None and now - cached[0] < cached[2]:
        return cached[1]
    # If a concurrent coroutine is already fetching the same URL, return whatever
    # is in the cache (possibly stale) rather than firing a duplicate HTTP call.
    if base_url in _inflight:
        return cached[1] if cached is not None else {}
    _inflight.add(base_url)
    try:
        prices = await _fetch(base_url)
        ttl = _TTL_SECONDS
    except Exception:  # noqa: BLE001 - pricing is optional; never fail the estimate
        prices = cached[1] if cached is not None else {}
        ttl = _FAILURE_TTL_SECONDS
    finally:
        _inflight.discard(base_url)
    _cache[base_url] = (now, prices, ttl)
    return prices
