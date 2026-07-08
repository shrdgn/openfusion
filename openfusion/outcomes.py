"""Routing outcome store: record request results and bias future routing decisions.

The store tracks an exponential moving average (EMA) of success rate per
(route_label, prompt_tier) key. The router consults it to nudge the heuristic
decision toward whichever path has been working best recently.

Design constraints:
- In-process only; no persistence. Restarts start fresh (warm-up period is short).
- Thread-safe via a simple lock (asyncio event loop is single-threaded, but
  background tasks may update from a different coroutine).
- No sampling bias correction: we count all requests equally, regardless of
  prompt difficulty, so the EMA reflects the mix of traffic the server sees.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from openfusion.config import Tier

# Smoothing factor for the EMA. Lower = slower to adapt, more stable.
# 0.05 means each new observation carries 5% of the weight.
_ALPHA = 0.05

# Minimum observations before the EMA is trusted for routing nudges.
_MIN_OBSERVATIONS = 10

# How much the EMA can shift the heuristic. If fuse EMA >> solo EMA by this
# margin, prefer fuse (and vice versa). Keeps the nudge conservative.
_NUDGE_THRESHOLD = 0.15


@dataclass
class _Bucket:
    ema: float = 0.5  # start neutral
    count: int = 0

    def update(self, success: bool) -> None:
        value = 1.0 if success else 0.0
        self.ema = _ALPHA * value + (1 - _ALPHA) * self.ema
        self.count += 1

    @property
    def reliable(self) -> bool:
        return self.count >= _MIN_OBSERVATIONS


class OutcomeStore:
    """Thread-safe store for per-(route_label, tier) EMA success rates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    def _key(self, route_label: str, tier: Tier | str) -> tuple[str, str]:
        return (route_label, str(tier))

    def record(self, route_label: str, tier: Tier | str, *, success: bool) -> None:
        """Record the outcome of one request."""
        key = self._key(route_label, tier)
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _Bucket()
            self._buckets[key].update(success)

    def ema(self, route_label: str, tier: Tier | str) -> float | None:
        """Return the EMA for a route/tier pair, or None if not enough data."""
        key = self._key(route_label, tier)
        with self._lock:
            bucket = self._buckets.get(key)
        if bucket is None or not bucket.reliable:
            return None
        return bucket.ema

    def prefer_fuse(self, tier: Tier | str) -> bool | None:
        """Return True to prefer FUSE, False to prefer SOLO, None if undecided.

        Compares the EMA of 'fusion' vs 'pass_through' for the given tier.
        Returns None when either bucket lacks enough observations.
        """
        fuse_ema = self.ema("fusion", tier)
        solo_ema = self.ema("pass_through", tier)
        if fuse_ema is None or solo_ema is None:
            return None
        if fuse_ema - solo_ema >= _NUDGE_THRESHOLD:
            return True
        if solo_ema - fuse_ema >= _NUDGE_THRESHOLD:
            return False
        return None  # too close to call

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable summary of current buckets (for /v1/metrics)."""
        with self._lock:
            return {
                f"{route}/{tier}": {"ema": round(b.ema, 4), "count": b.count}
                for (route, tier), b in sorted(self._buckets.items())
            }


# Module-level singleton — imported by server.py and router.py.
OUTCOMES = OutcomeStore()
