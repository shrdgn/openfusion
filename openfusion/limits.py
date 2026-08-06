"""In-process concurrency cap and per-key rate limiting.

Both limits default to off (0) to preserve MVP behavior. They are best-effort,
single-process guards — not a substitute for an edge proxy — but they keep a
public deployment from being trivially overwhelmed or drained. Because asyncio
is cooperative and these methods don't ``await`` between read and update, the
counters are safe without a lock. The rate limiter uses a sliding-window
approximation (see ``RequestLimiter.check_rate``) so it isn't a naive fixed
window a client could double-burst across a boundary.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from openfusion.config import LimitsConfig
from openfusion.errors import OverloadedError, RateLimitError

# Hard cap on distinct rate-limit keys tracked at once. The expiry-based prune
# below only runs when the *current* key's own window has expired, so a client
# that never reuses a key (e.g. rotating Bearer tokens, one per request) would
# otherwise grow _window without bound. This mirrors the LRU eviction used for
# ResponseCache and server.py's runtime_api_keys store.
_WINDOW_MAX_KEYS = 10_000


class RequestLimiter:
    def __init__(self, config: LimitsConfig) -> None:
        self._max_in_flight = config.max_in_flight
        self._rpm = config.rate_limit_per_minute
        self._in_flight = 0
        # key -> (prev_window_count, curr_window_count, curr_window_start_monotonic)
        self._window: OrderedDict[str, tuple[int, int, float]] = OrderedDict()

    def check_rate(self, key: str) -> None:
        """Raise RateLimitError if ``key`` exceeded its per-minute budget.

        Uses a sliding-window-counter approximation (the current fixed window
        plus a time-decayed weight of the previous one) rather than a naive
        fixed window. A naive fixed window lets a client send ``rpm`` requests
        right before a window boundary and another ``rpm`` right after,
        getting ~2x its budget in a couple of seconds; blending in the
        previous window's count keeps any 60s span close to ``rpm``.
        """
        if self._rpm <= 0:
            return
        now = time.monotonic()
        prev_count, curr_count, start = self._window.get(key, (0, 0, now))
        elapsed = now - start
        if elapsed >= 60.0:
            if elapsed >= 120.0:
                # More than two windows have passed; nothing to carry over.
                prev_count, curr_count, start = 0, 0, now
            else:
                prev_count, curr_count, start = curr_count, 0, start + 60.0
            elapsed = now - start
            # Prune entries whose windows are both fully expired to prevent
            # unbounded memory growth when many unique keys (e.g. rotating
            # Bearer tokens) hit a long-lived server.
            self._window = OrderedDict(
                (k, v) for k, v in self._window.items() if now - v[2] < 120.0
            )
        weight = max(0.0, 1.0 - elapsed / 60.0)
        if curr_count + prev_count * weight >= self._rpm:
            raise RateLimitError()
        self._window[key] = (prev_count, curr_count + 1, start)
        self._window.move_to_end(key)
        while len(self._window) > _WINDOW_MAX_KEYS:
            self._window.popitem(last=False)

    def acquire(self) -> bool:
        """Reserve a concurrency slot. Raise OverloadedError when full.

        Returns whether a slot was actually taken (False when unlimited), so the
        caller knows whether a matching ``release`` is needed.
        """
        if self._max_in_flight <= 0:
            return False
        if self._in_flight >= self._max_in_flight:
            raise OverloadedError()
        self._in_flight += 1
        return True

    def release(self, acquired: bool) -> None:
        if acquired and self._in_flight > 0:
            self._in_flight -= 1
