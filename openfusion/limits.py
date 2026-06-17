"""In-process concurrency cap and per-key rate limiting.

Both limits default to off (0) to preserve MVP behavior. They are best-effort,
single-process guards — not a substitute for an edge proxy — but they keep a
public deployment from being trivially overwhelmed or drained. Because asyncio
is cooperative and these methods don't ``await`` between read and update, the
counters are safe without a lock.
"""

from __future__ import annotations

import time

from openfusion.config import LimitsConfig
from openfusion.errors import OverloadedError, RateLimitError


class RequestLimiter:
    def __init__(self, config: LimitsConfig) -> None:
        self._max_in_flight = config.max_in_flight
        self._rpm = config.rate_limit_per_minute
        self._in_flight = 0
        # key -> (count_in_window, window_start_monotonic)
        self._window: dict[str, tuple[int, float]] = {}

    def check_rate(self, key: str) -> None:
        """Raise RateLimitError if ``key`` exceeded its per-minute budget."""
        if self._rpm <= 0:
            return
        now = time.monotonic()
        count, start = self._window.get(key, (0, now))
        if now - start >= 60.0:
            count, start = 0, now
        if count >= self._rpm:
            raise RateLimitError()
        self._window[key] = (count + 1, start)

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
