"""Unit tests for _watch_disconnect in server.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from openfusion.server import _watch_disconnect


class _MockRequest:
    """Minimal request stand-in with a programmable is_disconnected sequence."""

    def __init__(self, sequence: list[bool]) -> None:
        self._iter = iter(sequence)
        self.is_disconnected = AsyncMock(side_effect=lambda: next(self._iter, True))


async def test_watch_disconnect_sets_event_immediately() -> None:
    """Disconnect detected on the first poll → cancel_event is set and function returns."""
    req = _MockRequest([True])
    event = asyncio.Event()
    with patch("openfusion.server.asyncio.sleep"):
        await _watch_disconnect(req, event)
    assert event.is_set()
    req.is_disconnected.assert_called_once()


async def test_watch_disconnect_skips_poll_when_event_preset() -> None:
    """cancel_event already set before entry → loop body never runs."""
    req = _MockRequest([])
    req.is_disconnected = AsyncMock()
    event = asyncio.Event()
    event.set()
    with patch("openfusion.server.asyncio.sleep"):
        await _watch_disconnect(req, event)
    req.is_disconnected.assert_not_called()


async def test_watch_disconnect_polls_multiple_times_before_disconnect() -> None:
    """Two False polls then True → event set after three is_disconnected calls."""
    req = _MockRequest([False, False, True])
    event = asyncio.Event()
    with patch("openfusion.server.asyncio.sleep"):
        await _watch_disconnect(req, event)
    assert event.is_set()
    assert req.is_disconnected.call_count == 3


async def test_watch_disconnect_does_not_set_event_when_not_disconnected() -> None:
    """A single not-disconnected poll followed by a pre-set cancel → exits without override."""
    req = _MockRequest([False])
    event = asyncio.Event()

    call_count = 0

    async def fake_sleep(_: float) -> None:
        nonlocal call_count
        call_count += 1
        # Set the event on the first sleep so the while condition fails next iteration.
        event.set()

    with patch("openfusion.server.asyncio.sleep", side_effect=fake_sleep):
        await _watch_disconnect(req, event)

    # is_disconnected was called once (returned False), then sleep set the event,
    # so the while loop exited without a second poll.
    assert event.is_set()
    assert req.is_disconnected.call_count == 1
