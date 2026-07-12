"""Unit tests for server.py's _stream_with_cancellation.

Shared by _pipeline_stream and _fusion_stream: watches for client disconnect,
forwards lines from `make_lines` until cancelled, and records success/error via
`on_finish`. The real streamers (pipeline_and_stream, vote/ranked/synthesize_and_
stream) already stop yielding once their own cancel_event check trips, so the
wrapper's own `if cancel_event.is_set(): break` and exception-handling branches
need a `make_lines` that doesn't self-limit to be exercised directly.
"""

from __future__ import annotations

import pytest

from openfusion.server import _stream_with_cancellation


class _NeverDisconnectsRequest:
    """Minimal server.Request stand-in: the client never disconnects."""

    async def is_disconnected(self) -> bool:
        return False


async def test_stops_yielding_once_cancel_event_is_set() -> None:
    async def make_lines(cancel_event):
        yield "first\n"
        cancel_event.set()  # e.g. tripped by something other than disconnect-watch
        yield "second\n"
        yield "third\n"

    finished = []
    lines = [
        line
        async for line in _stream_with_cancellation(
            _NeverDisconnectsRequest(), make_lines, finished.append
        )
    ]

    assert lines == ["first\n"]
    assert finished == ["success"]


async def test_records_error_outcome_and_reraises_on_failure() -> None:
    async def make_lines(cancel_event):
        yield "first\n"
        raise RuntimeError("boom")

    finished = []
    lines = []
    with pytest.raises(RuntimeError, match="boom"):
        async for line in _stream_with_cancellation(
            _NeverDisconnectsRequest(), make_lines, finished.append
        ):
            lines.append(line)

    assert lines == ["first\n"]
    assert finished == ["error"]


async def test_records_success_outcome_when_stream_completes() -> None:
    async def make_lines(cancel_event):
        yield "only\n"

    finished = []
    lines = [
        line
        async for line in _stream_with_cancellation(
            _NeverDisconnectsRequest(), make_lines, finished.append
        )
    ]

    assert lines == ["only\n"]
    assert finished == ["success"]
