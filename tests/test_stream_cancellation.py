"""Unit tests for server.py's _stream_with_cancellation.

Shared by _pipeline_stream and _fusion_stream: watches for client disconnect and
the ``timeouts.total_seconds`` deadline, forwards lines from `make_lines` until
cancelled, and records success/error via `on_finish`. The real streamers
(pipeline_and_stream, vote/ranked/synthesize_and_stream) already stop yielding
once their own cancel_event check trips, so the wrapper's own
`if cancel_event.is_set(): break` and exception-handling branches need a
`make_lines` that doesn't self-limit to be exercised directly.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from openfusion.server import _stream_with_cancellation

# Large enough that _watch_deadline never fires during a fast unit test.
_NO_DEADLINE = 60.0


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
            _NeverDisconnectsRequest(), make_lines, finished.append, total_seconds=_NO_DEADLINE
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
            _NeverDisconnectsRequest(), make_lines, finished.append, total_seconds=_NO_DEADLINE
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
            _NeverDisconnectsRequest(), make_lines, finished.append, total_seconds=_NO_DEADLINE
        )
    ]

    assert lines == ["only\n"]
    assert finished == ["success"]


async def test_emits_timeout_error_chunk_and_records_error_when_deadline_elapses() -> None:
    """A `make_lines` that outlives the budget gets cut off with a clean SSE error
    chunk (the client is still there, unlike on disconnect) instead of hanging or
    being silently dropped."""

    async def make_lines(cancel_event):
        # Mirrors the real streamers: polls cancel_event between short waits
        # rather than blocking on one long upstream call.
        yield "first\n"
        while not cancel_event.is_set():
            await asyncio.sleep(0.01)

    finished = []
    lines = [
        line
        async for line in _stream_with_cancellation(
            _NeverDisconnectsRequest(), make_lines, finished.append, total_seconds=0.05
        )
    ]

    assert lines[0] == "first\n"
    assert lines[-1] == "data: [DONE]\n\n"
    error_line = lines[-2]
    assert error_line.startswith("data: ")
    payload = json.loads(error_line[len("data: ") :])
    assert payload["error"]["code"] == "total_timeout"
    assert payload["error"]["type"] == "timeout_error"
    assert finished == ["error"]


async def test_emits_timeout_error_chunk_when_deadline_already_elapsed() -> None:
    """A ``total_seconds`` of 0 (already past-due) hits the loop-top deadline
    check directly, without needing ``asyncio.wait_for`` to time out first."""

    async def make_lines(cancel_event):
        yield "unreachable\n"

    finished = []
    lines = [
        line
        async for line in _stream_with_cancellation(
            _NeverDisconnectsRequest(), make_lines, finished.append, total_seconds=0.0
        )
    ]

    assert lines[0].startswith("data: ")
    payload = json.loads(lines[0][len("data: ") :])
    assert payload["error"]["code"] == "total_timeout"
    assert lines[1] == "data: [DONE]\n\n"
    assert finished == ["error"]


async def test_disconnect_before_deadline_does_not_emit_timeout_chunk() -> None:
    """When disconnect (not the deadline) trips cancel_event, there's no client
    left to send an error chunk to — it must not be emitted, and the outcome
    is "success" (a disconnect isn't a server-side failure)."""

    class _DisconnectsImmediatelyRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def make_lines(cancel_event):
        yield "first\n"
        yield "second\n"

    finished = []
    lines = [
        line
        async for line in _stream_with_cancellation(
            _DisconnectsImmediatelyRequest(),
            make_lines,
            finished.append,
            total_seconds=_NO_DEADLINE,
        )
    ]

    assert not any("total_timeout" in line for line in lines)
    assert "data: [DONE]\n\n" not in lines
    assert finished == ["success"]
