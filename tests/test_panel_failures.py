"""Coverage for panel.py's degradation, cancellation, and defensive branches.

These exercise paths that are either rare (real timeouts, upstream cancellation)
or unreachable through the public API with a real HTTP client (a non-dict
payload, a member missing from the debate's overrides map) and so need direct
calls or monkeypatching to trigger, mirroring the existing 429-retry test.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from openfusion.config import (
    CacheConfig,
    DebateConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
    TimeoutsConfig,
)
from openfusion.errors import UpstreamError
from openfusion.panel import (
    MemberFailure,
    MemberResponse,
    PanelResult,
    _call_member,
    _run_debate_round,
    expand_panel_members,
    gather_panel,
)
from openfusion.upstream import UpstreamClient


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _member(label: str = "a") -> PanelMember:
    return PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m", label=label)


def _config(**kwargs) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[_member()],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_self_fusion_requires_a_panel_member() -> None:
    config = OpenFusionConfig(strategy=Strategy.SELF_FUSION, panel=[])
    with pytest.raises(UpstreamError, match="Self-fusion requires at least one panel member"):
        expand_panel_members(config)


@pytest.mark.asyncio
async def test_gather_panel_no_members_raises() -> None:
    config = OpenFusionConfig(strategy=Strategy.PANEL, panel=[])
    client = UpstreamClient()
    with pytest.raises(UpstreamError, match="No panel members configured"):
        await gather_panel({"messages": [{"role": "user", "content": "hi"}]}, config, client)
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_handles_empty_choices(mock_router) -> None:
    """An upstream response with no choices degrades to empty content, not a crash."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    client = UpstreamClient()
    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]}, _config(), client
    )
    assert len(result.responses) == 1
    assert result.responses[0].content == ""
    await client.aclose()


@pytest.mark.asyncio
async def test_call_member_marks_cache_breakpoint(mock_router) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user_message = payload["messages"][0]
        assert isinstance(user_message["content"], list)
        assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}
        return httpx.Response(200, json=_completion("ok"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=upstream_handler
    )
    client = UpstreamClient()
    response = await _call_member(
        client,
        _member(),
        {"messages": [{"role": "user", "content": "hi"}]},
        _config(cache=CacheConfig(enabled=True)),
        {},
        timeout=5,
        cancel_event=asyncio.Event(),
    )
    assert response.content == "ok"
    await client.aclose()


@pytest.mark.asyncio
async def test_call_member_raises_when_cancelled_up_front() -> None:
    cancel_event = asyncio.Event()
    cancel_event.set()
    client = UpstreamClient()
    with pytest.raises(asyncio.CancelledError):
        await _call_member(
            client,
            _member(),
            {"messages": [{"role": "user", "content": "hi"}]},
            _config(),
            {},
            timeout=5,
            cancel_event=cancel_event,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_call_member_raises_when_deadline_already_passed() -> None:
    client = UpstreamClient()
    with pytest.raises(TimeoutError, match="exceeded timeout"):
        await _call_member(
            client,
            _member(),
            {"messages": [{"role": "user", "content": "hi"}]},
            _config(),
            {},
            timeout=-1.0,
            cancel_event=asyncio.Event(),
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_call_member_raises_on_non_dict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming isn't requested, so a non-dict payload can only come from a client bug."""

    async def fake_chat_completion(*args, **kwargs):
        async def _gen():
            yield {}

        return _gen()

    client = UpstreamClient()
    monkeypatch.setattr(client, "chat_completion", fake_chat_completion)
    with pytest.raises(UpstreamError, match="Expected non-streaming upstream response"):
        await _call_member(
            client,
            _member(),
            {"messages": [{"role": "user", "content": "hi"}]},
            _config(),
            {},
            timeout=5,
            cancel_event=asyncio.Event(),
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_call_member_cancelled_during_429_backoff_raises(mock_router) -> None:
    cancel_event = asyncio.Event()

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        # Cancellation arrives while the request is in flight (e.g. the client
        # disconnected); the retry loop must notice it before sleeping and back off.
        cancel_event.set()
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        side_effect=upstream_handler
    )
    client = UpstreamClient()
    with pytest.raises(asyncio.CancelledError):
        await _call_member(
            client,
            _member(),
            {"messages": [{"role": "user", "content": "hi"}]},
            _config(),
            {},
            timeout=5,
            cancel_event=cancel_event,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_pre_cancelled_raises_cancelled_error(mock_router) -> None:
    """A cancel_event set before dispatch propagates as CancelledError, not a failure."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("ok"))
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[_member("a"), _member("b")],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    cancel_event = asyncio.Event()
    cancel_event.set()
    with pytest.raises(asyncio.CancelledError):
        await gather_panel(
            {"messages": [{"role": "user", "content": "hi"}]},
            config,
            client,
            cancel_event=cancel_event,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_member_timeout_degrades(mock_router) -> None:
    async def routed_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["model"] == "slow":
            await asyncio.sleep(0.2)
        return httpx.Response(200, json=_completion("ok"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=routed_handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(
                base_url="https://mock.upstream/v1", api_key="k", model="fast", label="fast"
            ),
            PanelMember(
                base_url="https://mock.upstream/v1", api_key="k", model="slow", label="slow"
            ),
        ],
        timeouts=TimeoutsConfig(member_seconds=0.05, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    result = await gather_panel({"messages": [{"role": "user", "content": "hi"}]}, config, client)
    assert len(result.responses) == 1
    assert len(result.failures) == 1
    assert result.failures[0].label == "slow"
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_generic_exception_degrades(
    mock_router, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-UpstreamError, non-timeout failure (e.g. a client bug) still degrades gracefully."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("ok"))
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[_member("a"), _member("b")],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    real_chat_completion = client.chat_completion

    async def flaky_chat_completion(member, *args, **kwargs):
        if member.label == "a":
            raise ValueError("boom")
        return await real_chat_completion(member, *args, **kwargs)

    monkeypatch.setattr(client, "chat_completion", flaky_chat_completion)
    result = await gather_panel({"messages": [{"role": "user", "content": "hi"}]}, config, client)
    assert len(result.responses) == 1
    assert len(result.failures) == 1
    assert result.failures[0].reason == "boom"
    await client.aclose()


@pytest.mark.asyncio
async def test_debate_round_skipped_when_messages_missing(mock_router) -> None:
    """Malformed request body (no 'messages' list) leaves the panel result untouched."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("first"))
    )
    config = OpenFusionConfig(
        strategy=Strategy.DEBATE,
        debate=DebateConfig(rounds=1),
        panel=[_member("a"), _member("b")],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    result = await gather_panel({"prompt": "no messages key here"}, config, client)
    assert all(r.content == "first" for r in result.responses)
    await client.aclose()


@pytest.mark.asyncio
async def test_debate_revision_failure_keeps_first_round_answer(mock_router) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        if "other independent experts" in text:
            return httpx.Response(500, json={"error": {"message": "revision failed"}})
        return httpx.Response(200, json=_completion("first"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.DEBATE,
        debate=DebateConfig(rounds=1),
        panel=[_member("a"), _member("b")],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]}, config, client
    )
    assert len(result.responses) == 2
    assert all(r.content == "first" for r in result.responses)
    await client.aclose()


@pytest.mark.asyncio
async def test_run_debate_round_keeps_response_missing_from_overrides_map() -> None:
    """Defensive branch: a response whose label isn't in the overrides map is left as-is."""
    response = MemberResponse(label="orphan", content="kept", model="m")
    panel = PanelResult(responses=[response, MemberResponse(label="b", content="peer", model="m")])
    result = await _run_debate_round(
        {"messages": [{"role": "user", "content": "hi"}]},
        panel,
        members_by_label={},  # "orphan" and "b" both missing
        config=_config(),
        client=UpstreamClient(),
        timeout=5,
        cancel_event=asyncio.Event(),
    )
    assert [r.content for r in result.responses] == ["kept", "peer"]


@pytest.mark.asyncio
async def test_run_debate_round_keeps_response_without_peers() -> None:
    """Defensive branch: a lone response in the round has no peers to consider."""
    response = MemberResponse(label="solo", content="kept", model="m")
    member, overrides = _member("solo"), {}
    result = await _run_debate_round(
        {"messages": [{"role": "user", "content": "hi"}]},
        PanelResult(responses=[response]),
        members_by_label={"solo": (member, overrides)},
        config=_config(),
        client=UpstreamClient(),
        timeout=5,
        cancel_event=asyncio.Event(),
    )
    assert result.responses[0].content == "kept"


@pytest.mark.asyncio
async def test_gather_panel_on_member_callback_error_marked_unknown(mock_router) -> None:
    """A broken on_member progress callback shouldn't crash gather_panel outright."""
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("ok"))
    )

    async def broken_on_member(label: str, model: str, ok: bool, content: str) -> None:
        raise RuntimeError("progress callback exploded")

    client = UpstreamClient()
    with pytest.raises(UpstreamError, match="All panel members failed"):
        await gather_panel(
            {"messages": [{"role": "user", "content": "hi"}]},
            _config(),
            client,
            on_member=broken_on_member,
        )
    await client.aclose()


def test_member_failure_and_response_are_plain_dataclasses() -> None:
    failure = MemberFailure(label="a", reason="boom", status_code=500)
    assert failure.status_code == 500


@pytest.mark.asyncio
async def test_debate_loop_stops_early_when_cancelled_between_rounds(mock_router) -> None:
    """Cancellation observed between debate rounds skips remaining rounds cleanly."""
    cancel_event = asyncio.Event()
    revision_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal revision_calls
        payload = json.loads(request.content)
        text = json.dumps(payload["messages"])
        if "other independent experts" in text:
            revision_calls += 1
            if revision_calls == 2:
                # Cancellation arrives (e.g. client disconnect) once both round-1
                # revisions are already in flight, so it can only take effect
                # before round 2. Gating on the second dispatch (rather than the
                # first) keeps this deterministic: _call_member's cancellation
                # check runs before a request is ever sent, so by the time this
                # handler sees a request at all, that request already passed it.
                cancel_event.set()
            return httpx.Response(200, json=_completion("revised"))
        return httpx.Response(200, json=_completion("first"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=handler)
    config = OpenFusionConfig(
        strategy=Strategy.DEBATE,
        debate=DebateConfig(rounds=2),
        panel=[_member("a"), _member("b")],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    result = await gather_panel(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        client,
        cancel_event=cancel_event,
    )
    # Round 1 ran (answers are "revised"); round 2 was skipped once cancellation was seen.
    assert all(r.content == "revised" for r in result.responses)
    await client.aclose()


@pytest.mark.asyncio
async def test_gather_panel_outer_cancellation_cancels_pending_member_tasks(mock_router) -> None:
    """Cancelling the caller's await cleans up still-running member tasks, not just itself."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=_completion("late"))

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=slow_handler)
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        panel=[_member("a"), _member("b")],
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
    )
    client = UpstreamClient()
    task = asyncio.create_task(
        gather_panel({"messages": [{"role": "user", "content": "hi"}]}, config, client)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
