"""Tests for automatic provider fallback."""

from __future__ import annotations

import httpx
import pytest
import respx

from openfusion.config import FallbackConfig, FallbackEntry, PanelMember
from openfusion.health import _DOWN_THRESHOLD
from openfusion.upstream import UpstreamClient


def _sse(text: str) -> str:
    return (
        f'data: {{"choices":[{{"delta":{{"role":"assistant","content":"{text}"}}'  # noqa: E501
        ',"finish_reason":null}}]}}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )


# ---------------------------------------------------------------------------
# chat_completion_with_fallback — no fallback configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_fallback_success() -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://primary.api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]},
            )
        )
        client = UpstreamClient()
        member = PanelMember(
            base_url="https://primary.api/v1", api_key="k", model="gpt-4o"
        )
        result = await client.chat_completion_with_fallback(
            member, {"messages": []}, stream=False
        )
        await client.aclose()
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_no_fallback_raises_on_error() -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://primary.api/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": {"message": "server error"}})
        )
        client = UpstreamClient()
        member = PanelMember(
            base_url="https://primary.api/v1", api_key="k", model="gpt-4o"
        )
        from openfusion.errors import UpstreamError

        with pytest.raises(UpstreamError):
            await client.chat_completion_with_fallback(
                member, {"messages": []}, stream=False
            )
        await client.aclose()


# ---------------------------------------------------------------------------
# chat_completion_with_fallback — with fallback chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_used_on_primary_failure() -> None:
    with respx.mock() as mock:
        mock.post("https://primary.api/v1/chat/completions").mock(
            return_value=httpx.Response(503, json={"error": {"message": "unavailable"}})
        )
        mock.post("https://backup.api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "fallback"}, "finish_reason": "stop"}]},
            )
        )
        client = UpstreamClient()
        member = PanelMember(
            base_url="https://primary.api/v1", api_key="k", model="gpt-4o"
        )
        fallback = FallbackConfig(
            chains={
                "gpt-4o": [
                    FallbackEntry(
                        base_url="https://backup.api/v1", api_key="k2", model="gpt-4o"
                    )
                ]
            }
        )
        result = await client.chat_completion_with_fallback(
            member, {"messages": []}, stream=False, fallback=fallback
        )
        await client.aclose()
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_fallback_skips_down_provider() -> None:
    """A fallback provider marked DOWN should be skipped."""
    from openfusion.health import HEALTH

    # Force backup provider to DOWN state
    for _ in range(_DOWN_THRESHOLD):
        HEALTH.record_failure("backup")

    with respx.mock() as mock:
        mock.post("https://primary.api/v1/chat/completions").mock(
            return_value=httpx.Response(503, json={"error": {"message": "unavailable"}})
        )
        # Backup is DOWN so should never be called — don't register it

        client = UpstreamClient()
        member = PanelMember(
            base_url="https://primary.api/v1", api_key="k", model="gpt-4o"
        )
        fallback = FallbackConfig(
            chains={
                "gpt-4o": [
                    FallbackEntry(
                        base_url="https://backup.ai/v1", api_key="k2", model="gpt-4o"
                    )
                ]
            }
        )
        from openfusion.errors import UpstreamError

        with pytest.raises(UpstreamError):
            await client.chat_completion_with_fallback(
                member, {"messages": []}, stream=False, fallback=fallback
            )
        await client.aclose()

    # Cleanup: reset HEALTH singleton for other tests
    with HEALTH._lock:
        HEALTH._buckets.pop("backup", None)


@pytest.mark.asyncio
async def test_fallback_second_entry_succeeds() -> None:
    """When first fallback fails, the second one should be tried."""
    with respx.mock() as mock:
        mock.post("https://primary.api/v1/chat/completions").mock(
            return_value=httpx.Response(503, json={"error": {"message": "down"}})
        )
        mock.post("https://backup1.api/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": {"message": "also down"}})
        )
        mock.post("https://backup2.api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )
        )
        client = UpstreamClient()
        member = PanelMember(
            base_url="https://primary.api/v1", api_key="k", model="gpt-4o"
        )
        fallback = FallbackConfig(
            chains={
                "gpt-4o": [
                    FallbackEntry(
                        base_url="https://backup1.api/v1", api_key="k2", model="gpt-4o"
                    ),
                    FallbackEntry(
                        base_url="https://backup2.api/v1", api_key="k3", model="gpt-4o"
                    ),
                ]
            }
        )
        result = await client.chat_completion_with_fallback(
            member, {"messages": []}, stream=False, fallback=fallback
        )
        await client.aclose()
    assert isinstance(result, dict)
