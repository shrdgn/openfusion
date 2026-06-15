"""OpenAI SSE framing and streaming orchestration."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openfusion.config import OpenFusionConfig
from openfusion.panel import PanelResult, gather_panel
from openfusion.synthesize import synthesize
from openfusion.upstream import UpstreamClient


def _sse_line(event: str | None, data: str) -> str:
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def _chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    else:
        choice["finish_reason"] = None
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }


async def synthesize_and_stream(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    """Gather the panel, synthesize with the judge, and emit OpenAI-compatible SSE."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = config.fusion_model_name

    yield _sse_line(
        "progress",
        json.dumps({"stage": "panel", "message": "Gathering panel responses"}),
    )

    panel = await gather_panel(
        request_body,
        config,
        client,
        cancel_event=cancel_event,
    )

    yield _sse_line(
        "progress",
        json.dumps(
            {
                "stage": "synthesis",
                "message": "Synthesizing final answer",
                "panel_count": len(panel.responses),
                "failed_count": len(panel.failures),
            }
        ),
    )

    role_sent = False
    finish_reason: str | None = None
    judge_usage: dict[str, float] | None = None

    try:
        async for delta, usage, reason in synthesize(
            request_body,
            panel,
            config,
            client,
            timeout=config.timeouts.judge_seconds,
        ):
            if reason:
                finish_reason = reason
            if usage:
                judge_usage = usage
            if not delta and not reason:
                continue

            chunk_delta: dict[str, Any] = {}
            if not role_sent:
                chunk_delta["role"] = "assistant"
                role_sent = True
            if delta:
                chunk_delta["content"] = delta

            payload = _chunk(
                chunk_id=chunk_id,
                created=created,
                model=model,
                delta=chunk_delta,
                finish_reason=reason,
            )
            yield _sse_line(None, json.dumps(payload))
    except Exception as exc:  # noqa: BLE001 - stream error chunk after 200
        error_payload = {
            "error": {
                "message": str(exc),
                "type": "upstream_error",
                "code": "judge_stream_error",
            }
        }
        yield _sse_line(None, json.dumps(error_payload))
        yield _sse_line(None, "[DONE]")
        return

    terminal = _chunk(
        chunk_id=chunk_id,
        created=created,
        model=model,
        delta={},
        finish_reason=finish_reason or "stop",
    )
    yield _sse_line(None, json.dumps(terminal))

    usage_payload = _build_usage_payload(panel, judge_usage)
    if usage_payload:
        yield _sse_line("usage", json.dumps(usage_payload))

    yield _sse_line(None, "[DONE]")


def _build_usage_payload(
    panel: PanelResult,
    judge_usage: dict[str, float] | None,
) -> dict[str, Any] | None:
    panel_usage = panel.usage_total
    if panel_usage is None and judge_usage is None:
        return None

    payload: dict[str, Any] = {
        "panel": [
            {"label": response.label, "usage": response.usage}
            for response in panel.responses
            if response.usage
        ],
        "panel_total": panel_usage,
        "judge": judge_usage,
    }
    if panel_usage and judge_usage:
        total = {
            "prompt_tokens": panel_usage.get("prompt_tokens", 0)
            + judge_usage.get("prompt_tokens", 0),
            "completion_tokens": panel_usage.get("completion_tokens", 0)
            + judge_usage.get("completion_tokens", 0),
            "total_tokens": panel_usage.get("total_tokens", 0) + judge_usage.get("total_tokens", 0),
        }
        if "cost" in panel_usage or "cost" in judge_usage:
            total["cost"] = panel_usage.get("cost", 0.0) + judge_usage.get("cost", 0.0)
        payload["total"] = total
    return payload


async def buffer_synthesis(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
) -> dict[str, Any]:
    """Non-streaming synthesis by buffering judge output."""
    content_parts: list[str] = []
    finish_reason = "stop"
    judge_usage: dict[str, float] | None = None

    panel = await gather_panel(request_body, config, client)
    async for delta, usage, reason in synthesize(
        request_body,
        panel,
        config,
        client,
        timeout=config.timeouts.judge_seconds,
    ):
        if delta:
            content_parts.append(delta)
        if reason:
            finish_reason = reason
        if usage:
            judge_usage = usage

    usage_payload = _build_usage_payload(panel, judge_usage)
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.fusion_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage_payload and usage_payload.get("total"):
        response["usage"] = usage_payload["total"]
    return response
