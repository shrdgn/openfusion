"""OpenAI SSE framing and streaming orchestration."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openfusion.config import OpenFusionConfig
from openfusion.panel import PanelResult, expand_panel_members, gather_panel
from openfusion.ranked import pick_best
from openfusion.synthesize import ANALYSIS_SENTINEL, synthesize
from openfusion.upstream import UpstreamClient
from openfusion.vote import majority_vote


def _sse_line(event: str | None, data: str) -> str:
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def _progress(payload: dict[str, Any]) -> str:
    return _sse_line("progress", json.dumps(payload))


async def gather_with_progress(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[str | PanelResult]:
    """Run the panel, yielding progress SSE lines per member, then the PanelResult.

    Yields `event: progress` strings as members finish, and finally yields the
    `PanelResult` (the consumer should `isinstance`-check the last item).
    """
    members = expand_panel_members(config)
    panel_models = [member.model for member, _ in members]
    total = len(members)
    yield _progress(
        {
            "stage": "panel",
            "message": f"Querying {total} model{'s' if total != 1 else ''}",
            "models": panel_models,
            "total": total,
            "judge": config.judge.model if config.judge else None,
        }
    )

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def on_member(label: str, model: str, ok: bool, content: str) -> None:
        await queue.put({"model": model, "label": label, "ok": ok, "content": content})

    task = asyncio.create_task(
        gather_panel(request_body, config, client, cancel_event=cancel_event, on_member=on_member)
    )
    done = 0
    while not task.done() or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.2)
        except TimeoutError:
            continue
        done += 1
        if config.expose_panel and item["ok"]:
            yield _sse_line(
                "panel_answer",
                json.dumps(
                    {"model": item["model"], "label": item["label"], "content": item["content"]}
                ),
            )
        yield _progress(
            {
                "stage": "panel_member",
                "model": item["model"],
                "ok": item["ok"],
                "completed": done,
                "total": total,
            }
        )
    yield await task


class _AnalysisSplitter:
    """Split a judge stream into the answer and a trailing analysis block.

    Everything before ``ANALYSIS_SENTINEL`` is the answer (streamed as content);
    everything after is collected as the structured analysis. A short tail is
    held back so a sentinel straddling two chunks is still detected.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._past_sentinel = False
        self.analysis = ""

    def feed(self, delta: str) -> str:
        if self._past_sentinel:
            self.analysis += delta
            return ""
        self._buffer += delta
        index = self._buffer.find(ANALYSIS_SENTINEL)
        if index != -1:
            answer = self._buffer[:index]
            self.analysis += self._buffer[index + len(ANALYSIS_SENTINEL) :]
            self._buffer = ""
            self._past_sentinel = True
            return answer
        safe = len(self._buffer) - (len(ANALYSIS_SENTINEL) - 1)
        if safe <= 0:
            return ""
        out = self._buffer[:safe]
        self._buffer = self._buffer[safe:]
        return out

    def flush(self) -> str:
        if self._past_sentinel:
            return ""
        out = self._buffer
        self._buffer = ""
        return out

    def analysis_payload(self) -> dict[str, Any] | None:
        text = self.analysis.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
        return parsed if isinstance(parsed, dict) else {"raw": text}


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

    panel: PanelResult | None = None
    try:
        async for item in gather_with_progress(
            request_body, config, client, cancel_event=cancel_event
        ):
            if isinstance(item, PanelResult):
                panel = item
            else:
                yield item
    except Exception as exc:  # noqa: BLE001 - panel failed; report and end the stream
        yield _sse_line(
            None,
            json.dumps(
                {"error": {"message": str(exc), "type": "upstream_error", "code": "panel_error"}}
            ),
        )
        yield _sse_line(None, "[DONE]")
        return
    assert panel is not None

    yield _progress(
        {
            "stage": "synthesis",
            "message": f"Synthesizing with {config.judge.model if config.judge else 'judge'}",
            "panel_count": len(panel.responses),
            "failed_count": len(panel.failures),
        }
    )

    role_sent = False
    finish_reason: str | None = None
    judge_usage: dict[str, float] | None = None
    splitter = _AnalysisSplitter() if config.analysis.emit else None

    def _content_chunk(text: str, reason: str | None) -> str:
        nonlocal role_sent
        chunk_delta: dict[str, Any] = {}
        if not role_sent:
            chunk_delta["role"] = "assistant"
            role_sent = True
        if text:
            chunk_delta["content"] = text
        return _sse_line(
            None,
            json.dumps(
                _chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta=chunk_delta,
                    finish_reason=reason,
                )
            ),
        )

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
            content = splitter.feed(delta) if splitter else delta
            if not content and not reason:
                continue
            yield _content_chunk(content, reason)
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

    if splitter:
        tail = splitter.flush()
        if tail:
            yield _content_chunk(tail, None)

    terminal = _chunk(
        chunk_id=chunk_id,
        created=created,
        model=model,
        delta={},
        finish_reason=finish_reason or "stop",
    )
    yield _sse_line(None, json.dumps(terminal))

    if splitter:
        analysis = splitter.analysis_payload()
        if analysis:
            yield _sse_line("analysis", json.dumps(analysis))

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


async def vote_and_stream(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    """Gather the panel, majority-vote, and emit the winning answer as SSE."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = config.fusion_model_name

    panel: PanelResult | None = None
    async for item in gather_with_progress(
        request_body, config, client, cancel_event=cancel_event
    ):
        if isinstance(item, PanelResult):
            panel = item
        else:
            yield item
    assert panel is not None
    content, vote_meta = majority_vote(panel)

    yield _sse_line(
        "progress",
        json.dumps(
            {
                "stage": "vote",
                "message": "Majority vote over panel answers",
                "panel_count": len(panel.responses),
                "failed_count": len(panel.failures),
                "agreement": vote_meta.get("agreement"),
            }
        ),
    )

    yield _sse_line(
        None,
        json.dumps(
            _chunk(
                chunk_id=chunk_id,
                created=created,
                model=model,
                delta={"role": "assistant", "content": content},
                finish_reason=None,
            )
        ),
    )
    yield _sse_line(
        None,
        json.dumps(
            _chunk(chunk_id=chunk_id, created=created, model=model, delta={}, finish_reason="stop")
        ),
    )

    usage_payload = _build_usage_payload(panel, None)
    if usage_payload:
        yield _sse_line("usage", json.dumps(usage_payload))

    yield _sse_line(None, "[DONE]")


async def buffer_vote(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
) -> dict[str, Any]:
    """Non-streaming majority vote over the panel."""
    panel = await gather_panel(request_body, config, client)
    content, _ = majority_vote(panel)

    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.fusion_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    panel_usage = panel.usage_total
    if panel_usage:
        response["usage"] = panel_usage
    return response


async def buffer_synthesis(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
) -> dict[str, Any]:
    """Non-streaming synthesis by buffering judge output."""
    content_parts: list[str] = []
    finish_reason = "stop"
    judge_usage: dict[str, float] | None = None
    splitter = _AnalysisSplitter() if config.analysis.emit else None

    panel = await gather_panel(request_body, config, client)
    async for delta, usage, reason in synthesize(
        request_body,
        panel,
        config,
        client,
        timeout=config.timeouts.judge_seconds,
    ):
        if delta:
            content_parts.append(splitter.feed(delta) if splitter else delta)
        if reason:
            finish_reason = reason
        if usage:
            judge_usage = usage
    if splitter:
        content_parts.append(splitter.flush())

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
    if splitter:
        analysis = splitter.analysis_payload()
        if analysis:
            response["analysis"] = analysis
    if usage_payload and usage_payload.get("total"):
        response["usage"] = usage_payload["total"]
    return response


async def ranked_and_stream(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    """Gather the panel, have the judge pick the best answer, and stream it."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = config.fusion_model_name

    panel: PanelResult | None = None
    async for item in gather_with_progress(
        request_body, config, client, cancel_event=cancel_event
    ):
        if isinstance(item, PanelResult):
            panel = item
        else:
            yield item
    assert panel is not None
    content, meta = await pick_best(
        request_body, panel, config, client, timeout=config.timeouts.judge_seconds
    )

    yield _sse_line(
        "progress",
        json.dumps(
            {
                "stage": "ranked",
                "message": "Selecting the best panel answer",
                "panel_count": len(panel.responses),
                "winner": meta.get("winner"),
            }
        ),
    )
    yield _sse_line(
        None,
        json.dumps(
            _chunk(
                chunk_id=chunk_id,
                created=created,
                model=model,
                delta={"role": "assistant", "content": content},
                finish_reason=None,
            )
        ),
    )
    yield _sse_line(
        None,
        json.dumps(
            _chunk(chunk_id=chunk_id, created=created, model=model, delta={}, finish_reason="stop")
        ),
    )
    usage_payload = _build_usage_payload(panel, None)
    if usage_payload:
        yield _sse_line("usage", json.dumps(usage_payload))
    yield _sse_line(None, "[DONE]")


async def buffer_ranked(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
) -> dict[str, Any]:
    """Non-streaming ranked-choice selection."""
    panel = await gather_panel(request_body, config, client)
    content, _ = await pick_best(
        request_body, panel, config, client, timeout=config.timeouts.judge_seconds
    )
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.fusion_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    panel_usage = panel.usage_total
    if panel_usage:
        response["usage"] = panel_usage
    return response
