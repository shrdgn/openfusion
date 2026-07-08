"""Shared upstream client for OpenAI-compatible and Anthropic native APIs."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from openfusion.config import FallbackConfig, FallbackEntry, JudgeConfig, PanelMember
from openfusion.errors import UpstreamError
from openfusion.health import HEALTH
from openfusion.metrics import METRICS

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
LOGGER = logging.getLogger("openfusion.upstream")


def _provider_id_from_url(base_url: str) -> str:
    """Derive a stable provider id from a base URL for health/latency tracking."""
    from urllib.parse import urlparse

    host = urlparse(base_url).netloc or base_url
    # Strip port and www. prefix; use the second-level domain as the id.
    host = host.split(":")[0].removeprefix("www.")
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host


class UpstreamClient:
    """HTTP client wrapper for panel members and judge calls."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completion_with_fallback(
        self,
        member: PanelMember | JudgeConfig,
        body: dict[str, Any],
        *,
        stream: bool,
        fallback: FallbackConfig | None = None,
        timeout: float | None = None,
        phase: str | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Call ``member``; on failure try each fallback entry in order.

        Fallback entries for ``member.model`` are looked up from ``fallback.chains``.
        Entries whose provider is currently DOWN (per :data:`HEALTH`) are skipped.
        If all attempts fail, the last exception is re-raised.
        """
        candidates: list[PanelMember | JudgeConfig | FallbackEntry] = [member]  # type: ignore[list-item]
        if fallback:
            chain = fallback.chains.get(member.model, [])
            candidates.extend(chain)

        last_exc: BaseException | None = None
        for candidate in candidates:
            provider_id = _provider_id_from_url(candidate.base_url)
            if last_exc is not None and not HEALTH.is_available(provider_id):
                LOGGER.warning(
                    "fallback: skipping provider %s (status=%s)",
                    provider_id, HEALTH.status(provider_id).value,
                )
                continue
            try:
                # Build a PanelMember-compatible target for each fallback entry
                call_target: PanelMember | JudgeConfig
                if isinstance(candidate, FallbackEntry):
                    call_target = PanelMember(
                        base_url=candidate.base_url,
                        api_key=candidate.api_key,
                        model=candidate.model,
                        provider=candidate.provider,
                    )
                else:
                    call_target = candidate
                return await self.chat_completion(
                    call_target, body, stream=stream, timeout=timeout, phase=phase
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if isinstance(candidate, FallbackEntry):
                    LOGGER.warning(
                        "fallback: primary/fallback call to %s failed (%s), trying next",
                        provider_id, exc,
                    )
                else:
                    LOGGER.warning(
                        "fallback: primary call to %s failed (%s), trying fallbacks",
                        provider_id, exc,
                    )

        if last_exc is not None:
            raise last_exc
        raise UpstreamError("No candidates available")

    async def chat_completion(
        self,
        member: PanelMember | JudgeConfig,
        body: dict[str, Any],
        *,
        stream: bool,
        timeout: float | None = None,
        phase: str | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        provider_id = _provider_id_from_url(member.base_url)

        if member.provider == "anthropic":
            return await self._anthropic_chat_completion(
                member, body, stream=stream, timeout=timeout, phase=phase,
                provider_id=provider_id,
            )

        url = f"{member.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {member.api_key}",
            "Content-Type": "application/json",
        }
        payload = {**body, "model": member.model, "stream": stream}
        request_timeout = httpx.Timeout(timeout) if timeout is not None else None

        if stream:
            return self._stream_chat_completion(
                url,
                headers,
                payload,
                request_timeout,
                label=getattr(member, "label", None),
                phase=phase,
                provider_id=provider_id,
            )
        return await self._json_chat_completion(
            url,
            headers,
            payload,
            request_timeout,
            label=getattr(member, "label", None),
            phase=phase,
            provider_id=provider_id,
        )

    # ------------------------------------------------------------------
    # Anthropic Messages API (native, non-OpenRouter)
    # ------------------------------------------------------------------

    _ANTHROPIC_VERSION = "2023-06-01"

    async def _anthropic_chat_completion(
        self,
        member: PanelMember | JudgeConfig,
        body: dict[str, Any],
        *,
        stream: bool,
        timeout: float | None,
        phase: str | None,
        provider_id: str,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        url = f"{member.base_url}/messages"
        headers = {
            "x-api-key": member.api_key,
            "anthropic-version": self._ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload = _openai_to_anthropic(body, member.model, stream=stream)
        request_timeout = httpx.Timeout(timeout) if timeout is not None else None

        if stream:
            return self._anthropic_stream(
                url, headers, payload, request_timeout,
                label=getattr(member, "label", None), phase=phase,
                provider_id=provider_id,
            )
        started = time.perf_counter()
        try:
            response = await self._client.post(
                url, headers=headers, json=payload, timeout=request_timeout
            )
        except Exception:
            HEALTH.record_failure(provider_id)
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            HEALTH.record_failure(provider_id)
            self._log_request(
                phase=phase, label=getattr(member, "label", None),
                model=member.model, stream=False,
                status_code=response.status_code, latency_ms=elapsed_ms,
                level=logging.WARNING,
            )
            raise self._build_upstream_error(response.status_code, response.content)
        HEALTH.record_success(provider_id, elapsed_ms)
        raw = response.json()
        converted = _anthropic_to_openai(raw)
        self._log_request(
            phase=phase, label=getattr(member, "label", None),
            model=member.model, stream=False,
            status_code=response.status_code, latency_ms=elapsed_ms,
            usage=self._extract_usage(converted),
        )
        return converted

    async def _anthropic_stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout | None,
        *,
        label: str | None,
        phase: str | None,
        provider_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        status_code: int | None = None
        usage: dict[str, Any] | None = None
        chunks = 0
        success = False
        try:
            async with self._client.stream(
                "POST", url, headers=headers, json=payload, timeout=timeout
            ) as response:
                status_code = response.status_code
                if response.status_code >= 400:
                    body = await response.aread()
                    HEALTH.record_failure(provider_id)
                    self._log_request(
                        phase=phase, label=label,
                        model=str(payload.get("model")), stream=True,
                        status_code=response.status_code,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        level=logging.WARNING,
                    )
                    raise self._build_upstream_error(response.status_code, body)

                try:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        chunk = _anthropic_stream_event_to_openai(event)
                        if chunk is None:
                            continue
                        if "usage" in chunk:
                            usage = chunk["usage"]
                        chunks += 1
                        yield chunk
                    success = True
                finally:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    if success:
                        HEALTH.record_success(provider_id, elapsed_ms)
                    self._log_request(
                        phase=phase, label=label,
                        model=str(payload.get("model")), stream=True,
                        status_code=status_code,
                        latency_ms=elapsed_ms,
                        usage=usage, chunks=chunks,
                    )
        except Exception:
            if not success and status_code is None:
                HEALTH.record_failure(provider_id)
            raise

    async def _json_chat_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout | None,
        *,
        label: str | None,
        phase: str | None,
        provider_id: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = await self._client.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception:
            HEALTH.record_failure(provider_id)
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            HEALTH.record_failure(provider_id)
            self._log_request(
                phase=phase,
                label=label,
                model=str(payload.get("model")),
                stream=False,
                status_code=response.status_code,
                latency_ms=elapsed_ms,
                level=logging.WARNING,
            )
            return self._parse_response(response)
        HEALTH.record_success(provider_id, elapsed_ms)
        parsed = self._parse_response(response)
        self._log_request(
            phase=phase,
            label=label,
            model=str(payload.get("model")),
            stream=False,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
            usage=self._extract_usage(parsed),
        )
        return parsed

    async def _stream_chat_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout | None,
        *,
        label: str | None,
        phase: str | None,
        provider_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        status_code: int | None = None
        usage: dict[str, Any] | None = None
        chunks = 0
        success = False
        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                status_code = response.status_code
                if response.status_code >= 400:
                    body = await response.aread()
                    HEALTH.record_failure(provider_id)
                    self._log_request(
                        phase=phase,
                        label=label,
                        model=str(payload.get("model")),
                        stream=True,
                        status_code=response.status_code,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        level=logging.WARNING,
                    )
                    raise self._build_upstream_error(response.status_code, body)

                try:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise UpstreamError(f"Invalid upstream SSE payload: {exc}") from exc
                        chunks += 1
                        usage = self._extract_usage(chunk) or usage
                        yield chunk
                    success = True
                finally:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    if success:
                        HEALTH.record_success(provider_id, elapsed_ms)
                    self._log_request(
                        phase=phase,
                        label=label,
                        model=str(payload.get("model")),
                        stream=True,
                        status_code=status_code,
                        latency_ms=elapsed_ms,
                        usage=usage,
                        chunks=chunks,
                    )
        except Exception:
            if not success and status_code is None:
                HEALTH.record_failure(provider_id)
            raise

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise self._build_upstream_error(response.status_code, response.content)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise UpstreamError("Upstream returned invalid JSON") from exc

    def _build_upstream_error(self, status_code: int, body: bytes) -> UpstreamError:
        message = body.decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
            if isinstance(payload, dict) and "error" in payload:
                error = payload["error"]
                if isinstance(error, dict) and "message" in error:
                    message = str(error["message"])
        except json.JSONDecodeError:
            pass
        return UpstreamError(
            f"Upstream error ({status_code}): {message}",
            status_code=status_code,
        )

    def _extract_usage(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        usage = payload.get("usage")
        return usage if isinstance(usage, dict) else None

    def _log_request(
        self,
        *,
        phase: str | None,
        label: str | None,
        model: str,
        stream: bool,
        status_code: int | None,
        latency_ms: int,
        usage: dict[str, Any] | None = None,
        chunks: int | None = None,
        level: int = logging.INFO,
    ) -> None:
        fields: dict[str, Any] = {
            "phase": phase,
            "label": label,
            "model": model,
            "stream": stream,
            "status_code": status_code,
            "latency_ms": latency_ms,
        }
        if chunks is not None:
            fields["chunks"] = chunks
        if usage:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
                if key in usage:
                    fields[key] = usage[key]
        LOGGER.log(level, "upstream_request %s", json.dumps(fields, sort_keys=True))
        if label is not None and phase is not None:
            METRICS.record_panel_member_latency(
                label=label, phase=phase, latency_ms=latency_ms
            )

        if phase:
            outcome = "success" if status_code is not None and status_code < 400 else "error"
            METRICS.record_upstream(
                phase=phase,
                outcome=outcome,
                latency_ms=latency_ms,
                usage=usage,
            )


# ---------------------------------------------------------------------------
# Anthropic ↔ OpenAI format translation
# ---------------------------------------------------------------------------

_ANTHROPIC_STOP_REASONS = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}

# tool_choice values that map 1-to-1
_TOOL_CHOICE_MAP = {"none": {"type": "auto"}, "auto": {"type": "auto"}, "required": {"type": "any"}}


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        fn = t.get("function", {})
        entry: dict[str, Any] = {
            "name": fn.get("name", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            entry["description"] = fn["description"]
        out.append(entry)
    return out


def _openai_message_to_anthropic(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single OpenAI message to Anthropic format.

    Returns None for system messages (caller handles separately).
    """
    role = msg.get("role")
    if role == "system":
        return None

    # tool result → user message with tool_result content block
    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content") or "",
                }
            ],
        }

    # assistant message with tool_calls → content blocks with tool_use entries
    if role == "assistant" and msg.get("tool_calls"):
        blocks: list[dict[str, Any]] = []
        if msg.get("content"):
            blocks.append({"type": "text", "text": msg["content"]})
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                inputs = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": inputs,
                }
            )
        return {"role": "assistant", "content": blocks}

    return {"role": role, "content": msg.get("content") or ""}


def _openai_to_anthropic(body: dict[str, Any], model: str, *, stream: bool) -> dict[str, Any]:
    """Translate an OpenAI chat/completions request body to Anthropic Messages API format."""
    messages = body.get("messages", [])
    system: str | None = None
    anthropic_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system = str(msg.get("content", ""))
            continue
        converted = _openai_message_to_anthropic(msg)
        if converted is not None:
            anthropic_messages.append(converted)

    payload: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": body.get("max_tokens") or 1024,
        "stream": stream,
    }
    if system:
        payload["system"] = system
    for key in ("temperature", "top_p", "stop"):
        if body.get(key) is not None:
            payload[key] = body[key]

    if body.get("tools"):
        payload["tools"] = _openai_tools_to_anthropic(body["tools"])
        tc = body.get("tool_choice")
        if isinstance(tc, dict) and tc.get("type") == "function":
            payload["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}
        elif isinstance(tc, str) and tc in _TOOL_CHOICE_MAP:
            payload["tool_choice"] = _TOOL_CHOICE_MAP[tc]

    return payload


def _anthropic_tool_use_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_calls = []
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        tool_calls.append(
            {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            }
        )
    return tool_calls


def _anthropic_to_openai(response: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages API response to OpenAI chat/completions format."""
    content_blocks = response.get("content", [])
    text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )
    stop_reason = _ANTHROPIC_STOP_REASONS.get(
        response.get("stop_reason", "end_turn"), "stop"
    )
    usage_raw = response.get("usage", {})
    usage = {
        "prompt_tokens": usage_raw.get("input_tokens", 0),
        "completion_tokens": usage_raw.get("output_tokens", 0),
        "total_tokens": (
            usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0)
        ),
    }
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    tool_calls = _anthropic_tool_use_to_openai(content_blocks)
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = text or None
    return {
        "id": response.get("id", ""),
        "object": "chat.completion",
        "model": response.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": stop_reason,
            }
        ],
        "usage": usage,
    }


def _anthropic_stream_event_to_openai(event: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single Anthropic streaming event to an OpenAI-style delta chunk.

    Returns None for events that don't map to content (ping, message_start, etc.).
    """
    event_type = event.get("type")

    if event_type == "content_block_start":
        block = event.get("content_block", {})
        if block.get("type") == "tool_use":
            index = event.get("index", 0)
            return {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": block.get("id", ""),
                                    "type": "function",
                                    "function": {"name": block.get("name", ""), "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        return None

    if event_type == "content_block_delta":
        delta = event.get("delta", {})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            return {
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": delta.get("text", "")}, "finish_reason": None}
                ],
            }
        if delta_type == "input_json_delta":
            index = event.get("index", 0)
            return {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "function": {"arguments": delta.get("partial_json", "")},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        return None

    if event_type == "message_delta":
        delta = event.get("delta", {})
        stop_reason = _ANTHROPIC_STOP_REASONS.get(delta.get("stop_reason", ""), "stop")
        usage_raw = event.get("usage", {})
        chunk: dict[str, Any] = {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": stop_reason}],
        }
        if usage_raw:
            chunk["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": usage_raw.get("output_tokens", 0),
                "total_tokens": usage_raw.get("output_tokens", 0),
            }
        return chunk

    if event_type == "message_start":
        usage_raw = event.get("message", {}).get("usage", {})
        if usage_raw:
            return {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "usage": {
                    "prompt_tokens": usage_raw.get("input_tokens", 0),
                    "completion_tokens": 0,
                    "total_tokens": usage_raw.get("input_tokens", 0),
                },
            }

    return None


def extract_response_usage(payload: dict[str, Any]) -> dict[str, float] | None:
    """Parse token/cost fields from an upstream response or streaming chunk.

    Returns a ``dict[str, float]`` with whichever of ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, and ``cost`` are present, or
    ``None`` if the payload carries no usage information.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    result: dict[str, float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        result["cost"] = float(cost)
    return result or None


def member_from_dict(
    base_url: str,
    api_key: str,
    model: str,
    label: str | None = None,
) -> PanelMember:
    return PanelMember(base_url=base_url, api_key=api_key, model=model, label=label)
