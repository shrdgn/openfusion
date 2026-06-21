"""Shared upstream client for OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from openfusion.config import PanelMember
from openfusion.errors import UpstreamError
from openfusion.metrics import METRICS

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
LOGGER = logging.getLogger("openfusion.upstream")


class UpstreamClient:
    """HTTP client wrapper for panel members and judge calls."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completion(
        self,
        member: PanelMember,
        body: dict[str, Any],
        *,
        stream: bool,
        timeout: float | None = None,
        phase: str | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
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
                label=member.label,
                phase=phase,
            )
        return await self._json_chat_completion(
            url,
            headers,
            payload,
            request_timeout,
            label=member.label,
            phase=phase,
        )

    async def _json_chat_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout | None,
        *,
        label: str | None,
        phase: str | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._client.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
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
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        status_code: int | None = None
        usage: dict[str, Any] | None = None
        chunks = 0
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
            finally:
                self._log_request(
                    phase=phase,
                    label=label,
                    model=str(payload.get("model")),
                    stream=True,
                    status_code=status_code,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    usage=usage,
                    chunks=chunks,
                )

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

        if phase:
            outcome = "success" if status_code is not None and status_code < 400 else "error"
            METRICS.record_upstream(
                phase=phase,
                outcome=outcome,
                latency_ms=latency_ms,
                usage=usage,
            )


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
