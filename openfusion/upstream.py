"""Shared upstream client for OpenAI-compatible APIs."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from openfusion.config import PanelMember
from openfusion.errors import UpstreamError

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


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
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        url = f"{member.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {member.api_key}",
            "Content-Type": "application/json",
        }
        payload = {**body, "model": member.model, "stream": stream}
        request_timeout = httpx.Timeout(timeout) if timeout is not None else None

        if stream:
            return self._stream_chat_completion(url, headers, payload, request_timeout)
        return await self._json_chat_completion(url, headers, payload, request_timeout)

    async def _json_chat_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout | None,
    ) -> dict[str, Any]:
        response = await self._client.post(url, headers=headers, json=payload, timeout=timeout)
        return self._parse_response(response)

    async def _stream_chat_completion(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout | None,
    ) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise self._build_upstream_error(response.status_code, body)

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
                yield chunk

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


def member_from_dict(
    base_url: str,
    api_key: str,
    model: str,
    label: str | None = None,
) -> PanelMember:
    return PanelMember(base_url=base_url, api_key=api_key, model=model, label=label)
