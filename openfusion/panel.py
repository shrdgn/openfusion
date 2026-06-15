"""Panel fan-out and graceful degradation."""

from __future__ import annotations

import asyncio
import copy
import random
from dataclasses import dataclass, field
from typing import Any

from openfusion.config import OpenFusionConfig, PanelMember, Strategy
from openfusion.errors import UpstreamError
from openfusion.upstream import UpstreamClient


@dataclass
class MemberFailure:
    label: str
    reason: str
    status_code: int | None = None


@dataclass
class MemberResponse:
    label: str
    content: str
    model: str
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None


@dataclass
class PanelResult:
    responses: list[MemberResponse] = field(default_factory=list)
    failures: list[MemberFailure] = field(default_factory=list)

    @property
    def usage_total(self) -> dict[str, int] | None:
        prompt = completion = 0
        found = False
        for response in self.responses:
            if response.usage:
                found = True
                prompt += response.usage.get("prompt_tokens", 0)
                completion += response.usage.get("completion_tokens", 0)
        if not found:
            return None
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


def expand_panel_members(config: OpenFusionConfig) -> list[tuple[PanelMember, dict[str, Any]]]:
    """Return panel members with per-member sampling overrides."""
    if config.strategy == Strategy.SELF_FUSION:
        if not config.panel:
            raise UpstreamError("Self-fusion requires at least one panel member in config")
        base_member = config.panel[0]
        spread = config.self_fusion.temperature_spread
        members: list[tuple[PanelMember, dict[str, Any]]] = []
        for index in range(config.self_fusion.n):
            temperature = spread[index % len(spread)]
            overrides: dict[str, Any] = {"temperature": temperature}
            if config.self_fusion.seed_offset:
                overrides["seed"] = index + 1
            label = base_member.label or base_member.model
            member = base_member.model_copy(
                update={"label": f"{label}-{index + 1}"},
            )
            members.append((member, overrides))
        return members

    return [(member, {}) for member in config.panel]


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result or None


async def _call_member(
    client: UpstreamClient,
    member: PanelMember,
    request_body: dict[str, Any],
    overrides: dict[str, Any],
    *,
    timeout: float,
    cancel_event: asyncio.Event,
) -> MemberResponse:
    body = copy.deepcopy(request_body)
    body.pop("model", None)
    body.pop("stream", None)
    for key in ("temperature", "seed"):
        if key in overrides:
            body[key] = overrides[key]

    deadline = asyncio.get_running_loop().time() + timeout
    attempt = 0
    while True:
        if cancel_event.is_set():
            raise asyncio.CancelledError()
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Member {member.label} exceeded timeout")

        try:
            payload = await asyncio.wait_for(
                client.chat_completion(member, body, stream=False, timeout=remaining),
                timeout=remaining,
            )
            if not isinstance(payload, dict):
                raise UpstreamError("Expected non-streaming upstream response")
            return MemberResponse(
                label=member.label or member.model,
                content=_extract_content(payload),
                model=member.model,
                usage=_extract_usage(payload),
                raw=payload,
            )
        except UpstreamError as exc:
            if exc.upstream_status_code == 429 and attempt < 3:
                attempt += 1
                backoff = min(2**attempt + random.random(), 8.0)
                if cancel_event.is_set():
                    raise asyncio.CancelledError() from exc
                await asyncio.sleep(backoff)
                continue
            raise


async def gather_panel(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    cancel_event: asyncio.Event | None = None,
) -> PanelResult:
    """Fan out to all panel members concurrently with graceful degradation."""
    members = expand_panel_members(config)
    if not members:
        raise UpstreamError("No panel members configured")

    cancel = cancel_event or asyncio.Event()
    timeout = config.timeouts.member_seconds

    async def run_member(
        member: PanelMember,
        overrides: dict[str, Any],
    ) -> MemberResponse | MemberFailure:
        label = member.label or member.model
        try:
            return await _call_member(
                client,
                member,
                request_body,
                overrides,
                timeout=timeout,
                cancel_event=cancel,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            return MemberFailure(label=label, reason=str(exc))
        except UpstreamError as exc:
            return MemberFailure(
                label=label,
                reason=str(exc),
                status_code=exc.upstream_status_code,
            )
        except Exception as exc:  # noqa: BLE001 - degrade per member
            return MemberFailure(label=label, reason=str(exc))

    tasks = [
        asyncio.create_task(run_member(member, overrides))
        for member, overrides in members
    ]
    try:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

    result = PanelResult()
    for outcome in outcomes:
        if isinstance(outcome, MemberResponse):
            result.responses.append(outcome)
        elif isinstance(outcome, MemberFailure):
            result.failures.append(outcome)
        elif isinstance(outcome, asyncio.CancelledError):
            raise outcome
        elif isinstance(outcome, Exception):
            result.failures.append(MemberFailure(label="unknown", reason=str(outcome)))

    if not result.responses:
        reasons = "; ".join(f"{failure.label}: {failure.reason}" for failure in result.failures)
        raise UpstreamError(f"All panel members failed: {reasons}")

    return result
