"""Sequential pipeline strategy: chain LLM steps, feeding each output to the next."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

from openfusion.config import OpenFusionConfig, PanelMember, PipelineStepUse
from openfusion.errors import InvalidRequestError, UpstreamError
from openfusion.panel import gather_panel
from openfusion.synthesize import synthesize
from openfusion.upstream import UpstreamClient, extract_response_usage


def _inject_step_outputs(
    template: str,
    outputs: dict[str, str],
) -> str:
    """Replace {step_name} placeholders in a system prompt with captured outputs."""
    result = template
    for name, text in outputs.items():
        result = result.replace(f"{{{name}}}", text)
    return result


def _build_step_messages(
    original_messages: list[dict[str, Any]],
    step_system: str | None,
    outputs: dict[str, str],
) -> list[dict[str, Any]]:
    """Build the message list for a step, injecting previous outputs into system prompt."""
    non_system = [m for m in original_messages if m.get("role") != "system"]

    if step_system is None:
        if outputs:
            # Append prior outputs as an assistant context block before the user turn.
            context = "\n\n".join(
                f"[{name} output]\n{text}" for name, text in outputs.items()
            )
            injected: dict[str, Any] = {
                "role": "system",
                "content": f"Context from previous steps:\n{context}",
            }
            return [injected, *non_system]
        return list(non_system)

    resolved = _inject_step_outputs(step_system, outputs)
    return [{"role": "system", "content": resolved}, *non_system]


async def _collect_stream(
    stream: AsyncIterator[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """Drain an async iterator of chunks, returning (full_text, last_usage)."""
    parts: list[str] = []
    usage: dict[str, Any] | None = None
    async for chunk in stream:
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
        u = extract_response_usage(chunk)
        if u:
            usage = u
    return "".join(parts), usage


async def _open_solo_stream(
    client: UpstreamClient,
    solo_member: PanelMember,
    solo_body: dict[str, Any],
    *,
    timeout: float | None,
    phase: str,
) -> AsyncIterator[dict[str, Any]]:
    """Issue a streaming solo-step chat completion; raise if the upstream didn't stream."""
    solo_body["stream"] = True
    stream = await client.chat_completion(
        solo_member, solo_body, stream=True, timeout=timeout, phase=phase
    )
    if not hasattr(stream, "__aiter__"):
        raise UpstreamError("Expected streaming response from solo step")
    return stream  # type: ignore[return-value]


async def run_pipeline(
    request_body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    timeout: float | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any] | None, str | None]]:
    """Execute pipeline steps in order; stream text deltas from the final step.

    Yields (text_delta, usage, finish_reason) matching the synthesize() interface.
    """
    steps = config.pipeline.steps
    if not steps:
        raise InvalidRequestError("Pipeline has no steps configured")

    original_messages: list[dict[str, Any]] = request_body.get("messages", [])
    outputs: dict[str, str] = {}

    for idx, step in enumerate(steps):
        step_messages = _build_step_messages(original_messages, step.system, outputs)
        step_body = copy.deepcopy(request_body)
        step_body["messages"] = step_messages
        is_last = idx == len(steps) - 1

        if step.use == PipelineStepUse.FUSE:
            if config.judge is None:
                raise InvalidRequestError(
                    f"Step '{step.name}' uses fuse but no judge is configured"
                )
            panel = await gather_panel(step_body, config, client)
            if is_last:
                # Stream the final synthesis to the caller.
                async for delta, usage, finish_reason in synthesize(
                    step_body, panel, config, client, timeout=timeout
                ):
                    yield delta, usage, finish_reason
                return
            # Intermediate fuse step: collect the synthesis output silently.
            synth_parts: list[str] = []
            async for delta, _usage, _reason in synthesize(
                step_body, panel, config, client, timeout=timeout
            ):
                if delta:
                    synth_parts.append(delta)
            outputs[step.name] = "".join(synth_parts)

        else:  # SOLO
            pass_through = config.resolved_pass_through()
            if step.model is not None:
                pass_through = pass_through.model_copy(update={"model": step.model})

            solo_member = PanelMember(
                base_url=pass_through.base_url,
                api_key=pass_through.api_key,
                model=pass_through.model,
            )
            solo_body = {**step_body, "model": solo_member.model}
            stream = await _open_solo_stream(
                client, solo_member, solo_body, timeout=timeout, phase=step.name
            )

            if is_last:
                async for chunk in stream:
                    choices = chunk.get("choices") or []
                    delta_text = ""
                    finish_reason = None
                    if choices:
                        delta_text = (choices[0].get("delta") or {}).get("content") or ""
                        finish_reason = choices[0].get("finish_reason")
                    usage = extract_response_usage(chunk)
                    if delta_text or usage or finish_reason:
                        yield delta_text, usage, finish_reason
                return

            text, _usage = await _collect_stream(stream)
            outputs[step.name] = text
