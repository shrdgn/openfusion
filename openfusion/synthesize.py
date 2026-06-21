"""Judge synthesis — yields text deltas only."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

from openfusion.config import JudgeConfig, OpenFusionConfig
from openfusion.cost import CostPolicy, RequestPhase
from openfusion.errors import UpstreamError
from openfusion.panel import PanelResult
from openfusion.tools import apply_web_tools
from openfusion.upstream import UpstreamClient, extract_response_usage

JUDGE_SYSTEM_PROMPT = (
    "You are the synthesizer. Below are N independent answers to the same user request. "
    "Identify points of consensus, contradictions, partial coverage, unique insights, and "
    "blind spots. Then write a single best answer grounded in that analysis. "
    "Be concise and focused: lead with the answer, keep only the most important points, and cut "
    "redundancy, hedging, and filler. Prefer the shortest response that fully answers, and match "
    "length to the question — do not pad. Use headings or lists only when they aid clarity. "
    "Honor the original user's output format and constraints exactly. "
    "Do not mention the panel or that multiple answers existed."
)

ANALYSIS_SENTINEL = "===ANALYSIS==="

ANALYSIS_INSTRUCTION = (
    "\n\nAfter the answer, output a line containing exactly "
    f"{ANALYSIS_SENTINEL} and then a single JSON object with keys: consensus, "
    "contradictions, partial_coverage, unique_insights, blind_spots. Each value is "
    "a short string or list of strings. Output nothing after the JSON."
)


def _estimate_tokens(text: str) -> int:
    # Rough heuristic until tokenizer integration is added.
    return max(1, len(text) // 4)


def _truncate_panel_responses(
    responses: list[tuple[str, str]],
    max_tokens: int,
) -> list[tuple[str, str]]:
    """Truncate longest answers first until the injected panel fits the budget."""
    remaining = list(responses)
    while remaining:
        total = sum(_estimate_tokens(content) for _, content in remaining)
        if total <= max_tokens:
            return remaining
        longest_index = max(range(len(remaining)), key=lambda idx: len(remaining[idx][1]))
        label, content = remaining[longest_index]
        if len(content) <= 200:
            break
        truncated = content[: max(200, len(content) // 2)] + "\n...[truncated]"
        remaining[longest_index] = (label, truncated)
    return remaining


def build_judge_messages(
    original_messages: list[dict[str, Any]],
    panel: PanelResult,
    judge: JudgeConfig,
) -> list[dict[str, Any]]:
    labeled_blocks = [(response.label, response.content) for response in panel.responses]
    capped = _truncate_panel_responses(labeled_blocks, judge.max_panel_tokens)
    panel_text = "\n\n".join(f"### {label}\n{content}" for label, content in capped)

    user_messages = [message for message in original_messages if message.get("role") != "system"]
    original_user_text = "\n".join(
        str(message.get("content", "")) for message in user_messages if message.get("content")
    )

    judge_user_prompt = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Original user request:\n{original_user_text}\n\n"
        f"Panel answers:\n{panel_text}"
    )

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt},
    ]


def _extract_delta_content(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _extract_finish_reason(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else None


async def synthesize(
    request_body: dict[str, Any],
    panel: PanelResult,
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    timeout: float | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any] | None, str | None]]:
    """Yield (text_delta, usage, finish_reason) tuples from the judge upstream."""
    judge = config.judge
    if judge is None:
        raise UpstreamError("Judge is not configured")

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise UpstreamError("Request messages must be a list")

    judge_member = judge.model_copy(update={"label": "judge"})
    judge_body = copy.deepcopy(request_body)
    judge_messages = build_judge_messages(messages, panel, judge)
    if config.analysis.emit and judge_messages:
        judge_messages[0]["content"] += ANALYSIS_INSTRUCTION
    judge_body["messages"] = judge_messages
    judge_body.pop("model", None)
    # The judge synthesizes the final text answer; it must not emit tool calls.
    # Client-supplied tools were already run by the panel (when server-executable).
    for tool_key in ("tools", "tool_choice", "functions", "function_call"):
        judge_body.pop(tool_key, None)
    judge_body["stream"] = True
    if request_body.get("stream_options"):
        judge_body["stream_options"] = request_body["stream_options"]
    judge_body = CostPolicy(config.cost_controls).apply_token_limit(
        judge_body,
        RequestPhase.JUDGE,
        reject_over_limit=True,
    )
    if config.tools.apply_to_judge:
        judge_body = apply_web_tools(judge_body, config.tools)

    stream = await client.chat_completion(
        judge_member,
        judge_body,
        stream=True,
        timeout=timeout,
        phase=RequestPhase.JUDGE,
    )
    if not hasattr(stream, "__aiter__"):
        raise UpstreamError("Expected streaming judge response")

    async for chunk in stream:
        delta = _extract_delta_content(chunk)
        usage = extract_response_usage(chunk)
        finish_reason = _extract_finish_reason(chunk)
        if delta:
            yield delta, usage, finish_reason
        elif usage or finish_reason:
            yield "", usage, finish_reason
