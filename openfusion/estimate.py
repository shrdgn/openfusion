"""Pre-run cost/usage estimate for a fusion request."""

from __future__ import annotations

from typing import Any

from openfusion.config import Aggregator, OpenFusionConfig
from openfusion.panel import expand_panel_members


def _estimate_input_tokens(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            # OpenAI multimodal content blocks: count text parts only; ignore images
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    chars += len(block["text"])
    return max(1, chars // 4)  # rough; ~4 chars/token


def build_estimate(body: dict[str, Any], config: OpenFusionConfig, prices: dict) -> dict[str, Any]:
    """Estimate the calls, tokens, and (when priced) dollars a fusion request costs."""
    input_tokens = _estimate_input_tokens(body.get("messages"))
    panel_cap = config.cost_controls.panel_max_tokens or 1024
    judge_cap = config.cost_controls.judge_max_tokens or 1024

    calls: list[dict[str, Any]] = []
    for member, _overrides in expand_panel_members(config):
        calls.append(
            {"model": member.model, "phase": "panel", "input": input_tokens, "output": panel_cap}
        )

    if config.aggregator in (Aggregator.JUDGE, Aggregator.RANKED) and config.judge is not None:
        # The judge sees the original prompt plus every panel answer.
        judge_input = input_tokens + panel_cap * len(calls)
        judge_output = 8 if config.aggregator == Aggregator.RANKED else judge_cap
        calls.append(
            {
                "model": config.judge.model,
                "phase": "judge",
                "input": judge_input,
                "output": judge_output,
            }
        )

    cost = 0.0
    fully_priced = bool(calls)
    for call in calls:
        price = prices.get(call["model"])
        if price is None:
            fully_priced = False
            continue
        cost += call["input"] * price["prompt"] + call["output"] * price["completion"]

    return {
        "calls": len(calls),
        "models": [call["model"] for call in calls],
        "input_tokens": input_tokens,
        "max_output_tokens": sum(call["output"] for call in calls),
        "cost_usd": round(cost, 5) if fully_priced else None,
    }
