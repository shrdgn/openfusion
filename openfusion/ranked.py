"""Ranked-choice aggregation: the judge picks the single best answer.

A cheaper middle ground between majority vote and full synthesis. One short judge
call selects the strongest panel answer, which is returned verbatim — no long
synthesis generation, so it costs a fraction of the judge aggregator while still
using a model's judgment (unlike string-matching vote).
"""

from __future__ import annotations

import copy
import re
from typing import Any

from openfusion.config import OpenFusionConfig
from openfusion.cost import RequestPhase
from openfusion.errors import UpstreamError
from openfusion.panel import PanelResult
from openfusion.upstream import UpstreamClient

_RANK_PROMPT = (
    "You are judging candidate answers to the same request. Reply with ONLY the "
    "number of the single best answer (most correct, complete, and useful). "
    "Output just the number, nothing else."
)


def _original_user_text(messages: list[dict[str, Any]]) -> str:
    user_messages = [m for m in messages if m.get("role") != "system"]
    return "\n".join(str(m.get("content", "")) for m in user_messages if m.get("content"))


def build_ranking_messages(
    original_messages: list[dict[str, Any]],
    panel: PanelResult,
) -> list[dict[str, Any]]:
    blocks = "\n\n".join(
        f"[{i + 1}] {response.content}" for i, response in enumerate(panel.responses)
    )
    prompt = (
        f"{_RANK_PROMPT}\n\n"
        f"Request:\n{_original_user_text(original_messages)}\n\n"
        f"Candidate answers:\n{blocks}"
    )
    return [{"role": "system", "content": _RANK_PROMPT}, {"role": "user", "content": prompt}]


def _parse_choice(text: str, n: int) -> int:
    """Return a 0-based index from the judge's reply; default to 0 on garbage."""
    match = re.search(r"\d+", text or "")
    if not match:
        return 0
    choice = int(match.group()) - 1
    if 0 <= choice < n:
        return choice
    return 0


async def pick_best(
    request_body: dict[str, Any],
    panel: PanelResult,
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    timeout: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (best_answer_text, metadata) using one judge ranking call."""
    judge = config.judge
    if judge is None:
        raise UpstreamError("Ranked-choice aggregation requires a judge")
    responses = panel.responses
    if len(responses) == 1:
        return responses[0].content, {"winner": 0, "members": 1}

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise UpstreamError("Request messages must be a list")

    judge_member = judge.model_copy(update={"label": "ranker"})
    body = copy.deepcopy(request_body)
    body["messages"] = build_ranking_messages(messages, panel)
    body.pop("model", None)
    for tool_key in ("tools", "tool_choice", "functions", "function_call"):
        body.pop(tool_key, None)
    body["max_tokens"] = 8
    body["stream"] = False

    payload = await client.chat_completion(
        judge_member, body, stream=False, timeout=timeout, phase=RequestPhase.JUDGE
    )
    if not isinstance(payload, dict):
        raise UpstreamError("Expected JSON ranking response")
    choices = payload.get("choices") or []
    text = ""
    if choices:
        text = (choices[0].get("message") or {}).get("content") or ""
    index = _parse_choice(text, len(responses))
    return responses[index].content, {"winner": index, "members": len(responses)}
