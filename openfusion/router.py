"""Per-prompt router gate: decide whether a request is worth fusing.

Fusion runs N panel calls plus a judge, so it earns its cost on hard, open-ended
prompts and wastes it on trivial ones. When `router.enabled` is set, this module
makes that call up front from cheap prompt-shape signals (no model call), so the
proxy can answer simple prompts with a single pass-through completion and reserve
the panel for prompts that look like they benefit from it.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from openfusion.config import RouteModel, RouterConfig, RouterMode, Tier
from openfusion.cost import RequestPhase
from openfusion.outcomes import OutcomeStore
from openfusion.upstream import UpstreamClient

_log = logging.getLogger(__name__)


def _record_fallback() -> None:
    from openfusion.metrics import METRICS  # local import avoids circular dependency

    METRICS.record_router_fallback()


class RouteDecision(StrEnum):
    FUSE = "fuse"  # run the panel + aggregator
    SOLO = "solo"  # answer with a single pass-through call


_CLASSIFY_PROMPT = (
    "Decide whether the user's request needs a panel of multiple expert models "
    "(answer FUSE) or can be answered well by a single model (answer SOLO). "
    "FUSE for open-ended research, analysis, design, or high-stakes questions; "
    "SOLO for simple, factual, or trivial ones. Reply with exactly one word: "
    "FUSE or SOLO."
)


def _user_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # OpenAI multimodal content blocks: collect text parts only.
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
    return "\n".join(parts)


_STRONG_KEYWORDS = (
    "analyze",
    "analyse",
    "compare",
    "evaluate",
    "design",
    "research",
    "explain why",
    "trade-off",
    "tradeoff",
    "prove",
    "debug",
    "architecture",
)


def prompt_tier(text: str) -> Tier:
    """Estimate how hard a prompt is, for single-model routing."""
    lowered = text.lower()
    if "```" in text or len(text) >= 600 or any(k in lowered for k in _STRONG_KEYWORDS):
        return Tier.STRONG
    if len(text) >= 200:
        return Tier.BALANCED
    return Tier.FAST


# When the ideal tier has no candidate, fall back to the nearest available band.
_TIER_FALLBACK: dict[Tier, list[Tier]] = {
    Tier.FAST: [Tier.FAST, Tier.BALANCED, Tier.STRONG],
    Tier.BALANCED: [Tier.BALANCED, Tier.STRONG, Tier.FAST],
    Tier.STRONG: [Tier.STRONG, Tier.BALANCED, Tier.FAST],
}


def select_model(body: dict[str, Any], config: RouterConfig) -> RouteModel | None:
    """Pick the best single model for this prompt, or None to use the default."""
    if not config.route_models:
        return None
    want = prompt_tier(_user_text(body))
    for tier in _TIER_FALLBACK[want]:
        for candidate in config.route_models:
            if candidate.tier == tier:
                return candidate
    return config.route_models[0]


def route(
    body: dict[str, Any],
    config: RouterConfig,
    outcomes: OutcomeStore | None = None,
) -> RouteDecision:
    """Return whether to FUSE or answer SOLO for this request.

    When an OutcomeStore is supplied, the historical success-rate EMA can nudge
    the heuristic decision — but only when the heuristic itself is uncertain
    (i.e. the prompt doesn't clearly call for fusion or clearly not).
    """
    if config.mode == RouterMode.ALWAYS:
        return RouteDecision.FUSE
    if config.mode == RouterMode.NEVER:
        return RouteDecision.SOLO

    text = _user_text(body)
    lowered = text.lower()

    # Strong signals: code blocks and explicit analytical keywords always fuse.
    if "```" in text:
        return RouteDecision.FUSE
    if any(keyword in lowered for keyword in config.fuse_keywords):
        return RouteDecision.FUSE
    # Long prompts tend to carry enough substance to benefit from synthesis.
    if len(text) >= config.min_chars:
        return RouteDecision.FUSE

    # Heuristic says SOLO — but check the outcome store for a learning-based nudge.
    if outcomes is not None:
        tier = prompt_tier(text)
        prefer = outcomes.prefer_fuse(tier)
        if prefer is True:
            _log.debug("router: outcome store nudging SOLO→FUSE for tier=%s", tier)
            return RouteDecision.FUSE

    return RouteDecision.SOLO


async def route_async(
    body: dict[str, Any],
    config: RouterConfig,
    client: UpstreamClient,
    outcomes: OutcomeStore | None = None,
) -> RouteDecision:
    """Async router that supports the model classifier; else delegates to route().

    On any classifier error the decision falls back to the heuristic, so routing
    never fails a request.
    """
    if config.mode != RouterMode.MODEL or config.classifier is None:
        return route(body, config, outcomes)

    classifier = config.classifier.model_copy(update={"label": "router"})
    request = {
        "messages": [
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": _user_text(body)[:4000]},
        ],
        "max_tokens": config.classifier_max_tokens,
        "temperature": 0,
    }
    try:
        payload = await client.chat_completion(
            classifier, request, stream=False, phase=RequestPhase.PASS_THROUGH
        )
    except Exception:  # noqa: BLE001 - never fail routing on a classifier error
        _log.warning("router: classifier call failed, falling back to heuristic")
        _record_fallback()
        return route(body, config)

    if not isinstance(payload, dict):
        _log.warning("router: classifier returned unexpected payload type, falling back to heuristic")  # noqa: E501  # noqa: E501
        _record_fallback()
        return route(body, config)
    choices = payload.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
    upper = text.upper()
    if "SOLO" in upper and "FUSE" not in upper:
        return RouteDecision.SOLO
    if "FUSE" in upper:
        return RouteDecision.FUSE
    _log.warning("router: classifier response %r not recognized, falling back to heuristic", text)
    _record_fallback()
    return route(body, config)


def _select_prompt(route_models: list[RouteModel]) -> str:
    options = "\n".join(f"- {rm.model} ({rm.tier.value})" for rm in route_models)
    return (
        "Choose how to answer the user's request. Reply with EITHER the single word FUSE "
        "(to use a full panel of models for a hard/open-ended request), OR the exact id of "
        "the cheapest single model below that can handle it well. Models (fast=cheap, "
        "strong=most capable):\n"
        f"{options}\n\nReply with just FUSE or one model id."
    )


async def _classify_route(
    body: dict[str, Any], config: RouterConfig, client: UpstreamClient
) -> tuple[RouteDecision, RouteModel | None] | None:
    """One classifier call that returns FUSE or a chosen model. None on failure."""
    classifier = config.classifier.model_copy(update={"label": "router"})  # type: ignore[union-attr]
    request = {
        "messages": [
            {"role": "system", "content": _select_prompt(config.route_models)},
            {"role": "user", "content": _user_text(body)[:4000]},
        ],
        "max_tokens": 24,
        "temperature": 0,
    }
    try:
        payload = await client.chat_completion(
            classifier, request, stream=False, phase=RequestPhase.PASS_THROUGH
        )
    except Exception:  # noqa: BLE001 - never fail routing on a classifier error
        _log.warning("router: classifier call failed, falling back to heuristic")
        _record_fallback()
        return None
    if not isinstance(payload, dict):
        _log.warning("router: classifier returned unexpected payload type, falling back to heuristic")  # noqa: E501
        _record_fallback()
        return None
    choices = payload.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
    if "FUSE" in text.upper():
        return RouteDecision.FUSE, None
    lowered = text.lower()
    for candidate in config.route_models:
        if candidate.model.lower() in lowered:
            return RouteDecision.SOLO, candidate
    _log.warning("router: classifier response %r matched no model, falling back to heuristic", text)
    _record_fallback()
    return None


async def route_request(
    body: dict[str, Any],
    config: RouterConfig,
    client: UpstreamClient,
    outcomes: OutcomeStore | None = None,
) -> tuple[RouteDecision, RouteModel | None]:
    """Decide fuse-vs-solo AND, for solo, which model — the single routing entry point.

    With ``mode: model`` and ``route_models`` set, one classifier call picks FUSE or a
    specific model. Otherwise the heuristic decides and ``select_model`` picks the model.
    Always falls back to the heuristic on any classifier error.

    ``outcomes`` is the module-level OutcomeStore; when supplied, the learning loop can
    nudge the heuristic toward whichever path has been winning recently.
    """
    if config.mode == RouterMode.MODEL and config.classifier is not None and config.route_models:
        result = await _classify_route(body, config, client)
        if result is not None:
            return result

    decision = await route_async(body, config, client, outcomes)
    if decision == RouteDecision.SOLO:
        return RouteDecision.SOLO, select_model(body, config)
    return RouteDecision.FUSE, None
