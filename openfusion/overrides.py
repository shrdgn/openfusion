"""Per-request overrides for panel / judge / preset / tools.

Mirrors OpenRouter Fusion's ability to override the panel (`analysis_models`)
and judge (`model`) per request. The playground uses this so its model selectors
are real. Overridden panel/judge models reuse the server's default upstream
credentials (base_url + api_key) — clients pick model *ids*, never keys — and
the result is still bounded by the server's cost ceilings, auth, and rate limits.
"""

from __future__ import annotations

from typing import Any

from openfusion.config import (
    _PRESETS,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    Preset,
    Strategy,
)

MAX_OVERRIDE_PANEL = 6


def _default_credentials(config: OpenFusionConfig) -> tuple[str, str]:
    if config.panel:
        return config.panel[0].base_url, config.panel[0].api_key
    if config.pass_through is not None:
        return config.pass_through.base_url, config.pass_through.api_key
    if config.judge is not None:
        return config.judge.base_url, config.judge.api_key
    raise ValueError("No upstream credentials available for request overrides")


def _panel_from_models(models: list[str], base_url: str, api_key: str) -> list[PanelMember]:
    trimmed = [str(m) for m in models if str(m).strip()][:MAX_OVERRIDE_PANEL]
    return [
        PanelMember(base_url=base_url, api_key=api_key, model=model, label=f"panel-{index}")
        for index, model in enumerate(trimmed)
    ]


def apply_overrides(config: OpenFusionConfig, override: dict[str, Any]) -> OpenFusionConfig:
    """Return a copy of ``config`` with per-request overrides applied.

    Cost controls, gateway auth, and limits are intentionally preserved from the
    base config so overrides can't widen those bounds.
    """
    if not override:
        return config

    new = config.model_copy(deep=True)
    base_url, api_key = _default_credentials(config)

    preset = override.get("preset")
    if isinstance(preset, str) and preset in (Preset.QUALITY.value, Preset.BUDGET.value):
        spec = _PRESETS[Preset(preset)]
        new.preset = Preset(preset)
        new.strategy = Strategy.PANEL
        new.panel = _panel_from_models(spec["panel_models"], base_url, api_key)
        new.judge = JudgeConfig(base_url=base_url, api_key=api_key, model=spec["judge_model"])
        new.tools.web_search = True
        new.tools.web_fetch = True

    panel = override.get("panel")
    if isinstance(panel, list) and panel:
        members = _panel_from_models(panel, base_url, api_key)
        if members:
            new.strategy = Strategy.PANEL
            new.panel = members

    judge = override.get("judge")
    if isinstance(judge, str) and judge.strip():
        if new.judge is not None:
            new.judge = new.judge.model_copy(update={"model": judge})
        else:
            new.judge = JudgeConfig(base_url=base_url, api_key=api_key, model=judge)

    tools = override.get("tools")
    if isinstance(tools, dict):
        if "web_search" in tools:
            new.tools.web_search = bool(tools["web_search"])
        if "web_fetch" in tools:
            new.tools.web_fetch = bool(tools["web_fetch"])

    return new
