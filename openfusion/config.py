"""Configuration loading for openfusion."""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Strategy(StrEnum):
    PANEL = "panel"
    SELF_FUSION = "self_fusion"
    DEBATE = "debate"


class RouterMode(StrEnum):
    """How the pre-panel router decides between fusing and a single model."""

    HEURISTIC = "heuristic"  # cheap prompt-shape signals decide
    MODEL = "model"  # a small classifier model decides (falls back to heuristic)
    ALWAYS = "always"  # always fuse
    NEVER = "never"  # always answer with a single pass-through call


class Preset(StrEnum):
    """One-word panel recipes, mirroring OpenRouter Fusion's Quality/Budget switch.

    A preset expands to a diverse OpenRouter panel + judge with web tools enabled
    (the regime where bench/FINDINGS.md shows synthesis actually beats the best
    single member). Anything the user sets explicitly in YAML wins over the preset.
    """

    QUALITY = "quality"
    BUDGET = "budget"


# Each preset is a diverse panel (different model families take different
# search/fetch trajectories → complementary evidence for the judge to fuse) plus
# a strong judge. Models match the repo's example configs and validated bench runs.
_PRESETS: dict[Preset, dict[str, Any]] = {
    Preset.QUALITY: {
        "panel_models": [
            "anthropic/claude-sonnet-4",
            "google/gemini-3-pro",
            "deepseek/deepseek-v4-pro",
        ],
        "judge_model": "anthropic/claude-sonnet-4",
        "pass_through_model": "anthropic/claude-sonnet-4",
    },
    Preset.BUDGET: {
        "panel_models": [
            "openai/gpt-4o-mini",
            "deepseek/deepseek-v4-pro",
            "moonshotai/kimi-k2.6",
        ],
        "judge_model": "deepseek/deepseek-v4-pro",
        "pass_through_model": "openai/gpt-4o-mini",
    },
}


class Aggregator(StrEnum):
    """How panel responses are combined into the final answer."""

    JUDGE = "judge"  # a judge model synthesizes one answer
    VOTE = "vote"  # majority vote over panel answers (self-consistency)
    RANKED = "ranked"  # the judge picks the single best answer (no synthesis)


class PanelMember(BaseModel):
    base_url: str
    api_key: str
    model: str
    label: str | None = None

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class JudgeConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    max_panel_tokens: int = Field(default=120_000, ge=1)

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class SelfFusionConfig(BaseModel):
    n: int = Field(default=3, ge=1, le=16)
    temperature_spread: list[float] = Field(default_factory=lambda: [0.3, 0.7, 1.0])
    seed_offset: bool = True


_DEFAULT_FUSE_KEYWORDS = [
    "compare",
    "trade-off",
    "tradeoff",
    "analyze",
    "analyse",
    "evaluate",
    "design",
    "research",
    "pros and cons",
    "why",
    "explain",
    "critique",
    "recommend",
]


class Tier(StrEnum):
    """Rough capability/cost band for model routing."""

    FAST = "fast"  # cheap, for simple prompts
    BALANCED = "balanced"
    STRONG = "strong"  # frontier, for hard prompts


class RouteModel(BaseModel):
    """A candidate the router can send a single prompt to (cost/quality band)."""

    model: str
    tier: Tier = Tier.BALANCED
    base_url: str | None = None  # falls back to the default upstream credentials
    api_key: str | None = None


class RouterConfig(BaseModel):
    """Per-prompt gate: send simple prompts to a single model, fuse hard ones.

    Disabled by default — when off, every `openfusion` request is fused. When on,
    short/simple prompts are answered by one pass-through call (cheaper, faster)
    and only prompts that look like they benefit from a panel are fused. When
    ``route_models`` is set, the single-model branch picks the best model for the
    prompt by difficulty (set ``mode: never`` for pure routing with no fusion).
    """

    enabled: bool = False
    mode: RouterMode = RouterMode.HEURISTIC
    min_chars: int = Field(default=280, ge=0)
    fuse_keywords: list[str] = Field(default_factory=lambda: list(_DEFAULT_FUSE_KEYWORDS))
    # For mode=model: a small/cheap model that answers FUSE or SOLO. Falls back
    # to the heuristic if unset or if the classification call fails.
    classifier: PanelMember | None = None
    classifier_max_tokens: int = Field(default=4, ge=1, le=64)
    # Candidates for the single-model (SOLO) branch; empty = use the default
    # pass-through model.
    route_models: list[RouteModel] = Field(default_factory=list)


class DebateConfig(BaseModel):
    """Multi-round panel: members revise after seeing each other's answers."""

    rounds: int = Field(default=1, ge=1, le=3)


class LimitsConfig(BaseModel):
    """Concurrency and per-key rate limits for serving public traffic.

    Both default to 0 (unlimited), preserving the MVP behavior. ``max_in_flight``
    caps concurrent fusion requests; ``rate_limit_per_minute`` is a fixed-window
    cap per gateway key (or per client when no key is presented).
    """

    max_in_flight: int = Field(default=0, ge=0)
    rate_limit_per_minute: int = Field(default=0, ge=0)


class CacheConfig(BaseModel):
    """Prompt caching for the self-fusion shared prefix (provider-dependent).

    When enabled, panel calls mark the prompt with OpenRouter/Anthropic-style
    ``cache_control`` breakpoints so the N self-fusion samples reuse the cached
    prefix. A no-op on providers that ignore the marker.
    """

    enabled: bool = False


class ResponseCacheConfig(BaseModel):
    """In-process cache of fused answers, keyed by prompt + recipe.

    When enabled, an identical request (same messages, panel, judge, aggregator,
    tools, and token cap) is served from memory — instant and free — until it
    expires. Off by default.
    """

    enabled: bool = False
    ttl_seconds: int = Field(default=300, ge=1)
    max_entries: int = Field(default=512, ge=1)


class AnalysisConfig(BaseModel):
    """Expose the judge's structured analysis as a separate SSE ``analysis`` event."""

    emit: bool = False


class TimeoutsConfig(BaseModel):
    member_seconds: float = Field(default=120.0, gt=0)
    judge_seconds: float = Field(default=180.0, gt=0)
    total_seconds: float = Field(default=300.0, gt=0)


class GatewayAuthConfig(BaseModel):
    api_keys: list[str] = Field(default_factory=list)


class CostControlsConfig(BaseModel):
    """Token ceilings used to keep live provider calls bounded."""

    pass_through_max_tokens: int | None = Field(default=None, ge=1)
    panel_max_tokens: int | None = Field(default=None, ge=1)
    judge_max_tokens: int | None = Field(default=None, ge=1)


class ToolsConfig(BaseModel):
    """Server-side tools made available to panel members (and optionally the judge).

    Uses OpenRouter's agentic server tools (`openrouter:web_search` /
    `openrouter:web_fetch`), which the model calls in a loop and OpenRouter
    executes upstream, returning a final content answer — so no client-side
    tool-call loop is needed. `excluded_domains` is forwarded to web_search
    (and as blocked_domains to web_fetch) to prevent benchmark contamination.
    """

    web_search: bool = False
    web_fetch: bool = True
    max_results: int = Field(default=5, ge=1, le=20)
    engine: str = "auto"  # auto | native | exa
    excluded_domains: list[str] = Field(default_factory=list)
    apply_to_judge: bool = False


class PassThroughConfig(BaseModel):
    """Single upstream used for non-fusion model pass-through and tool calls."""

    base_url: str
    api_key: str
    model: str

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class OpenFusionConfig(BaseModel):
    preset: Preset | None = None
    strategy: Strategy = Strategy.SELF_FUSION
    aggregator: Aggregator = Aggregator.JUDGE
    panel: list[PanelMember] = Field(default_factory=list)
    judge: JudgeConfig | None = None
    self_fusion: SelfFusionConfig = Field(default_factory=SelfFusionConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    response_cache: ResponseCacheConfig = Field(default_factory=ResponseCacheConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    gateway: GatewayAuthConfig = Field(default_factory=GatewayAuthConfig)
    cost_controls: CostControlsConfig = Field(default_factory=CostControlsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    pass_through: PassThroughConfig | None = None
    fusion_model_name: str = "openfusion"
    # Allow clients to override panel/judge/preset/tools per request via the
    # `openfusion` request field (used by the playground). Off by default: when
    # on it's still bounded by gateway auth, cost ceilings, and rate limits.
    allow_request_overrides: bool = False
    # Allow the upstream API key to be set at runtime from the playground UI.
    # On for the zero-config quick start; keep off for shared/hosted servers.
    allow_ui_api_key: bool = False
    # Emit each panel member's answer as an SSE `panel_answer` event (the
    # side-by-side view). Per-request via the `openfusion` override; off for the
    # plain API so intermediate answers aren't exposed by default.
    expose_panel: bool = False

    def resolved_pass_through(self) -> PassThroughConfig:
        if self.pass_through is not None:
            return self.pass_through
        if not self.panel:
            raise ValueError("pass_through is required when panel is empty")
        member = self.panel[0]
        return PassThroughConfig(
            base_url=member.base_url,
            api_key=member.api_key,
            model=member.model,
        )


def _apply_preset(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill panel/judge/tools defaults from a named preset.

    Runs before env expansion so the injected ``${OPENROUTER_API_KEY}``
    placeholders go through the same fail-fast expansion as hand-written YAML.
    Every value the user set explicitly takes precedence over the preset.
    """
    preset_name = raw.get("preset")
    if not preset_name:
        return raw

    spec = _PRESETS[Preset(preset_name)]
    raw.setdefault("strategy", Strategy.PANEL.value)
    raw.setdefault("aggregator", Aggregator.JUDGE.value)
    # Tools on by default: FINDINGS.md shows fusion only beats the best single
    # member when panelists do real research, so a preset should land there.
    tools = dict(raw.get("tools") or {})
    tools.setdefault("web_search", True)
    tools.setdefault("web_fetch", True)
    raw["tools"] = tools

    api_key = "${OPENROUTER_API_KEY}"
    if not raw.get("panel"):
        raw["panel"] = [
            {
                "base_url": OPENROUTER_BASE_URL,
                "api_key": api_key,
                "model": model,
                "label": f"panel-{index}",
            }
            for index, model in enumerate(spec["panel_models"])
        ]
    if not raw.get("judge"):
        raw["judge"] = {
            "base_url": OPENROUTER_BASE_URL,
            "api_key": api_key,
            "model": spec["judge_model"],
        }
    if not raw.get("pass_through"):
        raw["pass_through"] = {
            "base_url": OPENROUTER_BASE_URL,
            "api_key": api_key,
            "model": spec["pass_through_model"],
        }
    return raw


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replacer(match: re.Match[str]) -> str:
            env_name = match.group(1)
            env_value = os.environ.get(env_name)
            if env_value is None:
                raise ValueError(
                    f"Environment variable {env_name} is not set. "
                    f"Export it before starting, e.g. `export {env_name}=...`, "
                    f"or replace the ${{{env_name}}} placeholder in your config."
                )
            return env_value

        return _ENV_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def _load_gateway_keys(raw: dict[str, Any]) -> list[str]:
    gateway = raw.get("gateway", {})
    keys = list(gateway.get("api_keys", []))
    env_keys = os.environ.get("OPENFUSION_API_KEYS", "").strip()
    if env_keys:
        keys.extend(key.strip() for key in env_keys.split(",") if key.strip())
    return keys


def quickstart_config() -> OpenFusionConfig:
    """Zero-config default so `openfusion` boots with no YAML and no env key.

    Uses the Budget preset with web tools on. The OpenRouter key is read from
    OPENROUTER_API_KEY if present, otherwise left empty so the user can paste it
    into the playground. UI key entry and per-request overrides are enabled so
    the quick start "just works"; a hosted server should ship a real config.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        from openfusion.credentials import load_saved_key

        key = load_saved_key() or ""
    spec = _PRESETS[Preset.BUDGET]
    return OpenFusionConfig(
        preset=Preset.BUDGET,
        strategy=Strategy.PANEL,
        panel=[
            PanelMember(base_url=OPENROUTER_BASE_URL, api_key=key, model=model, label=f"panel-{i}")
            for i, model in enumerate(spec["panel_models"])
        ],
        judge=JudgeConfig(base_url=OPENROUTER_BASE_URL, api_key=key, model=spec["judge_model"]),
        pass_through=PassThroughConfig(
            base_url=OPENROUTER_BASE_URL, api_key=key, model=spec["pass_through_model"]
        ),
        tools=ToolsConfig(web_search=True, web_fetch=True),
        cost_controls=CostControlsConfig(
            pass_through_max_tokens=1024, panel_max_tokens=1024, judge_max_tokens=1024
        ),
        allow_ui_api_key=True,
        allow_request_overrides=True,
    )


def load_config(path: str | Path | None = None) -> OpenFusionConfig:
    """Load configuration from YAML and environment variables.

    With no config file and no explicit path, returns the zero-config quick-start
    default (Budget preset) so the server boots and the key can be added in the UI.
    """
    config_path = Path(path or os.environ.get("OPENFUSION_CONFIG", "openfusion.yaml"))
    if not config_path.exists():
        if path is None:
            return quickstart_config()
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy an example to get started, e.g. "
            "`cp examples/preset.yaml.example openfusion.yaml`, "
            "then set OPENROUTER_API_KEY."
        )

    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    raw = _apply_preset(raw)
    raw["gateway"] = {"api_keys": _load_gateway_keys(raw)}
    expanded = _expand_env(raw)
    return OpenFusionConfig.model_validate(expanded)
