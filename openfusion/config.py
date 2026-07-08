"""Configuration loading for openfusion."""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Strategy(StrEnum):
    PANEL = "panel"
    SELF_FUSION = "self_fusion"
    DEBATE = "debate"
    PIPELINE = "pipeline"


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


def _infer_provider(base_url: str) -> Literal["openai", "anthropic"]:
    """Infer the provider from the base URL when not explicitly set."""
    if "anthropic.com" in base_url:
        return "anthropic"
    return "openai"


def _strip_trailing_slash(value: str) -> str:
    return value.rstrip("/")


class PanelMember(BaseModel):
    base_url: str
    api_key: str
    model: str
    label: str | None = None
    provider: Literal["openai", "anthropic"] | None = None

    _normalize_base_url = field_validator("base_url")(_strip_trailing_slash)

    @model_validator(mode="after")
    def infer_provider(self) -> PanelMember:
        if self.provider is None:
            self.provider = _infer_provider(self.base_url)
        return self


class JudgeConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    max_panel_tokens: int = Field(default=120_000, ge=1)
    provider: Literal["openai", "anthropic"] | None = None

    _normalize_base_url = field_validator("base_url")(_strip_trailing_slash)

    @model_validator(mode="after")
    def infer_provider(self) -> JudgeConfig:
        if self.provider is None:
            self.provider = _infer_provider(self.base_url)
        return self


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


class PipelineStepUse(StrEnum):
    """How a pipeline step answers the prompt."""

    FUSE = "fuse"    # full panel + judge synthesis
    SOLO = "solo"    # single pass-through model (fastest / cheapest)


class PipelineStepConfig(BaseModel):
    """One step in a sequential pipeline."""

    name: str
    use: PipelineStepUse = PipelineStepUse.SOLO
    # Override the model for SOLO steps (falls back to pass_through model).
    model: str | None = None
    # System prompt for this step. Supports {step_name} placeholders that are
    # replaced with the text output of a previous step at runtime.
    system: str | None = None


class PipelineConfig(BaseModel):
    """Sequential chain of LLM steps; output of each feeds the next."""

    steps: list[PipelineStepConfig] = Field(default_factory=list)


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

    _normalize_base_url = field_validator("base_url")(_strip_trailing_slash)


class ProviderConfig(BaseModel):
    """User-defined provider entry that extends or overrides the built-in registry.

    ``id`` must match a known provider (to override its API key) or be a new id
    (to add a custom/self-hosted provider). ``api_key`` takes precedence over the
    provider's ``env_key`` environment variable.
    """

    id: str
    base_url: str | None = None       # required for new providers; optional for overrides
    api_key: str | None = None        # explicit key; falls back to env_key if absent
    format: Literal["openai", "anthropic"] = "openai"

    _normalize_base_url = field_validator("base_url")(_strip_trailing_slash)


class FallbackEntry(BaseModel):
    """One alternative target in a model fallback chain."""

    base_url: str
    api_key: str
    model: str
    provider: Literal["openai", "anthropic"] | None = None

    _normalize_base_url = field_validator("base_url")(_strip_trailing_slash)

    @model_validator(mode="after")
    def infer_provider(self) -> FallbackEntry:
        if self.provider is None:
            self.provider = _infer_provider(self.base_url)
        return self


class FallbackConfig(BaseModel):
    """Per-model ordered list of fallback targets tried when the primary call fails.

    Key is the model name as used in the request (e.g. ``anthropic/claude-sonnet-4-5``
    or the bare model name). Entries are tried in order; the first available
    (not DOWN) one is used.
    """

    chains: dict[str, list[FallbackEntry]] = Field(default_factory=dict)


class OpenFusionConfig(BaseModel):
    preset: Preset | None = None
    strategy: Strategy = Strategy.SELF_FUSION
    aggregator: Aggregator = Aggregator.JUDGE
    panel: list[PanelMember] = Field(default_factory=list)
    judge: JudgeConfig | None = None
    self_fusion: SelfFusionConfig = Field(default_factory=SelfFusionConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
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
    providers: list[ProviderConfig] = Field(default_factory=list)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
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

    @model_validator(mode="after")
    def resolve_from_registry(self) -> OpenFusionConfig:
        """Expand provider/model shorthand into full endpoint config via registry."""
        from openfusion.registry import ModelRegistry  # avoid circular at module level

        if not self.providers:
            return self

        # Build a per-provider api_key map from ProviderConfig entries
        api_keys: dict[str, str] = {}
        extra_providers: list[dict[str, Any]] = []
        for pc in self.providers:
            if pc.api_key:
                api_keys[pc.id] = pc.api_key
            if pc.base_url:
                extra_providers.append(
                    {"id": pc.id, "base_url": pc.base_url, "format": pc.format}
                )

        registry = ModelRegistry.load(extra_providers=extra_providers or None)

        def _fill(model: str, base_url: str | None, api_key: str) -> tuple[str, str, str] | None:
            """Return (base_url, api_key, bare_model) if registry can resolve, else None."""
            if base_url:
                return None  # already explicit; skip
            if not registry.is_registered(model):
                return None
            resolved = registry.resolve(model, api_keys)
            if resolved is None:
                return None
            return resolved.base_url, resolved.api_key or api_key, resolved.model_id

        # Resolve panel members
        new_panel = []
        for m in self.panel:
            result = _fill(m.model, m.base_url, m.api_key)
            if result:
                base_url, api_key, bare_model = result
                m = m.model_copy(
                    update={"base_url": base_url, "api_key": api_key, "model": bare_model}
                )
            new_panel.append(m)
        self.panel = new_panel

        # Resolve judge
        if self.judge is not None:
            result = _fill(self.judge.model, self.judge.base_url, self.judge.api_key)
            if result:
                base_url, api_key, bare_model = result
                self.judge = self.judge.model_copy(
                    update={"base_url": base_url, "api_key": api_key, "model": bare_model}
                )

        # Resolve pass_through
        if self.pass_through is not None:
            result = _fill(
                self.pass_through.model, self.pass_through.base_url, self.pass_through.api_key
            )
            if result:
                base_url, api_key, bare_model = result
                self.pass_through = self.pass_through.model_copy(
                    update={"base_url": base_url, "api_key": api_key, "model": bare_model}
                )

        return self

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
