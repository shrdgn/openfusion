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


class Strategy(StrEnum):
    PANEL = "panel"
    SELF_FUSION = "self_fusion"


class Aggregator(StrEnum):
    """How panel responses are combined into the final answer."""

    JUDGE = "judge"  # a judge model synthesizes one answer
    VOTE = "vote"  # majority vote over panel answers (self-consistency)


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

    Leans on the upstream's server-side web plugin (e.g. OpenRouter's Exa-backed
    `web` plugin), which executes search upstream and returns a normal content
    answer — so no client-side tool-call loop is needed.
    """

    web_search: bool = False
    max_results: int = Field(default=5, ge=1, le=20)
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
    strategy: Strategy = Strategy.SELF_FUSION
    aggregator: Aggregator = Aggregator.JUDGE
    panel: list[PanelMember] = Field(default_factory=list)
    judge: JudgeConfig | None = None
    self_fusion: SelfFusionConfig = Field(default_factory=SelfFusionConfig)
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    gateway: GatewayAuthConfig = Field(default_factory=GatewayAuthConfig)
    cost_controls: CostControlsConfig = Field(default_factory=CostControlsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    pass_through: PassThroughConfig | None = None
    fusion_model_name: str = "openfusion"

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


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replacer(match: re.Match[str]) -> str:
            env_name = match.group(1)
            env_value = os.environ.get(env_name)
            if env_value is None:
                raise ValueError(f"Environment variable {env_name} is not set")
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


def load_config(path: str | Path | None = None) -> OpenFusionConfig:
    """Load configuration from YAML and environment variables."""
    config_path = Path(path or os.environ.get("OPENFUSION_CONFIG", "openfusion.yaml"))
    if not config_path.exists():
        example = Path("openfusion.yaml.example")
        if example.exists() and path is None:
            config_path = example
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    raw["gateway"] = {"api_keys": _load_gateway_keys(raw)}
    expanded = _expand_env(raw)
    return OpenFusionConfig.model_validate(expanded)
