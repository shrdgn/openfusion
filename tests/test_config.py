"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    Strategy,
    load_config,
)


def test_load_example_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    example = Path("examples/default.yaml.example").read_text(encoding="utf-8")
    config_path.write_text(example, encoding="utf-8")

    config = load_config(config_path)

    assert config.strategy == Strategy.SELF_FUSION
    assert config.self_fusion.n == 3
    assert config.panel[0].api_key == "secret-key"
    assert config.judge is not None
    assert config.judge.api_key == "secret-key"
    assert config.cost_controls.pass_through_max_tokens == 1024
    assert config.cost_controls.panel_max_tokens == 512
    assert config.cost_controls.judge_max_tokens == 1024


def test_load_dev_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.dev.yaml"
    example = Path("examples/dev.yaml.example").read_text(encoding="utf-8")
    config_path.write_text(example, encoding="utf-8")

    config = load_config(config_path)

    assert config.self_fusion.n == 2
    assert config.panel[0].model == "openai/gpt-4o-mini"
    assert config.cost_controls.pass_through_max_tokens == 80


def test_gateway_keys_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setenv("OPENFUSION_API_KEYS", "alpha,beta")
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        """
strategy: panel
panel:
  - base_url: https://example.com/v1
    api_key: ${OPENROUTER_API_KEY}
    model: test
judge:
  base_url: https://example.com/v1
  api_key: ${OPENROUTER_API_KEY}
  model: judge
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert "alpha" in config.gateway.api_keys
    assert "beta" in config.gateway.api_keys


def test_missing_env_var_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        """
panel:
  - base_url: https://example.com/v1
    api_key: ${OPENROUTER_API_KEY}
    model: test
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        load_config(config_path)


def test_budget_preset_expands_panel_judge_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text("preset: budget\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.strategy == Strategy.PANEL
    assert config.aggregator == Aggregator.JUDGE
    assert len(config.panel) == 3
    assert all(member.api_key == "secret-key" for member in config.panel)
    assert config.judge is not None
    assert config.tools.web_search is True
    assert config.tools.web_fetch is True
    assert config.resolved_pass_through().model == "openai/gpt-4o-mini"


def test_explicit_config_overrides_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        """
preset: quality
tools:
  web_search: false
panel:
  - base_url: https://example.com/v1
    api_key: ${OPENROUTER_API_KEY}
    model: my-own-model
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    # User-set values win; preset only fills the gaps (judge here).
    assert len(config.panel) == 1
    assert config.panel[0].model == "my-own-model"
    assert config.tools.web_search is False
    assert config.judge is not None
    assert config.judge.model == "anthropic/claude-sonnet-4"


def test_resolved_pass_through_from_panel() -> None:
    config = OpenFusionConfig.model_validate(
        {
            "panel": [
                {
                    "base_url": "https://example.com/v1",
                    "api_key": "key",
                    "model": "solo",
                }
            ]
        }
    )
    resolved = config.resolved_pass_through()
    assert resolved.model == "solo"


def test_base_url_trailing_slash_is_stripped_everywhere() -> None:
    """PanelMember, JudgeConfig, and PassThroughConfig share one strip_trailing_slash
    validator; a trailing slash would otherwise produce a double slash against
    upstream paths like f"{base_url}/chat/completions".
    """
    panel = PanelMember(base_url="https://example.com/v1/", api_key="k", model="m")
    judge = JudgeConfig(base_url="https://example.com/v1/", api_key="k", model="m")
    pass_through = PassThroughConfig(base_url="https://example.com/v1/", api_key="k", model="m")

    assert panel.base_url == "https://example.com/v1"
    assert judge.base_url == "https://example.com/v1"
    assert pass_through.base_url == "https://example.com/v1"
