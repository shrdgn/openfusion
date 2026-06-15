"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from openfusion.config import OpenFusionConfig, Strategy, load_config


def test_load_example_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    example = Path("openfusion.yaml.example").read_text(encoding="utf-8")
    config_path.write_text(example, encoding="utf-8")

    config = load_config(config_path)

    assert config.strategy == Strategy.SELF_FUSION
    assert config.self_fusion.n == 3
    assert config.panel[0].api_key == "secret-key"
    assert config.judge is not None
    assert config.judge.api_key == "secret-key"


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
