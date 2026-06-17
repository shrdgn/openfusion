"""CLI summary and config-error friendliness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from openfusion.cli import _summarize_config, build_setup_yaml
from openfusion.config import load_config


def test_setup_yaml_loads_into_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(build_setup_yaml("budget", "sk-xyz"), encoding="utf-8")

    config = load_config(config_path)

    assert len(config.panel) == 3
    assert all(member.api_key == "sk-xyz" for member in config.panel)
    assert config.judge is not None and config.judge.api_key == "sk-xyz"
    assert config.tools.web_search is True


def test_summarize_config_reports_preset_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text("preset: budget\n", encoding="utf-8")
    config = load_config(config_path)

    summary = _summarize_config(config, "0.0.0.0", 8000)

    assert "preset=budget" in summary
    assert "web search+fetch" in summary
    assert 'model="openfusion"' in summary
    assert "http://0.0.0.0:8000" in summary


def test_missing_config_file_has_actionable_hint(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="cp examples/preset.yaml.example"):
        load_config(missing)


def test_missing_env_var_hint_includes_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    with pytest.raises(ValueError, match="export OPENROUTER_API_KEY"):
        load_config(config_path)
