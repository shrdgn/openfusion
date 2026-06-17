"""Per-request override merging."""

from __future__ import annotations

from openfusion.config import JudgeConfig, OpenFusionConfig, PanelMember, Strategy
from openfusion.overrides import MAX_OVERRIDE_PANEL, apply_overrides


def _base() -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        panel=[PanelMember(base_url="https://up/v1", api_key="k", model="base", label="b")],
        judge=JudgeConfig(base_url="https://up/v1", api_key="k", model="jbase"),
    )


def test_empty_override_is_noop() -> None:
    base = _base()
    assert apply_overrides(base, {}) is base


def test_panel_override_reuses_credentials() -> None:
    cfg = apply_overrides(_base(), {"panel": ["a", "b"]})
    assert cfg.strategy == Strategy.PANEL
    assert [m.model for m in cfg.panel] == ["a", "b"]
    assert all(m.api_key == "k" and m.base_url == "https://up/v1" for m in cfg.panel)


def test_judge_override() -> None:
    cfg = apply_overrides(_base(), {"judge": "newjudge"})
    assert cfg.judge is not None
    assert cfg.judge.model == "newjudge"
    assert cfg.judge.api_key == "k"


def test_preset_override_sets_panel_and_tools() -> None:
    cfg = apply_overrides(_base(), {"preset": "budget"})
    assert cfg.strategy == Strategy.PANEL
    assert len(cfg.panel) == 3
    assert cfg.tools.web_search is True


def test_tools_override() -> None:
    cfg = apply_overrides(_base(), {"tools": {"web_search": True}})
    assert cfg.tools.web_search is True


def test_panel_is_capped() -> None:
    cfg = apply_overrides(_base(), {"panel": [str(i) for i in range(20)]})
    assert len(cfg.panel) == MAX_OVERRIDE_PANEL


def test_base_config_not_mutated() -> None:
    base = _base()
    apply_overrides(base, {"panel": ["a", "b"], "judge": "x"})
    assert [m.model for m in base.panel] == ["base"]
    assert base.judge.model == "jbase"
