"""Per-request override merging."""

from __future__ import annotations

from openfusion.config import (
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    Strategy,
)
from openfusion.overrides import MAX_OVERRIDE_PANEL, apply_overrides, set_all_keys


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


def test_max_tokens_override_caps_cost_controls() -> None:
    cfg = apply_overrides(_base(), {"max_tokens": 256})
    assert cfg.cost_controls.panel_max_tokens == 256
    assert cfg.cost_controls.judge_max_tokens == 256
    assert cfg.cost_controls.pass_through_max_tokens == 256


def test_max_tokens_override_is_clamped() -> None:
    cfg = apply_overrides(_base(), {"max_tokens": 999_999})
    assert cfg.cost_controls.judge_max_tokens == 8192


def test_base_config_not_mutated() -> None:
    base = _base()
    apply_overrides(base, {"panel": ["a", "b"], "judge": "x"})
    assert [m.model for m in base.panel] == ["base"]
    assert base.judge.model == "jbase"


def test_set_all_keys_replaces_every_key() -> None:
    base = OpenFusionConfig(
        panel=[
            PanelMember(base_url="https://up/v1", api_key="old-panel", model="m1"),
            PanelMember(base_url="https://up/v1", api_key="old-panel", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://up/v1", api_key="old-judge", model="j"),
        pass_through=PassThroughConfig(
            base_url="https://up/v1", api_key="old-pass", model="p"
        ),
    )
    cfg = set_all_keys(base, "new-key")
    assert all(m.api_key == "new-key" for m in cfg.panel)
    assert cfg.judge is not None and cfg.judge.api_key == "new-key"
    assert cfg.pass_through is not None and cfg.pass_through.api_key == "new-key"


def test_set_all_keys_handles_no_judge_or_pass_through() -> None:
    base = OpenFusionConfig(
        panel=[PanelMember(base_url="https://up/v1", api_key="old", model="m")],
    )
    cfg = set_all_keys(base, "new-key")
    assert cfg.panel[0].api_key == "new-key"
    assert cfg.judge is None
    assert cfg.pass_through is None


def test_set_all_keys_does_not_mutate_base() -> None:
    base = _base()
    set_all_keys(base, "new-key")
    assert base.panel[0].api_key == "k"
    assert base.judge is not None and base.judge.api_key == "k"
