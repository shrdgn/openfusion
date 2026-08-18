"""Tests for the pairwise research eval logic (offline)."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from bench import research_eval
from bench.research_eval import ResearchTask, fusion_is_a, wilson_interval, winner_from_verdict
from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    Strategy,
    TimeoutsConfig,
)


def test_wilson_interval_basic() -> None:
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(6, 8)
    assert 0.0 <= lo < 0.75 < hi <= 1.0
    # A unanimous small sample still has a wide interval (not [1,1]).
    lo2, hi2 = wilson_interval(8, 8)
    assert lo2 < 1.0 and hi2 == 1.0


def test_fusion_is_a_is_deterministic() -> None:
    assert fusion_is_a("r1") == fusion_is_a("r1")
    # Mixed assignment across ids (not all the same slot).
    assignments = {fusion_is_a(f"r{i}") for i in range(10)}
    assert assignments == {True, False}


def test_winner_when_fusion_is_a() -> None:
    assert winner_from_verdict("A", fusion_was_a=True) == "fusion"
    assert winner_from_verdict("B", fusion_was_a=True) == "solo"
    assert winner_from_verdict("TIE", fusion_was_a=True) == "tie"


def test_winner_when_fusion_is_b() -> None:
    assert winner_from_verdict("A", fusion_was_a=False) == "solo"
    assert winner_from_verdict("B", fusion_was_a=False) == "fusion"


def test_winner_tolerates_verbose_verdicts() -> None:
    assert winner_from_verdict("A is better", fusion_was_a=True) == "fusion"
    assert winner_from_verdict("Answer B", fusion_was_a=False) == "fusion"
    assert winner_from_verdict("It's a tie", fusion_was_a=True) == "tie"


def test_winner_unparsed_when_no_signal() -> None:
    # Empty or garbled judge replies must surface as 'unparsed', not silent ties.
    assert winner_from_verdict("???", fusion_was_a=True) == "unparsed"
    assert winner_from_verdict("", fusion_was_a=True) == "unparsed"


def test_explicit_tie_is_tie_not_unparsed() -> None:
    assert winner_from_verdict("TIE", fusion_was_a=True) == "tie"


def _config(member_seconds: float, judge_seconds: float) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.JUDGE,
        panel=[PanelMember(base_url="https://example.com/v1", api_key="k", model="m1")],
        judge=JudgeConfig(base_url="https://example.com/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(base_url="https://example.com/v1", api_key="k", model="p"),
        timeouts=TimeoutsConfig(
            member_seconds=member_seconds,
            judge_seconds=judge_seconds,
            total_seconds=member_seconds + judge_seconds,
        ),
    )


def _args(**overrides: Any) -> argparse.Namespace:
    defaults = dict(
        config="unused.yaml",
        tasks="unused.jsonl",
        base_url="http://127.0.0.1:8000/v1",
        solo_model=None,
        judge_model=None,
        max_tokens=512,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_eval_sizes_request_timeout_from_configured_member_and_judge_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config that widens member/judge budgets — like panel-strongsynth.yaml.example
    and panel-tools.yaml.example's 180/180 (worst case 360s) — must not have its bench
    HTTP calls killed by bench.run._chat's shared 360s default timeout. The research
    eval derives its own request timeout from the config under test instead."""
    config = _config(member_seconds=180.0, judge_seconds=180.0)
    monkeypatch.setattr(research_eval, "load_config", lambda _path: config)
    monkeypatch.setattr(
        research_eval,
        "_load_tasks",
        lambda _path: [ResearchTask(id="r1", prompt="what is the CAGR?")],
    )

    seen_timeouts: list[float] = []

    def fake_chat(*_args: Any, **kwargs: Any) -> tuple[str, float, dict]:
        seen_timeouts.append(kwargs["timeout"])
        return "A", 0.1, {}

    monkeypatch.setattr(research_eval, "_chat", fake_chat)

    result = research_eval.run_eval(_args())

    # member_seconds (180) + judge_seconds (180) + the eval's own margin.
    assert seen_timeouts and all(t == pytest.approx(390.0) for t in seen_timeouts)
    assert len(seen_timeouts) == 3  # solo + fusion + judge, per task
    assert result["summary"]["n"] == 1
