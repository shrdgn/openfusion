"""Offline tests for bench/draco_eval.py's HTTP request timeout sizing."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from bench import draco_eval
from bench.draco import Criterion, DracoTask
from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    Strategy,
    TimeoutsConfig,
)


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
        limit=1,
        base_url="http://127.0.0.1:8000/v1",
        solo_model=None,
        judge_model=None,
        answer_tokens=2048,
        grading_runs=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_eval_sizes_request_timeout_from_configured_member_and_judge_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config that widens member/judge budgets (like draco.yaml.example's 240/180 for
    slow agentic tool loops) must not have its bench HTTP calls killed by a smaller
    hardcoded client timeout — the DRACO harness derives its own timeout from the
    same config it's testing instead of assuming bench.run._chat's 360s default."""
    config = _config(member_seconds=240.0, judge_seconds=180.0)
    monkeypatch.setattr(draco_eval, "load_config", lambda _path: config)
    monkeypatch.setattr(
        draco_eval,
        "load_draco",
        lambda _limit: [
            DracoTask(
                id="t1",
                domain="finance",
                problem="what is the CAGR?",
                criteria=[Criterion(text="States the correct CAGR", weight=10.0)],
            )
        ],
    )

    seen_timeouts: list[float] = []

    def fake_chat_with_retry(*_args: Any, **kwargs: Any) -> tuple[str, float, dict]:
        seen_timeouts.append(kwargs["timeout"])
        return "the answer", 0.1, {}

    monkeypatch.setattr(draco_eval, "_chat_with_retry", fake_chat_with_retry)

    result = draco_eval.run_eval(_args())

    # member_seconds (240) + judge_seconds (180) + the harness's own margin.
    assert seen_timeouts and all(t == pytest.approx(450.0) for t in seen_timeouts)
    assert result["summary"]["n"] == 1
