"""Unit tests for CostPolicy (cost.py)."""

from __future__ import annotations

import pytest

from openfusion.config import CostControlsConfig
from openfusion.cost import CostPolicy, RequestPhase
from openfusion.errors import InvalidRequestError


def _policy(**kwargs: int | None) -> CostPolicy:
    return CostPolicy(CostControlsConfig(**kwargs))


# ---------------------------------------------------------------------------
# validate_max_tokens
# ---------------------------------------------------------------------------


def test_validate_max_tokens_none_is_ok() -> None:
    _policy().validate_max_tokens({})  # no max_tokens key → no raise


def test_validate_max_tokens_valid_int() -> None:
    _policy().validate_max_tokens({"max_tokens": 100})  # should not raise


def test_validate_max_tokens_bool_raises() -> None:
    with pytest.raises(InvalidRequestError, match="must be an integer"):
        _policy().validate_max_tokens({"max_tokens": True})


def test_validate_max_tokens_zero_raises() -> None:
    with pytest.raises(InvalidRequestError, match="greater than 0"):
        _policy().validate_max_tokens({"max_tokens": 0})


def test_validate_max_tokens_negative_raises() -> None:
    with pytest.raises(InvalidRequestError, match="greater than 0"):
        _policy().validate_max_tokens({"max_tokens": -5})


# ---------------------------------------------------------------------------
# apply_token_limit — no limit configured
# ---------------------------------------------------------------------------


def test_apply_no_limit_returns_deep_copy() -> None:
    policy = _policy()
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 500}
    result = policy.apply_token_limit(body, RequestPhase.PANEL, reject_over_limit=False)
    assert result == body
    assert result is not body  # deep copy, not the same object


def test_apply_no_limit_injects_nothing() -> None:
    policy = _policy()
    result = policy.apply_token_limit({}, RequestPhase.JUDGE, reject_over_limit=False)
    assert "max_tokens" not in result


# ---------------------------------------------------------------------------
# apply_token_limit — limit set, various max_tokens combinations
# ---------------------------------------------------------------------------


def test_apply_injects_limit_when_max_tokens_absent() -> None:
    policy = _policy(panel_max_tokens=256)
    result = policy.apply_token_limit({}, RequestPhase.PANEL, reject_over_limit=False)
    assert result["max_tokens"] == 256


def test_apply_leaves_max_tokens_when_within_limit() -> None:
    policy = _policy(panel_max_tokens=500)
    result = policy.apply_token_limit(
        {"max_tokens": 100}, RequestPhase.PANEL, reject_over_limit=False
    )
    assert result["max_tokens"] == 100


def test_apply_equal_to_limit_is_allowed() -> None:
    policy = _policy(panel_max_tokens=100)
    result = policy.apply_token_limit(
        {"max_tokens": 100}, RequestPhase.PANEL, reject_over_limit=False
    )
    assert result["max_tokens"] == 100


def test_apply_caps_over_limit_when_not_rejecting() -> None:
    policy = _policy(panel_max_tokens=100)
    result = policy.apply_token_limit(
        {"max_tokens": 500}, RequestPhase.PANEL, reject_over_limit=False
    )
    assert result["max_tokens"] == 100


def test_apply_raises_when_reject_over_limit() -> None:
    policy = _policy(panel_max_tokens=100)
    with pytest.raises(InvalidRequestError, match="exceeds"):
        policy.apply_token_limit({"max_tokens": 500}, RequestPhase.PANEL, reject_over_limit=True)


# ---------------------------------------------------------------------------
# apply_token_limit — all three phases route to the correct limit
# ---------------------------------------------------------------------------


def test_apply_pass_through_phase() -> None:
    policy = _policy(pass_through_max_tokens=50)
    result = policy.apply_token_limit({}, RequestPhase.PASS_THROUGH, reject_over_limit=False)
    assert result["max_tokens"] == 50


def test_apply_panel_phase_ignores_other_limits() -> None:
    # Only panel_max_tokens should apply for the PANEL phase
    policy = _policy(pass_through_max_tokens=10, panel_max_tokens=200, judge_max_tokens=30)
    result = policy.apply_token_limit({}, RequestPhase.PANEL, reject_over_limit=False)
    assert result["max_tokens"] == 200


def test_apply_judge_phase() -> None:
    policy = _policy(judge_max_tokens=200)
    result = policy.apply_token_limit({}, RequestPhase.JUDGE, reject_over_limit=False)
    assert result["max_tokens"] == 200


# ---------------------------------------------------------------------------
# apply_token_limit — original body is not mutated
# ---------------------------------------------------------------------------


def test_apply_does_not_mutate_original_body() -> None:
    policy = _policy(panel_max_tokens=100)
    body: dict = {"max_tokens": 500, "temperature": 0.7}
    _ = policy.apply_token_limit(body, RequestPhase.PANEL, reject_over_limit=False)
    assert body["max_tokens"] == 500  # deep-copied; original unchanged


# ---------------------------------------------------------------------------
# _limit_for — defensive branch for an unrecognized phase
# ---------------------------------------------------------------------------


def test_limit_for_unknown_phase_raises() -> None:
    # RequestPhase is a closed StrEnum, so this is only reachable if a caller
    # passes something other than one of its three members.
    policy = _policy()
    with pytest.raises(ValueError, match="Unknown request phase"):
        policy._limit_for("bogus")
