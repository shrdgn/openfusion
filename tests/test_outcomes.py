"""Tests for the routing outcome store and learning loop integration."""

from __future__ import annotations

import threading

from openfusion.config import RouterConfig, RouterMode, Tier
from openfusion.outcomes import _ALPHA, _MIN_OBSERVATIONS, OutcomeStore
from openfusion.router import RouteDecision, route

# ---------------------------------------------------------------------------
# OutcomeStore.record / EMA update
# ---------------------------------------------------------------------------


def test_record_first_observation_creates_bucket() -> None:
    store = OutcomeStore()
    store.record("fusion", Tier.FAST, success=True)
    assert store.ema("fusion", Tier.FAST) is None  # not reliable yet


def test_ema_none_before_min_observations() -> None:
    store = OutcomeStore()
    for _ in range(_MIN_OBSERVATIONS - 1):
        store.record("fusion", Tier.FAST, success=True)
    assert store.ema("fusion", Tier.FAST) is None


def test_ema_available_at_min_observations() -> None:
    store = OutcomeStore()
    for _ in range(_MIN_OBSERVATIONS):
        store.record("fusion", Tier.FAST, success=True)
    assert store.ema("fusion", Tier.FAST) is not None


def test_ema_converges_toward_all_success() -> None:
    store = OutcomeStore()
    for _ in range(100):
        store.record("fusion", Tier.FAST, success=True)
    ema = store.ema("fusion", Tier.FAST)
    assert ema is not None
    assert ema > 0.9


def test_ema_converges_toward_all_failure() -> None:
    store = OutcomeStore()
    for _ in range(100):
        store.record("fusion", Tier.FAST, success=False)
    ema = store.ema("fusion", Tier.FAST)
    assert ema is not None
    assert ema < 0.1


def test_ema_update_formula() -> None:
    store = OutcomeStore()
    store.record("r", Tier.FAST, success=True)
    # After one observation starting from 0.5: ema = alpha*1 + (1-alpha)*0.5
    expected = _ALPHA * 1.0 + (1 - _ALPHA) * 0.5
    with store._lock:
        bucket = store._buckets[("r", str(Tier.FAST))]
    assert abs(bucket.ema - expected) < 1e-9


def test_buckets_are_independent_per_route_and_tier() -> None:
    store = OutcomeStore()
    for _ in range(_MIN_OBSERVATIONS):
        store.record("fusion", Tier.FAST, success=True)
    for _ in range(_MIN_OBSERVATIONS):
        store.record("pass_through", Tier.FAST, success=False)

    fuse_ema = store.ema("fusion", Tier.FAST)
    solo_ema = store.ema("pass_through", Tier.FAST)
    assert fuse_ema is not None
    assert solo_ema is not None
    assert fuse_ema > solo_ema


# ---------------------------------------------------------------------------
# OutcomeStore.prefer_fuse
# ---------------------------------------------------------------------------


def test_prefer_fuse_none_when_insufficient_data() -> None:
    store = OutcomeStore()
    assert store.prefer_fuse(Tier.FAST) is None


def test_prefer_fuse_none_when_only_one_side_has_data() -> None:
    store = OutcomeStore()
    for _ in range(_MIN_OBSERVATIONS):
        store.record("fusion", Tier.FAST, success=True)
    assert store.prefer_fuse(Tier.FAST) is None


def test_prefer_fuse_true_when_fuse_dominates() -> None:
    store = OutcomeStore()
    for _ in range(200):
        store.record("fusion", Tier.FAST, success=True)
    for _ in range(200):
        store.record("pass_through", Tier.FAST, success=False)
    assert store.prefer_fuse(Tier.FAST) is True


def test_prefer_fuse_false_when_solo_dominates() -> None:
    store = OutcomeStore()
    for _ in range(200):
        store.record("fusion", Tier.FAST, success=False)
    for _ in range(200):
        store.record("pass_through", Tier.FAST, success=True)
    assert store.prefer_fuse(Tier.FAST) is False


def test_prefer_fuse_none_when_too_close() -> None:
    store = OutcomeStore()
    # Both sides identical — difference is 0, below _NUDGE_THRESHOLD
    for _ in range(100):
        store.record("fusion", Tier.FAST, success=True)
        store.record("pass_through", Tier.FAST, success=True)
    assert store.prefer_fuse(Tier.FAST) is None


def test_prefer_fuse_respects_threshold() -> None:
    store = OutcomeStore()
    # Drive fusion EMA high, solo EMA just below threshold
    for _ in range(200):
        store.record("fusion", Tier.BALANCED, success=True)
    for _ in range(200):
        store.record("pass_through", Tier.BALANCED, success=True)
    # Both converge to ~1.0 so difference is near 0 — should be None
    assert store.prefer_fuse(Tier.BALANCED) is None


# ---------------------------------------------------------------------------
# OutcomeStore.snapshot
# ---------------------------------------------------------------------------


def test_snapshot_empty() -> None:
    store = OutcomeStore()
    assert store.snapshot() == {}


def test_snapshot_contains_recorded_buckets() -> None:
    store = OutcomeStore()
    store.record("fusion", Tier.FAST, success=True)
    snap = store.snapshot()
    assert f"fusion/{Tier.FAST}" in snap
    entry = snap[f"fusion/{Tier.FAST}"]
    assert "ema" in entry
    assert "count" in entry
    assert entry["count"] == 1


def test_snapshot_rounds_ema() -> None:
    store = OutcomeStore()
    store.record("fusion", Tier.FAST, success=True)
    snap = store.snapshot()
    ema_val = snap[f"fusion/{Tier.FAST}"]["ema"]
    # Should be rounded to 4 decimal places
    assert ema_val == round(ema_val, 4)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_records_are_safe() -> None:
    store = OutcomeStore()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(50):
                store.record("fusion", Tier.FAST, success=True)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    with store._lock:
        bucket = store._buckets[("fusion", str(Tier.FAST))]
    assert bucket.count == 500


# ---------------------------------------------------------------------------
# router.route() integration with OutcomeStore
# ---------------------------------------------------------------------------


def _auto_config() -> RouterConfig:
    return RouterConfig(mode=RouterMode.HEURISTIC)


def test_route_solo_without_outcomes() -> None:
    config = _auto_config()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert route(body, config) == RouteDecision.SOLO


def test_route_outcomes_nudge_solo_to_fuse() -> None:
    store = OutcomeStore()
    # Simulate many successful fusions, many failed solo
    for _ in range(200):
        store.record("fusion", Tier.FAST, success=True)
    for _ in range(200):
        store.record("pass_through", Tier.FAST, success=False)

    config = _auto_config()
    body = {"messages": [{"role": "user", "content": "hi"}]}  # short → FAST tier, heuristic = SOLO
    assert route(body, config, outcomes=store) == RouteDecision.FUSE


def test_route_outcomes_do_not_nudge_when_insufficient_data() -> None:
    store = OutcomeStore()
    config = _auto_config()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert route(body, config, outcomes=store) == RouteDecision.SOLO


def test_route_strong_signal_ignores_outcomes() -> None:
    """Code blocks always fuse regardless of what the outcome store says."""
    store = OutcomeStore()
    for _ in range(200):
        store.record("pass_through", Tier.STRONG, success=True)
    for _ in range(200):
        store.record("fusion", Tier.STRONG, success=False)

    config = _auto_config()
    body = {"messages": [{"role": "user", "content": "```python\nprint('hi')\n```"}]}
    # Strong signal (code block) → FUSE regardless of outcome store
    assert route(body, config, outcomes=store) == RouteDecision.FUSE


def test_route_always_mode_ignores_outcomes() -> None:
    store = OutcomeStore()
    config = RouterConfig(mode=RouterMode.ALWAYS)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert route(body, config, outcomes=store) == RouteDecision.FUSE


def test_route_never_mode_ignores_outcomes() -> None:
    store = OutcomeStore()
    config = RouterConfig(mode=RouterMode.NEVER)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert route(body, config, outcomes=store) == RouteDecision.SOLO
