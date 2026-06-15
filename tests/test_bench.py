"""Benchmark tests (offline logic only)."""

from __future__ import annotations

from bench.run import Task, _normalize, _score


def test_exact_match_scoring() -> None:
    task = Task(id="1", prompt="p", expected="45", match="exact")
    assert _score(task, "45")
    assert not _score(task, "46")


def test_contains_match_scoring() -> None:
    task = Task(id="2", prompt="p", expected="carbon dioxide", match="contains")
    assert _score(task, "Plants absorb carbon dioxide from the air.")
    assert not _score(task, "oxygen")


def test_normalize_strips_case_and_whitespace() -> None:
    assert _normalize("  YES \n") == "yes"
