"""Debate strategy benchmark tests (offline logic only).

These tests verify the bench framework handles debate-mode responses correctly:
scoring, summarization, and per-mode result aggregation.
"""

from __future__ import annotations

from bench.run import Task, TaskResult, _score

# ---------------------------------------------------------------------------
# Scoring helpers shared with other bench tests
# ---------------------------------------------------------------------------


def test_exact_match_accepts_correct_debate_answer() -> None:
    task = Task(id="d1", prompt="p", expected="Paris", match="exact")
    assert _score(task, "Paris")
    assert not _score(task, "London")


def test_contains_match_for_open_ended_debate_response() -> None:
    task = Task(id="d2", prompt="p", expected="photosynthesis", match="contains")
    assert _score(task, "Plants use photosynthesis to convert light into energy.")
    assert not _score(task, "Plants breathe oxygen.")


def test_numeric_match_for_debate_math_task() -> None:
    task = Task(id="d3", prompt="p", expected="42", match="numeric")
    assert _score(task, "After debate, the answer is 42.")
    assert _score(task, "The final answer: $42.00")
    assert not _score(task, "The answer is 43.")


# ---------------------------------------------------------------------------
# Result aggregation mirrors what run_benchmark.summarize() produces;
# verify the debate mode would be counted correctly if wired into the runner.
# ---------------------------------------------------------------------------


def _make_debate_results(
    correct: list[bool],
    latencies: list[float],
    tokens: list[int],
) -> list[TaskResult]:
    return [
        TaskResult(
            task_id=f"d{i}",
            mode="debate",
            correct=ok,
            answer="some answer",
            latency_seconds=lat,
            total_tokens=tok,
            cost_usd=0.0,
        )
        for i, (ok, lat, tok) in enumerate(zip(correct, latencies, tokens, strict=True), 1)
    ]


def _summarize(results: list[TaskResult]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    avg_latency = sum(r.latency_seconds for r in results) / max(total, 1)
    total_tokens = sum(r.total_tokens for r in results)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "avg_latency_seconds": avg_latency,
        "total_tokens": total_tokens,
    }


def test_debate_all_correct() -> None:
    results = _make_debate_results(
        correct=[True, True, True],
        latencies=[1.0, 1.2, 0.9],
        tokens=[100, 120, 110],
    )
    summary = _summarize(results)
    assert summary["correct"] == 3
    assert summary["total"] == 3
    assert summary["accuracy"] == 1.0
    assert abs(summary["avg_latency_seconds"] - (1.0 + 1.2 + 0.9) / 3) < 1e-9
    assert summary["total_tokens"] == 330


def test_debate_partial_correct() -> None:
    results = _make_debate_results(
        correct=[True, False, True],
        latencies=[1.0, 2.0, 1.5],
        tokens=[100, 200, 150],
    )
    summary = _summarize(results)
    assert summary["correct"] == 2
    assert summary["total"] == 3
    assert abs(summary["accuracy"] - 2 / 3) < 1e-9


def test_debate_no_correct() -> None:
    results = _make_debate_results(
        correct=[False, False],
        latencies=[1.0, 1.0],
        tokens=[50, 50],
    )
    summary = _summarize(results)
    assert summary["correct"] == 0
    assert summary["accuracy"] == 0.0


def test_debate_outperforms_solo_comparison() -> None:
    """Demonstrate the comparison pattern used in run_benchmark."""
    solo = _make_debate_results(
        correct=[True, False, False, True],
        latencies=[0.5, 0.5, 0.5, 0.5],
        tokens=[50] * 4,
    )
    debate = _make_debate_results(
        correct=[True, True, False, True],
        latencies=[2.0, 2.0, 2.0, 2.0],
        tokens=[200] * 4,
    )
    solo_summary = _summarize(solo)
    debate_summary = _summarize(debate)

    assert debate_summary["accuracy"] > solo_summary["accuracy"]
    assert debate_summary["avg_latency_seconds"] > solo_summary["avg_latency_seconds"]
    assert debate_summary["total_tokens"] > solo_summary["total_tokens"]
