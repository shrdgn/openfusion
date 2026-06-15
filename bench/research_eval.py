#!/usr/bin/env python3
"""Open-ended deep-research proof: blind pairwise LLM-judge of solo vs fusion.

Deep-research tasks are open-ended, so exact-match scoring does not apply.
Instead, for each question we get a solo answer and a fusion answer, then ask a
judge model which is better in a blind, position-randomized A/B comparison.
This isolates whether the fusion synthesis step lifts open-ended answer quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from bench.run import _chat
from openfusion.config import load_config

GRADE_PROMPT = (
    "You are a strict, impartial evaluator comparing two answers to a technical "
    "question. Judge on factual accuracy, completeness, sound reasoning, and "
    "absence of errors. Do NOT reward length or verbosity; a confident wrong or "
    "padded answer should lose.\n\n"
    "Question:\n{question}\n\n"
    "Answer A:\n{answer_a}\n\n"
    "Answer B:\n{answer_b}\n\n"
    "Which answer is better overall? Pick a winner; only answer TIE if the two "
    "are genuinely indistinguishable in quality. Reply with exactly one token: "
    "A, B, or TIE."
)


@dataclass
class ResearchTask:
    id: str
    prompt: str


@dataclass
class PairResult:
    task_id: str
    winner: str  # "fusion" | "solo" | "tie"
    fusion_was_a: bool
    verdict: str
    solo_cost_usd: float
    fusion_cost_usd: float
    solo_tokens: int
    fusion_tokens: int


def _load_tasks(path: Path) -> list[ResearchTask]:
    tasks: list[ResearchTask] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        tasks.append(ResearchTask(id=str(payload["id"]), prompt=str(payload["prompt"])))
    return tasks


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion (no SciPy needed)."""
    if n == 0:
        return None
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def fusion_is_a(task_id: str) -> bool:
    """Deterministically assign fusion to slot A or B to cancel position bias."""
    digest = hashlib.md5(task_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 2 == 0


def winner_from_verdict(verdict: str, fusion_was_a: bool) -> str:
    """Map a raw A/B/TIE verdict to 'fusion' | 'solo' | 'tie' | 'unparsed'.

    Uses word-boundary matching so verbose verdicts like "Answer B" are not
    misread (the 'A' in 'Answer' must not count). An explicit TIE returns
    'tie'; a verdict with no A/B/TIE signal at all (e.g. an empty or garbled
    judge reply) returns 'unparsed' so it is surfaced rather than silently
    counted as a tie. If both A and B appear, the earliest letter wins.
    """
    text = verdict.strip().upper()
    if re.search(r"\bTIE\b", text):
        return "tie"
    match_a = re.search(r"\bA\b", text)
    match_b = re.search(r"\bB\b", text)
    if match_a and match_b:
        choice = "A" if match_a.start() < match_b.start() else "B"
    elif match_a:
        choice = "A"
    elif match_b:
        choice = "B"
    else:
        return "unparsed"
    if choice == "A":
        return "fusion" if fusion_was_a else "solo"
    return "solo" if fusion_was_a else "fusion"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    tasks = _load_tasks(Path(args.tasks))
    solo_model = args.solo_model or config.panel[0].model
    judge_model = args.judge_model or config.resolved_pass_through().model
    gateway_key = config.gateway.api_keys[0] if config.gateway.api_keys else "bench-key"

    results: list[PairResult] = []
    with httpx.Client() as client:
        for task in tasks:
            solo_answer, _, solo_usage = _chat(
                client,
                base_url=args.base_url,
                model=solo_model,
                prompt=task.prompt,
                api_key=gateway_key,
                max_tokens=args.max_tokens,
            )
            fusion_answer, _, fusion_usage = _chat(
                client,
                base_url=args.base_url,
                model=config.fusion_model_name,
                prompt=task.prompt,
                api_key=gateway_key,
                max_tokens=args.max_tokens,
            )

            fa = fusion_is_a(task.id)
            answer_a, answer_b = (
                (fusion_answer, solo_answer) if fa else (solo_answer, fusion_answer)
            )
            # Give the judge room to emit a token (thinking models need more
            # than a few), and a small positive temperature since some models
            # (e.g. Gemini) degrade at temperature 0.
            verdict, _, _ = _chat(
                client,
                base_url=args.base_url,
                model=judge_model,
                prompt=GRADE_PROMPT.format(
                    question=task.prompt, answer_a=answer_a, answer_b=answer_b
                ),
                api_key=gateway_key,
                max_tokens=16,
                temperature=0.2,
            )
            results.append(
                PairResult(
                    task_id=task.id,
                    winner=winner_from_verdict(verdict, fa),
                    fusion_was_a=fa,
                    verdict=verdict.strip(),
                    solo_cost_usd=float(solo_usage.get("cost", 0.0) or 0.0),
                    fusion_cost_usd=float(fusion_usage.get("cost", 0.0) or 0.0),
                    solo_tokens=int(solo_usage.get("total_tokens", 0) or 0),
                    fusion_tokens=int(fusion_usage.get("total_tokens", 0) or 0),
                )
            )

    return summarize(args, solo_model, judge_model, config.fusion_model_name, results)


def summarize(
    args: argparse.Namespace,
    solo_model: str,
    judge_model: str,
    fusion_model: str,
    results: list[PairResult],
) -> dict[str, Any]:
    fusion_wins = sum(1 for r in results if r.winner == "fusion")
    solo_wins = sum(1 for r in results if r.winner == "solo")
    ties = sum(1 for r in results if r.winner == "tie")
    unparsed = sum(1 for r in results if r.winner == "unparsed")
    decided = fusion_wins + solo_wins
    ci = wilson_interval(fusion_wins, decided)
    return {
        "eval": "research_pairwise",
        "tasks_file": str(args.tasks),
        "config": str(args.config),
        "solo_model": solo_model,
        "fusion_model": fusion_model,
        "judge_model": judge_model,
        "max_tokens": args.max_tokens,
        "summary": {
            "n": len(results),
            "fusion_wins": fusion_wins,
            "solo_wins": solo_wins,
            "ties": ties,
            "unparsed": unparsed,
            "decided": decided,
            "fusion_win_rate_decided": (fusion_wins / decided) if decided else None,
            "fusion_win_rate_ci95": list(ci) if ci else None,
            "solo": {
                "total_tokens": sum(r.solo_tokens for r in results),
                "total_cost_usd": sum(r.solo_cost_usd for r in results),
            },
            "fusion": {
                "total_tokens": sum(r.fusion_tokens for r in results),
                "total_cost_usd": sum(r.fusion_cost_usd for r in results),
            },
        },
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise solo-vs-fusion research eval")
    parser.add_argument("--config", default="openfusion.bench.yaml.example")
    parser.add_argument("--tasks", default="bench/tasks/research.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--solo-model", default=None)
    parser.add_argument("--judge-model", default=None, help="Defaults to the pass-through model.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    report = run_eval(args)
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    s = report["summary"]
    rate = s["fusion_win_rate_decided"]
    rate_str = f"{rate * 100:.0f}%" if rate is not None else "n/a"
    ci = s.get("fusion_win_rate_ci95")
    ci_str = f" [95% CI {ci[0] * 100:.0f}-{ci[1] * 100:.0f}%]" if ci else ""
    print(
        f"\nfusion {s['fusion_wins']} / solo {s['solo_wins']} / tie {s['ties']} "
        f"/ unparsed {s['unparsed']} "
        f"(fusion win rate on {s['decided']} decided: {rate_str}{ci_str})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
