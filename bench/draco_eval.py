#!/usr/bin/env python3
"""DRACO eval: rubric-graded solo vs fusion on deep-research tasks.

For each task we generate a solo answer and a fusion answer (both with the same
web tools), then grade each against the task's weighted rubric with an
LLM-as-judge and compute the DRACO normalized score. We report mean normalized
score per system (overall and per domain) and the fusion-minus-solo delta.

Deviation from the paper for cost: criteria are graded in one batched judge call
per response (a JSON array of met/not-met) rather than one call per criterion,
and grading defaults to a single run (configurable via --grading-runs).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from bench.draco import Criterion, DracoTask, load_draco, normalized_score
from bench.run import _chat
from openfusion.config import load_config
from openfusion.tools import build_web_tools

GRADE_PROMPT = (
    "You are a strict, impartial grader applying a rubric to a response.\n\n"
    "Research task:\n{problem}\n\n"
    "Response to grade:\n{answer}\n\n"
    "Rubric criteria (numbered, one per line):\n{criteria}\n\n"
    "For EACH criterion decide whether the response satisfies it. A criterion "
    "that describes an error or unsafe content counts as met ONLY if the "
    "response actually commits it. Do not reward verbosity. Reply with ONLY a "
    'JSON object of the form {{"met": [true, false, ...]}} containing exactly '
    "{n} booleans in criterion order — no prose, no markdown."
)

_ANSWER_CHAR_CAP = 16000
_CRITERION_CHAR_CAP = 300


@dataclass
class TaskScore:
    task_id: str
    domain: str
    solo_score: float | None
    fusion_score: float | None


def _format_criteria(criteria: list[Criterion]) -> str:
    return "\n".join(
        f"{i + 1}. {c.text[:_CRITERION_CHAR_CAP]}" for i, c in enumerate(criteria)
    )


def _parse_met(verdict: str, n: int) -> list[bool] | None:
    """Extract exactly ``n`` booleans from a judge verdict, else None."""
    match = re.search(r"\{.*\}", verdict, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    met = obj.get("met") if isinstance(obj, dict) else None
    if not isinstance(met, list) or len(met) != n:
        return None
    return [bool(x) for x in met]


def _grade(
    client: httpx.Client,
    *,
    base_url: str,
    judge_model: str,
    api_key: str,
    task: DracoTask,
    answer: str,
    grading_runs: int,
) -> float | None:
    """Mean normalized score over grading runs, or None if all runs unparsed."""
    prompt = GRADE_PROMPT.format(
        problem=task.problem,
        answer=answer[:_ANSWER_CHAR_CAP],
        criteria=_format_criteria(task.criteria),
        n=len(task.criteria),
    )
    scores: list[float] = []
    for _ in range(grading_runs):
        verdict, _, _ = _chat(
            client,
            base_url=base_url,
            model=judge_model,
            prompt=prompt,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.2,
        )
        met = _parse_met(verdict, len(task.criteria))
        if met is not None:
            scores.append(normalized_score(task.criteria, met))
    return statistics.mean(scores) if scores else None


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    tasks = load_draco(args.limit)
    solo_model = args.solo_model or config.panel[0].model
    judge_model = args.judge_model or config.resolved_pass_through().model
    gateway_key = config.gateway.api_keys[0] if config.gateway.api_keys else "bench-key"

    solo_tools = build_web_tools(config.tools)
    solo_extra = {"tools": solo_tools} if solo_tools else None

    scores: list[TaskScore] = []
    errors: list[str] = []
    with httpx.Client() as client:
        for task in tasks:
            if not task.criteria:
                errors.append(f"{task.id}:no-rubric")
                continue
            try:
                solo_answer, _, _ = _chat(
                    client,
                    base_url=args.base_url,
                    model=solo_model,
                    prompt=task.problem,
                    api_key=gateway_key,
                    max_tokens=args.answer_tokens,
                    extra_body=solo_extra,
                )
                fusion_answer, _, _ = _chat(
                    client,
                    base_url=args.base_url,
                    model=config.fusion_model_name,
                    prompt=task.problem,
                    api_key=gateway_key,
                    max_tokens=args.answer_tokens,
                )
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                print(f"task {task.id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                errors.append(f"{task.id}:answer-error")
                continue

            solo_score = _grade(
                client,
                base_url=args.base_url,
                judge_model=judge_model,
                api_key=gateway_key,
                task=task,
                answer=solo_answer,
                grading_runs=args.grading_runs,
            )
            fusion_score = _grade(
                client,
                base_url=args.base_url,
                judge_model=judge_model,
                api_key=gateway_key,
                task=task,
                answer=fusion_answer,
                grading_runs=args.grading_runs,
            )
            scores.append(TaskScore(task.id, task.domain, solo_score, fusion_score))

    return summarize(args, solo_model, judge_model, config.fusion_model_name, scores, errors)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(
    args: argparse.Namespace,
    solo_model: str,
    judge_model: str,
    fusion_model: str,
    scores: list[TaskScore],
    errors: list[str],
) -> dict[str, Any]:
    solo_vals = [s.solo_score for s in scores if s.solo_score is not None]
    fusion_vals = [s.fusion_score for s in scores if s.fusion_score is not None]
    paired = [
        (s.solo_score, s.fusion_score)
        for s in scores
        if s.solo_score is not None and s.fusion_score is not None
    ]
    deltas = [f - s for s, f in paired]

    domains: dict[str, dict[str, list[float]]] = {}
    for s in scores:
        d = domains.setdefault(s.domain, {"solo": [], "fusion": []})
        if s.solo_score is not None:
            d["solo"].append(s.solo_score)
        if s.fusion_score is not None:
            d["fusion"].append(s.fusion_score)

    return {
        "eval": "draco_rubric",
        "config": str(args.config),
        "solo_model": solo_model,
        "fusion_model": fusion_model,
        "judge_model": judge_model,
        "solo_tools": [t["type"] for t in (build_web_tools(load_config(args.config).tools))],
        "answer_tokens": args.answer_tokens,
        "grading_runs": args.grading_runs,
        "summary": {
            "n": len(scores),
            "errors": len(errors),
            "solo_mean": _mean(solo_vals),
            "fusion_mean": _mean(fusion_vals),
            "delta_mean": _mean(deltas),
            "paired_n": len(paired),
            "fusion_better": sum(1 for d in deltas if d > 0),
            "solo_better": sum(1 for d in deltas if d < 0),
            "by_domain": {
                dom: {"solo": _mean(v["solo"]), "fusion": _mean(v["fusion"])}
                for dom, v in sorted(domains.items())
            },
        },
        "scores": [asdict(s) for s in scores],
        "error_ids": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DRACO rubric-graded solo-vs-fusion eval")
    parser.add_argument("--config", default="openfusion.draco.yaml.example")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--solo-model", default=None)
    parser.add_argument("--judge-model", default=None, help="Defaults to the pass-through model.")
    parser.add_argument("--answer-tokens", type=int, default=2048)
    parser.add_argument("--grading-runs", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_eval(args)
    output = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    s = report["summary"]

    def pct(value: float | None) -> str:
        return f"{value:.1f}%" if value is not None else "n/a"

    print(
        f"\nDRACO: solo {pct(s['solo_mean'])} | fusion {pct(s['fusion_mean'])} "
        f"| delta {pct(s['delta_mean'])} (paired {s['paired_n']}, "
        f"fusion better on {s['fusion_better']}/{s['paired_n']}, errors {s['errors']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
