#!/usr/bin/env python3
"""Head-to-head benchmark: self-fusion vs solo model."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from bench.datasets import LOADERS
from openfusion.config import load_config


@dataclass
class Task:
    id: str
    prompt: str
    expected: str
    match: str = "exact"


@dataclass
class TaskResult:
    task_id: str
    mode: str
    correct: bool
    answer: str
    latency_seconds: float
    total_tokens: int = 0
    cost_usd: float = 0.0


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"[\s\.,;:!?]+$", "", normalized.strip())
    return re.sub(r"\s+", " ", normalized.lower())


_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    cleaned = matches[-1].replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _score(task: Task, answer: str) -> bool:
    if task.match == "numeric":
        got = _last_number(answer)
        expected = _last_number(task.expected)
        return got is not None and expected is not None and abs(got - expected) < 1e-6
    if task.match == "contains":
        return _normalize(task.expected) in _normalize(answer)
    return _normalize(answer) == _normalize(task.expected)


def _load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        tasks.append(
            Task(
                id=str(payload["id"]),
                prompt=str(payload["prompt"]),
                expected=str(payload["expected"]),
                match=str(payload.get("match", "exact")),
            )
        )
    return tasks


def _chat(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = 360.0,
) -> tuple[str, float, dict[str, Any]]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return str(answer), time.perf_counter() - started, usage


def _build_tasks(args: argparse.Namespace) -> tuple[list[Task], str]:
    if args.dataset:
        loader = LOADERS[args.dataset]
        raw = loader(args.limit)
        tasks = [Task(id=t.id, prompt=t.prompt, expected=t.expected, match=t.match) for t in raw]
        return tasks, f"{args.dataset}[:{args.limit}]"
    return _load_tasks(Path(args.tasks)), str(args.tasks)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    tasks, tasks_label = _build_tasks(args)
    solo_model = args.solo_model or config.panel[0].model
    gateway_key = config.gateway.api_keys[0] if config.gateway.api_keys else "bench-key"

    results: list[TaskResult] = []
    with httpx.Client() as client:
        for task in tasks:
            solo_answer, solo_latency, solo_usage = _chat(
                client,
                base_url=args.base_url,
                model=solo_model,
                prompt=task.prompt,
                api_key=gateway_key,
                max_tokens=args.max_tokens,
            )
            results.append(
                TaskResult(
                    task_id=task.id,
                    mode="solo",
                    correct=_score(task, solo_answer),
                    answer=solo_answer,
                    latency_seconds=solo_latency,
                    total_tokens=int(solo_usage.get("total_tokens", 0) or 0),
                    cost_usd=float(solo_usage.get("cost", 0.0) or 0.0),
                )
            )

            fusion_answer, fusion_latency, fusion_usage = _chat(
                client,
                base_url=args.base_url,
                model=config.fusion_model_name,
                prompt=task.prompt,
                api_key=gateway_key,
                max_tokens=args.max_tokens,
            )
            results.append(
                TaskResult(
                    task_id=task.id,
                    mode="fusion",
                    correct=_score(task, fusion_answer),
                    answer=fusion_answer,
                    latency_seconds=fusion_latency,
                    total_tokens=int(fusion_usage.get("total_tokens", 0) or 0),
                    cost_usd=float(fusion_usage.get("cost", 0.0) or 0.0),
                )
            )

    def summarize(mode: str) -> dict[str, Any]:
        mode_results = [item for item in results if item.mode == mode]
        correct = sum(1 for item in mode_results if item.correct)
        total = len(mode_results)
        avg_latency = sum(item.latency_seconds for item in mode_results) / max(total, 1)
        total_tokens = sum(item.total_tokens for item in mode_results)
        total_cost = sum(item.cost_usd for item in mode_results)
        return {
            "mode": mode,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "avg_latency_seconds": avg_latency,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            # Quality-per-dollar: lower is better. None when nothing was correct.
            "cost_per_correct_usd": (total_cost / correct) if correct else None,
            "tokens_per_correct": (total_tokens / correct) if correct else None,
        }

    return {
        "tasks_file": tasks_label,
        "config": str(args.config),
        "solo_model": solo_model,
        "fusion_model": config.fusion_model_name,
        "max_tokens": args.max_tokens,
        "solo": summarize("solo"),
        "fusion": summarize("fusion"),
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark openfusion vs solo model")
    parser.add_argument("--config", default="openfusion.yaml.example")
    parser.add_argument("--tasks", default="bench/tasks/sample.jsonl")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(LOADERS),
        help="Load tasks from a public dataset loader instead of --tasks.",
    )
    parser.add_argument("--limit", type=int, default=40, help="Max tasks when using --dataset.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--solo-model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if fusion accuracy is below solo (off by default; "
        "a completed run is success, the table tells the story).",
    )
    args = parser.parse_args()

    report = run_benchmark(args)
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    if args.fail_on_regression and report["fusion"]["accuracy"] < report["solo"]["accuracy"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
