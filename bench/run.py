#!/usr/bin/env python3
"""Head-to-head benchmark: self-fusion vs solo model."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _score(task: Task, answer: str) -> bool:
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
) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload["choices"][0]["message"]["content"]
    return str(answer), time.perf_counter() - started


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    tasks = _load_tasks(Path(args.tasks))
    solo_model = args.solo_model or config.panel[0].model
    gateway_key = config.gateway.api_keys[0] if config.gateway.api_keys else "bench-key"

    results: list[TaskResult] = []
    with httpx.Client() as client:
        for task in tasks:
            solo_answer, solo_latency = _chat(
                client,
                base_url=args.base_url,
                model=solo_model,
                prompt=task.prompt,
                api_key=gateway_key,
            )
            results.append(
                TaskResult(
                    task_id=task.id,
                    mode="solo",
                    correct=_score(task, solo_answer),
                    answer=solo_answer,
                    latency_seconds=solo_latency,
                )
            )

            fusion_answer, fusion_latency = _chat(
                client,
                base_url=args.base_url,
                model=config.fusion_model_name,
                prompt=task.prompt,
                api_key=gateway_key,
            )
            results.append(
                TaskResult(
                    task_id=task.id,
                    mode="fusion",
                    correct=_score(task, fusion_answer),
                    answer=fusion_answer,
                    latency_seconds=fusion_latency,
                )
            )

    def summarize(mode: str) -> dict[str, Any]:
        mode_results = [item for item in results if item.mode == mode]
        correct = sum(1 for item in mode_results if item.correct)
        total = len(mode_results)
        avg_latency = sum(item.latency_seconds for item in mode_results) / max(total, 1)
        return {
            "mode": mode,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "avg_latency_seconds": avg_latency,
        }

    return {
        "tasks_file": str(args.tasks),
        "config": str(args.config),
        "solo_model": solo_model,
        "fusion_model": config.fusion_model_name,
        "solo": summarize("solo"),
        "fusion": summarize("fusion"),
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark openfusion vs solo model")
    parser.add_argument("--config", default="openfusion.yaml.example")
    parser.add_argument("--tasks", default="bench/tasks/sample.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--solo-model", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_benchmark(args)
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    solo_acc = report["solo"]["accuracy"]
    fusion_acc = report["fusion"]["accuracy"]
    if fusion_acc < solo_acc:
        sys.exit(2)


if __name__ == "__main__":
    main()
