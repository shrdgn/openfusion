#!/usr/bin/env python3
"""DRACO deep-research benchmark: loader, rubric model, and scoring.

DRACO (Perplexity) is a single `test.jsonl` of 100 deep-research tasks. Each
row has `id`, `domain`, `problem` (the query), and `answer` — a JSON-encoded,
task-specific rubric of ~40 weighted criteria across four axes (factual
accuracy, breadth/depth, presentation, citation). Criteria can carry negative
weights: meeting one means the response contains an error.

Scoring (per the paper):
    raw        = sum(w_i for each criterion i that is MET)
    normalized = clamp(raw / sum(w_i for w_i > 0), 0, 1) * 100

Run this module directly to probe the live schema without spending on models:
    python -m bench.draco --probe --limit 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from bench.datasets import _fetch_jsonl

DRACO_URL = "https://huggingface.co/datasets/perplexity-ai/draco/resolve/main/test.jsonl"

# Hosts of the benchmark/rubric themselves — excluded from panel/solo web tools
# so models can't retrieve the grading rubric (the contamination risk OpenRouter
# flagged). Wire these into a config's tools.excluded_domains for DRACO runs.
RUBRIC_HOST_DOMAINS = [
    "huggingface.co",
    "arxiv.org",
    "perplexity.ai",
    "r2cdn.perplexity.ai",
    "research.perplexity.ai",
    "researchgate.net",
]

_TEXT_KEYS = ("criterion", "text", "description", "statement", "requirement", "name")
_WEIGHT_KEYS = ("weight", "points", "score", "value")
_CATEGORY_KEYS = ("category", "axis", "type", "dimension", "section")
_CRITERIA_KEYS = ("criteria", "rubric", "rubric_items", "items", "rubrics", "checklist")


@dataclass
class Criterion:
    text: str
    weight: float
    category: str | None = None


@dataclass
class DracoTask:
    id: str
    domain: str
    problem: str
    criteria: list[Criterion]


def _find_criteria_list(rubric: object) -> list[dict]:
    """Locate the list of criterion objects inside a parsed rubric."""
    if isinstance(rubric, list):
        return [item for item in rubric if isinstance(item, dict)]
    if not isinstance(rubric, dict):
        return []
    for key in _CRITERIA_KEYS:
        value = rubric.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    # Fallback: first list-of-dicts value anywhere in the object.
    for value in rubric.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def _first(d: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _criterion_from(d: dict) -> Criterion:
    text = _first(d, _TEXT_KEYS)
    weight = _first(d, _WEIGHT_KEYS)
    category = _first(d, _CATEGORY_KEYS)
    try:
        weight_val = float(weight) if isinstance(weight, (int, float)) else 0.0
    except (TypeError, ValueError):
        weight_val = 0.0
    return Criterion(
        text=str(text) if text is not None else "",
        weight=weight_val,
        category=str(category) if category is not None else None,
    )


def _parse_rubric(answer: object) -> list[Criterion]:
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            return []
    return [_criterion_from(c) for c in _find_criteria_list(answer)]


def load_draco(limit: int, url: str = DRACO_URL) -> list[DracoTask]:
    """Return the first ``limit`` DRACO tasks with parsed weighted rubrics."""
    rows = _fetch_jsonl(url, "draco-test.jsonl")[:limit]
    tasks: list[DracoTask] = []
    for index, row in enumerate(rows):
        problem = row.get("problem") or row.get("prompt") or row.get("query") or ""
        criteria = _parse_rubric(row.get("answer") or row.get("rubric"))
        tasks.append(
            DracoTask(
                id=str(row.get("id", f"draco-{index + 1}")),
                domain=str(row.get("domain", "")),
                problem=str(problem),
                criteria=criteria,
            )
        )
    return tasks


def normalized_score(criteria: list[Criterion], met: list[bool]) -> float:
    """DRACO normalized score in [0, 100] for one graded response."""
    positive = sum(c.weight for c in criteria if c.weight > 0)
    if positive <= 0:
        return 0.0
    raw = sum(c.weight for c, is_met in zip(criteria, met, strict=False) if is_met)
    return max(0.0, min(1.0, raw / positive)) * 100.0


def _probe(limit: int) -> None:
    """Fetch the dataset and print its structure — no model calls."""
    rows = _fetch_jsonl(DRACO_URL, "draco-test.jsonl")
    print(f"rows: {len(rows)}")
    print(f"top-level keys: {sorted(rows[0].keys())}")
    for row in rows[:limit]:
        criteria = _parse_rubric(row.get("answer") or row.get("rubric"))
        weights = [c.weight for c in criteria]
        negatives = sum(1 for w in weights if w < 0)
        print(f"\n--- task {row.get('id')} [{row.get('domain')}] ---")
        print(f"problem: {str(row.get('problem'))[:160]}...")
        print(f"criteria parsed: {len(criteria)} (negative weights: {negatives})")
        if criteria:
            c = criteria[0]
            print(f"first criterion: weight={c.weight} category={c.category!r}")
            print(f"  text: {c.text[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRACO loader / schema probe")
    parser.add_argument("--probe", action="store_true", help="Print dataset structure and exit.")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    if args.probe:
        _probe(args.limit)
    else:
        for task in load_draco(args.limit):
            print(f"{task.id} [{task.domain}] criteria={len(task.criteria)}")


if __name__ == "__main__":
    main()
