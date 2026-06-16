"""Offline tests for the DRACO loader, rubric parsing, and scoring."""

from __future__ import annotations

import json

from bench.draco import (
    Criterion,
    _criterion_from,
    _find_criteria_list,
    _parse_rubric,
    normalized_score,
)


def test_parse_rubric_from_json_string() -> None:
    answer = json.dumps(
        {
            "id": "some-slug",
            "criteria": [
                {"criterion": "States the correct CAGR", "weight": 10, "category": "factual"},
                {"criterion": "Cites a primary source", "weight": 5, "category": "citation"},
                {"criterion": "Gives dangerous advice", "weight": -25, "category": "factual"},
            ],
        }
    )
    criteria = _parse_rubric(answer)
    assert [c.weight for c in criteria] == [10.0, 5.0, -25.0]
    assert criteria[0].text == "States the correct CAGR"
    assert criteria[2].category == "factual"


def test_criterion_from_alternate_field_names() -> None:
    c = _criterion_from({"text": "foo", "points": 7, "axis": "presentation"})
    assert c == Criterion(text="foo", weight=7.0, category="presentation")


def test_find_criteria_list_fallback_to_any_list_of_dicts() -> None:
    rubric = {"meta": "x", "weird_key": [{"description": "c1", "weight": 1}]}
    found = _find_criteria_list(rubric)
    assert found and found[0]["description"] == "c1"


def test_normalized_score_basic() -> None:
    criteria = [Criterion("a", 10), Criterion("b", 5), Criterion("c", 5)]
    # 10 of 20 positive weight met -> 50.
    assert normalized_score(criteria, [True, False, False]) == 50.0
    # 15 of 20 -> 75.
    assert normalized_score(criteria, [True, True, False]) == 75.0
    assert normalized_score(criteria, [True, True, True]) == 100.0
    assert normalized_score(criteria, [False, False, False]) == 0.0


def test_normalized_score_negative_weight_penalizes_and_clamps() -> None:
    criteria = [Criterion("good", 10), Criterion("error", -25)]
    # Meeting the negative criterion drives raw below zero -> clamped to 0.
    assert normalized_score(criteria, [True, True]) == 0.0
    # Only the positive criterion met -> full positive weight.
    assert normalized_score(criteria, [True, False]) == 100.0


def test_normalized_score_no_positive_weight_is_zero() -> None:
    assert normalized_score([Criterion("x", -5)], [False]) == 0.0
