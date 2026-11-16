"""Tests for majority-vote aggregation."""

from __future__ import annotations

import httpx

from openfusion.panel import MemberResponse, PanelResult
from openfusion.vote import answer_key, majority_vote


def _panel(*contents: str) -> PanelResult:
    result = PanelResult()
    for i, content in enumerate(contents):
        result.responses.append(
            MemberResponse(label=f"m{i}", content=content, model="m", usage=None)
        )
    return result


def test_answer_key_prefers_last_number() -> None:
    assert answer_key("First 3, then ... The answer is 42") == "42"
    assert answer_key("The total is $1,200.") == "1200"


def test_answer_key_canonicalizes_equal_numbers() -> None:
    assert answer_key("The answer is 42.0") == "42"
    assert answer_key("The answer is 42.00") == "42"
    assert answer_key("The answer is 42") == "42"
    assert answer_key("about 3.50 dollars") == "3.5"
    assert answer_key("exactly 100") == "100"  # no "1E+2"
    assert answer_key("it is -0.0") == "0"
    assert answer_key("x = -5") == "-5"


def test_answer_key_does_not_read_hyphen_as_minus_sign() -> None:
    assert answer_key("somewhere between 5-7") == "7"
    assert answer_key("Deadline 2024-03-15") == "15"


def test_majority_vote_agrees_across_number_formatting() -> None:
    _, meta = majority_vote(_panel("answer: 18", "answer: 18.0", "answer: 20"))
    assert meta["winner"] == "18"
    assert meta["agreement"] == 2 / 3


def test_answer_key_falls_back_to_last_line() -> None:
    assert answer_key("Reasoning here\nyes") == "yes"
    assert answer_key("") == ""


def test_majority_vote_picks_the_agreed_answer() -> None:
    panel = _panel(
        "work... The answer is 18",
        "different path, answer is 18",
        "I think the answer is 20",
    )
    chosen, meta = majority_vote(panel)
    assert meta["winner"] == "18"
    assert meta["agreement"] == 2 / 3
    assert "18" in chosen


def test_majority_vote_tie_breaks_by_order() -> None:
    panel = _panel("answer is 5", "answer is 9")
    chosen, meta = majority_vote(panel)
    # Tie -> earliest member wins, deterministically.
    assert meta["winner"] == "5"
    assert "5" in chosen


def test_majority_vote_empty_panel() -> None:
    chosen, meta = majority_vote(PanelResult())
    assert chosen == ""
    assert meta["members"] == 0


async def test_vote_aggregator_end_to_end(
    client: httpx.AsyncClient, mock_router, test_config
) -> None:
    # Switch the app's config to vote aggregation.
    from openfusion.config import Aggregator

    test_config.aggregator = Aggregator.VOTE

    answers = iter(["The answer is 18", "so it is 18", "maybe 25"])

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": next(answers)}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=upstream_handler)

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "9+9?"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "18" in body["choices"][0]["message"]["content"]
