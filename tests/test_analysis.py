"""Analysis splitter: separate the answer from the trailing analysis block."""

from __future__ import annotations

from openfusion.stream import _AnalysisSplitter


def test_splits_answer_from_analysis_across_chunks() -> None:
    splitter = _AnalysisSplitter()
    out = ""
    out += splitter.feed("Hello ")
    out += splitter.feed("world ===ANA")  # sentinel straddles two chunks
    out += splitter.feed("LYSIS===")
    out += splitter.feed('{"consensus": "x"}')
    out += splitter.flush()

    assert out.strip() == "Hello world"
    assert splitter.analysis_payload() == {"consensus": "x"}


def test_no_sentinel_is_all_content() -> None:
    splitter = _AnalysisSplitter()
    out = splitter.feed("Just an answer.") + splitter.flush()
    assert out == "Just an answer."
    assert splitter.analysis_payload() is None


def test_non_json_analysis_falls_back_to_raw() -> None:
    splitter = _AnalysisSplitter()
    out = splitter.feed("answer ===ANALYSIS=== not json") + splitter.flush()
    assert out.strip() == "answer"
    assert splitter.analysis_payload() == {"raw": "not json"}
