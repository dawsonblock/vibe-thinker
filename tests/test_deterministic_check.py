"""Pytest tests for deterministic answer checking and structured routing."""

import pytest

from vibe_clr_async import VibeThinkerCLRAsync


@pytest.fixture
def clr():
    return VibeThinkerCLRAsync(server_url="http://localhost:0", k=1)


class TestBoxedExtraction:
    def test_extract_boxed(self, clr):
        text = "Some reasoning... \\boxed{42} and more text"
        assert clr._extract_boxed_answer(text) == "42"

    def test_extract_last_boxed(self, clr):
        text = "\\boxed{first} then \\boxed{final}"
        assert clr._extract_boxed_answer(text) == "final"

    def test_no_boxed_returns_none(self, clr):
        assert clr._extract_boxed_answer("no answer here") is None


class TestNumericNormalization:
    @pytest.mark.parametrize("s,expected", [
        ("42", 42.0),
        ("3.14", 3.14),
        ("1,000", 1000.0),
        ("7/2", 3.5),
        ("\\frac{1}{2}", 0.5),
        ("not a number", None),
    ])
    def test_normalize_numeric(self, clr, s, expected):
        result = clr._normalize_numeric(s)
        if expected is None:
            assert result is None
        else:
            assert abs(result - expected) < 1e-6


class TestDeterministicCheck:
    def test_consistent_numeric_answers_boost(self, clr):
        trajectories = [
            {"answer_present": True, "raw_trace": "reasoning \\boxed{42}"},
            {"answer_present": True, "raw_trace": "other reasoning \\boxed{42}"},
            {"answer_present": True, "raw_trace": "third \\boxed{42}"},
        ]
        result = clr._check_answer_deterministic("42", trajectories)
        assert result is True

    def test_contradictory_answers_flagged(self, clr):
        trajectories = [
            {"answer_present": True, "raw_trace": "reasoning \\boxed{42}"},
            {"answer_present": True, "raw_trace": "other \\boxed{99}"},
            {"answer_present": True, "raw_trace": "third \\boxed{99}"},
        ]
        result = clr._check_answer_deterministic("42", trajectories)
        assert result is False

    def test_insufficient_data_returns_none(self, clr):
        trajectories = [{"answer_present": True, "raw_trace": "\\boxed{42}"}]
        result = clr._check_answer_deterministic("42", trajectories)
        assert result is None

    def test_non_numeric_string_match(self, clr):
        trajectories = [
            {"answer_present": True, "raw_trace": "\\boxed{Paris}"},
            {"answer_present": True, "raw_trace": "\\boxed{Paris}"},
        ]
        result = clr._check_answer_deterministic("Paris", trajectories)
        assert result is True
