"""Pytest tests for the CLR scoring logic (no model servers needed)."""

import pytest

from vibe_clr_async import VibeThinkerCLRAsync


@pytest.fixture
def clr():
    """VibeThinkerCLRAsync without needing a real server."""
    return VibeThinkerCLRAsync(server_url="http://localhost:0", k=1)


class TestReliabilityScoring:
    def test_empty_verdicts_returns_zero(self, clr):
        assert clr._calculate_reliability([]) == 0.0

    def test_no_answer_returns_zero(self, clr):
        # No answer_present flag -> score 0, even with 5 verified claims
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        assert clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=False) == 0.0

    def test_fewer_than_min_claims_returns_zero(self, clr):
        # Only 2 meaningful claims — below MIN_CLAIMS_FOR_SCORING=5
        claims = ["a" * 20, "b" * 20]
        assert clr._calculate_reliability([1, 1], claims=claims, answer_present=True) == 0.0

    def test_single_claim_returns_zero(self, clr):
        # The smoking gun from the audit: 1 verified claim -> 1.0
        # Now it must return 0.0
        assert clr._calculate_reliability([1], claims=["a meaningful claim here"], answer_present=True) == 0.0

    def test_garbage_claims_rejected(self, clr):
        # The exact garbage from the audit: "by step reasoning."
        claims = ["by step reasoning.", "by step.", "by step reasoning. So we can elaborate."]
        # All are garbage -> filtered out -> 0 meaningful -> score 0
        assert clr._calculate_reliability([1, 1, 1], claims=claims, answer_present=True) == 0.0

    def test_short_claims_rejected(self, clr):
        # Claims shorter than MIN_CLAIM_LENGTH (15 chars) are too trivial
        claims = ["short", "tiny", "x"]
        assert clr._calculate_reliability([1, 1, 1], claims=claims, answer_present=True) == 0.0

    def test_any_failed_verdict_capped(self, clr):
        # One wrong claim out of 5 -> score capped at 0.3
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability([1, 1, 1, 1, 0], claims=claims, answer_present=True)
        assert score <= 0.3
        assert score > 0.0  # not zero, but heavily penalized

    def test_all_verified_meaningful_claims_high_score(self, clr):
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=True)
        assert score == 1.0

    def test_mixed_garbage_and_real_claims(self, clr):
        # 2 garbage + 5 real, all verified -> only 5 count, score 1.0
        claims = ["by step.", "short",
                  "real claim one here", "real claim two here",
                  "real claim three here", "real claim four here",
                  "real claim five here"]
        score = clr._calculate_reliability([1, 1, 1, 1, 1, 1, 1], claims=claims, answer_present=True)
        assert score == 1.0

    def test_deterministic_check_boost(self, clr):
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score_base = clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=True)
        score_boosted = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True, deterministic_check=True
        )
        assert score_boosted > score_base or score_boosted == 1.0

    def test_deterministic_check_contradiction_halves(self, clr):
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score_base = clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=True)
        score_contradicted = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True, deterministic_check=False
        )
        assert score_contradicted == score_base * 0.5


class TestIsMeaningfulClaim:
    @pytest.mark.parametrize("claim,expected", [
        ("by step reasoning.", False),
        ("by step.", False),
        ("by step reasoning. So we can elaborate.", False),
        ("step by step.", False),
        ("none", False),
        ("null", False),
        ("n/a", False),
        ("short", False),
        ("ab", False),
        ("...", False),
        ("123", False),
        ("The recurrence relation produces values 2, 3, 7, 43, 1807", True),
        ("We compute a_2 = 2^2 - 2 + 1 = 3", True),
        ("The geometric series converges to 3/2", True),
    ])
    def test_meaningful_claim_filter(self, clr, claim, expected):
        assert clr._is_meaningful_claim(claim) == expected
