"""Pytest tests for the CLR scoring logic (no model servers needed)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync


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

    def test_self_claims_only_is_capped_at_065(self, clr):
        """The most important test: 5 self-verified claims must NOT reach 1.0.
        Self-verification alone is capped at 0.65 — model self-agreement is
        not proof of correctness."""
        claims = [
            "This is a meaningful claim with enough detail one.",
            "This is a meaningful claim with enough detail two.",
            "This is a meaningful claim with enough detail three.",
            "This is a meaningful claim with enough detail four.",
            "This is a meaningful claim with enough detail five.",
        ]
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True,
            deterministic_check=None,
        )
        assert score <= 0.65, f"Self-claims-only score {score} exceeds 0.65 cap"

    def test_all_verified_meaningful_claims_capped_without_verifier(self, clr):
        """Without a deterministic verifier, even perfect self-verification
        cannot exceed 0.65."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=True)
        assert score <= 0.65

    def test_deterministic_check_allows_above_065(self, clr):
        """With deterministic verification, score CAN exceed 0.65."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True,
            deterministic_check=True,
        )
        # 1.0 * 0.7 + 1.0 * 0.3 = 1.0
        assert score > 0.65

    def test_deterministic_check_refutation_scores_zero(self, clr):
        """If a deterministic verifier refutes the answer, score must be 0."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True,
            deterministic_check=False,
        )
        assert score == 0.0

    def test_mixed_garbage_and_real_claims_capped(self, clr):
        # 2 garbage + 5 real, all verified -> only 5 count, but capped at 0.65
        claims = ["by step.", "short",
                  "real claim one here", "real claim two here",
                  "real claim three here", "real claim four here",
                  "real claim five here"]
        score = clr._calculate_reliability([1, 1, 1, 1, 1, 1, 1], claims=claims, answer_present=True)
        assert score <= 0.65


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


class TestFailClosedRun:
    """Tests for the fail-closed behavior of VibeThinkerCLRAsync.run().

    A dead model server is infrastructure failure, not a low-confidence answer.
    """

    @pytest.mark.asyncio
    async def test_all_trajectories_transport_fail_raises(self, clr):
        """All trajectories fail with transport exceptions -> RuntimeError."""
        async def boom(*args, **kwargs):
            raise RuntimeError("Connection refused")
        with patch.object(clr, "_generate_one_trajectory", new=AsyncMock(side_effect=boom)):
            with pytest.raises(RuntimeError, match="All CLR trajectories failed"):
                await clr.run("test problem")

    @pytest.mark.asyncio
    async def test_partial_trajectory_failure_still_returns_with_metadata(self):
        """Some trajectories fail, some succeed -> continue with warning metadata."""
        clr = VibeThinkerCLRAsync(server_url="http://localhost:0", k=4)
        good_traj = {
            "score": 1.0,
            "answer": "42",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{42}",
            "answer_present": True,
        }

        call_count = 0
        async def mixed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Connection refused")
            return good_traj

        with patch.object(clr, "_generate_one_trajectory", new=AsyncMock(side_effect=mixed)):
            result = await clr.run("test problem")
        assert result.partial_failure is True
        assert result.transport_failures > 0
        assert result.best_answer == "42"

    @pytest.mark.asyncio
    async def test_successful_empty_answer_returns_zero_score_completed(self, clr):
        """Trajectories succeed but none produce a final answer -> score 0, completed."""
        empty_traj = {
            "score": 0.0,
            "answer": None,
            "claims": [],
            "verdicts": [],
            "raw_trace": "reasoning with no boxed answer",
            "answer_present": False,
        }
        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=empty_traj)):
            result = await clr.run("test problem")
        assert result.best_score == 0.0
        assert result.best_answer == "No clear answer found"
        assert result.failure_reason is None  # not an infrastructure failure


class TestVerifierIntegration:
    """Tests for the verifier integration in CLR run().

    A deterministic verifier is the ONLY path that allows the final score
    to exceed the self-claims-only cap of 0.65.
    """

    @pytest.mark.asyncio
    async def test_no_verifier_caps_at_065(self, clr):
        """Without a verifier, score is capped at 0.65 even with perfect claims."""
        good_traj = {
            "score": 0.65,
            "answer": "42",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{42}",
            "answer_present": True,
        }
        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            result = await clr.run("test problem", verifier=None)
        assert result.verification_method == "self_claims_only"
        assert result.verified is False
        assert result.best_score <= 0.65

    @pytest.mark.asyncio
    async def test_math_verifier_allows_above_065(self, clr):
        """With a passing math verifier, score CAN exceed 0.65."""
        from verifiers import MathVerifier
        from verifiers.base import VerificationResult

        good_traj = {
            "score": 0.65,
            "answer": "4",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{4}",
            "answer_present": True,
        }
        # Mock the math verifier to return verified=True
        verifier = MathVerifier()
        original_verify = verifier.verify
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=True, score=1.0, method="numeric_comparison",
                evidence={"candidate": 4.0, "expected": 4.0},
            )
        verifier.verify = mock_verify

        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            result = await clr.run("What is 2+2?", verifier=verifier, task_type="math")
        assert result.verification_method == "math_verifier"
        assert result.verified is True
        assert result.best_score > 0.65

    @pytest.mark.asyncio
    async def test_verifier_refutation_scores_zero(self, clr):
        """If a verifier refutes the answer, score must be 0."""
        from verifiers import MathVerifier
        from verifiers.base import VerificationResult

        good_traj = {
            "score": 0.65,
            "answer": "5",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{5}",
            "answer_present": True,
        }
        verifier = MathVerifier()
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=False, score=0.0, method="numeric_comparison",
                evidence={"candidate": 5.0, "expected": 4.0},
                error="5.0 != expected 4.0",
            )
        verifier.verify = mock_verify

        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            result = await clr.run("What is 2+2?", verifier=verifier, task_type="math")
        assert result.verified is False
        assert result.best_score == 0.0

    @pytest.mark.asyncio
    async def test_verifier_error_falls_back_to_self_claims(self, clr):
        """If a verifier raises an exception, fall back to self-claims-only."""
        from verifiers import MathVerifier

        good_traj = {
            "score": 0.65,
            "answer": "4",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{4}",
            "answer_present": True,
        }
        verifier = MathVerifier()
        async def boom_verify(query, answer, context):
            raise RuntimeError("verifier crashed")
        verifier.verify = boom_verify

        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            result = await clr.run("What is 2+2?", verifier=verifier, task_type="math")
        assert result.verification_method == "self_claims_only"
        assert result.verified is False
        assert result.best_score <= 0.65

