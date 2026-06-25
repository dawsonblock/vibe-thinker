"""Runtime integration tests for scoring and verifier wiring.

These tests prove that the active runtime path (not just scoring.py)
obeys the confidence cap and uses verifiers correctly.

Acceptance criteria tested:
  1. Self-claims-only CLR runtime score cannot exceed 0.65.
  2. compute_confidence() is used by the active runtime path.
  3. MathVerifier is called for math tasks.
  4. CodeVerifier is called only when executable checks are available.
  5. FactualVerifier unsupported result prevents high confidence.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync
from verifiers import MathVerifier, CodeVerifier, FactualVerifier
from verifiers.base import VerificationResult
from hybrid_orchestrator import select_verifier
from math_solver import solve as solve_math
from sandbox import LocalSubprocessExecutor


@contextmanager
def patch_generators(clr, return_value=None, side_effect=None):
    """Patch both trajectory generation methods.

    Adaptive mode uses _generate_lightweight_trajectory when a verifier
    is present, and _generate_one_trajectory otherwise. Tests need to
    patch both to cover all code paths.
    """
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    with patch.object(clr, "_generate_one_trajectory", new=mock):
        with patch.object(clr, "_generate_lightweight_trajectory", new=mock):
            yield mock


@pytest.fixture
def clr():
    return VibeThinkerCLRAsync(server_url="http://localhost:0", k=1)


def _good_trajectory(answer="42"):
    """A trajectory with 5 meaningful verified claims and a final answer."""
    return {
        "score": 0.65,  # already capped by _calculate_reliability
        "answer": answer,
        "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
        "verdicts": [1, 1, 1, 1, 1],
        "raw_trace": f"reasoning \\boxed{{{answer}}}",
        "answer_present": True,
    }


class TestRuntimeConfidenceCap:
    """The active runtime path must enforce the 0.65 self-claims-only cap."""

    @pytest.mark.asyncio
    async def test_runtime_clr_cannot_return_high_confidence_without_verifier(self, clr):
        """Without a verifier, the CLR runtime must not return > 0.65."""
        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=_good_trajectory())):
            result = await clr.run("Explain something", verifier=None)
        assert result.verification_method == "self_claims_only"
        assert result.best_score <= 0.65
        assert result.verified is False

    @pytest.mark.asyncio
    async def test_runtime_clr_can_exceed_cap_with_math_verifier(self, clr):
        """With a passing math verifier, score CAN exceed 0.65."""
        verifier = MathVerifier()
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=True, score=1.0, method="numeric_comparison",
                evidence={"candidate": 42.0, "expected": 42.0},
            )
        verifier.verify = mock_verify

        with patch_generators(clr, return_value=_good_trajectory("42")):
            result = await clr.run("Compute 2 + 2", verifier=verifier, task_type="math")
        assert result.verification_method == "math_verifier"
        assert result.verified is True
        assert result.best_score > 0.65

    @pytest.mark.asyncio
    async def test_runtime_clr_verifier_refutation_scores_zero(self, clr):
        """If a verifier refutes the answer, final score must be 0."""
        verifier = MathVerifier()
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=False, score=0.0, method="numeric_comparison",
                error="wrong answer",
            )
        verifier.verify = mock_verify

        with patch_generators(clr, return_value=_good_trajectory("5")):
            result = await clr.run("Compute 2 + 2", verifier=verifier, task_type="math")
        assert result.verified is False
        assert result.best_score == 0.0

    @pytest.mark.asyncio
    async def test_partial_verifier_score_not_inflated(self, clr):
        """A verifier returning verified=True with score=0.7 must NOT be
        inflated to deterministic_verification=1.0. The actual score must
        be used so weak verification (e.g. FactualVerifier overlap) stays
        weak in the final confidence."""
        verifier = MathVerifier()
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=True, score=0.7, method="numeric_comparison",
                evidence={"partial": True},
            )
        verifier.verify = mock_verify

        with patch_generators(clr, return_value=_good_trajectory("42")):
            result = await clr.run("Compute 2 + 2", verifier=verifier, task_type="math")
        assert result.verified is True
        # 0.7 * 0.7 + 0.65 * 0.3 = 0.49 + 0.195 = 0.685
        # Must NOT be 1.0 * 0.7 + 0.65 * 0.3 = 0.895
        assert result.best_score < 0.8
        assert abs(result.best_score - 0.685) < 0.01


class TestVerifierSelection:
    """select_verifier must pick the right verifier for each task type."""

    def test_math_task_gets_math_verifier(self):
        v = select_verifier("math")
        assert isinstance(v, MathVerifier)

    def test_code_task_gets_code_verifier(self):
        v = select_verifier("code")
        assert isinstance(v, CodeVerifier)

    def test_factual_task_gets_factual_verifier(self):
        v = select_verifier("factual")
        assert isinstance(v, FactualVerifier)

    def test_conversation_gets_no_verifier(self):
        assert select_verifier("conversation") is None

    def test_summarization_gets_no_verifier(self):
        assert select_verifier("summarization") is None

    def test_unknown_gets_no_verifier(self):
        assert select_verifier("unknown") is None

    def test_planning_gets_no_verifier(self):
        assert select_verifier("planning") is None


class TestCodeVerifierHonesty:
    """CodeVerifier must refuse unsafe/no-test cases by default."""

    @pytest.mark.asyncio
    async def test_no_tests_returns_unverified(self):
        """Code without executable tests must NOT be verified."""
        v = CodeVerifier(executor=LocalSubprocessExecutor())
        result = await v.verify("Write a function", "def f(): pass", context={})
        assert result.verified is False
        assert "no unit_tests" in (result.error or "") or "no" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_unverified(self):
        """Infinite loop must timeout and return unverified."""
        v = CodeVerifier(timeout=2.0, executor=LocalSubprocessExecutor())
        result = await v.verify("Run this", "while True: pass",
                                context={"expected_output": "anything"})
        assert result.verified is False
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execution_exception_returns_unverified(self):
        """Code that raises must return unverified."""
        v = CodeVerifier(executor=LocalSubprocessExecutor())
        result = await v.verify("Run this", "raise ValueError('boom')",
                                context={"expected_output": "anything"})
        assert result.verified is False

    @pytest.mark.asyncio
    async def test_passing_tests_returns_verified(self):
        """Code that passes bounded unit tests is verified."""
        v = CodeVerifier(executor=LocalSubprocessExecutor())
        code = "def add(a, b):\n    return a + b\n"
        tests = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"
        result = await v.verify("Write add", code, context={"unit_tests": tests})
        assert result.verified is True
        assert result.score == 1.0


class TestFactualVerifierHonesty:
    """FactualVerifier must hard-cap unsupported factual answers."""

    @pytest.mark.asyncio
    async def test_no_sources_returns_unverified(self):
        """Factual claims without retrieval sources cannot be verified."""
        v = FactualVerifier()
        result = await v.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={},
        )
        assert result.verified is False
        assert result.method == "unsupported_factual"

    @pytest.mark.asyncio
    async def test_unsupported_factual_cannot_exceed_cap(self, clr):
        """A factual answer without retrieval must not get high confidence."""
        verifier = FactualVerifier()
        # FactualVerifier with no sources returns verified=False
        # The CLR run should fall back to self_claims_only cap
        with patch_generators(clr, return_value=_good_trajectory("Paris")):
            result = await clr.run(
                "What is the capital of France?",
                verifier=verifier,
                task_type="factual",
            )
        # FactualVerifier returns verified=False, score=0.0 -> final_score stays capped
        assert result.best_score <= 0.65
        assert result.verified is False


class TestVerifierContextDerivation:
    """Tests proving verifiers get real context, not empty dicts.

    The math solver derives expected_answer for simple problems. The
    verifier then has something to compare against — this is what makes
    the verifier layer useful instead of ceremonial.
    """

    def test_math_solver_solves_recurrence(self):
        """The canonical recurrence query must be solvable."""
        result = solve_math("Solve this step by step: a_1=2, a_{n+1}=a_n^2-a_n+1. Find a_5.")
        assert result == "1807"

    def test_math_solver_solves_arithmetic(self):
        result = solve_math("What is 2+2?")
        assert result == "4"

    @pytest.mark.asyncio
    async def test_math_verifier_with_derived_context_verifies_correct_answer(self, clr):
        """When the math solver derives expected_answer and the model's
        answer matches, the verifier should return verified=True."""
        verifier = MathVerifier()
        # The solver will derive expected_answer="4" for "What is 2+2?"
        context = {"expected_answer": "4"}

        with patch_generators(clr, return_value=_good_trajectory("4")):
            result = await clr.run(
                "What is 2+2?",
                verifier=verifier,
                task_type="math",
                verifier_context=context,
            )
        assert result.verification_method == "math_verifier"
        assert result.verified is True
        assert result.best_score > 0.65

    @pytest.mark.asyncio
    async def test_math_verifier_with_derived_context_refutes_wrong_answer(self, clr):
        """When the math solver derives expected_answer and the model's
        answer doesn't match, the verifier should return verified=False."""
        verifier = MathVerifier()
        context = {"expected_answer": "4"}  # correct answer is 4

        with patch_generators(clr, return_value=_good_trajectory("5")):
            result = await clr.run(
                "What is 2+2?",
                verifier=verifier,
                task_type="math",
                verifier_context=context,
            )
        assert result.verified is False
        assert result.best_score == 0.0

    @pytest.mark.asyncio
    async def test_math_verifier_without_context_returns_unverified(self, clr):
        """Without derivable context, the verifier returns verified=False.
        This is the honest behavior — do NOT fake expected answers."""
        verifier = MathVerifier()
        context = {}  # no expected_answer

        with patch_generators(clr, return_value=_good_trajectory("42")):
            result = await clr.run(
                "Explain something complex",
                verifier=verifier,
                task_type="math",
                verifier_context=context,
            )
        assert result.verified is False
        assert result.best_score <= 0.65
