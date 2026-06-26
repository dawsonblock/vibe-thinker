"""Pytest tests for structured routing output."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hybrid_orchestrator import HybridReasoningOrchestrator
from vibe_clr_async import CLRResult
from verifiers.base import VerificationResult


@pytest.fixture
def orch():
    """Orchestrator with keyword routing (no embedding deps needed)."""
    return HybridReasoningOrchestrator(
        vibe_endpoint="http://localhost:0",
        generalist_endpoint="http://localhost:0",
        use_clr=False,
        use_embedding_router=False,
        use_clr_cache=False,
        use_trajectory_store=False,
    )


class TestStructuredRouting:
    def test_math_query(self, orch):
        decision = orch.route_structured("Solve the recurrence a_{n+1} = a_n^2 - a_n + 1")
        assert decision["route"] == "specialist"
        assert decision["task_type"] == "math"
        assert "deterministic_check" in decision["requires_tools"]
        assert decision["requires_model"] is True
        assert "reason" in decision
        assert "confidence" in decision

    def test_code_query(self, orch):
        decision = orch.route_structured("Write a Python function to sort a list efficiently")
        assert decision["route"] in ("specialist", "hybrid")
        assert decision["task_type"] == "code"
        assert "python_exec" in decision["requires_tools"]

    def test_conversation_query(self, orch):
        decision = orch.route_structured("Explain the history of the Riemann Hypothesis")
        assert decision["route"] == "generalist"
        assert decision["task_type"] == "conversation"
        assert decision["requires_tools"] == []

    def test_summarization_query(self, orch):
        decision = orch.route_structured("Summarize the key ideas in The Selfish Gene")
        assert decision["task_type"] == "summarization"

    def test_unknown_query(self, orch):
        decision = orch.route_structured("xyzzy foobar quux")
        assert decision["task_type"] == "unknown"
        assert decision["requires_human_review"] is True

    def test_low_confidence_triggers_human_review(self, orch):
        decision = orch.route_structured("xyzzy foobar quux")
        assert decision["requires_human_review"] is True

    def test_high_confidence_no_human_review(self, orch):
        decision = orch.route_structured("Solve the recurrence relation step by step")
        assert decision["requires_human_review"] is False

    def test_decision_has_all_fields(self, orch):
        decision = orch.route_structured("Calculate the integral of x^2")
        required_fields = {"route", "confidence", "task_type", "requires_tools",
                          "requires_model", "requires_human_review", "reason"}
        assert required_fields.issubset(decision.keys())


class TestRoutingFalsePositives:
    """Tests for false-positive routing that the v0.3 hardening fixes.

    "code of conduct" should NOT route to code.
    "sum of human knowledge" should NOT route to math.
    """

    def test_code_of_conduct_not_programming(self, orch):
        decision = orch.route_structured("What is a code of conduct?")
        assert decision["task_type"] != "code"

    def test_code_of_ethics_not_programming(self, orch):
        decision = orch.route_structured("Explain the code of ethics for engineers")
        assert decision["task_type"] != "code"

    def test_dress_code_not_programming(self, orch):
        decision = orch.route_structured("What is the dress code for the event?")
        assert decision["task_type"] != "code"

    def test_building_code_not_programming(self, orch):
        decision = orch.route_structured("Does this meet the building code?")
        assert decision["task_type"] != "code"

    def test_sum_of_human_knowledge_not_math(self, orch):
        decision = orch.route_structured("The sum of human knowledge is vast")
        assert decision["task_type"] != "math"

    def test_world_series_not_math(self, orch):
        decision = orch.route_structured("Who won the World Series in 2024?")
        assert decision["task_type"] != "math"

    def test_compute_sum_routes_math(self, orch):
        decision = orch.route_structured("Compute the sum of 1 + 2 + 3 + 4 + 5")
        assert decision["task_type"] == "math"

    def test_debug_python_routes_code(self, orch):
        decision = orch.route_structured("Debug this Python function for me")
        assert decision["task_type"] == "code"

    def test_leetcode_routes_code(self, orch):
        decision = orch.route_structured("Solve LeetCode hard: two sum problem")
        assert decision["task_type"] == "code"

    def test_solve_equation_routes_math(self, orch):
        decision = orch.route_structured("Solve the equation 2x + 3 = 7")
        assert decision["task_type"] == "math"

    def test_area_code_not_programming(self, orch):
        decision = orch.route_structured("What is the area code for New York?")
        assert decision["task_type"] != "code"

    def test_legal_code_not_programming(self, orch):
        decision = orch.route_structured("Explain the legal code for property rights")
        assert decision["task_type"] != "code"


class TestRouteClassification:
    """Tests that actual route (specialist/generalist/hybrid) agrees with
    task_type. The route must NOT send generalist tasks to specialist CLR."""

    def test_code_of_conduct_routes_generalist(self, orch):
        decision = orch.route_structured("What is a code of conduct?")
        assert decision["route"] == "generalist"

    def test_code_of_ethics_routes_generalist(self, orch):
        decision = orch.route_structured("Explain the code of ethics for engineers")
        assert decision["route"] == "generalist"

    def test_dress_code_routes_generalist(self, orch):
        decision = orch.route_structured("What is the dress code for the event?")
        assert decision["route"] == "generalist"

    def test_sum_of_human_knowledge_routes_generalist_or_hybrid(self, orch):
        decision = orch.route_structured("The sum of human knowledge is vast")
        assert decision["route"] in {"generalist", "hybrid"}
        assert decision["route"] != "specialist"

    def test_world_series_routes_generalist_or_hybrid(self, orch):
        decision = orch.route_structured("Who won the World Series in 2024?")
        assert decision["route"] in {"generalist", "hybrid"}

    def test_compute_sum_routes_specialist(self, orch):
        decision = orch.route_structured("Compute the sum of 1 + 2 + 3 + 4 + 5")
        assert decision["route"] == "specialist"

    def test_debug_python_routes_specialist(self, orch):
        decision = orch.route_structured("Debug this Python function for me")
        assert decision["route"] == "specialist"

    def test_solve_equation_routes_specialist(self, orch):
        decision = orch.route_structured("Solve the equation 2x + 3 = 7")
        assert decision["route"] == "specialist"

    def test_task_type_and_route_agree_for_math(self, orch):
        """If task_type is math, route must be specialist."""
        decision = orch.route_structured("Calculate the integral of x^2")
        if decision["task_type"] == "math":
            assert decision["route"] == "specialist"

    def test_task_type_and_route_agree_for_conversation(self, orch):
        """If task_type is conversation, route must be generalist."""
        decision = orch.route_structured("Explain quantum mechanics in simple terms")
        if decision["task_type"] == "conversation":
            assert decision["route"] == "generalist"

    def test_recurrence_query_routes_math_specialist(self, orch):
        """The canonical recurrence query from test_full_stack.py must route
        to specialist with task_type=math, not hybrid/unknown."""
        q = "Solve this step by step: a_1=2, a_{n+1}=a_n^2 - a_n + 1. Find a_5."
        decision = orch.route_structured(q)
        assert decision["task_type"] == "math", \
            f"Expected math, got {decision['task_type']}"
        assert decision["route"] == "specialist", \
            f"Expected specialist, got {decision['route']}"

    def test_indexed_variable_routes_math(self, orch):
        """Queries with indexed variable notation should route to math."""
        decision = orch.route_structured("Given a_1=3, find a_4")
        assert decision["task_type"] == "math"

    def test_find_a5_routes_math(self, orch):
        """'find a_5' pattern should trigger math intent."""
        decision = orch.route_structured("Find a_5 where a_1=1")
        assert decision["task_type"] == "math"


class TestCodeSpecialistRouting:
    """Tests for the optional dedicated code-specialist endpoint (ruvltra).

    When code_specialist_endpoint is configured, code tasks route to it.
    With a code_verifier (default), the multi-candidate sandbox-verified loop
    runs: generalist writes tests, ruvltra generates N candidates, CodeVerifier
    picks the winner. Math/reasoning still uses the VibeThinker CLR path.
    """

    def test_endpoint_defaults_to_none(self, orch):
        assert orch.code_specialist_endpoint is None

    def test_endpoint_stored_and_stripped(self):
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082/",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        assert o.code_specialist_endpoint == "http://127.0.0.1:8082"

    def test_code_candidates_default(self):
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False, use_embedding_router=False, use_clr_cache=False,
            use_trajectory_store=False,
        )
        assert o.code_candidates == 6  # default raised for fast 0.5B models

    @pytest.mark.asyncio
    async def test_code_task_verified_loop_first_candidate_passes(self):
        """Code task -> multi-candidate loop -> first candidate passes verification."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=3,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        o._call_code_specialist = AsyncMock(return_value="```python\ndef square(n): return n*n\n```")
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=True, score=1.0, method="unit_tests",
        ))
        o._run_clr_with_cache = AsyncMock(side_effect=AssertionError("CLR should not run for code"))

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_verified"
        assert result.clr_score == 1.0
        assert "ruvltra" in result.specialist_used
        assert "def square" in result.final_answer
        o._run_clr_with_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_code_task_verified_loop_no_candidate_passes(self):
        """All candidates fail verification -> unverified, score 0.0 (fail-closed)."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=2,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        o._call_code_specialist = AsyncMock(return_value="```python\ndef square(n): return n+1\n```")
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests", error="assertion failed",
        ))

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        assert result.raw_traces["verified"] is False

    @pytest.mark.asyncio
    async def test_code_task_no_test_spec_falls_back_unverified(self):
        """Generalist can't produce tests -> single-candidate unverified, score 0.0."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value=None)
        o._call_code_specialist = AsyncMock(return_value="def square(n): return n*n")

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        o._call_code_specialist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_code_task_no_verifier_plain_generation(self):
        """With code_verifier=None, code tasks use plain single-candidate generation."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_verifier=None,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._call_code_specialist = AsyncMock(return_value="def sort(l): return sorted(l)")
        o._run_clr_with_cache = AsyncMock(side_effect=AssertionError("CLR should not run for code"))

        result = await o.run("Write a Python function to sort a list efficiently")
        assert result.route_taken == "code_specialist"
        assert "ruvltra" in result.specialist_used
        assert result.final_answer == "def sort(l): return sorted(l)"
        o._call_code_specialist.assert_awaited_once()
        o._run_clr_with_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_math_task_does_not_use_code_specialist(self):
        """Math tasks must stay on the VibeThinker CLR path, not ruvltra."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._call_code_specialist = AsyncMock(side_effect=AssertionError("ruvltra should not run for math"))
        o._generate_test_spec = AsyncMock(side_effect=AssertionError("test spec should not run for math"))
        fake_clr = CLRResult(best_answer="42", best_score=0.9, best_raw_trace="")
        o._run_clr_with_cache = AsyncMock(return_value=(fake_clr, False))

        result = await o.run("Solve the equation 2x + 3 = 7")
        assert result.route_taken != "code_specialist"
        assert result.final_answer == "42"
        o._call_code_specialist.assert_not_awaited()
        o._run_clr_with_cache.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_code_without_specialist_falls_back_to_clr(self):
        """With no code_specialist_endpoint, code tasks use the normal CLR path."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint=None,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        fake_clr = CLRResult(best_answer="code here", best_score=0.8, best_raw_trace="")
        o._run_clr_with_cache = AsyncMock(return_value=(fake_clr, False))

        result = await o.run("Write a Python function to sort a list efficiently")
        assert result.route_taken != "code_specialist"
        o._run_clr_with_cache.assert_awaited_once()

    def test_extract_python_block_with_fence(self):
        text = "Here:\n```python\ndef foo(): pass\n```\nDone."
        assert HybridReasoningOrchestrator._extract_python_block(text) == "def foo(): pass"

    def test_extract_python_block_without_fence(self):
        assert HybridReasoningOrchestrator._extract_python_block("def foo(): pass") == "def foo(): pass"

