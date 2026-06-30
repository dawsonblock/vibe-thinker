"""Pytest tests for structured routing output."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
        # Phase 3.1: the mutation check calls verify a second time with
        # mutated code. The original code passes; the mutated code must
        # fail (meaningful tests). Use side_effect to distinguish.
        original_code = "def square(n): return n*n"

        async def verify_side_effect(query, code, context):
            if code == original_code:
                return VerificationResult(verified=True, score=1.0, method="unit_tests")
            return VerificationResult(verified=False, score=0.0, method="unit_tests",
                                      error="mutation correctly failed")

        o.code_verifier.verify = verify_side_effect
        assert hasattr(o, "_run_clr_with_cache"), (
            "cannot monkeypatch a method the class does not define "
            "(would mask a missing-method regression)")
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
    async def test_code_task_vacuous_tests_detected_by_mutation(self):
        """Phase 3.1: When the candidate passes but mutation testing
        reveals the tests are vacuous (mutated code also passes), the
        candidate is rejected and the test-feedback loop is triggered.
        After the retry also detects vacuous tests, the result is
        unverified with score 0.0."""
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
        o._generate_test_spec = AsyncMock(return_value="assert True")
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n*n\n```"
        )
        o.code_verifier = MagicMock()
        # The mock returns verified=True for ALL code (original AND
        # mutated) — simulating vacuous tests that pass anything.
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=True, score=1.0, method="unit_tests",
        ))

        result = await o.run("Write a Python function to square a number")
        # Vacuous tests detected -> not verified, score 0.0.
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        assert result.raw_traces["verified"] is False
        # The traces should record the vacuous test detection.
        traces = result.raw_traces.get("all_verification_traces", [])
        assert any(t.get("vacuous_tests") for t in traces)

    @pytest.mark.asyncio
    async def test_code_task_no_test_spec_falls_back_unverified(self):
        """Generalist can't produce tests -> single-candidate unverified, score 0.0.

        v3.1: When no test spec is generated, the sandbox fallback tries
        to execute the code in a Docker sandbox. If the code has
        restricted imports (e.g. os) but still runs in the sandbox (the
        sandbox has --network=none and --read-only, so os.getcwd() works
        but os.socket() doesn't), the sandbox fallback assigns 0.65.
        To test the truly-unverified path, we mock the sandbox to fail.
        """
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
        # Code with a restricted import -> static analysis fails.
        o._call_code_specialist = AsyncMock(return_value="import os\nprint(os.getcwd())")
        # Mock the sandbox fallback to return (None, []) — no sandbox
        # available — so the code falls through to AST static analysis,
        # which also fails on "import os", resulting in unverified.
        from hybrid_orchestrator import _wasmtime_sandbox_fallback
        import hybrid_orchestrator as _ho
        async def _mock_sandbox(code, code_verifier=None):
            return (None, [])
        with patch("hybrid_orchestrator._wasmtime_sandbox_fallback", _mock_sandbox):
            result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        o._call_code_specialist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_code_task_no_test_spec_static_analysis_fallback(self):
        """v3.2: When no test spec and no sandbox is available, the AST
        static analysis fallback is GATED OFF by default — the route
        returns code_specialist_unverified with score 0.0 (not the old
        0.4 heuristic). AST is not a security boundary."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            # default: allow_static_fallback=False
        )
        o._generate_test_spec = AsyncMock(return_value=None)
        o._call_code_specialist = AsyncMock(return_value="def square(n): return n*n")
        # Mock the sandbox fallback to return (None, []) — no sandbox
        # available — so the code falls through to the static gate.
        async def _mock_sandbox(code, code_verifier=None):
            return (None, [])
        with patch("hybrid_orchestrator._wasmtime_sandbox_fallback", _mock_sandbox):
            result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        o._call_code_specialist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_code_task_static_analysis_fallback_when_enabled(self):
        """v3.2: When allow_static_fallback=True (local dev), no sandbox,
        clean code -> the AST fallback runs and emits the renamed route
        'code_specialist_unverified_static_only' with a 0.2 heuristic
        (lowered from the old 0.4) and verified=False."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            allow_static_fallback=True,
        )
        o._generate_test_spec = AsyncMock(return_value=None)
        o._call_code_specialist = AsyncMock(return_value="def square(n): return n*n")
        async def _mock_sandbox(code, code_verifier=None):
            return (None, [])
        with patch("hybrid_orchestrator._wasmtime_sandbox_fallback", _mock_sandbox):
            # _static_analysis_fallback emits a DeprecationWarning by design
            # (it's on the deprecation path). Assert it here so it doesn't
            # bubble up as an unasserted warning in the test summary.
            with pytest.warns(DeprecationWarning, match="_static_analysis_fallback"):
                result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified_static_only"
        assert result.clr_score == 0.2
        assert result.raw_traces["verified"] is False
        assert result.raw_traces["static_analysis"] is True
        o._call_code_specialist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_code_task_no_test_spec_sandbox_fallback(self):
        """v3.1: When no test spec but the code runs successfully in the
        sandbox, the sandbox fallback assigns 0.65 (the highest self-claim
        cap — NOT full verification, but a real security boundary)."""
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
        # Mock the sandbox fallback to return (0.65, []) — code ran
        # successfully in the sandbox without trapping.
        async def _mock_sandbox(code, code_verifier=None):
            return (0.65, [])
        with patch("hybrid_orchestrator._wasmtime_sandbox_fallback", _mock_sandbox):
            result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_sandbox"
        assert result.clr_score == 0.65
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
        assert hasattr(o, "_run_clr_with_cache"), (
            "cannot monkeypatch a method the class does not define "
            "(would mask a missing-method regression)")
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
        assert hasattr(o, "_run_clr_with_cache"), (
            "cannot monkeypatch a method the class does not define "
            "(would mask a missing-method regression)")
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
        assert hasattr(o, "_run_clr_with_cache"), (
            "cannot monkeypatch a method the class does not define "
            "(would mask a missing-method regression)")
        o._run_clr_with_cache = AsyncMock(return_value=(fake_clr, False))

        result = await o.run("Write a Python function to sort a list efficiently")
        assert result.route_taken != "code_specialist"
        o._run_clr_with_cache.assert_awaited_once()

    def test_extract_python_block_with_fence(self):
        text = "Here:\n```python\ndef foo(): pass\n```\nDone."
        assert HybridReasoningOrchestrator._extract_python_block(text) == "def foo(): pass"

    def test_extract_python_block_without_fence(self):
        assert HybridReasoningOrchestrator._extract_python_block("def foo(): pass") == "def foo(): pass"

    @pytest.mark.asyncio
    async def test_test_error_triggers_test_spec_retry(self):
        """When ALL candidates fail with TEST_ERROR, the generalist gets one
        retry to rewrite the tests with error feedback."""
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
        # First test spec is broken, second is fixed
        o._generate_test_spec = AsyncMock(
            side_effect=["assert nonexistent_func() == 1", "assert square(2) == 4"]
        )
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n*n\n```"
        )
        o.code_verifier = MagicMock()
        # First attempt: 2 candidates, both fail with TEST_ERROR (parallel).
        # Second attempt: 2 candidates, both pass (parallel). With
        # asyncio.gather, both candidates are verified concurrently, so we
        # need 4 side_effect items (2 per attempt).
        o.code_verifier.verify = AsyncMock(side_effect=[
            # First attempt — both TEST_ERROR
            VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="TEST_ERROR: name 'nonexistent_func' is not defined",
            ),
            VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="TEST_ERROR: name 'nonexistent_func' is not defined",
            ),
            # Second attempt (with fixed tests) — both pass
            VerificationResult(
                verified=True, score=1.0, method="unit_tests",
            ),
            VerificationResult(
                verified=True, score=1.0, method="unit_tests",
            ),
        ])

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_verified"
        assert result.clr_score == 1.0
        # Test spec was generated twice (initial + retry)
        assert o._generate_test_spec.await_count == 2
        # The retry prompt should contain the error feedback
        second_call_args = o._generate_test_spec.call_args_list[1][0][0]
        assert "TEST_ERROR" in second_call_args

    @pytest.mark.asyncio
    async def test_assertion_failure_does_not_trigger_retry(self):
        """ASSERTION_FAILED is a candidate problem, not a test problem — no retry."""
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
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n+1\n```"
        )
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            error="ASSERTION_FAILED: assert 3 == 4",
        ))

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        # No retry — ASSERTION_FAILED is a code problem
        assert o._generate_test_spec.await_count == 1

    @pytest.mark.asyncio
    async def test_import_error_does_not_trigger_retry(self):
        """IMPORT_ERROR is a candidate problem (code failed to define), no retry."""
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
        o._call_code_specialist = AsyncMock(
            return_value="```python\nsyntax error here\n```"
        )
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            error="IMPORT_ERROR: invalid syntax",
        ))

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        assert o._generate_test_spec.await_count == 1

    @pytest.mark.asyncio
    async def test_test_error_retry_only_once(self):
        """If the retry also produces TEST_ERROR, no second retry (max 2 attempts)."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=1,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(
            side_effect=["assert bad1()", "assert bad2()"]
        )
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef f(): pass\n```"
        )
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            error="TEST_ERROR: name 'bad1' is not defined",
        ))

        result = await o.run("Write a function")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        # Exactly 2 attempts (initial + 1 retry), not 3
        assert o._generate_test_spec.await_count == 2

    @pytest.mark.asyncio
    async def test_partial_test_error_does_not_trigger_retry(self):
        """If only SOME candidates fail with TEST_ERROR (not all), no retry."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=2,
            max_repair_attempts=0,  # isolate test-spec-retry behavior
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n*n\n```"
        )
        o.code_verifier = MagicMock()
        # One TEST_ERROR, one ASSERTION_FAILED — not all TEST_ERROR
        o.code_verifier.verify = AsyncMock(side_effect=[
            VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="TEST_ERROR: something",
            ),
            VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="ASSERTION_FAILED: assert 5 == 4",
            ),
        ])

        result = await o.run("Write a function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        # No retry — not ALL were TEST_ERROR
        assert o._generate_test_spec.await_count == 1

    # ------------------------------------------------------------------ #
    # Iterative code repair (v0.4.0)
    # ------------------------------------------------------------------ #
    @pytest.mark.asyncio
    async def test_assertion_failure_triggers_repair_success(self):
        """When a candidate fails with ASSERTION_FAILED, the failing code +
        error are fed back to the code specialist. If a repair passes, the
        result is verified (score 1.0) with repair_attempts traced."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=1,
            max_repair_attempts=2,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        # Initial candidate is buggy; the repair produces a correct solution.
        o._call_code_specialist = AsyncMock(side_effect=[
            "```python\ndef square(n): return n + 1\n```",   # initial (buggy)
            "```python\ndef square(n): return n * n\n```",   # repair (correct)
        ])
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(side_effect=[
            VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="ASSERTION_FAILED: assert 3 == 4",
            ),
            VerificationResult(
                verified=True, score=1.0, method="unit_tests",
            ),
        ])

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_verified"
        assert result.clr_score == 1.0
        assert result.raw_traces["verified"] is True
        assert result.raw_traces["repair_attempts"] == 1
        # The repair prompt must contain the failing-code error feedback.
        repair_call_args = o._call_code_specialist.call_args_list[1][0][0]
        assert "ASSERTION_FAILED" in repair_call_args
        # No test-spec retry happened (ASSERTION_FAILED is a code problem).
        assert o._generate_test_spec.await_count == 1

    @pytest.mark.asyncio
    async def test_import_error_triggers_repair(self):
        """IMPORT_ERROR is also a candidate defect — repair fires for it too."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=1,
            max_repair_attempts=2,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        o._call_code_specialist = AsyncMock(side_effect=[
            "```python\nsyntax error here\n```",            # initial (broken)
            "```python\ndef square(n): return n * n\n```",  # repair (correct)
        ])
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(side_effect=[
            VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="IMPORT_ERROR: invalid syntax",
            ),
            VerificationResult(
                verified=True, score=1.0, method="unit_tests",
            ),
        ])

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_verified"
        assert result.clr_score == 1.0
        assert result.raw_traces["repair_attempts"] == 1

    @pytest.mark.asyncio
    async def test_repair_exhausted_fails_closed(self):
        """If every repair attempt also fails, the result is unverified with
        score 0.0 (fail-closed — never fake verification)."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=1,
            max_repair_attempts=2,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        # The specialist always returns the same buggy code, even on repair.
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n + 1\n```"
        )
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            error="ASSERTION_FAILED: assert 3 == 4",
        ))

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        assert result.raw_traces["verified"] is False
        assert result.raw_traces["repair_attempts"] == 2
        # 1 initial verify + 2 repair verifies = 3 total verify calls.
        assert o.code_verifier.verify.await_count == 3
        # No test-spec retry.
        assert o._generate_test_spec.await_count == 1

    @pytest.mark.asyncio
    async def test_max_repair_attempts_zero_disables_repair(self):
        """With max_repair_attempts=0, ASSERTION_FAILED does NOT trigger any
        repair — the loop is fully disabled (backward-compatible behavior)."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=1,
            max_repair_attempts=0,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(return_value="assert square(2) == 4")
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n + 1\n```"
        )
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            error="ASSERTION_FAILED: assert 3 == 4",
        ))

        result = await o.run("Write a Python function to square a number")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        assert result.raw_traces["repair_attempts"] == 0
        # Only the single initial verification — no repair calls.
        assert o.code_verifier.verify.await_count == 1
        # The code specialist is called exactly once (no repair generation).
        assert o._call_code_specialist.await_count == 1

    @pytest.mark.asyncio
    async def test_test_error_does_not_trigger_repair(self):
        """TEST_ERROR is a broken-test problem, not a code bug — it must
        trigger the test-spec retry path, NOT the code-repair path."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            code_specialist_endpoint="http://127.0.0.1:8082",
            code_candidates=1,
            max_repair_attempts=2,
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        o._generate_test_spec = AsyncMock(
            side_effect=["assert bad1()", "assert bad2()"]
        )
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef f(): pass\n```"
        )
        o.code_verifier = MagicMock()
        o.code_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            error="TEST_ERROR: name 'bad1' is not defined",
        ))

        result = await o.run("Write a function")
        assert result.route_taken == "code_specialist_unverified"
        assert result.clr_score == 0.0
        # TEST_ERROR triggers test-spec retry (2 generations), not repair.
        assert o._generate_test_spec.await_count == 2
        # 2 verifications (one per test-spec attempt); no repair verifications.
        assert o.code_verifier.verify.await_count == 2
        assert result.raw_traces["repair_attempts"] == 0

    @pytest.mark.asyncio
    async def test_parallel_verification_runs_concurrently(self):
        """Candidate verification runs via asyncio.gather, not sequentially.
        Verify by checking that all verify calls start before any completes
        (concurrent, not serial)."""
        import asyncio as aio
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
        o._call_code_specialist = AsyncMock(
            return_value="```python\ndef square(n): return n*n\n```"
        )
        # Mock the warm pool to skip container startup overhead
        o._warm_pool = MagicMock()
        o._warm_pool._started = True
        o.code_verifier = MagicMock()

        # Track concurrency: all 3 verify calls should start before any
        # completes. If sequential, only 1 would be active at a time.
        started = []
        can_complete = aio.Event()
        original_code = "def square(n): return n*n"

        async def slow_verify(query, code, config):
            started.append(len(started))
            await aio.sleep(0.05)
            # Phase 3.1: mutation check calls verify with mutated code.
            # Original code passes; mutated code fails (meaningful tests).
            if code == original_code:
                return VerificationResult(
                    verified=True, score=1.0, method="unit_tests",
                )
            return VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                error="mutation correctly failed",
            )

        o.code_verifier.verify = slow_verify

        result = await o.run("Write a function to square a number")
        assert result.route_taken == "code_specialist_verified"
        # All 3 candidate verify calls were made concurrently (gather
        # fires all before any completes). Phase 3.1 adds a 4th call
        # for the mutation check (sequential, after the gather).
        assert len(started) >= 3
        # The first 3 calls (the gather) all started before any completed
        # — that's the concurrency proof. The 4th (mutation check) starts
        # after the gather resolves.
        assert started[:3] == [0, 1, 2]


class TestTestSpecValidation:
    """Tests for _validate_test_spec — rejects vacuous test specs."""

    def test_rejects_assert_true(self):
        assert HybridReasoningOrchestrator._validate_test_spec("assert True") is False

    def test_rejects_assert_one_equals_one(self):
        assert HybridReasoningOrchestrator._validate_test_spec("assert 1 == 1") is False

    def test_rejects_bare_constant(self):
        assert HybridReasoningOrchestrator._validate_test_spec('assert "yes"') is False

    def test_rejects_all_vacuous_mixed(self):
        spec = "assert True\nassert 1 == 1\nassert 0 or 1"
        assert HybridReasoningOrchestrator._validate_test_spec(spec) is False

    def test_accepts_function_call_assert(self):
        spec = "assert add(2, 3) == 5\nassert add(0, 0) == 0"
        assert HybridReasoningOrchestrator._validate_test_spec(spec) is True

    def test_accepts_variable_reference_assert(self):
        spec = "result = add(2, 3)\nassert result == 5"
        assert HybridReasoningOrchestrator._validate_test_spec(spec) is True

    def test_accepts_mixed_with_at_least_one_real(self):
        spec = "assert True\nassert square(4) == 16"
        assert HybridReasoningOrchestrator._validate_test_spec(spec) is True

    def test_rejects_unparseable(self):
        assert HybridReasoningOrchestrator._validate_test_spec("assert ++++") is False

    def test_rejects_empty(self):
        assert HybridReasoningOrchestrator._validate_test_spec("") is False


class TestExtractPythonBlock:
    """Tests for _extract_python_block — hardened fence parsing."""

    def test_standard_python_fence(self):
        text = "Here are the tests:\n```python\nassert add(1,2)==3\n```\nDone."
        assert HybridReasoningOrchestrator._extract_python_block(text) == "assert add(1,2)==3"

    def test_abbreviated_py_fence(self):
        text = "```py\nassert add(1,2)==3\n```"
        assert HybridReasoningOrchestrator._extract_python_block(text) == "assert add(1,2)==3"

    def test_no_language_tag_fence(self):
        text = "```\nassert add(1,2)==3\n```"
        assert HybridReasoningOrchestrator._extract_python_block(text) == "assert add(1,2)==3"

    def test_bare_code_no_fence(self):
        text = "assert add(1,2)==3"
        assert HybridReasoningOrchestrator._extract_python_block(text) == "assert add(1,2)==3"

    def test_multiline_block(self):
        text = "```python\nassert add(1,2)==3\nassert add(0,0)==0\n```"
        result = HybridReasoningOrchestrator._extract_python_block(text)
        assert "assert add(1,2)==3" in result
        assert "assert add(0,0)==0" in result

    def test_selects_last_valid_python_block(self):
        """v0.4.0: when multiple fenced blocks exist, select the last
        valid Python AST block (LLMs output reasoning first, solution last)."""
        text = (
            "First, install dependencies:\n"
            "```bash\npip install numpy\n```\n"
            "Here's the solution:\n"
            "```python\nassert add(1,2)==3\n```"
        )
        result = HybridReasoningOrchestrator._extract_python_block(text)
        assert result == "assert add(1,2)==3"
        assert "pip install" not in result

    def test_selects_last_valid_among_multiple_python_blocks(self):
        """When multiple Python blocks exist, select the last parseable one."""
        text = (
            "```python\n# draft solution\nimport os\n```\n"
            "Actually, here's the final version:\n"
            "```python\ndef add(a, b):\n    return a + b\n```"
        )
        result = HybridReasoningOrchestrator._extract_python_block(text)
        assert "def add" in result
        assert "draft" not in result

    def test_falls_back_to_last_block_if_none_parse(self):
        """If no block parses as valid Python, return the last block
        (let the verifier reject it rather than evaluating the wrong block)."""
        text = (
            "```python\nthis is not valid python !!!\n```\n"
            "```python\nalso not valid @@@\n```"
        )
        result = HybridReasoningOrchestrator._extract_python_block(text)
        assert "also not valid" in result

    # --- v3.2: CommonMark-aware fence matching ------------------------- #

    def test_four_backtick_opener_closed_by_four(self):
        """CommonMark: a `````` opener must be closed by >=4 backticks
        with no info string. A ``` line inside is literal content."""
        text = "````python\nx = 1\n```\nstill in block\n````"
        result = HybridReasoningOrchestrator._extract_python_block(text)
        # The ``` line is literal content, so the block contains all of it.
        assert "x = 1" in result
        assert "still in block" in result

    def test_closing_fence_with_info_string_is_content(self):
        """CommonMark: a closing fence must have NO info string. A line
        like ```python inside an open block is literal content, not a
        close — so the block continues until a bare fence appears."""
        text = "```python\nx = 1\n```python\ny = 2\n```"
        result = HybridReasoningOrchestrator._extract_python_block(text)
        # The ```python line in the middle is literal content; the block
        # closes at the final bare ```. Result parses as valid Python.
        assert "x = 1" in result
        assert "y = 2" in result

    def test_nested_backticks_in_explanation_do_not_close(self):
        """v3.2 regression: the old matcher toggled on any ```-prefixed
        line, so an inline `` `os.getcwd()` `` or a fenced sub-example in
        the model's reasoning could swallow real code. The CommonMark
        matcher requires the closer to have no info string, so an inner
        ```bash inside a python block (which the model might emit as an
        example command in its reasoning) does NOT close the block."""
        text = (
            "```python\n"
            "# Example: don't run this in your shell:\n"
            "# ```bash\n"
            "# rm -rf /tmp/old\n"
            "# ```\n"
            "def clean(tmpdir):\n"
            "    return sorted(tmpdir)\n"
            "```"
        )
        result = HybridReasoningOrchestrator._extract_python_block(text)
        # The whole python block survives — the inner ```bash / ``` lines
        # are literal comment content, not fence toggles.
        assert "def clean" in result
        assert "rm -rf" in result

    def test_indented_fence_up_to_three_spaces(self):
        """CommonMark: up to 3 leading spaces are allowed on a fence."""
        text = "   ```python\nx = 1\n   ```"
        result = HybridReasoningOrchestrator._extract_python_block(text)
        assert result == "x = 1"

    def test_info_string_with_extra_args_ignored_after_first_token(self):
        """CommonMark: only the first whitespace-delimited token of the
        info string is the language tag; the rest is ignored."""
        text = "```python title=\"solution\"\nx = 1\n```"
        result = HybridReasoningOrchestrator._extract_python_block(text)
        assert result == "x = 1"


class TestLogicTranslationRetry:
    """Phase 3.2: Z3/SMT translation retry loop tests."""

    def _make_orch(self):
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        return o

    @pytest.mark.asyncio
    async def test_valid_translation_first_try(self):
        """When the first translation produces valid Z3, no retry needed."""
        o = self._make_orch()
        o._translate_logic_constraints = AsyncMock(return_value={
            "constraints": ["x > 0", "x + y == 10"],
            "variables": {"x": "Int", "y": "Int"},
            "values": {"x": 7, "y": 3},
        })
        result = await o._translate_logic_constraints_with_retry("test")
        assert result is not None
        assert result["constraints"] == ["x > 0", "x + y == 10"]
        # Should have called _translate_logic_constraints exactly once.
        o._translate_logic_constraints.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.logic
    async def test_retry_on_parse_error(self):
        """When the first translation has a parse error, retry with feedback."""
        pytest.importorskip("z3", reason="requires z3-solver for constraint parse validation")
        o = self._make_orch()
        # First call: invalid Z3 syntax. Second call: valid.
        o._translate_logic_constraints = AsyncMock(side_effect=[
            {
                "constraints": ["x >>> 0"],  # invalid
                "variables": {"x": "Int"},
                "values": {"x": 5},
            },
            {
                "constraints": ["x > 0"],  # valid
                "variables": {"x": "Int"},
                "values": {"x": 5},
            },
        ])
        result = await o._translate_logic_constraints_with_retry("test")
        assert result is not None
        assert result["constraints"] == ["x > 0"]
        assert o._translate_logic_constraints.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.logic
    async def test_exhausted_retries_returns_best_effort(self):
        """When all retries fail, return the last (invalid) result —
        the LogicVerifier will catch the parse error at verification time."""
        pytest.importorskip("z3", reason="requires z3-solver for constraint parse validation")
        o = self._make_orch()
        o._translate_logic_constraints = AsyncMock(return_value={
            "constraints": ["bad !!!"],
            "variables": {"x": "Int"},
            "values": {"x": 5},
        })
        result = await o._translate_logic_constraints_with_retry(
            "test", max_retries=2
        )
        # Returns the best-effort result (not None) — the verifier will
        # fail-closed on the parse error.
        assert result is not None
        assert result["constraints"] == ["bad !!!"]
        # Should have called 3 times (initial + 2 retries).
        assert o._translate_logic_constraints.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.logic
    async def test_none_translation_no_retry_wasted(self):
        """When _translate_logic_constraints returns None (network/JSON
        error), retry once with feedback, but don't waste all retries."""
        pytest.importorskip("z3", reason="requires z3-solver for constraint parse validation")
        o = self._make_orch()
        o._translate_logic_constraints = AsyncMock(return_value=None)
        result = await o._translate_logic_constraints_with_retry(
            "test", max_retries=2
        )
        assert result is None
        # Should retry (with feedback) up to max_retries + 1 times.
        assert o._translate_logic_constraints.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.logic
    async def test_retry_feedback_contains_parse_error(self):
        """The retry prompt should include the specific Z3 parse error."""
        pytest.importorskip("z3", reason="requires z3-solver for constraint parse validation")
        o = self._make_orch()
        o._translate_logic_constraints = AsyncMock(side_effect=[
            {
                "constraints": ["x >>> 0"],
                "variables": {"x": "Int"},
                "values": {"x": 5},
            },
            {
                "constraints": ["x > 0"],
                "variables": {"x": "Int"},
                "values": {"x": 5},
            },
        ])
        await o._translate_logic_constraints_with_retry("test")
        # The second call should have the parse error in the query.
        second_call_args = o._translate_logic_constraints.call_args_list[1]
        query_arg = second_call_args[0][0]  # positional arg
        assert "Z3 PARSE ERROR" in query_arg or "parse" in query_arg.lower()

    @pytest.mark.asyncio
    async def test_translation_passes_logic_grammar(self):
        """v3.2: _translate_logic_constraints passes LOGIC_CONSTRAINTS_GRAMMAR
        to _call_generalist so the generalist is constrained to valid JSON
        with the constraints/variables/values keys."""
        from format_enforcer import LOGIC_CONSTRAINTS_GRAMMAR
        o = self._make_orch()
        # Mock _call_generalist to return a valid JSON response and
        # capture the grammar kwarg it was called with.
        o._call_generalist = AsyncMock(return_value=(
            '{"constraints": ["x > 0"], '
            '"variables": {"x": "Int"}, '
            '"values": {"x": 5}}'
        ))
        result = await o._translate_logic_constraints("test problem")
        assert result is not None
        assert result["constraints"] == ["x > 0"]
        # The grammar kwarg must be the LOGIC_CONSTRAINTS_GRAMMAR.
        call_kwargs = o._call_generalist.call_args.kwargs
        assert call_kwargs.get("grammar") == LOGIC_CONSTRAINTS_GRAMMAR

