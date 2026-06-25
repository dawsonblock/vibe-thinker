"""Pytest tests for structured routing output."""

import pytest

from hybrid_orchestrator import HybridReasoningOrchestrator


@pytest.fixture
def orch():
    """Orchestrator with keyword routing (no embedding deps needed)."""
    return HybridReasoningOrchestrator(
        vibe_endpoint="http://localhost:0",
        generalist_endpoint="http://localhost:0",
        use_clr=False,
        use_embedding_router=False,
        use_clr_cache=False,
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

