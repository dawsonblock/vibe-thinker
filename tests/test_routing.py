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
