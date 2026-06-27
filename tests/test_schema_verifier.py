"""Tests for SchemaVerifier and LogicVerifier (Phase 1.3)."""

import json
import pytest

from verifiers.schema_verifier import SchemaVerifier, _YAML_AVAILABLE
from verifiers.logic_verifier import LogicVerifier, _Z3_AVAILABLE
from verifiers.base import VerificationResult


# ---------------------------------------------------------------------- #
# SchemaVerifier
# ---------------------------------------------------------------------- #
class TestSchemaVerifierJSON:
    @pytest.mark.asyncio
    async def test_valid_json_object(self):
        v = SchemaVerifier()
        answer = json.dumps({"name": "Alice", "age": 30})
        result = await v.verify("q", answer, {
            "schema": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                },
            }
        })
        assert result.verified is True
        assert result.score == 1.0
        assert result.method == "schema_validation"

    @pytest.mark.asyncio
    async def test_missing_required_key(self):
        v = SchemaVerifier()
        answer = json.dumps({"name": "Alice"})
        result = await v.verify("q", answer, {
            "schema": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {"name": {"type": "string"},
                               "age": {"type": "integer"}},
            }
        })
        assert result.verified is False
        assert "missing required keys" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type(self):
        v = SchemaVerifier()
        answer = json.dumps({"name": 123, "age": 30})
        result = await v.verify("q", answer, {
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "age": {"type": "integer"}},
            }
        })
        assert result.verified is False
        assert "expected type 'string'" in result.error

    @pytest.mark.asyncio
    async def test_numeric_minimum_violation(self):
        v = SchemaVerifier()
        answer = json.dumps({"age": -5})
        result = await v.verify("q", answer, {
            "schema": {"type": "object",
                       "properties": {"age": {"type": "integer", "minimum": 0}}}
        })
        assert result.verified is False
        assert "minimum" in result.error

    @pytest.mark.asyncio
    async def test_array_items(self):
        v = SchemaVerifier()
        answer = json.dumps([1, 2, 3])
        result = await v.verify("q", answer, {
            "schema": {"type": "array", "items": {"type": "integer"}}
        })
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_array_items_violation(self):
        v = SchemaVerifier()
        answer = json.dumps([1, "two", 3])
        result = await v.verify("q", answer, {
            "schema": {"type": "array", "items": {"type": "integer"}}
        })
        assert result.verified is False
        assert "expected type 'integer'" in result.error

    @pytest.mark.asyncio
    async def test_enum_violation(self):
        v = SchemaVerifier()
        answer = json.dumps("purple")
        result = await v.verify("q", answer, {
            "schema": {"type": "string", "enum": ["red", "green", "blue"]}
        })
        assert result.verified is False
        assert "enum" in result.error

    @pytest.mark.asyncio
    async def test_additional_properties_rejected(self):
        v = SchemaVerifier()
        answer = json.dumps({"name": "Alice", "extra": "no"})
        result = await v.verify("q", answer, {
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        })
        assert result.verified is False
        assert "additional properties" in result.error

    @pytest.mark.asyncio
    async def test_bool_not_integer(self):
        """bool is a subclass of int in Python — must not pass integer check."""
        v = SchemaVerifier()
        answer = json.dumps({"flag": True})
        result = await v.verify("q", answer, {
            "schema": {"type": "object",
                       "properties": {"flag": {"type": "integer"}}}
        })
        assert result.verified is False

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        v = SchemaVerifier()
        result = await v.verify("q", "{not valid json", {"schema": {"type": "object"}})
        assert result.verified is False
        assert "JSON parse error" in result.error

    @pytest.mark.asyncio
    async def test_no_schema_returns_false(self):
        v = SchemaVerifier()
        result = await v.verify("q", "{}", {})
        assert result.verified is False
        assert "no schema" in result.error


class TestSchemaVerifierRegex:
    @pytest.mark.asyncio
    async def test_pattern_match(self):
        v = SchemaVerifier()
        result = await v.verify("q", "ABC123", {"pattern": r"[A-Z]+\d+"})
        assert result.verified is True
        assert result.evidence["pattern_matched"] == r"[A-Z]+\d+"

    @pytest.mark.asyncio
    async def test_pattern_no_match(self):
        v = SchemaVerifier()
        result = await v.verify("q", "abc", {"pattern": r"[A-Z]+\d+"})
        assert result.verified is False
        assert "does not match pattern" in result.error

    @pytest.mark.asyncio
    async def test_invalid_regex(self):
        v = SchemaVerifier()
        result = await v.verify("q", "abc", {"pattern": r"[unclosed"})
        assert result.verified is False
        assert "invalid regex pattern" in result.error

    @pytest.mark.asyncio
    async def test_pattern_and_schema_both_must_pass(self):
        v = SchemaVerifier()
        answer = json.dumps({"id": "X123"})
        # Pattern matches but schema fails (id should be integer)
        result = await v.verify("q", answer, {
            "pattern": r'\{.*\}',
            "schema": {"type": "object",
                       "properties": {"id": {"type": "integer"}}},
        })
        assert result.verified is False


class TestSchemaVerifierExpectedKeys:
    @pytest.mark.asyncio
    async def test_expected_keys_present(self):
        v = SchemaVerifier()
        result = await v.verify("q", json.dumps({"a": 1, "b": 2}),
                                 {"expected_keys": ["a", "b"]})
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_expected_keys_missing(self):
        v = SchemaVerifier()
        result = await v.verify("q", json.dumps({"a": 1}),
                                 {"expected_keys": ["a", "b"]})
        assert result.verified is False
        assert "missing required keys" in result.error

    @pytest.mark.asyncio
    async def test_expected_keys_on_non_dict(self):
        v = SchemaVerifier()
        result = await v.verify("q", json.dumps([1, 2]),
                                 {"expected_keys": ["a"]})
        assert result.verified is False
        assert "expected a dict" in result.error


@pytest.mark.skipif(not _YAML_AVAILABLE, reason="PyYAML not installed")
class TestSchemaVerifierYAML:
    @pytest.mark.asyncio
    async def test_valid_yaml(self):
        v = SchemaVerifier()
        answer = "name: Alice\nage: 30\n"
        result = await v.verify("q", answer, {
            "format": "yaml",
            "schema": {"type": "object",
                       "required": ["name", "age"],
                       "properties": {"name": {"type": "string"},
                                      "age": {"type": "integer"}}},
        })
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_invalid_yaml(self):
        v = SchemaVerifier()
        result = await v.verify("q", "name: [unclosed", {
            "format": "yaml",
            "schema": {"type": "object"},
        })
        assert result.verified is False
        assert "YAML parse error" in result.error


# ---------------------------------------------------------------------- #
# LogicVerifier
# ---------------------------------------------------------------------- #
@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
class TestLogicVerifier:
    @pytest.mark.asyncio
    async def test_satisfiable_and_values_match(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=7, y=3", {
            "constraints": ["x > 0", "x + y == 10", "y < x"],
            "variables": {"x": "Int", "y": "Int"},
            "values": {"x": 7, "y": 3},
        })
        assert result.verified is True
        assert result.score == 1.0
        assert result.evidence["satisfiable"] is True

    @pytest.mark.asyncio
    async def test_satisfiable_but_values_violate(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=1, y=9", {
            "constraints": ["x > 0", "x + y == 10", "y < x"],
            "variables": {"x": "Int", "y": "Int"},
            "values": {"x": 1, "y": 9},
        })
        assert result.verified is False
        assert "violate constraints" in result.error
        assert "y < x" in result.evidence["failing_constraints"]

    @pytest.mark.asyncio
    async def test_unsat_constraints(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=5", {
            "constraints": ["x > 10", "x < 5"],
            "variables": {"x": "Int"},
            "values": {"x": 5},
        })
        assert result.verified is False
        assert "UNSAT" in result.error

    @pytest.mark.asyncio
    async def test_no_constraints(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=5", {})
        assert result.verified is False
        assert "no constraints" in result.error

    @pytest.mark.asyncio
    async def test_sat_but_no_values(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=5", {
            "constraints": ["x > 0"],
            "variables": {"x": "Int"},
        })
        assert result.verified is False
        assert "no values provided" in result.error

    @pytest.mark.asyncio
    async def test_real_arithmetic(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=1.5", {
            "constraints": ["x > 0", "x < 2"],
            "variables": {"x": "Real"},
            "values": {"x": 1.5},
        })
        assert result.verified is True


class TestLogicVerifierUnavailable:
    @pytest.mark.skipif(_Z3_AVAILABLE, reason="z3-solver IS installed — test the fail-closed path only when absent")
    @pytest.mark.asyncio
    async def test_z3_unavailable_fail_closed(self):
        v = LogicVerifier()
        result = await v.verify("q", "x=5", {
            "constraints": ["x > 0"],
            "variables": {"x": "Int"},
            "values": {"x": 5},
        })
        assert result.verified is False
        assert result.method == "smt_unavailable"
        assert "z3-solver not installed" in result.error


# ---------------------------------------------------------------------- #
# Logic constraint translation (v0.4.1)
# ---------------------------------------------------------------------- #
class TestLogicConstraintTranslation:
    """Test the _translate_logic_constraints method that translates
    natural-language logic problems into Z3-compatible constraints via
    the generalist model."""

    @pytest.mark.asyncio
    async def test_valid_constraint_translation(self):
        """When the generalist returns valid JSON, it's parsed correctly."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        generalist_response = json.dumps({
            "constraints": ["x > 0", "x + y == 10", "y < x"],
            "variables": {"x": "Int", "y": "Int"},
            "values": {"x": 7, "y": 3},
        })

        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=generalist_response,
        ):
            # Create a minimal orchestrator instance (we only need the
            # _translate_logic_constraints method, not a full init).
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints(
                "If x is positive and x + y = 10 and y < x, what are x and y?"
            )
        assert result is not None
        assert result["constraints"] == ["x > 0", "x + y == 10", "y < x"]
        assert result["variables"] == {"x": "Int", "y": "Int"}
        assert result["values"] == {"x": 7, "y": 3}

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        """When the generalist returns malformed JSON, return None (fail-closed)."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value="This is not JSON at all.",
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints("some query")
        assert result is None

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json_parsed(self):
        """JSON wrapped in markdown code fences is still parsed."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        response = '```json\n{"constraints": ["a > 0"], "variables": {"a": "Int"}, "values": {"a": 5}}\n```'
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints("query")
        assert result is not None
        assert result["constraints"] == ["a > 0"]

    @pytest.mark.asyncio
    async def test_empty_constraints_returns_none(self):
        """When the generalist returns empty constraints, return None."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        response = json.dumps({"constraints": [], "variables": {}, "values": {}})
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints("query")
        assert result is None

    @pytest.mark.asyncio
    async def test_generalist_error_returns_none(self):
        """When the generalist call raises, return None (fail-closed)."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, side_effect=RuntimeError("connection refused"),
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints("query")
        assert result is None

    @pytest.mark.asyncio
    async def test_arithmetic_word_problem(self):
        """Arithmetic word problem: 'I have 3 apples and give 1 away'."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        generalist_response = json.dumps({
            "constraints": ["apples == 3", "given == 1", "remaining == apples - given"],
            "variables": {"apples": "Int", "given": "Int", "remaining": "Int"},
            "values": {"apples": 3, "given": 1, "remaining": 2},
        })
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=generalist_response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints(
                "I have 3 apples and give 1 away. How many do I have?"
            )
        assert result is not None
        assert len(result["constraints"]) == 3
        assert result["values"]["remaining"] == 2

    @pytest.mark.asyncio
    async def test_boolean_logic_problem(self):
        """Boolean logic: 'If raining then wet. Not wet. Is it raining?'"""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        generalist_response = json.dumps({
            "constraints": ["Implies(raining, wet)", "Not(wet)"],
            "variables": {"raining": "Bool", "wet": "Bool"},
            "values": {"raining": 0},
        })
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=generalist_response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints(
                "If it is raining then the ground is wet. The ground is not wet. Is it raining?"
            )
        assert result is not None
        assert "Implies" in result["constraints"][0]
        assert result["variables"]["raining"] == "Bool"

    @pytest.mark.asyncio
    async def test_real_arithmetic_problem(self):
        """Real-valued variables: 'x is between 1.5 and 2.5'."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        generalist_response = json.dumps({
            "constraints": ["x > 1.5", "x < 2.5"],
            "variables": {"x": "Real"},
            "values": {"x": 2.0},
        })
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=generalist_response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints(
                "x is a real number between 1.5 and 2.5. What is x?"
            )
        assert result is not None
        assert result["variables"]["x"] == "Real"
        assert result["values"]["x"] == 2.0

    @pytest.mark.asyncio
    async def test_extra_text_around_json(self):
        """Generalist returns JSON with extra text before/after — still parsed."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        response = (
            'Sure, here is the translation:\n'
            '{"constraints": ["a > 0"], "variables": {"a": "Int"}, "values": {"a": 5}}\n'
            'Hope this helps!'
        )
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints("query")
        assert result is not None
        assert result["constraints"] == ["a > 0"]

    @pytest.mark.asyncio
    async def test_valid_json_but_wrong_z3_syntax(self):
        """Valid JSON with invalid Z3 syntax is still returned (the verifier
        will catch the syntax error — translation doesn't validate Z3)."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from unittest.mock import AsyncMock, patch

        generalist_response = json.dumps({
            "constraints": ["this is not valid z3"],
            "variables": {"x": "Int"},
            "values": {"x": 5},
        })
        with patch.object(
            HybridReasoningOrchestrator, "_call_generalist",
            new_callable=AsyncMock, return_value=generalist_response,
        ):
            orch = HybridReasoningOrchestrator.__new__(HybridReasoningOrchestrator)
            result = await orch._translate_logic_constraints("query")
        # Translation returns the constraints as-is — the Z3 verifier will
        # catch the syntax error and return verified=False.
        assert result is not None
        assert result["constraints"] == ["this is not valid z3"]
