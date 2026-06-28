"""Verifier golden-set regression suite (v3.2).

A curated set of (verifier, query, answer, context, expected_verified)
tuples that exercise each verifier's core logic. This guards against
regressions: when a verifier changes, this suite must still pass. If a
verifier starts accepting a hallucinated answer or rejecting a correct
one, this suite catches it.

The golden set is deliberately small and high-signal — each case
represents a class of error the verifier is supposed to catch or
accept. New cases should be added when a new failure mode is found in
production (with a comment naming the failure mode).

Why this matters (from the critique): the plan adds capabilities (DPO
fine-tuning, multi-turn dialog, federation reputation) that all depend
on the verifiers being trustworthy. Without a golden-set regression
suite, every other improvement is built on sand — a silent verifier
regression could mark hallucinated outputs as verified and poison the
flywheel. This suite is the floor under all of that.
"""

import pytest

from verifiers.math_verifier import MathVerifier
from verifiers.logic_verifier import LogicVerifier, _Z3_AVAILABLE
from verifiers.schema_verifier import SchemaVerifier
from verifiers.base import VerificationResult


# ---------------------------------------------------------------------- #
# MathVerifier golden set
# ---------------------------------------------------------------------- #
_MATH_GOLDEN = [
    # (query, answer, context, expected_verified, comment)
    # CORRECT answers -> verified=True
    ("What is 6*7?", "\\boxed{42}", {"expected_answer": 42}, True,
     "exact integer match"),
    ("What is 6*7?", "42", {"expected_answer": 42}, True,
     "integer match without boxed"),
    ("What is 1/3 as a decimal?", "0.333333", {"expected_answer": 0.333333,
     "tolerance": 1e-5}, True, "float within tolerance"),
    ("What is 100?", "1e2", {"expected_answer": 100}, True,
     "scientific notation (1e2 == 100) — fixed in the numeric parser"),
    ("What is 0.0015?", "1.5e-3", {"expected_answer": 0.0015,
     "tolerance": 1e-9}, True,
     "scientific notation with negative exponent (1.5e-3 == 0.0015)"),
    ("What is 2000?", "\\boxed{2e3}", {"expected_answer": 2000}, True,
     "scientific notation inside \\boxed (2e3 == 2000)"),
    ("What is 123000?", "1.23e5", {"expected_answer": 123000}, True,
     "scientific notation with positive exponent (1.23e5 == 123000)"),
    # HALLUCINATED answers -> verified=False
    ("What is 6*7?", "\\boxed{43}", {"expected_answer": 42}, False,
     "wrong integer (off by one)"),
    ("What is 6*7?", "\\boxed{420}", {"expected_answer": 42}, False,
     "wrong by order of magnitude"),
    ("What is 6*7?", "the answer is forty-two", {"expected_answer": 42}, False,
     "non-numeric answer cannot be verified"),
    ("What is 6*7?", "\\boxed{}", {"expected_answer": 42}, False,
     "empty boxed answer"),
    # No expected answer -> verified=False (honest, can't confirm)
    ("What is 6*7?", "\\boxed{42}", {}, False,
     "no expected_answer -> can only confirm numeric, not correct"),
]


class TestMathVerifierGoldenSet:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,answer,context,expected_verified,comment", _MATH_GOLDEN,
        ids=[c[:40] for _, _, _, _, c in _MATH_GOLDEN],
    )
    async def test_math_golden(self, query, answer, context,
                               expected_verified, comment):
        v = MathVerifier()
        result = await v.verify(query, answer, context)
        assert result.verified == expected_verified, (
            f"FAIL [{comment}]: expected verified={expected_verified}, "
            f"got verified={result.verified}, error={result.error}"
        )


# ---------------------------------------------------------------------- #
# LogicVerifier golden set (skipped if z3 not installed)
# ---------------------------------------------------------------------- #
_LOGIC_GOLDEN = [
    # (query, answer, context, expected_verified, comment)
    # CORRECT: values satisfy all constraints
    ("x is positive, x+y=10, y<x", "x=7, y=3",
     {"constraints": ["x > 0", "x + y == 10", "y < x"],
      "variables": {"x": "Int", "y": "Int"},
      "values": {"x": 7, "y": 3}}, True,
     "values satisfy all constraints"),
    # HALLUCINATED: values violate a constraint
    ("x is positive, x+y=10, y<x", "x=3, y=7",
     {"constraints": ["x > 0", "x + y == 10", "y < x"],
      "variables": {"x": "Int", "y": "Int"},
      "values": {"x": 3, "y": 7}}, False,
     "y < x violated (y=7 > x=3)"),
    ("x is positive, x+y=10, y<x", "x=-2, y=12",
     {"constraints": ["x > 0", "x + y == 10", "y < x"],
      "variables": {"x": "Int", "y": "Int"},
      "values": {"x": -2, "y": 12}}, False,
     "x > 0 violated (x=-2)"),
    # No constraints -> can't verify
    ("some problem", "x=5", {"variables": {"x": "Int"}, "values": {"x": 5}},
     False, "no constraints provided"),
    # Unsat constraints -> the problem itself is broken
    ("x > 5 and x < 3", "x=4",
     {"constraints": ["x > 5", "x < 3"],
      "variables": {"x": "Int"},
      "values": {"x": 4}}, False,
     "constraints are unsatisfiable"),
]


@pytest.mark.skipif(not _Z3_AVAILABLE,
                    reason="z3-solver not installed")
class TestLogicVerifierGoldenSet:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,answer,context,expected_verified,comment", _LOGIC_GOLDEN,
        ids=[c[:40] for _, _, _, _, c in _LOGIC_GOLDEN],
    )
    async def test_logic_golden(self, query, answer, context,
                                expected_verified, comment):
        v = LogicVerifier()
        result = await v.verify(query, answer, context)
        assert result.verified == expected_verified, (
            f"FAIL [{comment}]: expected verified={expected_verified}, "
            f"got verified={result.verified}, error={result.error}"
        )


# ---------------------------------------------------------------------- #
# SchemaVerifier golden set
# ---------------------------------------------------------------------- #
# SchemaVerifier is deterministic (no model calls): it parses the answer
# (JSON/YAML/text) and checks it against a schema/pattern/expected_keys.
# Each case below represents a class of structural error the verifier must
# catch or accept. Golden cases guard against regressions where a schema
# check silently passes a malformed answer (poisoning the flywheel) or
# rejects a well-formed one.
import json as _json

_SCHEMA_GOLDEN = [
    # (query, answer, context, expected_verified, comment)
    # CORRECT structural conformance -> verified=True
    ("return a user object",
     _json.dumps({"name": "Alice", "age": 30}),
     {"schema": {"type": "object", "required": ["name", "age"],
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer", "minimum": 0}}}},
     True, "valid object matching schema"),
    ("return a user object",
     _json.dumps({"name": "Alice", "age": 30, "email": "a@b.com"}),
     {"schema": {"type": "object", "required": ["name", "age"],
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer", "minimum": 0}},
                 "additionalProperties": True}},
     True, "extra properties allowed when additionalProperties not False"),
    ("return a list of ints", "[1, 2, 3]",
     {"schema": {"type": "array", "items": {"type": "integer"},
                 "minItems": 1}}, True, "valid array of integers"),
    ("match the pattern", "ABC-1234",
     {"pattern": r"[A-Z]{3}-\d{4}"}, True, "regex fullmatch succeeds"),
    ("return keys", _json.dumps({"id": 1, "name": "x"}),
     {"expected_keys": ["id", "name"]}, True, "expected_keys shortcut passes"),
    # HALLUCINATED / malformed -> verified=False
    ("return a user object",
     _json.dumps({"name": "Alice"}),  # missing required "age"
     {"schema": {"type": "object", "required": ["name", "age"],
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer"}}}},
     False, "missing required key"),
    ("return a user object",
     _json.dumps({"name": "Alice", "age": "thirty"}),  # age is string not int
     {"schema": {"type": "object", "required": ["name", "age"],
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer"}}}},
     False, "wrong type for property (string instead of integer)"),
    ("return a user object",
     _json.dumps({"name": "Alice", "age": -1}),  # violates minimum: 0
     {"schema": {"type": "object", "required": ["name", "age"],
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer", "minimum": 0}}}},
     False, "numeric constraint violated (age < minimum)"),
    ("return a user object",
     _json.dumps({"name": "Alice", "age": 30, "x": 1}),
     {"schema": {"type": "object", "required": ["name", "age"],
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer"}},
                 "additionalProperties": False}},
     False, "additional properties rejected when additionalProperties is False"),
    ("return a list of ints", "[]",
     {"schema": {"type": "array", "items": {"type": "integer"},
                 "minItems": 1}}, False, "array below minItems"),
    ("return a list of ints", "[1, \"two\", 3]",
     {"schema": {"type": "array", "items": {"type": "integer"}}},
     False, "array item with wrong type"),
    ("match the pattern", "abc-1234",  # lowercase, won't fullmatch
     {"pattern": r"[A-Z]{3}-\d{4}"}, False, "regex fullmatch fails"),
    ("return keys", _json.dumps({"id": 1}),  # missing "name"
     {"expected_keys": ["id", "name"]}, False, "expected_keys missing one key"),
    ("return a user object", "not json at all",
     {"schema": {"type": "object"}}, False, "unparseable JSON"),
    # No criteria -> verified=False (honest, can't verify structure)
    ("return anything", "{\"x\": 1}", {}, False,
     "no schema/pattern/expected_keys -> cannot verify"),
]


class TestSchemaVerifierGoldenSet:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,answer,context,expected_verified,comment", _SCHEMA_GOLDEN,
        ids=[c[:40] for _, _, _, _, c in _SCHEMA_GOLDEN],
    )
    async def test_schema_golden(self, query, answer, context,
                                 expected_verified, comment):
        v = SchemaVerifier()
        result = await v.verify(query, answer, context)
        assert result.verified == expected_verified, (
            f"FAIL [{comment}]: expected verified={expected_verified}, "
            f"got verified={result.verified}, error={result.error}"
        )


class TestGoldenSetCompleteness:
    """Meta-test: the golden set must cover both True and False outcomes
    for each verifier. A golden set that only tests one direction is
    vacuous — it would pass even if the verifier always returned True
    (or always False)."""

    def test_math_golden_has_both_directions(self):
        verified_true = sum(1 for _, _, _, e, _ in _MATH_GOLDEN if e)
        verified_false = sum(1 for _, _, _, e, _ in _MATH_GOLDEN if not e)
        assert verified_true > 0, "math golden set has no verified=True cases"
        assert verified_false > 0, "math golden set has no verified=False cases"

    @pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3 not installed")
    def test_logic_golden_has_both_directions(self):
        verified_true = sum(1 for _, _, _, e, _ in _LOGIC_GOLDEN if e)
        verified_false = sum(1 for _, _, _, e, _ in _LOGIC_GOLDEN if not e)
        assert verified_true > 0, "logic golden set has no verified=True cases"
        assert verified_false > 0, "logic golden set has no verified=False cases"

    def test_schema_golden_has_both_directions(self):
        verified_true = sum(1 for _, _, _, e, _ in _SCHEMA_GOLDEN if e)
        verified_false = sum(1 for _, _, _, e, _ in _SCHEMA_GOLDEN if not e)
        assert verified_true > 0, "schema golden set has no verified=True cases"
        assert verified_false > 0, "schema golden set has no verified=False cases"
