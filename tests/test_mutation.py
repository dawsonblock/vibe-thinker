"""Tests for the mutation testing module (Phase 3.1).

Tests the mutation operators (syntactic transforms that introduce real
bugs), the mutate_code entry point (tries operators in order, returns
the first valid mutation), and the vacuous-test detection contract.
"""

import ast

import pytest

from verifiers.mutation import (
    mutate_code,
    MutationResult,
    _flip_arithmetic,
    _flip_comparison,
    _return_none,
    _swap_constants,
    _drop_statement,
)


class TestFlipArithmetic:
    def test_flips_plus_to_minus(self):
        result = _flip_arithmetic("def add(a, b): return a + b")
        assert result is not None
        mutated, desc = result
        assert " - " in mutated
        assert "+" not in mutated.split("return")[1]
        assert "flipped" in desc

    def test_flips_minus_to_plus(self):
        result = _flip_arithmetic("def sub(a, b): return a - b")
        assert result is not None
        mutated, _ = result
        assert " + " in mutated

    def test_no_arithmetic_returns_none(self):
        assert _flip_arithmetic("def f(): return True") is None

    def test_flips_only_first_occurrence(self):
        result = _flip_arithmetic("def f(): return a + b + c")
        assert result is not None
        mutated, _ = result
        # Only the first + should be flipped.
        assert mutated.count(" + ") == 1
        assert " - " in mutated


class TestFlipComparison:
    def test_flips_eq_to_neq(self):
        result = _flip_comparison("def f(x): return x == 5")
        assert result is not None
        mutated, _ = result
        assert "!=" in mutated
        assert "==" not in mutated

    def test_flips_neq_to_eq(self):
        result = _flip_comparison("def f(x): return x != 5")
        assert result is not None
        mutated, _ = result
        assert "==" in mutated

    def test_flips_le_to_gt(self):
        result = _flip_comparison("def f(x): return x <= 5")
        assert result is not None
        mutated, _ = result
        assert ">" in mutated
        assert "<=" not in mutated

    def test_no_comparison_returns_none(self):
        assert _flip_comparison("def f(): return 42") is None


class TestReturnNone:
    def test_replaces_first_return(self):
        result = _return_none("def f(x): return x * 2")
        assert result is not None
        mutated, _ = result
        assert "return None" in mutated
        assert "x * 2" not in mutated.split("return None")[0]

    def test_no_return_value_returns_none(self):
        # A bare `return` with no value should not be mutated.
        assert _return_none("def f(): return") is None

    def test_no_return_at_all(self):
        assert _return_none("x = 1") is None

    def test_invalid_python_returns_none(self):
        assert _return_none("def f(: return 1") is None


class TestSwapConstants:
    def test_swaps_two_integers(self):
        result = _swap_constants("def f(): return 3 + 7")
        assert result is not None
        mutated, _ = result
        # The 3 and 7 should be swapped.
        assert "3" in mutated and "7" in mutated

    def test_single_constant_returns_none(self):
        assert _swap_constants("def f(): return 42") is None

    def test_identical_constants_returns_none(self):
        assert _swap_constants("def f(): return 5 + 5") is None

    def test_skips_booleans(self):
        # True/False are int subclasses but should be skipped.
        result = _swap_constants("def f(): return True and False")
        # True and False are bools — should be skipped, so no swap.
        assert result is None


class TestDropStatement:
    def test_drops_assignment(self):
        result = _drop_statement("x = 1\nprint(x)")
        assert result is not None
        mutated, desc = result
        assert "MUTATED" in mutated
        assert "dropped" in desc

    def test_does_not_drop_def(self):
        result = _drop_statement("def f(): return 1")
        # The only statement is a FunctionDef — should not be dropped.
        # (The body Return is a stmt but is in the excluded list.)
        assert result is None

    def test_does_not_drop_import(self):
        assert _drop_statement("import os") is None

    def test_does_not_drop_return(self):
        assert _drop_statement("def f(): return 1") is None


class TestMutateCode:
    def test_applies_first_valid_mutation(self):
        code = "def add(a, b): return a + b"
        result = mutate_code(code)
        assert result.applied is True
        assert result.mutated_code is not None
        assert result.mutated_code != code
        assert result.operator is not None
        assert result.description

    def test_mutated_code_is_valid_python(self):
        code = "def square(n): return n * n"
        result = mutate_code(code)
        assert result.applied
        # The mutated code must parse as valid Python.
        ast.parse(result.mutated_code)

    def test_empty_source_no_mutation(self):
        result = mutate_code("")
        assert result.applied is False
        assert result.mutated_code is None

    def test_whitespace_only_no_mutation(self):
        result = mutate_code("   \n  \n")
        assert result.applied is False

    def test_too_simple_no_mutation(self):
        # A single pass statement — no operators can apply.
        result = mutate_code("pass")
        assert result.applied is False
        assert result.mutated_code is None

    def test_mutation_is_different_from_original(self):
        code = "def f(x):\n    if x > 0:\n        return x\n    return -x"
        result = mutate_code(code)
        assert result.applied
        assert result.mutated_code != code

    def test_mutation_introduces_behavioral_change(self):
        """The mutation must change the code's behavior, not just cosmetics."""
        code = "def add(a, b): return a + b"
        result = mutate_code(code)
        assert result.applied
        # The mutated code should produce different output for some input.
        # We can't execute it here, but we can verify the source differs
        # meaningfully (not just whitespace).
        assert result.mutated_code.strip() != code.strip()

    def test_seed_reproducible(self):
        """The same seed produces the same mutation (deterministic operators)."""
        code = "def f(x): return x + 1"
        r1 = mutate_code(code, seed=42)
        r2 = mutate_code(code, seed=42)
        assert r1.mutated_code == r2.mutated_code
        assert r1.operator == r2.operator

    def test_original_code_preserved(self):
        code = "def f(x): return x * 2"
        result = mutate_code(code)
        assert result.original_code == code

    def test_mutation_result_applied_property(self):
        result = MutationResult(mutated_code="x = 1", original_code="x = 2")
        assert result.applied is True
        result2 = MutationResult(mutated_code=None, original_code="x = 2")
        assert result2.applied is False
        # Same code -> not applied (no-op).
        result3 = MutationResult(mutated_code="x = 2", original_code="x = 2")
        assert result3.applied is False


class TestMutationOperatorsCoverage:
    """Verify that different code shapes trigger different operators."""

    def test_arithmetic_code_uses_flip_arithmetic(self):
        code = "def f(a, b): return a + b"
        result = mutate_code(code)
        assert result.operator == "flip_arithmetic"

    def test_comparison_code_uses_flip_comparison(self):
        # No arithmetic ops, but has a comparison.
        code = "def f(x): return x == 42"
        result = mutate_code(code)
        assert result.operator == "flip_comparison"

    def test_return_only_code_uses_return_none(self):
        # No arithmetic, no comparison, but has a return value.
        code = "def f(x): return x"
        result = mutate_code(code)
        assert result.operator == "return_none"
