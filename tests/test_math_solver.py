"""Tests for the deterministic math solver."""

import pytest

from math_solver import solve


class TestRecurrenceSolver:
    def test_recurrence_a5(self):
        """The canonical recurrence from test_full_stack.py."""
        result = solve("Solve this step by step: a_1=2, a_{n+1}=a_n^2-a_n+1. Find a_5.")
        assert result == "1807"

    def test_recurrence_a4(self):
        result = solve("a_1=1, a_{n+1}=2*a_n+1, find a_4")
        assert result == "15"

    def test_recurrence_simple(self):
        result = solve("a_1=3, a_{n+1}=a_n+2, find a_5")
        assert result == "11"

    def test_recurrence_no_init_returns_none(self):
        result = solve("a_{n+1}=a_n^2, find a_5")
        assert result is None

    def test_recurrence_no_target_returns_none(self):
        result = solve("a_1=2, a_{n+1}=a_n^2")
        assert result is None

    def test_recurrence_unsafe_expr_returns_none(self):
        """Expressions with non-arithmetic operations must be rejected."""
        result = solve("a_1=2, a_{n+1}=import os, find a_5")
        assert result is None


class TestFiniteSumSolver:
    def test_simple_sum(self):
        result = solve("Compute the sum of 1 + 2 + 3 + 4 + 5")
        assert result == "15"

    def test_sum_with_fractions(self):
        result = solve("sum of 1 + 1/2 + 1/4 + 1/8 + 1/16")
        assert result == "1.9375"

    def test_sum_single_term_returns_none(self):
        result = solve("sum of 42")
        assert result is None


class TestArithmeticSolver:
    def test_addition(self):
        result = solve("What is 2+2?")
        assert result == "4"

    def test_multiplication(self):
        result = solve("What is 3 * 4?")
        assert result == "12"

    def test_subtraction(self):
        result = solve("Calculate 15 - 7")
        assert result == "8"

    def test_complex_expression(self):
        result = solve("What is (2 + 3) * 4?")
        assert result == "20"

    def test_non_arithmetic_returns_none(self):
        result = solve("What is the meaning of life?")
        assert result is None


class TestUnsolvableProblems:
    def test_word_problem_returns_none(self):
        result = solve("Explain quantum mechanics")
        assert result is None

    def test_empty_string_returns_none(self):
        result = solve("")
        assert result is None

    def test_proof_returns_none(self):
        result = solve("Prove that the sum of two evens is even")
        assert result is None
