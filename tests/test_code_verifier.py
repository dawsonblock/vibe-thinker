"""Pytest tests for the code verifier."""

import pytest

from verifiers.code_verifier import CodeVerifier


@pytest.fixture
def verifier():
    return CodeVerifier(timeout=5.0)


class TestCodeVerifier:
    @pytest.mark.asyncio
    async def test_passes_valid_function(self, verifier):
        code = """
def add(a, b):
    return a + b
"""
        tests = """
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0
"""
        result = await verifier.verify(
            "Write an add function", code,
            context={"unit_tests": tests},
        )
        assert result.verified is True
        assert result.score == 1.0
        assert result.method == "unit_tests"

    @pytest.mark.asyncio
    async def test_fails_wrong_function(self, verifier):
        code = """
def add(a, b):
    return a - b  # wrong!
"""
        tests = """
assert add(2, 3) == 5
"""
        result = await verifier.verify(
            "Write an add function", code,
            context={"unit_tests": tests},
        )
        assert result.verified is False
        assert "ASSERTION_FAILED" in (result.error or "") or "success marker" in (result.error or "")

    @pytest.mark.asyncio
    async def test_times_out_infinite_loop(self, verifier):
        code = "while True:\n    pass\n"
        result = await verifier.verify(
            "Run this code", code,
            context={"expected_output": "anything"},
        )
        assert result.verified is False
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_captures_exception(self, verifier):
        code = "raise ValueError('boom')\n"
        result = await verifier.verify(
            "Run this code", code,
            context={"expected_output": "anything"},
        )
        assert result.verified is False
        assert "exited with code" in (result.error or "") or "IMPORT_ERROR" in (result.error or "")

    @pytest.mark.asyncio
    async def test_stdout_comparison_passes(self, verifier):
        code = "print('hello world')"
        result = await verifier.verify(
            "Print hello world", code,
            context={"expected_output": "hello world"},
        )
        assert result.verified is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_stdout_comparison_fails(self, verifier):
        code = "print('goodbye world')"
        result = await verifier.verify(
            "Print hello world", code,
            context={"expected_output": "hello world"},
        )
        assert result.verified is False
        assert "stdout mismatch" in (result.error or "")

    @pytest.mark.asyncio
    async def test_no_criteria_returns_unverified(self, verifier):
        result = await verifier.verify(
            "Write a function", "def f(): pass",
            context={},
        )
        assert result.verified is False
        assert "no unit_tests" in (result.error or "")

    @pytest.mark.asyncio
    async def test_import_error_handled(self, verifier):
        code = "import nonexistent_module_xyz\n"
        tests = "assert True\n"
        result = await verifier.verify(
            "Write code", code,
            context={"unit_tests": tests},
        )
        assert result.verified is False
