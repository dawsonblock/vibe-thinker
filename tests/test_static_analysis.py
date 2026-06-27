"""Tests for the static analysis fallback (v2.0, hardened v3.0) and
the v3.1 wasmtime sandbox fallback.

When the Generalist fails to generate unit tests, the code verification
loop falls back to a sandboxed execution pass (v3.1) or, if no sandbox
is available, the deprecated AST static analysis pass (v2.0).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hybrid_orchestrator import _static_analysis_fallback, _wasmtime_sandbox_fallback


class TestStaticAnalysisFallback:
    """Verify the static analysis fallback scoring."""

    def test_clean_code_gets_partial_score(self):
        """Code that parses cleanly and has no restricted imports
        gets a partial heuristic score of 0.4."""
        code = "def add(a, b):\n    return a + b\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.4
        assert issues == []

    def test_syntax_error_gets_zero(self):
        """Code with a syntax error gets score 0.0."""
        code = "def add(a, b)\n    return a + b\n"  # missing colon
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert len(issues) == 1
        assert "syntax error" in issues[0].lower()

    def test_restricted_import_os_gets_zero(self):
        """Code importing 'os' gets score 0.0."""
        code = "import os\nprint(os.getcwd())\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("os" in i for i in issues)

    def test_restricted_import_subprocess_gets_zero(self):
        """Code importing 'subprocess' gets score 0.0."""
        code = "import subprocess\nsubprocess.run(['ls'])\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("subprocess" in i for i in issues)

    def test_restricted_import_socket_gets_zero(self):
        """Code importing 'socket' gets score 0.0."""
        code = "import socket\ns = socket.socket()\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("socket" in i for i in issues)

    def test_restricted_import_from_gets_zero(self):
        """Code using 'from os import path' gets score 0.0."""
        code = "from os import path\nprint(path.exists('.'))\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("os" in i for i in issues)

    def test_safe_imports_get_partial_score(self):
        """Code with safe imports (math, json, re) gets 0.4."""
        code = "import math\nimport json\nimport re\nprint(math.pi)\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.4
        assert issues == []

    def test_empty_code_gets_partial_score(self):
        """Empty code parses cleanly (it's valid Python) — score 0.4."""
        score, issues = _static_analysis_fallback("")
        assert score == 0.4
        assert issues == []

    def test_class_definition_gets_partial_score(self):
        """A class definition with methods parses cleanly — score 0.4."""
        code = (
            "class Solution:\n"
            "    def two_sum(self, nums, target):\n"
            "        seen = {}\n"
            "        for i, n in enumerate(nums):\n"
            "            if target - n in seen:\n"
            "                return [seen[target - n], i]\n"
            "            seen[n] = i\n"
            "        return []\n"
        )
        score, issues = _static_analysis_fallback(code)
        assert score == 0.4
        assert issues == []

    def test_multiple_restricted_imports_listed(self):
        """Multiple restricted imports are all listed in issues."""
        code = "import os\nimport subprocess\nimport socket\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("os" in i for i in issues)
        assert any("subprocess" in i for i in issues)
        assert any("socket" in i for i in issues)

    def test_score_capped_at_04(self):
        """The static analysis score is always capped at 0.4 — never 1.0."""
        code = "def perfect_solution():\n    return 42\n"
        score, _ = _static_analysis_fallback(code)
        assert score == 0.4  # NOT 1.0 — this is a heuristic, not full verification
        assert score < 0.5  # Below the cache trust threshold


class TestStaticAnalysisEvasionVectors:
    """v3.0: Verify dynamic import evasion vectors are caught."""

    def test_dunder_import_call_gets_zero(self):
        """__import__('os') bypasses ast.Import but is caught by ast.Call check."""
        code = "mod = __import__('os')\nmod.system('rm -rf /')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("__import__" in i for i in issues)

    def test_importlib_import_module_gets_zero(self):
        """importlib.import_module('os') bypasses ast.Import but is caught."""
        code = (
            "import importlib\n"
            "mod = importlib.import_module('os')\n"
            "mod.system('ls')\n"
        )
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        # importlib is both a restricted import AND an evasion vector
        assert any("importlib" in i for i in issues)

    def test_importlib_reference_without_import_gets_zero(self):
        """Even referencing importlib without importing it is flagged."""
        code = "x = importlib.import_module('os')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("importlib" in i for i in issues)

    def test_exec_call_gets_zero(self):
        """exec() calls are flagged as dynamic code execution."""
        code = "exec('import os')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("exec" in i for i in issues)

    def test_eval_call_gets_zero(self):
        """eval() calls are flagged as dynamic code execution."""
        code = "eval('__import__(\"os\")')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("eval" in i for i in issues)

    def test_builtins_reference_gets_zero(self):
        """__builtins__ reference is flagged as a reflection vector."""
        code = "f = getattr(__builtins__, 'eval')\nf('1+1')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("__builtins__" in i for i in issues)

    def test_clean_code_with_no_evasion_still_gets_partial(self):
        """Code without any evasion vectors still gets 0.4."""
        code = (
            "def solve(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        result.append(item * 2)\n"
            "    return result\n"
        )
        score, issues = _static_analysis_fallback(code)
        assert score == 0.4
        assert issues == []

    def test_mixed_evasion_and_restricted_both_reported(self):
        """Both restricted imports and evasion vectors are reported."""
        code = "import os\nmod = __import__('subprocess')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        # Restricted import (os) is found first, so it's reported.
        # The evasion (__import__) is also detected.
        assert any("os" in i for i in issues)

    def test_getattr_builtins_evasion_gets_zero(self):
        """getattr(__builtins__, 'eval') is flagged as evasion."""
        code = "f = getattr(__builtins__, 'eval')\nf('1+1')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("getattr" in i or "__builtins__" in i for i in issues)

    def test_builtins_dunder_import_evasion_gets_zero(self):
        """builtins.__import__('os') is flagged as evasion."""
        code = "import builtins\nmod = builtins.__import__('os')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        # builtins is both a restricted import AND an evasion vector
        assert any("builtins" in i for i in issues)

    def test_builtins_reference_gets_zero(self):
        """Referencing the builtins module is flagged."""
        code = "x = builtins.eval('1+1')\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("builtins" in i for i in issues)


# ---------------------------------------------------------------------- #
# Wasmtime sandbox fallback (v3.1)
# ---------------------------------------------------------------------- #
class TestWasmtimeSandboxFallback:
    """v3.1: Tests for the sandboxed execution fallback."""

    @pytest.mark.asyncio
    async def test_no_sandbox_returns_none(self):
        """When no sandbox is available (no wasmtime, no Docker), returns
        (None, []) so the caller falls back to AST."""
        score, issues = await _wasmtime_sandbox_fallback(
            "print('hello')", code_verifier=None,
        )
        assert score is None
        assert issues == []

    @pytest.mark.asyncio
    async def test_docker_sandbox_success(self):
        """When the Docker sandbox executes the code successfully (exit 0),
        returns (0.65, [])."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stderr = ""
        mock_executor.execute = AsyncMock(return_value=mock_result)
        mock_verifier = MagicMock()
        mock_verifier.executor = mock_executor

        score, issues = await _wasmtime_sandbox_fallback(
            "print('hello')", code_verifier=mock_verifier,
        )
        assert score == 0.65
        assert issues == []

    @pytest.mark.asyncio
    async def test_docker_sandbox_failure(self):
        """When the code errors in the sandbox (non-zero exit), returns
        (0.0, [error])."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 1
        mock_result.stderr = "NameError: name 'x' is not defined"
        mock_executor.execute = AsyncMock(return_value=mock_result)
        mock_verifier = MagicMock()
        mock_verifier.executor = mock_executor

        score, issues = await _wasmtime_sandbox_fallback(
            "print(x)", code_verifier=mock_verifier,
        )
        assert score == 0.0
        assert len(issues) == 1
        assert "exit code 1" in issues[0]

    @pytest.mark.asyncio
    async def test_docker_sandbox_execution_exception(self):
        """When the sandbox execution itself raises, returns (0.0, [error])."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=RuntimeError("sandbox down"))
        mock_verifier = MagicMock()
        mock_verifier.executor = mock_executor

        score, issues = await _wasmtime_sandbox_fallback(
            "print('hello')", code_verifier=mock_verifier,
        )
        assert score == 0.0
        assert len(issues) == 1
        assert "sandbox down" in issues[0]

    @pytest.mark.asyncio
    async def test_verifier_with_no_executor_returns_none(self):
        """When the code_verifier has no executor, returns (None, [])."""
        mock_verifier = MagicMock()
        mock_verifier.executor = None
        score, issues = await _wasmtime_sandbox_fallback(
            "print('hello')", code_verifier=mock_verifier,
        )
        assert score is None

    @pytest.mark.asyncio
    async def test_score_capped_at_065(self):
        """The sandbox score is always 0.65 — never 1.0."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_executor.execute = AsyncMock(return_value=mock_result)
        mock_verifier = MagicMock()
        mock_verifier.executor = mock_executor

        score, _ = await _wasmtime_sandbox_fallback(
            "def perfect(): return 42", code_verifier=mock_verifier,
        )
        assert score == 0.65
        assert score < 0.7  # Below the cache trust threshold
