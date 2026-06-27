"""Tests for the static analysis fallback (v2.0, hardened v3.0).

When the Generalist fails to generate unit tests, the code verification
loop falls back to a static analysis pass. This tests the
_static_analysis_fallback function directly.
"""

import pytest

from hybrid_orchestrator import _static_analysis_fallback


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
