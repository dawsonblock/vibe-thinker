"""Tests for the static analysis fallback (v2.0).

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
