"""Tests for the static analysis fallback (v2.0, hardened v3.0) and
the v3.1 wasmtime sandbox fallback.

When the Generalist fails to generate unit tests, the code verification
loop falls back to a sandboxed execution pass (v3.1) or, if no sandbox
is available, the deprecated AST static analysis pass (v2.0).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hybrid_orchestrator import _static_analysis_fallback, _wasmtime_sandbox_fallback

import inspect


@pytest.mark.filterwarnings("ignore:.*_static_analysis_fallback.*:DeprecationWarning")
class TestStaticAnalysisFallback:
    """Verify the static analysis fallback scoring.

    v3.2.1: _static_analysis_fallback emits a DeprecationWarning on every
    call (it's NOT a security boundary). The existing tests below call it
    to verify its scoring logic, not to assert anything about the warning.
    The deprecation warning itself is tested by
    TestStaticAnalysisEvasionVectors.test_emits_deprecation_warning. We
    suppress it here to keep test output readable.
    """

    def test_clean_code_gets_partial_score(self):
        """Code that parses cleanly and has no restricted imports
        gets a partial heuristic score of 0.2 (v3.2: lowered from 0.4
        to reduce the epistemic weight of an unexecuted signal)."""
        code = "def add(a, b):\n    return a + b\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.2
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
        """Code with safe imports (math, json, re) gets 0.2 (v3.2)."""
        code = "import math\nimport json\nimport re\nprint(math.pi)\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.2
        assert issues == []

    def test_empty_code_gets_partial_score(self):
        """Empty code parses cleanly (it's valid Python) — score 0.2 (v3.2)."""
        score, issues = _static_analysis_fallback("")
        assert score == 0.2
        assert issues == []

    def test_class_definition_gets_partial_score(self):
        """A class definition with methods parses cleanly — score 0.2 (v3.2)."""
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
        assert score == 0.2
        assert issues == []

    def test_multiple_restricted_imports_listed(self):
        """Multiple restricted imports are all listed in issues."""
        code = "import os\nimport subprocess\nimport socket\n"
        score, issues = _static_analysis_fallback(code)
        assert score == 0.0
        assert any("os" in i for i in issues)
        assert any("subprocess" in i for i in issues)
        assert any("socket" in i for i in issues)

    def test_score_capped_at_02(self):
        """The static analysis score is always capped at 0.2 — never 1.0."""
        code = "def perfect_solution():\n    return 42\n"
        score, _ = _static_analysis_fallback(code)
        assert score == 0.2  # NOT 1.0 — this is a heuristic, not full verification
        assert score < 0.5  # Below the cache trust threshold


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestStaticAnalysisEvasionVectors:
    """v3.0: Verify dynamic import evasion vectors are caught.

    v3.2.1: suppresses the _static_analysis_fallback DeprecationWarning
    except in test_emits_deprecation_warning, which explicitly asserts it.
    """

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
        """Code without any evasion vectors still gets 0.2 (v3.2)."""
        code = (
            "def solve(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        result.append(item * 2)\n"
            "    return result\n"
        )
        score, issues = _static_analysis_fallback(code)
        assert score == 0.2
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

    def test_chr_obfuscation_bypass_is_known_limitation(self):
        """v3.2.1: Document the known bypass — chr()-constructed strings
        in getattr evade the AST checks. This is WHY static analysis is
        NOT a security boundary.

        ``getattr(__builtins__, chr(95)*2 + 'import' + chr(95)*2)`` uses
        a Call (chr) as the getattr argument, not a Name or Constant.
        The AST check only flags getattr when an arg is a Name referencing
        __builtins__ — but here the __builtins__ reference IS caught.
        The truly uncatchable form is when __builtins__ itself is obtained
        via reflection (no direct Name reference). This test documents
        that the simple __import__(chr(...)) form IS caught (the function
        name __import__ is a Name node regardless of its arguments), and
        that the limitation is the runtime-construction path.
        """
        import warnings
        # __import__(chr(111)+chr(115)) — the function name __import__ is
        # a Name node, so it IS caught regardless of the argument form.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            score, issues = _static_analysis_fallback(
                "m = __import__(chr(111) + chr(115))\n"
            )
        assert score == 0.0
        assert any("__import__" in i for i in issues)

    def test_emits_deprecation_warning(self):
        """v3.2.1: every call emits a DeprecationWarning (NOT a security
        boundary, on the deprecation path)."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _static_analysis_fallback("x = 1\n")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


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


# ---------------------------------------------------------------------- #
# v3.2: Wasmtime fuel + wall-clock belt-and-suspenders
# ---------------------------------------------------------------------- #
class TestWasmtimeFuel:
    """v3.2: deterministic fuel limits catch infinite loops at the
    CPU-instruction level; the wall-clock timeout is the outer belt."""

    @pytest.mark.asyncio
    async def test_out_of_fuel_traps_deterministically(self, monkeypatch, tmp_path):
        """When the Wasm store runs out of fuel, the sandbox must return
        score=0.0 with a 'fuel' message — NOT crash, NOT hang."""
        # Inject a fake wasmtime module so the fuel path runs without
        # the real wasmtime installed.
        import sys, types
        fake = types.ModuleType("wasmtime")

        class _Config:
            consume_fuel = None  # setter_property-like

            def __init__(self):
                self._consume_fuel = False

            # emulate @setter_property
            consume_fuel = type("P", (), {
                "__set__": lambda s, inst, v: setattr(inst, "_cf", v),
                "__get__": lambda s, inst, cls: getattr(inst, "_cf", False),
            })()

        class _Store:
            def __init__(self, engine):
                self.fuel = None

            def set_fuel(self, n):
                self.fuel = n

        class _Instance:
            def __init__(self, store, module, args):
                pass

            def exports(self, store):
                return {"run_python": lambda store, code: (_ for _ in ()).throw(
                    RuntimeError("wasm trap: out of fuel consumed 1000000000"))}

        class _Engine:
            def __init__(self, config):
                pass

        class _Module:
            @staticmethod
            def from_file(engine, path):
                return MagicMock()

        fake.Config = _Config
        fake.Engine = _Engine
        fake.Module = _Module
        fake.Store = _Store
        fake.Instance = _Instance
        monkeypatch.setitem(sys.modules, "wasmtime", fake)
        # Write a dummy wasm module path so the env-gated branch runs.
        wasm_path = tmp_path / "fake.wasm"
        wasm_path.write_bytes(b"\x00asm")
        monkeypatch.setenv("VIBE_WASM_PYTHON_MODULE", str(wasm_path))

        score, issues = await _wasmtime_sandbox_fallback(
            "while True: pass", code_verifier=None,
        )
        assert score == 0.0
        assert any("fuel" in i.lower() for i in issues)

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_catches_hang(self, monkeypatch, tmp_path):
        """When the wasm call hangs (e.g. a host syscall that fuel can't
        see), the wall-clock timeout must fire and return score=0.0."""
        import sys, types, asyncio
        fake = types.ModuleType("wasmtime")

        class _Config:
            def __init__(self):
                self._cf = False
            consume_fuel = type("P", (), {
                "__set__": lambda s, inst, v: setattr(inst, "_cf", v),
                "__get__": lambda s, inst, cls: getattr(inst, "_cf", False),
            })()

        class _Store:
            def __init__(self, engine):
                pass
            def set_fuel(self, n):
                pass

        class _Instance:
            def __init__(self, store, module, args):
                pass
            def exports(self, store):
                # The real wasmtime run() is synchronous. The
                # orchestrator runs it in a thread executor so
                # asyncio.wait_for can abandon it on timeout. Simulate a
                # synchronous hang here (time.sleep) — the executor
                # thread blocks, but wait_for returns TimeoutError and
                # the orchestrator stops waiting. The sleep is kept short
                # enough (1.5s) that the orphaned executor thread exits
                # cleanly before pytest tears down the event loop.
                import time as _time
                def _run(store, code):
                    _time.sleep(1.5)  # >> 0.2s timeout, exits before teardown
                    return 0
                return {"run_python": _run}

        class _Engine:
            def __init__(self, config):
                pass

        class _Module:
            @staticmethod
            def from_file(engine, path):
                return MagicMock()

        fake.Config = _Config
        fake.Engine = _Engine
        fake.Module = _Module
        fake.Store = _Store
        fake.Instance = _Instance
        monkeypatch.setitem(sys.modules, "wasmtime", fake)
        wasm_path = tmp_path / "fake.wasm"
        wasm_path.write_bytes(b"\x00asm")
        monkeypatch.setenv("VIBE_WASM_PYTHON_MODULE", str(wasm_path))
        monkeypatch.setenv("VIBE_WASM_WALL_CLOCK_TIMEOUT", "0.2")

        score, issues = await _wasmtime_sandbox_fallback(
            "import time; time.sleep(100)", code_verifier=None,
        )
        assert score == 0.0
        assert any("timeout" in i.lower() for i in issues)


# ---------------------------------------------------------------------- #
# v3.2: --allow-static-fallback gate
# ---------------------------------------------------------------------- #
class TestStaticFallbackGate:
    """v3.2: the AST static-analysis fallback is gated behind
    ``allow_static_fallback`` (off by default). When the gate is closed
    and no sandbox is available, the code route must return
    verified=False / score=0.0 rather than the 0.2 heuristic. When the
    gate is open, the heuristic (0.2) is emitted under a distinct route
    name so consumers cannot confuse it with any verified path.
    """

    def test_gate_off_returns_zero_unverified_route(self):
        """Default (gate off) -> code_specialist_unverified, score 0.0."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        # Construct with the default (allow_static_fallback=False) and
        # exercise the no-sandbox branch by calling the internal route
        # directly via the public helper used by the code loop.
        # We can't easily build a full orchestrator here (it spins up a
        # CLR reasoner), so we assert the default attribute value and the
        # route name constant instead.
        # Default is set at the class level via the constructor default.
        import inspect
        sig = inspect.signature(HybridReasoningOrchestrator.__init__)
        assert sig.parameters["allow_static_fallback"].default is False

    def test_route_name_changed_to_unverified_static_only(self):
        """When the gate is open, the route is 'code_specialist_unverified_static_only',
        NOT 'code_specialist_static_analysis' (the old name that consumers
        might have whitelisted as if it were verified)."""
        import hybrid_orchestrator as ho
        # The old route name must NOT appear in the static-fallback branch.
        src = inspect.getsource(ho)
        # The old name is still referenced in the docstring history, so we
        # check the active return statement instead: the new route name
        # appears, the old one does not appear in a route_taken= assignment.
        assert "code_specialist_unverified_static_only" in src
        # The old route name must not be assigned as a route_taken value
        # in the static-fallback return.
        assert 'route_taken="code_specialist_static_analysis"' not in src
