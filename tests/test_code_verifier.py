"""Pytest tests for the code verifier.

Tests use LocalSubprocessExecutor explicitly for speed and CI
portability. The verifier's sandbox selection logic is tested
separately to verify it refuses to run without a sandbox.
"""

import pytest

from verifiers.code_verifier import CodeVerifier, select_executor
from sandbox import DockerSandboxExecutor, DockerSbxExecutor, LocalSubprocessExecutor, WarmDockerPool


@pytest.fixture
def executor():
    """Use local subprocess executor for tests (trusted test code)."""
    return LocalSubprocessExecutor(timeout=5.0)


@pytest.fixture
def verifier(executor):
    return CodeVerifier(timeout=5.0, executor=executor)


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
        # With the sandbox executor, an exception produces empty stdout,
        # so the verifier reports a stdout mismatch (correct behavior).
        assert "stdout mismatch" in (result.error or "") or \
               "exited with code" in (result.error or "") or \
               "IMPORT_ERROR" in (result.error or "")

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


class TestSandboxSelection:
    """Tests for the sandbox executor selection logic.

    The verifier must refuse to run untrusted code if no sandbox is
    available — it must NOT fall back to host execution by default.
    """

    def test_select_executor_returns_none_by_default_if_no_docker(self):
        """Without Docker or sbx, and without allow_unsafe, return None."""
        # This test verifies the logic, not the actual availability.
        # We mock is_available to simulate no Docker/sbx/warm-pool.
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: False)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: False)
            m.setattr(WarmDockerPool, "is_available", lambda self: False)
            result = select_executor(allow_unsafe=False)
            assert result is None

    def test_select_executor_allows_unsafe_with_flag(self):
        """With allow_unsafe=True, fall back to LocalSubprocessExecutor."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: False)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: False)
            m.setattr(WarmDockerPool, "is_available", lambda self: False)
            result = select_executor(allow_unsafe=True)
            assert isinstance(result, LocalSubprocessExecutor)

    def test_select_executor_prefers_warm_pool_by_default(self):
        """Warm pool is preferred over cold Docker (5x faster)."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: True)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: True)
            m.setattr(WarmDockerPool, "is_available", lambda self: True)
            result = select_executor()
            assert isinstance(result, WarmDockerPool)

    def test_select_executor_prefers_docker_when_warm_pool_disabled(self):
        """With prefer_warm_pool=False, Docker container is used."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: True)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: True)
            m.setattr(WarmDockerPool, "is_available", lambda self: True)
            result = select_executor(prefer_warm_pool=False)
            assert isinstance(result, DockerSandboxExecutor)

    def test_select_executor_prefers_sbx_with_flag(self):
        """With prefer_sbx=True, sbx is preferred over Docker and warm pool."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: True)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: True)
            m.setattr(WarmDockerPool, "is_available", lambda self: True)
            result = select_executor(prefer_sbx=True)
            assert isinstance(result, DockerSbxExecutor)

    def test_select_executor_falls_back_to_sbx(self):
        """If Docker and warm pool are not available but sbx is, use sbx."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: False)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: True)
            m.setattr(WarmDockerPool, "is_available", lambda self: False)
            result = select_executor()
            assert isinstance(result, DockerSbxExecutor)


class TestRefuseWithoutSandbox:
    """The verifier must refuse to run if no sandbox is available."""

    @pytest.mark.asyncio
    async def test_refuses_without_sandbox(self):
        """Without a sandbox executor, the verifier returns verified=False
        with an error explaining that no sandbox was found."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: False)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: False)
            m.setattr(WarmDockerPool, "is_available", lambda self: False)
            v = CodeVerifier(timeout=5.0)  # no executor, no allow_unsafe
            assert v.executor is None

            result = await v.verify(
                "Run this", "print('hello')",
                context={"expected_output": "hello"},
            )
            assert result.verified is False
            assert "no sandbox executor available" in (result.error or "")
            assert "refusing" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_refuses_without_sandbox_even_with_tests(self):
        """Even with unit tests, refuse without a sandbox."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(DockerSandboxExecutor, "is_available", lambda self: False)
            m.setattr(DockerSbxExecutor, "is_available", lambda self: False)
            m.setattr(WarmDockerPool, "is_available", lambda self: False)
            v = CodeVerifier(timeout=5.0)

            result = await v.verify(
                "Write add", "def add(a,b): return a+b",
                context={"unit_tests": "assert add(1,2)==3"},
            )
            assert result.verified is False
            assert "no sandbox executor available" in (result.error or "")


class TestExecutorEvidence:
    """The verifier should record which executor ran the code."""

    @pytest.mark.asyncio
    async def test_evidence_includes_executor_name(self, verifier):
        code = "print('hello')"
        result = await verifier.verify(
            "Print hello", code,
            context={"expected_output": "hello"},
        )
        assert result.verified is True
        assert "executor" in result.evidence
        assert result.evidence["executor"] == "local_subprocess_unsafe"


class TestNonceAntiSpoofing:
    """Security: a candidate that prints the old success marker string
    must NOT be verified. The verifier requires a per-execution nonce
    that candidate code cannot know."""

    @pytest.mark.asyncio
    async def test_candidate_printing_old_marker_not_verified(self, verifier):
        """A candidate that prints 'ALL_TESTS_PASSED' and then fails the
        tests must NOT be verified (the old bypass vulnerability)."""
        code = (
            "def add(a, b):\n"
            "    return a - b  # wrong\n"
            "print('ALL_TESTS_PASSED')  # spoof attempt\n"
        )
        tests = "assert add(2, 3) == 5\n"
        result = await verifier.verify(
            "Write add", code,
            context={"unit_tests": tests},
        )
        assert result.verified is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_candidate_overriding_sys_exit_not_verified(self, verifier):
        """A candidate that neutralizes sys.exit so a failing test still
        exits 0 must NOT be verified — the nonce marker is only printed
        if the test block completes without raising."""
        code = (
            "import sys\n"
            "sys.exit = lambda code=0: None  # neutralize exit\n"
            "def add(a, b):\n"
            "    return a - b  # wrong\n"
        )
        tests = "assert add(2, 3) == 5\n"
        result = await verifier.verify(
            "Write add", code,
            context={"unit_tests": tests},
        )
        assert result.verified is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_legitimate_pass_still_verified(self, verifier):
        """A correct solution must still pass verification with the nonce."""
        code = "def add(a, b):\n    return a + b\n"
        tests = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"
        result = await verifier.verify(
            "Write add", code,
            context={"unit_tests": tests},
        )
        assert result.verified is True
        assert result.score == 1.0
        assert result.evidence.get("nonce_verified") is True
