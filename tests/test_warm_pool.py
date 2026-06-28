"""Tests for the WarmDockerPool executor.

These tests verify:
  - Pool starts and creates warm containers
  - Code execution works via docker exec
  - Test execution (code + asserts) works
  - Cleanup removes containers
  - Fallback when Docker is unavailable
  - select_executor prefers warm pool when available

Tests that need Docker are marked with @pytest.mark.skipif.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from sandbox import WarmDockerPool
from sandbox.warm_pool_executor import WarmDockerPool as _WarmDockerPool
from verifiers.code_verifier import select_executor


def _docker_available():
    """Check if Docker is running."""
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


HAS_DOCKER = _docker_available()


class TestWarmPoolUnit:
    """Unit tests that don't need Docker running."""

    def test_pool_creation(self):
        pool = WarmDockerPool(pool_size=2)
        assert pool.pool_size == 2
        assert pool.name == "warm_docker_pool"
        assert len(pool._containers) == 0

    def test_is_available_returns_bool(self):
        pool = WarmDockerPool(pool_size=1)
        # Should return a bool, not raise
        result = pool.is_available()
        assert isinstance(result, bool)

    def test_get_container_returns_none_when_empty(self):
        pool = WarmDockerPool(pool_size=2)
        container, idx = pool._get_container()
        assert container is None
        assert idx == -1

    def test_get_container_round_robin(self):
        pool = WarmDockerPool(pool_size=3)
        pool._containers = [
            {"name": "c0", "uses": 0},
            {"name": "c1", "uses": 0},
            {"name": "c2", "uses": 0},
        ]
        # Round-robin: c0, c1, c2, c0, c1, ...
        results = [pool._get_container() for _ in range(7)]
        names = [r[0]["name"] for r in results]
        assert names == ["c0", "c1", "c2", "c0", "c1", "c2", "c0"]

    @pytest.mark.asyncio
    async def test_cleanup_uses_docker_command(self):
        """Regression test: cleanup() must include 'docker' in the command.

        Before the fix, cleanup() passed ["rm", "-f", name] to
        _run_docker, which tried to execute the binary "rm" instead of
        "docker rm -f name". Containers leaked on every shutdown.
        """
        pool = WarmDockerPool(
            pool_size=2, container_prefix="test_vt_regression"
        )
        pool._containers = [
            {"name": "test_container_0", "uses": 0},
            {"name": "test_container_1", "uses": 0},
        ]
        # Mock _run_docker to capture the commands
        commands = []

        async def mock_run_docker(cmd, timeout=10.0):
            commands.append(cmd)
            from sandbox.base import ExecutionResult
            return ExecutionResult(
                exit_code=0, stdout="", stderr="",
                executor="warm_docker_pool", duration_ms=10,
            )

        pool._run_docker = mock_run_docker
        await pool.cleanup()
        # Must have called "docker rm -f" for each container, not just "rm -f"
        assert len(commands) == 2
        for cmd in commands:
            assert cmd[0] == "docker", (
                f"Expected 'docker' as first arg, got {cmd[0]}"
            )
            assert cmd[1] == "rm"
            assert cmd[2] == "-f"
        assert len(pool._containers) == 0
        assert pool._started is False

    @pytest.mark.asyncio
    async def test_recycle_failure_removes_from_pool(self):
        """If container restart fails twice, the container is removed from the
        pool rather than leaving an invalid entry."""
        pool = WarmDockerPool(pool_size=2, container_prefix="test_vt_recycle")
        pool._containers = [
            {"name": "good_container", "uses": 0},
            {"name": "bad_container", "uses": 5},
        ]
        call_count = 0

        async def mock_run_docker(cmd, timeout=10.0):
            nonlocal call_count
            call_count += 1
            from sandbox.base import ExecutionResult
            # rm succeeds, but docker run always fails
            if "run" in cmd:
                return ExecutionResult(
                    exit_code=1, stdout="", stderr="docker run failed",
                    executor="warm_docker_pool", duration_ms=10,
                )
            return ExecutionResult(
                exit_code=0, stdout="", stderr="",
                executor="warm_docker_pool", duration_ms=10,
            )

        pool._run_docker = mock_run_docker
        # Recycle the bad container (index 1)
        await pool._recycle_container(1)
        # The bad container should be removed from the pool
        assert len(pool._containers) == 1
        assert pool._containers[0]["name"] == "good_container"

    @pytest.mark.asyncio
    async def test_timeout_triggers_recycle(self):
        """When an execution times out, the container is recycled (not left
        in an unknown state)."""
        pool = WarmDockerPool(pool_size=1, container_prefix="test_vt_timeout")
        pool._containers = [{"name": "timeout_container", "uses": 0}]
        recycle_called = False

        async def mock_recycle(idx):
            nonlocal recycle_called
            recycle_called = True

        pool._recycle_container = mock_recycle

        # Mock _run_docker for /tmp clean (called before execution)
        async def mock_run_docker(cmd, timeout=10.0):
            from sandbox.base import ExecutionResult
            return ExecutionResult(
                exit_code=0, stdout="", stderr="",
                executor="warm_docker_pool", duration_ms=10,
            )

        pool._run_docker = mock_run_docker

        # Mock create_subprocess_exec to return a fake process whose
        # communicate() times out (matching the real code path).
        # The production code calls asyncio.wait_for(proc.communicate(),
        # timeout=timeout). We patch wait_for to raise TimeoutError
        # immediately. To avoid "coroutine was never awaited" warnings
        # from the un-awaited communicate() coroutine, we use a Mock
        # (not AsyncMock) that returns a dummy value — wait_for raises
        # before the coroutine is ever scheduled.
        mock_proc = MagicMock()
        mock_proc.returncode = None
        # communicate() returns a dummy coroutine that is never awaited
        # because wait_for raises first. Using MagicMock (not AsyncMock)
        # means no real coroutine object is created.
        mock_proc.communicate = MagicMock(return_value=(b"", b""))
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec",
                   return_value=mock_proc), \
             patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await pool.execute("print('hello')", timeout=0.1)

        assert result.timed_out is True
        assert recycle_called is True


@pytest.mark.skipif(not HAS_DOCKER, reason="Docker not available")
class TestWarmPoolIntegration:
    """Integration tests that need Docker running."""

    @pytest.mark.asyncio
    async def test_pool_start_and_execute(self):
        pool = WarmDockerPool(pool_size=2, container_prefix="test_vt_exec")
        try:
            await pool.start()
            assert len(pool._containers) == 2
            result = await pool.execute("print(1 + 1)", timeout=10.0)
            assert result.exit_code == 0
            assert "2" in result.stdout
            assert result.executor == "warm_docker_pool"
            assert result.sandbox_name is not None
        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_execute_tests(self):
        pool = WarmDockerPool(pool_size=1, container_prefix="test_vt_tests")
        try:
            await pool.start()
            code = "def double(n):\n    return n * 2\n"
            tests = "assert double(5) == 10\nassert double(0) == 0\n"
            result = await pool.execute_tests(code, tests, timeout=10.0)
            assert result.exit_code == 0
            nonce = result.evidence.get("test_nonce")
            assert nonce is not None
            assert f"VT_PASS_{nonce}" in result.stdout
        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_failing_tests(self):
        pool = WarmDockerPool(pool_size=1, container_prefix="test_vt_fail")
        try:
            await pool.start()
            code = "def double(n):\n    return n + 1  # wrong\n"
            tests = "assert double(5) == 10\n"
            result = await pool.execute_tests(code, tests, timeout=10.0)
            assert result.exit_code != 0
            nonce = result.evidence.get("test_nonce")
            assert nonce is not None
            assert f"VT_PASS_{nonce}" not in result.stdout
        finally:
            await pool.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_removes_containers(self):
        pool = WarmDockerPool(pool_size=2, container_prefix="test_vt_clean")
        await pool.start()
        assert len(pool._containers) == 2
        await pool.cleanup()
        assert len(pool._containers) == 0
        assert pool._started is False

    @pytest.mark.asyncio
    async def test_isolation_between_executions(self):
        """Ensure /tmp is cleaned between executions (no state leakage)."""
        pool = WarmDockerPool(pool_size=1, container_prefix="test_vt_iso")
        try:
            await pool.start()
            # First execution writes a file
            await pool.execute(
                "open('/tmp/leak.txt', 'w').write('secret')", timeout=10.0
            )
            # Second execution should NOT find the file
            result = await pool.execute(
                "import os\nprint(os.path.exists('/tmp/leak.txt'))",
                timeout=10.0,
            )
            assert "False" in result.stdout
        finally:
            await pool.cleanup()


class TestSelectExecutorPreference:
    """Test that select_executor prefers warm pool when available."""

    def test_prefers_warm_pool_by_default(self):
        if not HAS_DOCKER:
            pytest.skip("Docker not available")
        executor = select_executor()
        # Should be a WarmDockerPool
        # (preferred over cold DockerSandboxExecutor)
        assert isinstance(executor, WarmDockerPool)

    def test_can_disable_warm_pool(self):
        if not HAS_DOCKER:
            pytest.skip("Docker not available")
        executor = select_executor(prefer_warm_pool=False)
        # Should fall back to DockerSandboxExecutor
        assert not isinstance(executor, WarmDockerPool)
