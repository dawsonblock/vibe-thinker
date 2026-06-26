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
        assert pool._get_container() is None

    def test_get_container_round_robin(self):
        pool = WarmDockerPool(pool_size=3)
        pool._containers = [
            {"name": "c0", "uses": 0},
            {"name": "c1", "uses": 0},
            {"name": "c2", "uses": 0},
        ]
        c0 = pool._get_container()
        c1 = pool._get_container()
        c2 = pool._get_container()
        c3 = pool._get_container()  # wraps around
        assert c0["name"] == "c0"
        assert c1["name"] == "c1"
        assert c2["name"] == "c2"
        assert c3["name"] == "c0"  # round-robin


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
            assert "ALL_TESTS_PASSED" in result.stdout
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
            assert "ALL_TESTS_PASSED" not in result.stdout
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
        # Should be a WarmDockerPool (preferred over cold DockerSandboxExecutor)
        assert isinstance(executor, WarmDockerPool)

    def test_can_disable_warm_pool(self):
        if not HAS_DOCKER:
            pytest.skip("Docker not available")
        executor = select_executor(prefer_warm_pool=False)
        # Should fall back to DockerSandboxExecutor
        assert not isinstance(executor, WarmDockerPool)
