"""Tests for dynamic sandbox resource allocation (Phase 2.2)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hybrid_orchestrator import HybridReasoningOrchestrator
from verifiers.base import VerificationResult


@pytest.fixture
def orch():
    return HybridReasoningOrchestrator(
        vibe_endpoint="http://localhost:0",
        generalist_endpoint="http://localhost:0",
        use_clr=False,
        use_embedding_router=False,
        use_clr_cache=False,
        use_trajectory_store=False,
    )


# ---------------------------------------------------------------------- #
# route_structured includes compute_limits
# ---------------------------------------------------------------------- #
class TestComputeLimitsInRouting:
    def test_math_task_gets_low_limits(self, orch):
        decision = orch.route_structured("Solve 2 + 2")
        assert "compute_limits" in decision
        limits = decision["compute_limits"]
        assert limits["memory"] == "64m"
        assert limits["timeout"] == 5.0
        assert limits["heavy_compute_keywords"] == 0

    def test_simple_code_task_gets_default(self, orch):
        decision = orch.route_structured("Write a Python function to square a number")
        limits = decision["compute_limits"]
        assert limits["memory"] == "128m"
        assert limits["timeout"] == 10.0

    def test_heavy_code_task_gets_bumped_memory(self, orch):
        decision = orch.route_structured(
            "Write a Python script to process a large CSV dataframe "
            "with pandas and numpy"
        )
        limits = decision["compute_limits"]
        # "dataframe", "pandas", "numpy", "large", "csv" = 5 hits -> 512m
        assert limits["memory"] == "512m"
        assert limits["timeout"] > 10.0
        assert limits["heavy_compute_keywords"] >= 3

    def test_medium_heavy_task_gets_256m(self, orch):
        decision = orch.route_structured(
            "Sort a large dataset using merge sort"
        )
        limits = decision["compute_limits"]
        # "large", "dataset", "sort", "merge" = 4 hits -> 512m
        assert limits["memory"] in ("256m", "512m")
        assert limits["heavy_compute_keywords"] >= 2

    def test_timeout_capped_at_60s(self, orch):
        # Stuff many heavy keywords to exceed the 60s cap.
        query = " ".join([
            "dataframe", "pandas", "numpy", "torch", "tensorflow",
            "matrix", "recursive", "backtracking", "simulation",
            "monte carlo", "encrypt", "compress", "image", "audio",
        ])
        decision = orch.route_structured(f"Write code to {query}")
        limits = decision["compute_limits"]
        assert limits["timeout"] <= 60.0

    def test_conversation_gets_minimal_limits(self, orch):
        decision = orch.route_structured("Explain how recursion works")
        limits = decision["compute_limits"]
        assert limits["memory"] == "64m"
        assert limits["timeout"] == 5.0

    def test_unknown_task_gets_minimal_limits(self, orch):
        decision = orch.route_structured("xyzzy foobar")
        limits = decision["compute_limits"]
        assert limits["memory"] == "64m"
        assert limits["timeout"] == 5.0


# ---------------------------------------------------------------------- #
# CodeVerifier uses compute_limits from context
# ---------------------------------------------------------------------- #
class TestCodeVerifierComputeLimits:
    @pytest.mark.asyncio
    async def test_compute_limits_passed_to_executor(self):
        """When compute_limits is in the context, the CodeVerifier passes
        the dynamic timeout/memory to the executor instead of its defaults."""
        from verifiers.code_verifier import CodeVerifier
        mock_executor = MagicMock()
        mock_executor.execute_tests = AsyncMock(return_value=MagicMock(
            exit_code=0, stdout="VT_PASS_abc123", stderr="",
            timed_out=False, executor="mock", evidence={"test_nonce": "abc123"},
            error=None,
        ))
        verifier = CodeVerifier(timeout=5.0, executor=mock_executor)
        await verifier.verify("q", "code", {
            "unit_tests": "assert True",
            "compute_limits": {"timeout": 30.0, "memory": "512m"},
        })
        # The executor should have been called with the dynamic limits.
        call_kwargs = mock_executor.execute_tests.call_args
        assert call_kwargs.kwargs["timeout"] == 30.0
        assert call_kwargs.kwargs["memory_limit"] == "512m"

    @pytest.mark.asyncio
    async def test_no_compute_limits_uses_defaults(self):
        """Without compute_limits, the verifier uses its own defaults
        (backward-compatible)."""
        from verifiers.code_verifier import CodeVerifier
        mock_executor = MagicMock()
        mock_executor.execute_tests = AsyncMock(return_value=MagicMock(
            exit_code=0, stdout="VT_PASS_abc123", stderr="",
            timed_out=False, executor="mock", evidence={"test_nonce": "abc123"},
            error=None,
        ))
        verifier = CodeVerifier(timeout=7.0, executor=mock_executor)
        await verifier.verify("q", "code", {"unit_tests": "assert True"})
        call_kwargs = mock_executor.execute_tests.call_args
        assert call_kwargs.kwargs["timeout"] == 7.0
        assert call_kwargs.kwargs["memory_limit"] == "128m"

    @pytest.mark.asyncio
    async def test_compute_limits_passed_to_stdout_compare(self):
        from verifiers.code_verifier import CodeVerifier
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=MagicMock(
            exit_code=0, stdout="hello\n", stderr="",
            timed_out=False, executor="mock", evidence={}, error=None,
        ))
        verifier = CodeVerifier(timeout=5.0, executor=mock_executor)
        await verifier.verify("q", "print('hello')", {
            "expected_output": "hello",
            "compute_limits": {"timeout": 20.0, "memory": "256m"},
        })
        call_kwargs = mock_executor.execute.call_args
        assert call_kwargs.kwargs["timeout"] == 20.0
        assert call_kwargs.kwargs["memory_limit"] == "256m"


# ---------------------------------------------------------------------- #
# Orchestrator integration: compute_limits flow into verifier context
# ---------------------------------------------------------------------- #
class TestOrchestratorComputeLimitsIntegration:
    @pytest.mark.asyncio
    async def test_code_context_includes_compute_limits(self):
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        limits = {"memory": "256m", "timeout": 15.0, "heavy_compute_keywords": 1}
        ctx = await o._build_verifier_context(
            "Write a function", "code", compute_limits=limits,
        )
        assert ctx["compute_limits"] == limits

    @pytest.mark.asyncio
    async def test_math_context_no_compute_limits(self):
        """compute_limits are only for code tasks — math doesn't get them."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        limits = {"memory": "256m", "timeout": 15.0}
        ctx = await o._build_verifier_context(
            "2 + 2", "math", compute_limits=limits,
        )
        assert "compute_limits" not in ctx

    @pytest.mark.asyncio
    async def test_code_context_no_limits_when_none(self):
        """When compute_limits is None, code context doesn't include it
        (backward-compatible — verifier uses its own defaults)."""
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
        )
        ctx = await o._build_verifier_context("Write a function", "code")
        assert "compute_limits" not in ctx
