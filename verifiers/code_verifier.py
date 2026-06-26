"""Deterministic code verifier.

Executes Python snippets in an isolated sandbox and captures
stdout/stderr. Can run provided unit tests against a candidate solution.
Fails closed on timeout or execution error.

Security: code is run in a sandbox executor (Docker container by default,
sbx microVM for maximum isolation). The verifier does NOT fall back to
host execution for untrusted code — if no sandbox is available, it
returns verified=False with an error explaining that no sandbox was found.

Capabilities:
  - run small Python snippets in sandbox
  - timeout execution (default 5 seconds)
  - capture stdout/stderr
  - run provided unit tests
  - fail closed on timeout/error
  - fail closed if no sandbox executor available

Sandbox selection order:
  1. DockerSandboxExecutor (if Docker is available)
  2. DockerSbxExecutor (if sbx is available)
  3. LocalSubprocessExecutor (ONLY if allow_unsafe=True explicitly)
  4. None -> refuse verification
"""

import textwrap
from typing import Any, Dict, Optional

from verifiers.base import VerificationResult
from sandbox import DockerSandboxExecutor, DockerSbxExecutor, LocalSubprocessExecutor, WarmDockerPool
from sandbox.base import ExecutionResult, SandboxExecutor


def select_executor(
    allow_unsafe: bool = False,
    prefer_sbx: bool = False,
    prefer_warm_pool: bool = True,
    warm_pool_size: int = 3,
) -> Optional[SandboxExecutor]:
    """Select the best available sandbox executor.

    Selection order:
      1. DockerSbxExecutor (if prefer_sbx=True and sbx available)
      2. WarmDockerPool (if prefer_warm_pool=True and Docker available)
         — 5x faster than cold docker run for repeated verifications
      3. DockerSandboxExecutor (if Docker available)
      4. DockerSbxExecutor (if sbx available)
      5. LocalSubprocessExecutor (ONLY if allow_unsafe=True)
      6. None (refuse verification)

    Args:
        allow_unsafe: if True, allow LocalSubprocessExecutor as a fallback.
            Default False — do not run untrusted code on the host.
        prefer_sbx: if True, prefer sbx microVM over Docker container.
        prefer_warm_pool: if True, prefer WarmDockerPool over cold
            DockerSandboxExecutor. Default True — warm pools are 5x faster
            for the multi-candidate code loop.
        warm_pool_size: number of warm containers to keep running.

    Returns:
        A SandboxExecutor instance, or None if no safe executor is available.
    """
    if prefer_sbx:
        sbx = DockerSbxExecutor()
        if sbx.is_available():
            return sbx

    if prefer_warm_pool:
        pool = WarmDockerPool(pool_size=warm_pool_size)
        if pool.is_available():
            return pool

    docker = DockerSandboxExecutor()
    if docker.is_available():
        return docker

    sbx = DockerSbxExecutor()
    if sbx.is_available():
        return sbx

    if allow_unsafe:
        return LocalSubprocessExecutor()

    return None


class CodeVerifier:
    """Deterministic verifier that executes Python code in a sandbox.

    The verifier uses a SandboxExecutor for isolation. By default, it
    prefers Docker containers. If Docker is not available, it tries sbx.
    If neither is available, it REFUSES to verify — it does not fall back
    to host execution unless allow_unsafe=True is explicitly passed.
    """

    name = "code_verifier"

    def __init__(
        self,
        timeout: float = 5.0,
        executor: Optional[SandboxExecutor] = None,
        allow_unsafe: bool = False,
    ):
        """Initialize the code verifier.

        Args:
            timeout: execution timeout in seconds.
            executor: explicit sandbox executor to use. If None, one is
                selected automatically via select_executor().
            allow_unsafe: if True and no sandbox is available, fall back
                to LocalSubprocessExecutor. Default False — refuse to
                verify rather than running untrusted code on the host.
        """
        self.timeout = timeout
        self._explicit_executor = executor is not None
        self.executor: Optional[SandboxExecutor] = executor or select_executor(
            allow_unsafe=allow_unsafe,
        )

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        unit_tests = context.get("unit_tests")
        expected_output = context.get("expected_output")

        # Check that we have a sandbox executor
        if self.executor is None:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="python_exec",
                evidence={"candidate_code": answer[:200]},
                error="no sandbox executor available; refusing to run "
                      "untrusted code on host. Install Docker or sbx, "
                      "or pass allow_unsafe=True explicitly.",
            )

        # If unit tests are provided, run them against the candidate code.
        if unit_tests:
            return await self._run_unit_tests(answer, unit_tests)

        # If expected_output is provided, run the code and compare stdout.
        if expected_output is not None:
            return await self._run_and_compare_stdout(answer, str(expected_output))

        # No verification criteria provided — be honest.
        return VerificationResult(
            verified=False,
            score=0.0,
            method="python_exec",
            evidence={"candidate_code": answer[:200]},
            error="no unit_tests or expected_output provided; cannot verify",
        )

    async def _run_unit_tests(
        self, code: str, unit_tests: str
    ) -> VerificationResult:
        """Run unit tests against the candidate code in the sandbox."""
        result = await self.executor.execute_tests(
            code, unit_tests,
            timeout=self.timeout,
            network=False,
        )
        return self._interpret_test_result(result, {
            "code_length": len(code),
            "tests_length": len(unit_tests),
            "executor": result.executor,
        })

    async def _run_and_compare_stdout(
        self, code: str, expected_output: str
    ) -> VerificationResult:
        """Run code in sandbox and compare stdout to expected output."""
        result = await self.executor.execute(
            code, timeout=self.timeout, network=False,
        )
        if result.timed_out:
            return VerificationResult(
                verified=False, score=0.0, method="python_exec",
                evidence={"timeout": self.timeout, "executor": result.executor},
                error=f"execution timed out after {self.timeout}s",
            )
        if result.error and "failed" in result.error:
            return VerificationResult(
                verified=False, score=0.0, method="python_exec",
                evidence={"executor": result.executor},
                error=result.error,
            )
        actual = result.stdout.strip()
        expected = expected_output.strip()
        if actual == expected:
            return VerificationResult(
                verified=True, score=1.0, method="python_exec",
                evidence={"stdout": actual, "expected": expected,
                          "executor": result.executor},
            )
        return VerificationResult(
            verified=False, score=0.0, method="python_exec",
            evidence={"stdout": actual, "expected": expected,
                      "executor": result.executor},
            error=f"stdout mismatch: got {actual!r}, expected {expected!r}",
        )

    def _interpret_test_result(
        self, result: ExecutionResult, evidence: Dict[str, Any]
    ) -> VerificationResult:
        """Interpret a sandbox execution result for unit tests."""
        if result.timed_out:
            return VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                evidence={**evidence, "timeout": self.timeout},
                error=f"execution timed out after {self.timeout}s",
            )
        if result.error and "failed" in result.error.lower():
            return VerificationResult(
                verified=False, score=0.0, method="unit_tests",
                evidence={**evidence, "error": result.error},
                error=result.error,
            )

        stdout = result.stdout
        stderr = result.stderr
        returncode = result.exit_code

        if "ALL_TESTS_PASSED" in stdout:
            return VerificationResult(
                verified=True, score=1.0, method="unit_tests",
                evidence={**evidence, "stdout": stdout,
                          "returncode": returncode},
            )

        # Non-zero exit without success marker -> include the error detail
        error_msg = "success marker 'ALL_TESTS_PASSED' not found in stdout"
        if returncode != 0:
            for marker in ("ASSERTION_FAILED", "IMPORT_ERROR", "TEST_ERROR"):
                if marker in stdout:
                    error_msg = stdout.strip()
                    break
            else:
                error_msg = f"process exited with code {returncode}: {error_msg}"

        return VerificationResult(
            verified=False, score=0.0, method="unit_tests",
            evidence={**evidence, "stdout": stdout,
                      "stderr": stderr, "returncode": returncode},
            error=error_msg,
        )
