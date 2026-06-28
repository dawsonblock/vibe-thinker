"""Local subprocess executor — UNSAFE, for development only.

This executor runs Python code directly on the host in a subprocess.
It provides timeout protection but NO filesystem, network, or memory
isolation. It exists for development and testing where Docker is not
available.

WARNING: Do NOT use this executor for untrusted model output.
The entire point of the sandbox layer is to prevent running generated
code directly on the host. This executor is explicitly marked unsafe
and should only be used:
  - in CI environments that are already isolated
  - in development with trusted test code
  - as a fallback when Docker is not available AND the code is trusted

For any real use, use DockerSandboxExecutor or DockerSbxExecutor.
"""

import asyncio
import textwrap
import time
from typing import Any, Dict, Optional

from sandbox.base import ExecutionResult


class LocalSubprocessExecutor:
    """Execute Python code in a local subprocess.

    UNSAFE: No filesystem, network, or memory isolation.
    For development/testing only.
    """

    name = "local_subprocess_unsafe"

    def __init__(self, timeout: float = 5.0):
        self.default_timeout = timeout

    async def execute(
        self,
        script: str,
        *,
        timeout: float = 5.0,
        network: bool = False,
        memory_limit: str = "128m",
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script in a local subprocess.

        The network and memory_limit parameters are accepted for API
        compatibility but are NOT enforced — this executor cannot
        isolate network or memory.
        """
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = int((time.monotonic() - start) * 1000)
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    timed_out=True,
                    executor=self.name,
                    duration_ms=elapsed,
                    error=f"execution timed out after {timeout}s",
                )

            elapsed = int((time.monotonic() - start) * 1000)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                executor=self.name,
                duration_ms=elapsed,
            )
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="",
                executor=self.name,
                duration_ms=elapsed,
                error=f"execution failed: {e}",
            )

    async def execute_tests(
        self,
        code: str,
        tests: str,
        *,
        timeout: float = 5.0,
        network: bool = False,
        memory_limit: str = "128m",
    ) -> ExecutionResult:
        """Execute unit tests against candidate code locally."""
        from sandbox.base import VT_TEST_NONCE_ENV, build_test_harness
        script, nonce = build_test_harness(code, tests)
        result = await self.execute(
            script, timeout=timeout, network=network, memory_limit=memory_limit,
            env={VT_TEST_NONCE_ENV: nonce},
        )
        result.evidence["test_nonce"] = nonce
        return result

    def is_available(self) -> bool:
        """Always available — python3 is required for the system itself."""
        return True

    async def cleanup(self) -> None:
        """No persistent resources to clean up."""
        pass
