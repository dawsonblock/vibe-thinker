"""Base types for sandbox execution.

The SandboxExecutor protocol is the abstraction layer between
CodeVerifier and the actual isolation backend (Docker, sbx, Shuru).
The control plane should not care which backend runs the command —
it cares about policy, evidence, exit code, and logs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ExecutionResult:
    """Result of executing code in a sandbox.

    Attributes:
        exit_code: process exit code (0 = success).
        stdout: captured stdout.
        stderr: captured stderr.
        timed_out: True if the execution hit the timeout limit.
        executor: name of the executor backend that ran this.
        sandbox_name: name of the sandbox/container if applicable.
        duration_ms: wall-clock execution time in milliseconds.
        evidence: additional structured evidence (diff summary, hashes, etc.).
        error: None if no infrastructure error, else a description.
    """
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    executor: str = "unknown"
    sandbox_name: Optional[str] = None
    duration_ms: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True if exit_code == 0 and not timed out and no infrastructure error."""
        return self.exit_code == 0 and not self.timed_out and self.error is None


@runtime_checkable
class SandboxExecutor(Protocol):
    """Protocol for isolated code execution backends.

    Implementations:
      - DockerSandboxExecutor: runs code in a Docker container with
        network isolation, memory limits, read-only filesystem.
      - DockerSbxExecutor: runs code in an sbx microVM (heavier weight,
        full VM isolation, private Docker daemon).
      - LocalSubprocessExecutor: UNSAFE — runs code directly on host.
        Only for development/testing where Docker is not available.

    The protocol is async so backends that need I/O (Docker API, sbx CLI)
    can implement it naturally.
    """

    name: str

    async def execute(
        self,
        script: str,
        *,
        timeout: float = 10.0,
        network: bool = False,
        memory_limit: str = "128m",
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script in an isolated environment.

        Args:
            script: Python source code to execute.
            timeout: maximum execution time in seconds.
            network: if True, allow network access. Default False.
                Network access should be opt-in, never default.
            memory_limit: memory limit for the execution environment.
            env: optional environment variables to set.

        Returns:
            An :class:`ExecutionResult` with stdout, stderr, exit code,
            and timing information.
        """
        ...

    async def execute_tests(
        self,
        code: str,
        tests: str,
        *,
        timeout: float = 10.0,
        network: bool = False,
        memory_limit: str = "128m",
    ) -> ExecutionResult:
        """Execute unit tests against candidate code.

        Args:
            code: the candidate Python code to test.
            tests: Python test code that imports/uses the candidate.
            timeout: maximum execution time in seconds.
            network: if True, allow network access. Default False.
            memory_limit: memory limit for the execution environment.

        Returns:
            An :class:`ExecutionResult`. Exit code 0 with
            "ALL_TESTS_PASSED" in stdout means tests passed.
        """
        ...

    def is_available(self) -> bool:
        """Check if this executor backend is available on the current system.

        Returns:
            True if the backend can be used (Docker installed, sbx
            installed, etc.), False otherwise.
        """
        ...

    async def cleanup(self) -> None:
        """Clean up any resources (containers, temp files, etc.)."""
        ...
