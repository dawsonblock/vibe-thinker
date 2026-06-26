"""Docker-based sandbox executor.

Runs Python code in a Docker container with hardening:
  - --network=none (no network access by default)
  - --memory=128m (memory limit)
  - --read-only (read-only root filesystem)
  - --security-opt=no-new-privileges (no privilege escalation)
  - --cap-drop=ALL (drop all Linux capabilities)
  - --tmpfs /tmp (writable temp dir for execution)
  - --rm (auto-remove container after exit)
  - --pids-limit=64 (process count limit)

This is the inner layer of defense-in-depth. The outer layer is the
sbx microVM that runs the entire vibe-thinker orchestrator.

Requires Docker to be installed and running on the host (or inside the
sbx microVM, which has its own Docker daemon).
"""

import asyncio
import json
import textwrap
import time
from typing import Any, Dict, Optional

from sandbox.base import ExecutionResult, SandboxExecutor


class DockerSandboxExecutor:
    """Execute Python code in a hardened Docker container.

    This is the production executor for CodeVerifier. It provides:
      - filesystem isolation (read-only root, writable /tmp only)
      - network isolation (no network by default)
      - memory limits (default 128m)
      - process limits (64 PIDs)
      - no privilege escalation
      - automatic cleanup (--rm)

    The container uses python:3.12-slim as the base image.
    """

    name = "docker_sandbox"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: float = 10.0,
    ):
        self.image = image
        self.default_timeout = timeout

    async def execute(
        self,
        script: str,
        *,
        timeout: float = 10.0,
        network: bool = False,
        memory_limit: str = "128m",
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script in a Docker container."""
        start = time.monotonic()

        # Build the docker command
        cmd = [
            "docker", "run", "--rm",
            "--network", "none" if not network else "default",
            "--memory", memory_limit,
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "64",
            "--tmpfs", "/tmp:rw,size=10m",
            "--workdir", "/tmp",
        ]

        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["--env", f"{key}={value}"])

        # Add image and command
        cmd.extend([self.image, "python3", "-c", script])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
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
                error=f"docker execution failed: {e}",
            )

    async def execute_tests(
        self,
        code: str,
        tests: str,
        *,
        timeout: float = 10.0,
        network: bool = False,
        memory_limit: str = "128m",
    ) -> ExecutionResult:
        """Execute unit tests against candidate code in Docker."""
        from sandbox.base import VT_TEST_NONCE_ENV, build_test_harness
        script, nonce = build_test_harness(code, tests)
        result = await self.execute(
            script, timeout=timeout, network=network, memory_limit=memory_limit,
            env={VT_TEST_NONCE_ENV: nonce},
        )
        result.evidence["test_nonce"] = nonce
        return result

    def is_available(self) -> bool:
        """Check if Docker is installed and the daemon is running."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    async def cleanup(self) -> None:
        """No persistent resources to clean up — containers use --rm."""
        pass
