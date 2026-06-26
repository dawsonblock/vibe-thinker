"""Docker Sandbox (sbx) executor — microVM-based isolation.

This executor uses the sbx CLI to run code inside a microVM. Each
sandbox gets its own kernel, filesystem, and network — this is the
strongest isolation layer available.

Unlike DockerSandboxExecutor (which runs a container), DockerSbxExecutor
runs a full microVM with:
  - separate kernel per sandbox
  - no shared memory/processes with host
  - private Docker Engine inside sandbox
  - no path to host Docker daemon
  - network traffic mediated by host proxy
  - credentials injected by proxy, not copied into VM

This is heavier weight than DockerSandboxExecutor. Use it when:
  - running untrusted agent code that needs full VM isolation
  - the agent needs its own Docker daemon (e.g. to build containers)
  - you want the strongest isolation boundary available

For simple code verification (run a snippet, check output),
DockerSandboxExecutor is sufficient and faster.

Requires sbx to be installed and authenticated:
  brew install docker/tap/sbx
  sbx login
"""

import asyncio
import os
import tempfile
import time
from typing import Any, Dict, Optional

from sandbox.base import ExecutionResult


class DockerSbxExecutor:
    """Execute code in an sbx microVM sandbox.

    This is the outer layer of defense-in-depth. The sbx microVM
    isolates the entire execution environment from the host.

    Policy defaults (matching the verdict's hard policy):
      - clone mode required (agent writes only inside sandbox clone)
      - deny-all network by default
      - timeout required
      - no global secrets for code verification
    """

    name = "docker_sbx"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: float = 30.0,
        network_policy: str = "deny-all",
    ):
        self.image = image
        self.default_timeout = timeout
        self.network_policy = network_policy

    async def execute(
        self,
        script: str,
        *,
        timeout: float = 30.0,
        network: bool = False,
        memory_limit: str = "128m",
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script in an sbx microVM.

        This creates a temporary workspace, writes the script to it,
        launches an sbx sandbox, runs the script, and captures output.
        """
        start = time.monotonic()

        # Create a temporary workspace with the script
        with tempfile.TemporaryDirectory(prefix="sbx_exec_") as workspace:
            script_path = os.path.join(workspace, "verify_script.py")
            with open(script_path, "w") as f:
                f.write(script)

            # Initialize a git repo in the workspace (sbx --clone requires it)
            init_result = await self._run_command(
                ["git", "init"], cwd=workspace, timeout=5,
            )
            if init_result.exit_code != 0:
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr=init_result.stderr,
                    executor=self.name,
                    error=f"git init failed: {init_result.stderr}",
                )

            # Commit the script so clone mode can see it
            await self._run_command(
                ["git", "add", "-A"], cwd=workspace, timeout=5,
            )
            await self._run_command(
                ["git", "commit", "-m", "verify script"],
                cwd=workspace, timeout=5,
            )

            sandbox_name = f"verify-{int(start)}"

            # Build the sbx run command
            # Use --clone so the sandbox gets an isolated copy
            # Use a shell agent to run the script
            sbx_cmd = [
                "sbx", "run", "--clone",
                "--name", sandbox_name,
                "--no-attach",
            ]
            # Forward environment variables into the sandbox (e.g. the
            # verification nonce). sbx forwards these to the agent process.
            if env:
                for key, value in env.items():
                    sbx_cmd.extend(["--env", f"{key}={value}"])
            sbx_cmd.extend(["shell", "--",
                "python3", "/workspace/verify_script.py"])

            try:
                proc = await asyncio.create_subprocess_exec(
                    *sbx_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    # Clean up the sandbox
                    await self._run_command(
                        ["sbx", "rm", sandbox_name], timeout=10,
                    )
                    elapsed = int((time.monotonic() - start) * 1000)
                    return ExecutionResult(
                        exit_code=-1, stdout="", stderr="",
                        timed_out=True, executor=self.name,
                        sandbox_name=sandbox_name,
                        duration_ms=elapsed,
                        error=f"sbx execution timed out after {timeout}s",
                    )

                elapsed = int((time.monotonic() - start) * 1000)
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

                # Clean up the sandbox
                await self._run_command(
                    ["sbx", "rm", sandbox_name], timeout=10,
                )

                return ExecutionResult(
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    executor=self.name,
                    sandbox_name=sandbox_name,
                    duration_ms=elapsed,
                )

            except Exception as e:
                elapsed = int((time.monotonic() - start) * 1000)
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr="",
                    executor=self.name,
                    duration_ms=elapsed,
                    error=f"sbx execution failed: {e}",
                )

    async def execute_tests(
        self,
        code: str,
        tests: str,
        *,
        timeout: float = 30.0,
        network: bool = False,
        memory_limit: str = "128m",
    ) -> ExecutionResult:
        """Execute unit tests against candidate code in an sbx microVM."""
        from sandbox.base import VT_TEST_NONCE_ENV, build_test_harness
        script, nonce = build_test_harness(code, tests)
        result = await self.execute(
            script, timeout=timeout, network=network, memory_limit=memory_limit,
            env={VT_TEST_NONCE_ENV: nonce},
        )
        result.evidence["test_nonce"] = nonce
        return result

    async def _run_command(
        self, cmd: list, cwd: Optional[str] = None, timeout: float = 10.0,
    ) -> ExecutionResult:
        """Run a command and capture output."""
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr="",
                    timed_out=True, executor=self.name,
                    error=f"command timed out after {timeout}s",
                )
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                executor=self.name,
                duration_ms=elapsed,
            )
        except Exception as e:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                executor=self.name,
                error=f"command failed: {e}",
            )

    def is_available(self) -> bool:
        """Check if sbx is installed and authenticated."""
        import subprocess
        try:
            result = subprocess.run(
                ["sbx", "ls", "--format", "json"],
                capture_output=True, text=True, timeout=10,
            )
            # sbx ls returns 0 if authenticated, non-zero if not logged in
            return result.returncode == 0
        except Exception:
            return False

    async def cleanup(self) -> None:
        """No persistent resources — sandboxes are removed after each execution."""
        pass
