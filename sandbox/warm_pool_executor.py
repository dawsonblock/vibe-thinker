"""Warm-pool Docker executor — keeps containers running for fast exec.

The standard DockerSandboxExecutor runs `docker run --rm` for every
verification, which incurs ~1.8s cold-start overhead per call (container
creation + Python interpreter startup). For the multi-candidate code loop
where the 0.5B model generates code in ~200ms, this Docker overhead is the
dominant bottleneck — 5x slower than the model itself.

The WarmDockerPool keeps N pre-started containers running in the background
(`docker run -d ... sleep infinity`) and executes code via `docker exec`,
which takes ~0.35s — a 5x speedup.

Security: warm containers use the same hardening as cold ones:
  - --network=none (no network access)
  - --read-only (read-only root filesystem)
  - --security-opt=no-new-privileges
  - --cap-drop=ALL
  - --pids-limit=64
  - --tmpfs /tmp (writable temp dir for execution only)

The pool reuses containers across verifications. Between executions, the
container's /tmp is cleaned to prevent state leakage between candidates.
Containers are recycled after max_uses executions to prevent resource
accumulation (file handles, zombie processes).
"""

import asyncio
import textwrap
import time
from typing import Any, Dict, List, Optional

from sandbox.base import ExecutionResult, SandboxExecutor


class WarmDockerPool:
    """Pool of warm Docker containers for fast code execution.

    Keeps `pool_size` containers running and dispatches executions to them
    round-robin via `docker exec`. Containers are recycled after
    `max_uses_per_container` executions.
    """

    name = "warm_docker_pool"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        pool_size: int = 3,
        timeout: float = 10.0,
        max_uses_per_container: int = 50,
        container_prefix: str = "vibe_sbx",
    ):
        self.image = image
        self.pool_size = pool_size
        self.default_timeout = timeout
        self.max_uses_per_container = max_uses_per_container
        self.container_prefix = container_prefix

        self._containers: List[Dict[str, Any]] = []
        self._next_idx = 0
        self._started = False

    async def start(self) -> None:
        """Start the warm container pool."""
        if self._started:
            return
        for i in range(self.pool_size):
            name = f"{self.container_prefix}_{i}"
            # Remove stale container if it exists (ignore errors)
            await self._run_docker(["docker", "rm", "-f", name], timeout=10.0)
            # Brief pause to let Docker daemon release the name
            await asyncio.sleep(0.1)
            # Start a warm container with the same hardening as cold runs
            cmd = [
                "docker", "run", "-d",
                "--name", name,
                "--network", "none",
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
                self.image, "sleep", "3600",
            ]
            result = await self._run_docker(cmd, timeout=30.0)
            if result.exit_code == 0:
                self._containers.append({
                    "name": name,
                    "uses": 0,
                })
                print(f"[WarmPool] Started container {name}")
            else:
                print(f"[WarmPool] Failed to start {name}: {result.stderr}")
        if self._containers:
            self._started = True
        else:
            print(f"[WarmPool] No containers started — will fall back to cold runs")

    async def _run_docker(
        self, cmd: List[str], timeout: float = 10.0
    ) -> ExecutionResult:
        """Run a docker CLI command and capture output."""
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = int((time.monotonic() - start) * 1000)
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr="",
                    timed_out=True, executor=self.name,
                    duration_ms=elapsed,
                    error=f"docker command timed out after {timeout}s",
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
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                executor=self.name, duration_ms=elapsed,
                error=f"docker command failed: {e}",
            )

    def _get_container(self) -> Optional[Dict[str, Any]]:
        """Get the next available container (round-robin)."""
        if not self._containers:
            return None
        container = self._containers[self._next_idx % len(self._containers)]
        self._next_idx += 1
        return container

    async def _recycle_container(self, idx: int) -> None:
        """Recycle a container that exceeded max_uses."""
        container = self._containers[idx]
        name = container["name"]
        await self._run_docker(["docker", "rm", "-f", name], timeout=10.0)
        # Restart it
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--network", "none",
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "64",
            "--tmpfs", "/tmp:rw,size=10m",
            "--workdir", "/tmp",
            self.image, "sleep", "3600",
        ]
        result = await self._run_docker(cmd, timeout=30.0)
        if result.exit_code == 0:
            self._containers[idx] = {"name": name, "uses": 0}
            print(f"[WarmPool] Recycled container {name}")

    async def execute(
        self,
        script: str,
        *,
        timeout: float = 10.0,
        network: bool = False,
        memory_limit: str = "128m",
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script in a warm container via docker exec."""
        if not self._containers:
            # Fallback: start the pool if not started
            await self.start()
            if not self._containers:
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr="",
                    executor=self.name,
                    error="no warm containers available",
                )

        container = self._get_container()
        if container is None:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                executor=self.name,
                error="no warm containers available",
            )

        name = container["name"]
        start = time.monotonic()

        # Clean /tmp between executions to prevent state leakage.
        # Use find instead of rm -rf /tmp/* to also remove hidden files.
        await self._run_docker(
            ["docker", "exec", name, "find", "/tmp", "-mindepth", "1", "-delete"],
            timeout=5.0,
        )

        # Build the exec command
        cmd = ["docker", "exec"]
        if env:
            for key, value in env.items():
                cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([name, "python3", "-c", script])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = int((time.monotonic() - start) * 1000)
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr="",
                    timed_out=True, executor=self.name,
                    sandbox_name=name, duration_ms=elapsed,
                    error=f"execution timed out after {timeout}s",
                )

            elapsed = int((time.monotonic() - start) * 1000)
            container["uses"] += 1

            # Recycle if needed
            if container["uses"] >= self.max_uses_per_container:
                idx = self._containers.index(container)
                await self._recycle_container(idx)

            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                executor=self.name,
                sandbox_name=name,
                duration_ms=elapsed,
            )
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                executor=self.name, sandbox_name=name,
                duration_ms=elapsed,
                error=f"docker exec failed: {e}",
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
        """Execute unit tests against candidate code in a warm container."""
        code_clean = textwrap.dedent(code).strip()
        tests_clean = textwrap.dedent(tests).strip()
        script = (
            "import sys, traceback\n"
            "try:\n"
            + textwrap.indent(code_clean, "    ") + "\n"
            "except Exception as e:\n"
            "    print(f'IMPORT_ERROR: {e}')\n"
            "    sys.exit(1)\n"
            "try:\n"
            + textwrap.indent(tests_clean, "    ") + "\n"
            "    print('ALL_TESTS_PASSED')\n"
            "except AssertionError as e:\n"
            "    print(f'ASSERTION_FAILED: {e}')\n"
            "    sys.exit(1)\n"
            "except Exception as e:\n"
            "    print(f'TEST_ERROR: {e}')\n"
            "    traceback.print_exc()\n"
            "    sys.exit(1)\n"
        )
        return await self.execute(
            script, timeout=timeout, network=network, memory_limit=memory_limit,
        )

    def is_available(self) -> bool:
        """Check if Docker is installed and running."""
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
        """Remove all warm containers."""
        for container in self._containers:
            await self._run_docker(
                ["rm", "-f", container["name"]], timeout=10.0
            )
            print(f"[WarmPool] Removed container {container['name']}")
        self._containers = []
        self._started = False
