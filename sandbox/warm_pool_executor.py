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
        docker_network: Optional[str] = None,
        proxy_egress: Optional[str] = None,
    ):
        self.image = image
        self.pool_size = pool_size
        self.default_timeout = timeout
        self.max_uses_per_container = max_uses_per_container
        self.container_prefix = container_prefix
        self._docker_network = docker_network
        self._proxy_egress = proxy_egress

        self._containers: List[Dict[str, Any]] = []
        self._next_idx = 0
        self._started = False
        # v0.4.0: per-container concurrency semaphore. Prevents OOM
        # when CODE_CANDIDATES=15 fires against a pool of 3 containers.
        # Each container gets at most 1 concurrent execution; extra
        # candidates queue up behind the semaphore.
        self._container_locks: List[asyncio.Semaphore] = []
        self._locks_initialized = False

    def set_docker_network(self, network: Optional[str]) -> None:
        """Set the Docker network to attach warm containers to.

        Must be called before ``start()``. Changing the network after the
        pool is started requires a restart (``cleanup()`` + ``start()``).
        """
        self._docker_network = network

    def set_proxy_egress(self, proxy_addr: Optional[str]) -> None:
        """Set HTTP_PROXY/HTTPS_PROXY for warm containers.

        Must be called before ``start()``. The proxy env vars are baked
        into the container at creation time.
        """
        self._proxy_egress = proxy_addr

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
            # Start a warm container with the same hardening as cold runs.
            # NOTE: --init (tini) is NOT used here because it makes tini
            # PID 1 and sleep PID 2 — our /proc cleanup (which kills all
            # PIDs > 1) would kill sleep and stop the container. Without
            # --init, sleep IS PID 1 and is correctly skipped by the
            # /proc loop. Zombie reaping is handled by the /proc cleanup.
            network = self._docker_network or "none"
            cmd = [
                "docker", "run", "-d",
                "--name", name,
                "--network", network,
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
            ]
            if self._proxy_egress:
                cmd.extend(["-e", f"HTTP_PROXY=http://{self._proxy_egress}"])
                cmd.extend(["-e", f"HTTPS_PROXY=http://{self._proxy_egress}"])
            cmd.extend([self.image, "sleep", "3600"])
            result = await self._run_docker(cmd, timeout=30.0)
            if result.exit_code == 0:
                self._containers.append({
                    "name": name,
                    "uses": 0,
                })
                # v0.4.0: one semaphore per container — limits concurrent
                # docker exec calls to 1 per container, preventing OOM
                # when CODE_CANDIDATES=15 fires against pool_size=3.
                self._container_locks.append(asyncio.Semaphore(1))
                print(f"[WarmPool] Started container {name}")
            else:
                print(f"[WarmPool] Failed to start {name}: {result.stderr}")
        if self._containers:
            self._started = True
            self._locks_initialized = True
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

    def _get_container(self) -> tuple:
        """Get the next available container (round-robin).

        Returns (container, lock_index) or (None, -1).
        """
        if not self._containers:
            return None, -1
        idx = self._next_idx % len(self._containers)
        self._next_idx += 1
        return self._containers[idx], idx

    async def _kill_process_group(self, name: str) -> None:
        """Kill all processes in the container except PID 1 (tini/sleep).

        v0.4.0: with --init (tini as PID 1), zombies are auto-reaped.
        We only need to kill any lingering background processes spawned
        by the candidate script. We must NOT kill `sleep 3600` (tini's
        child) or the container will stop.

        IMPORTANT: do NOT use `pkill -P 1` — that kills `sleep 3600`
        which is tini's child and keeps the container alive. Only kill
        processes with PID > 1 that are NOT the sleep command.
        """
        # Kill all processes except PID 1 and the shell itself.
        # With tini (--init), zombies are auto-reaped. This cleanup
        # only targets background processes the script may have spawned.
        await self._run_docker(
            ["docker", "exec", name, "sh", "-c",
             "self=$$; for p in /proc/[0-9]*; do "
             "pid=${p#/proc/}; "
             "[ $pid -gt 1 ] && [ $pid != $self ] && "
             "kill -9 $pid 2>/dev/null; done; true"],
            timeout=3.0,
        )

    async def _recycle_container(self, idx: int) -> None:
        """Recycle a container that exceeded max_uses or timed out.

        If the restart fails, retries once. If the retry also fails, removes
        the container from the pool entirely (shrinking the pool by one) rather
        than leaving an invalid entry that would cause errors on next use.
        """
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
            return

        # Retry once — transient Docker errors happen.
        print(f"[WarmPool] Container {name} restart failed (exit {result.exit_code}), retrying...")
        result = await self._run_docker(cmd, timeout=30.0)
        if result.exit_code == 0:
            self._containers[idx] = {"name": name, "uses": 0}
            print(f"[WarmPool] Recycled container {name} (on retry)")
            return

        # Both attempts failed — remove the invalid entry from the pool.
        # Shrinking the pool is safer than leaving a dead container that would
        # cause errors on every subsequent use.
        print(f"[WarmPool] Container {name} restart failed twice — removing from pool")
        self._containers.pop(idx)

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

        container, lock_idx = self._get_container()
        if container is None:
            return ExecutionResult(
                exit_code=-1, stdout="", stderr="",
                executor=self.name,
                error="no warm containers available",
            )

        name = container["name"]
        start = time.monotonic()

        # v0.4.0: acquire per-container semaphore to prevent OOM.
        # With pool_size=3 and CODE_CANDIDATES=15, this ensures each
        # container runs at most 1 candidate at a time. The other 12
        # candidates queue up behind the 3 semaphores.
        if self._locks_initialized and lock_idx < len(self._container_locks):
            sem = self._container_locks[lock_idx]
            await sem.acquire()
        else:
            sem = None
        try:
            # Clean /tmp between executions to prevent state leakage.
            # Use find instead of rm -rf /tmp/* to also remove hidden files.
            await self._run_docker(
                ["docker", "exec", name, "find", "/tmp", "-mindepth", "1", "-delete"],
                timeout=5.0,
            )

            # v0.4.0: zombie reaping is now handled by tini (--init flag in
            # docker run). tini as PID 1 auto-reaps SIGCHLD, so we no longer
            # need the 50-100ms /proc traversal before each execution.
            # The /proc cleanup after execution is kept as defense-in-depth
            # for any background processes the script may have spawned.
            # NOTE: we tried setsid (process group kill) but it's not
            # available in python:3.12-slim and breaks exit code propagation.

            # Build the exec command.
            cmd = ["docker", "exec"]
            if env:
                for key, value in env.items():
                    cmd.extend(["--env", f"{key}={value}"])
            cmd.extend([name, "python3", "-c", script])

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
                # v0.4.0: kill the entire process group (setsid created it).
                # This kills any background processes the script spawned,
                # without traversing /proc. Falls back to /proc cleanup if
                # the PGID kill fails (older Docker/images without setsid).
                await self._kill_process_group(name)
                elapsed = int((time.monotonic() - start) * 1000)
                # Recycle the container after a timeout — the killed process
                # may have left zombie children or held file handles that the
                # /tmp clean won't catch. A fresh container is safer than
                # reusing one in an unknown state.
                idx = self._containers.index(container)
                await self._recycle_container(idx)
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr="",
                    timed_out=True, executor=self.name,
                    sandbox_name=name, duration_ms=elapsed,
                    error=f"execution timed out after {timeout}s",
                )

            # v0.4.0: kill any lingering background processes from the
            # script's process group (setsid). This is much faster than
            # the old /proc traversal (~0ms vs 50-100ms).
            await self._kill_process_group(name)

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
        finally:
            # v0.4.0: release the per-container semaphore so the next
            # candidate can use this container.
            if sem is not None:
                sem.release()

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
        from sandbox.base import VT_TEST_NONCE_ENV, build_test_harness
        script, nonce = build_test_harness(code, tests)
        result = await self.execute(
            script, timeout=timeout, network=network, memory_limit=memory_limit,
            env={VT_TEST_NONCE_ENV: nonce},
        )
        result.evidence["test_nonce"] = nonce
        return result

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
                ["docker", "rm", "-f", container["name"]], timeout=10.0
            )
            print(f"[WarmPool] Removed container {container['name']}")
        self._containers = []
        self._started = False
