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
  - --entrypoint python3 (bypass sandbox image entrypoint, v2.0)

## NetworkMode (v0.4.2)

The executor's network behavior is controlled by :class:`NetworkMode`:

  - ``DISABLED`` (default): ``--network none``. No network access at all.
    The safest mode. Used for all untrusted code execution.
  - ``BEST_EFFORT_PROXY``: ``--network default`` with HTTP_PROXY/HTTPS_PROXY
    env vars pointing at an SNI-aware proxy. NOT a security boundary —
    clients that ignore proxy env vars (raw sockets, direct IP) can bypass
    it. Use only with trusted code that respects proxy conventions.
  - ``ENFORCED_GATEWAY``: the container runs on a Docker ``--internal``
    network with no direct internet access. The executor automatically
    starts a gateway container running the SNI egress proxy, connected to
    both the default bridge (for internet access) and the internal
    network (for sandbox access). The sandbox routes traffic through the
    gateway via ``HTTP_PROXY``/``HTTPS_PROXY`` env vars. Raw socket
    egress from the sandbox is blocked by the ``--internal`` network.
    The allowlisted gateway/proxy egress path is validated by integration
    tests: allowlisted domains are reachable, non-allowlisted domains are
    blocked (403), and raw socket bypass is blocked. Fail-closes to
    ``--network none`` if the gateway network or container cannot be
    created. Requires Docker.

When a NetworkAllowList is provided AND network_mode is BEST_EFFORT_PROXY
or ENFORCED_GATEWAY, the executor uses domain-level egress filtering.
When network_mode is DISABLED, the allow-list is ignored — no network
access is granted regardless.

## Security architecture (v3.1)

The container runs with --cap-drop ALL, --security-opt
no-new-privileges, and --user 1000:1000 (the sandbox user). The
--entrypoint python3 flag bypasses the sandbox image's legacy iptables
entrypoint (which required SETGID/SETUID caps for runuser). In proxy
mode, the container has no special capabilities at all and runs as a
non-root user — the proxy handles all filtering externally. Running as
uid 1000 (not root-with-no-caps) is strict defense-in-depth: even a
kernel cap-bypass or a misconfigured cap grant leaves the process
non-root.

When an allow-list is present, the executor resolves allow-listed
domains on the HOST at startup and injects them into the container via
Docker's ``--add-host`` flag (e.g.,
``--add-host pypi.org:151.101.0.223``). This reduces the need for DNS
(port 53) access inside the container for allow-listed domains. NOTE:
this does NOT fully disable DNS — the container may still have a DNS
resolver configured by Docker's default network settings. The
``--add-host`` entries supplement ``/etc/hosts`` but do not guarantee
that all DNS queries are blocked.

DNS can be restricted to a specific resolver via the `dns_resolver`
parameter, which is passed to the SNI proxy for DNS resolution pinning
(v2.0 wildcard DNS loophole fix).

This is the inner layer of defense-in-depth. The outer layer is the
sbx microVM that runs the entire vibe-thinker orchestrator.

Requires Docker to be installed and running on the host (or inside the
sbx microVM, which has its own Docker daemon).
"""

import asyncio
import os
import time
import uuid
from typing import Dict, Optional, TYPE_CHECKING

from sandbox.base import ExecutionResult, NetworkMode

if TYPE_CHECKING:
    from sandbox.network_allowlist import NetworkAllowList


# The sandbox image. The v0.4.0 iptables-based image is no longer used
# — v2.0 uses the SNI-proxy/Envoy egress path exclusively. Falls back
# to python:3.12-slim if the custom image is not available.
SANDBOX_IMAGE = "vibe-thinker-sandbox:latest"
FALLBACK_IMAGE = "python:3.12-slim"

# Default SNI-proxy egress address (v1.2+). When an allow-list is present,
# the executor routes traffic through a proxy at this address. The proxy
# inspects the TLS SNI / HTTP Host header (domain-level filtering) instead
# of IP-based iptables rules — solving CDN IP rotation. Override with
# --proxy-egress.
DEFAULT_PROXY_EGRESS = "127.0.0.1:8888"

# Docker internal network name for ENFORCED_GATEWAY mode. The executor
# creates this network if it doesn't exist. The network is --internal
# (no direct internet access); egress goes through a gateway container.
GATEWAY_NETWORK_NAME = "vibe-thinker-gateway-net"

# Gateway container image — uses python:3.12-slim with the sandbox/
# directory mounted as a read-only volume. No custom image build needed;
# the SNI proxy uses only stdlib (asyncio, socket, json).
GATEWAY_IMAGE = "python:3.12-slim"
GATEWAY_PORT = 8888


class DockerSandboxExecutor:
    """Execute Python code in a hardened Docker container.

    This is the default executor for CodeVerifier. It provides:
      - filesystem isolation (read-only root, writable /tmp only)
      - network isolation (controlled by NetworkMode)
      - memory limits (default 128m)
      - process limits (64 PIDs)
      - no privilege escalation
      - automatic cleanup (--rm)
      - privilege dropping (candidate runs as non-root sandbox user)

    Args:
        image: Docker image to use. Defaults to the purpose-built
            `vibe-thinker-sandbox` image.
        timeout: default execution timeout in seconds.
        allowlist: optional NetworkAllowList for granular egress
            filtering. When set AND network_mode allows network access,
            the executor uses domain-level egress filtering via an
            SNI-aware proxy.
        dns_resolver: optional IP address of a DNS resolver to restrict
            DNS queries to (prevents DNS-based data exfiltration).
        network_mode: controls the Docker network configuration.
            See :class:`NetworkMode` for the three modes. Default is
            ``NetworkMode.DISABLED`` (no network access).
        gateway_network: name of the Docker --internal network to use
            for ENFORCED_GATEWAY mode. Defaults to
            ``vibe-thinker-gateway-net``. The executor creates this
            network if it doesn't exist.
    """

    name = "docker_sandbox"

    def __init__(
        self,
        image: str = SANDBOX_IMAGE,
        timeout: float = 10.0,
        allowlist: Optional["NetworkAllowList"] = None,
        dns_resolver: Optional[str] = None,
        network_mode: Optional[NetworkMode] = None,
        gateway_network: str = GATEWAY_NETWORK_NAME,
    ):
        self.image = image
        self.default_timeout = timeout
        self._allowlist = allowlist
        self._dns_resolver = dns_resolver
        self._proxy_egress: Optional[str] = None
        # network_mode=None means auto-detect: BEST_EFFORT_PROXY when an
        # allow-list is present (backward compat with pre-v0.4.2 behavior),
        # DISABLED otherwise. Explicitly setting a mode overrides this.
        self._network_mode = network_mode
        self._gateway_network = gateway_network
        self._gateway_network_created = False
        # Gateway container lifecycle (ENFORCED_GATEWAY mode).
        self._gateway_container_name = f"vibe-thinker-gw-{uuid.uuid4().hex[:8]}"
        self._gateway_ip: Optional[str] = None
        self._gateway_started = False

    def _effective_network_mode(self) -> NetworkMode:
        """Resolve the effective network mode.

        If an explicit mode was set, use it. Otherwise auto-detect:
        BEST_EFFORT_PROXY when an allow-list is present (preserving the
        pre-v0.4.2 behavior where allow-list → proxy mode), DISABLED
        otherwise.
        """
        if self._network_mode is not None:
            return self._network_mode
        if self._allowlist is not None and not self._allowlist.is_empty:
            return NetworkMode.BEST_EFFORT_PROXY
        return NetworkMode.DISABLED

    def set_allowlist(self, allowlist: Optional["NetworkAllowList"]) -> None:
        """Update the network allow-list (e.g. from a CLI flag after init)."""
        self._allowlist = allowlist

    def set_dns_resolver(self, resolver: Optional[str]) -> None:
        """Update the DNS resolver restriction."""
        self._dns_resolver = resolver

    def set_proxy_egress(self, proxy_addr: Optional[str]) -> None:
        """Set the SNI proxy egress address (e.g. '127.0.0.1:8888').

        When set, the sandbox container routes traffic through the proxy.
        The proxy inspects the TLS SNI / HTTP Host header and allows/denies
        based on the domain — solving CDN IP rotation.
        """
        self._proxy_egress = proxy_addr

    def set_network_mode(self, mode: Optional[NetworkMode]) -> None:
        """Update the network mode (e.g. from a CLI flag after init).

        Pass None to revert to auto-detect mode (BEST_EFFORT_PROXY when
        an allow-list is present, DISABLED otherwise).
        """
        self._network_mode = mode

    async def _ensure_gateway_network(self) -> Optional[str]:
        """Create the Docker --internal network for ENFORCED_GATEWAY mode.

        Returns the network name if successful, None if Docker is not
        available or network creation failed. The network is --internal
        (no direct internet access); egress must go through a gateway
        container that the operator runs separately.
        """
        if self._gateway_network_created:
            return self._gateway_network
        try:
            # Check if the network already exists.
            check = await asyncio.create_subprocess_exec(
                "docker", "network", "inspect", self._gateway_network,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await check.wait()
            if check.returncode == 0:
                self._gateway_network_created = True
                return self._gateway_network
            # Create the internal network.
            create = await asyncio.create_subprocess_exec(
                "docker", "network", "create",
                "--internal",
                "--driver", "bridge",
                self._gateway_network,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await create.communicate()
            if create.returncode == 0:
                self._gateway_network_created = True
                print(f"[DockerSandbox] Created internal gateway network: "
                      f"{self._gateway_network}")
                return self._gateway_network
            print(f"[DockerSandbox] Failed to create gateway network: "
                  f"{stderr.decode().strip()}")
        except Exception as e:
            print(f"[DockerSandbox] Gateway network setup failed: {e}")
        return None

    async def _start_gateway(self) -> Optional[str]:
        """Start the SNI proxy gateway container.

        The gateway runs the Python SNI egress proxy (sandbox.sni_proxy)
        on ``python:3.12-slim``. It is connected to two networks:

          1. The default Docker bridge (for internet access — the proxy
             needs to resolve and tunnel to allow-listed upstreams).
          2. The ``--internal`` gateway network (so the sandbox container
             can reach the proxy).

        The sandbox container, attached only to the ``--internal``
        network, routes traffic through the gateway via
        ``HTTP_PROXY``/``HTTPS_PROXY`` env vars. Raw socket egress from
        the sandbox is blocked by the ``--internal`` network (no route
        to the outside).

        Returns the gateway's IP on the internal network, or None if
        startup failed (caller fail-closes to ``--network none``).
        """
        if self._gateway_started and self._gateway_ip:
            return self._gateway_ip
        if not self._allowlist or self._allowlist.is_empty:
            return None

        # Serialize the allowlist for the SNI proxy CLI.
        allowlist_str = ",".join(e.raw for e in self._allowlist.entries)

        # Absolute path to the sandbox/ directory for the volume mount.
        # The gateway container needs sni_proxy.py, network_allowlist.py,
        # and __init__.py — all in the sandbox/ package.
        sandbox_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            # 1. Start the gateway on the default bridge (internet access).
            cmd = [
                "docker", "run", "-d",
                "--name", self._gateway_container_name,
                "--network", "bridge",
                "--restart", "no",
                "-v", f"{sandbox_dir}:/app/sandbox:ro",
                "-w", "/app",
                GATEWAY_IMAGE,
                "python3", "-u", "-m", "sandbox.sni_proxy",
                "--host", "0.0.0.0",
                "--port", str(GATEWAY_PORT),
                "--allowlist", allowlist_str,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                print(f"[DockerSandbox] Failed to start gateway container: "
                      f"{stderr.decode().strip()}")
                return None

            # 2. Connect the gateway to the internal network.
            connect = await asyncio.create_subprocess_exec(
                "docker", "network", "connect",
                self._gateway_network, self._gateway_container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await connect.communicate()
            if connect.returncode != 0:
                print(f"[DockerSandbox] Failed to connect gateway to "
                      f"internal network: {stderr.decode().strip()}")
                await self._stop_gateway()
                return None

            # 3. Get the gateway's IP on the internal network.
            # Use index() in the Go template because the network name
            # contains hyphens (Go templates interpret hyphens as
            # subtraction operators in dot-notation field access).
            inspect = await asyncio.create_subprocess_exec(
                "docker", "inspect",
                "--format",
                f"{{{{(index .NetworkSettings.Networks \"{self._gateway_network}\").IPAddress}}}}",
                self._gateway_container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await inspect.communicate()
            if inspect.returncode != 0:
                print(f"[DockerSandbox] Failed to get gateway IP: "
                      f"{stderr.decode().strip()}")
                await self._stop_gateway()
                return None

            ip = stdout.decode().strip()
            if not ip:
                print("[DockerSandbox] Gateway IP is empty on internal network")
                await self._stop_gateway()
                return None

            # 4. Wait for the SNI proxy to be ready (check container logs
            # for the "Listening" message, with a timeout).
            ready = await self._wait_gateway_ready(timeout=10.0)
            if not ready:
                print("[DockerSandbox] Gateway proxy did not become ready "
                      "within timeout")
                await self._stop_gateway()
                return None

            self._gateway_ip = ip
            self._gateway_started = True
            print(f"[DockerSandbox] Gateway started at {ip}:{GATEWAY_PORT}")
            return ip

        except Exception as e:
            print(f"[DockerSandbox] Gateway startup failed: {e}")
            await self._stop_gateway()
            return None

    async def _wait_gateway_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the SNI proxy to print its 'Listening' message.

        Polls ``docker logs`` for the startup line. Returns True if the
        proxy is ready, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "logs", self._gateway_container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                output = (stdout + stderr).decode("utf-8", errors="replace")
                if "Listening on" in output:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    async def _stop_gateway(self) -> None:
        """Stop and remove the gateway container."""
        if not self._gateway_started and not self._gateway_container_name:
            return
        for cmd_args in (
            ["docker", "stop", self._gateway_container_name],
            ["docker", "rm", "-f", self._gateway_container_name],
        ):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except Exception:
                pass
        self._gateway_started = False
        self._gateway_ip = None

    async def execute(
        self,
        script: str,
        *,
        timeout: float = 10.0,
        network: bool = False,
        memory_limit: str = "128m",
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a Python script in a Docker container.

        The network configuration is determined by ``self._network_mode``:
          - DISABLED: ``--network none`` (ignores the ``network`` flag
            and any allow-list — no network access, period).
          - BEST_EFFORT_PROXY: ``--network default`` with proxy env vars
            when an allow-list is present. Falls back to ``--network none``
            if no allow-list is configured.
          - ENFORCED_GATEWAY: ``--network <internal-net>`` with no direct
            internet access. Egress goes through a gateway container.
        """
        start = time.monotonic()

        use_allowlist = (
            self._allowlist is not None
            and not self._allowlist.is_empty
        )
        proxy_addr = self._proxy_egress or DEFAULT_PROXY_EGRESS
        effective_mode = self._effective_network_mode()

        # Determine the Docker network configuration based on the
        # effective network mode.
        if effective_mode == NetworkMode.DISABLED:
            # No network access at all. This is the default and the safest
            # mode. The `network` flag and allow-list are both ignored.
            docker_network = "none"
            use_proxy = False
            use_gateway = False
        elif effective_mode == NetworkMode.ENFORCED_GATEWAY:
            # Internal network with no direct internet access. Egress
            # goes through a gateway container. This IS a security
            # boundary (when the gateway is correctly configured).
            gateway_net = await self._ensure_gateway_network()
            if gateway_net is not None:
                docker_network = gateway_net
                use_gateway = True
                use_proxy = use_allowlist  # proxy env vars for the gateway
                if use_proxy:
                    # Start the gateway container and get its IP on the
                    # internal network. The sandbox will route traffic
                    # through the gateway via HTTP_PROXY/HTTPS_PROXY.
                    gateway_ip = await self._start_gateway()
                    if gateway_ip is not None:
                        proxy_addr = f"{gateway_ip}:{GATEWAY_PORT}"
                    else:
                        # Gateway failed — fail-closed to no network.
                        print("[DockerSandbox] WARNING: Gateway container "
                              "startup failed — falling back to "
                              "--network none (fail-closed).")
                        docker_network = "none"
                        use_proxy = False
                        use_gateway = False
            else:
                # Gateway network setup failed — fail-closed to no network.
                print("[DockerSandbox] WARNING: ENFORCED_GATEWAY network "
                      "setup failed — falling back to --network none "
                      "(fail-closed).")
                docker_network = "none"
                use_proxy = False
                use_gateway = False
        else:
            # BEST_EFFORT_PROXY: use proxy env vars when an allow-list
            # is present. Falls back to --network none if no allow-list.
            use_proxy = use_allowlist
            use_gateway = False
            if use_proxy:
                docker_network = "default"
            else:
                docker_network = "none" if not network else "default"

        if use_proxy:
            # SNI proxy egress mode. The container routes traffic
            # through the proxy via HTTP_PROXY/HTTPS_PROXY env vars. The
            # proxy inspects the TLS SNI / HTTP Host header and allows/denies
            # based on the domain — solving CDN IP rotation. No iptables
            # needed, no NET_ADMIN needed.
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", docker_network,
                "--memory", memory_limit,
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--user", "1000:1000",
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
                "--entrypoint", "python3",
            ]
            # DNS exfiltration fix — inject allow-listed domains
            # via --add-host so the container can route to them without
            # DNS access.
            if self._allowlist is not None:
                cmd.extend(self._allowlist.generate_add_host_args())
            # Set proxy env vars so HTTP clients in the container use
            # the proxy.
            proxy_url = f"http://{proxy_addr}"
            proxy_env = {
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "NO_PROXY": "localhost,127.0.0.1",
                "no_proxy": "localhost,127.0.0.1",
            }
            merged_env = {**proxy_env}
            if env:
                merged_env.update(env)
            for key, value in merged_env.items():
                cmd.extend(["--env", f"{key}={value}"])
            cmd.extend([self.image, "-c", script])
        else:
            # No proxy — either DISABLED mode, or BEST_EFFORT_PROXY with
            # no allow-list. Use the determined docker_network.
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", docker_network,
                "--memory", memory_limit,
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--user", "1000:1000",
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
                "--entrypoint", "python3",
            ]
            if env:
                for key, value in env.items():
                    cmd.extend(["--env", f"{key}={value}"])
            cmd.extend([self.image, "-c", script])

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

            result = ExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                executor=self.name,
                duration_ms=elapsed,
            )

            # Attach audit evidence for the network configuration.
            result.evidence["network_mode"] = effective_mode.value
            result.evidence["docker_network"] = docker_network
            if use_proxy:
                result.evidence["proxy_egress"] = proxy_addr
                result.evidence["dns_restricted"] = bool(self._dns_resolver)
                result.evidence["dns_injection"] = (
                    self._allowlist is not None
                    and not self._allowlist.is_empty
                )
            if use_gateway:
                result.evidence["gateway_network"] = self._gateway_network
                result.evidence["enforced"] = True

            return result

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
            script,
            timeout=timeout,
            network=network,
            memory_limit=memory_limit,
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
        """Stop the gateway container if running. Sandbox containers use --rm."""
        await self._stop_gateway()
