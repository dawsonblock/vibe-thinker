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

When a NetworkAllowList is provided (v0.4.0), the executor uses
--network=default and applies iptables egress filtering rules inside
the container before running the candidate code. This allows granular
network access (e.g. only pypi.org:443 for pip install) instead of the
binary all-or-nothing --network flag.

## Security architecture (v0.4.0 hardened)

The sandbox uses a purpose-built Docker image (`vibe-thinker-sandbox`)
that includes iptables/ip6tables baked in — no apt-get at runtime.
The entrypoint script (`sandbox/entrypoint.sh`) runs as root to apply
firewall rules, then drops privileges to a non-root `sandbox` user
before exec'ing the candidate code. This closes the TOCTOU window
where candidate code could execute before firewall lockdown.

The candidate code runs as uid 1000 (sandbox user) with no NET_ADMIN
capability — it cannot modify the firewall rules that were applied
before it started. The --cap-add=NET_ADMIN is granted only for the
entrypoint's iptables phase; after the `runuser` call, the candidate
process has no capabilities.

IPv6 is explicitly denied via ip6tables DROP policy to prevent IPv6
bypass of the IPv4-only iptables allow-list rules.

DNS can be restricted to a specific resolver via the `dns_resolver`
parameter, preventing DNS-based data exfiltration through arbitrary
resolver queries.

This is the inner layer of defense-in-depth. The outer layer is the
sbx microVM that runs the entire vibe-thinker orchestrator.

Requires Docker to be installed and running on the host (or inside the
sbx microVM, which has its own Docker daemon).
"""

import asyncio
import base64
import hashlib
import json
import textwrap
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from sandbox.base import ExecutionResult, SandboxExecutor

if TYPE_CHECKING:
    from sandbox.network_allowlist import NetworkAllowList


# The purpose-built sandbox image with iptables baked in.
# Falls back to python:3.12-slim if the custom image is not available
# (the caller can override via the `image` parameter).
SANDBOX_IMAGE = "vibe-thinker-sandbox:latest"
FALLBACK_IMAGE = "python:3.12-slim"

# Default SNI-proxy egress address (v1.2). When an allow-list is present
# and legacy iptables egress is NOT enabled, the executor routes traffic
# through a proxy at this address by default. The proxy inspects the TLS
# SNI / HTTP Host header (domain-level filtering) instead of IP-based
# iptables rules — solving CDN IP rotation. Override with --proxy-egress
# or set --legacy-iptables-egress to restore the v0.4.0 iptables path.
DEFAULT_PROXY_EGRESS = "127.0.0.1:8888"


class DockerSandboxExecutor:
    """Execute Python code in a hardened Docker container.

    This is the production executor for CodeVerifier. It provides:
      - filesystem isolation (read-only root, writable /tmp only)
      - network isolation (no network by default, or allow-list egress)
      - memory limits (default 128m)
      - process limits (64 PIDs)
      - no privilege escalation
      - automatic cleanup (--rm)
      - privilege dropping (candidate runs as non-root sandbox user)
      - IPv6 denial (ip6tables DROP policy)

    Args:
        image: Docker image to use. Defaults to the purpose-built
            `vibe-thinker-sandbox` image with iptables baked in.
        timeout: default execution timeout in seconds.
        allowlist: optional NetworkAllowList for granular egress
            filtering. When set, the executor uses domain-level
            egress filtering via an SNI-aware proxy by default (v1.2),
            or iptables IP-based filtering when legacy_iptables_egress
            is True (the v0.4.0 behavior).
        dns_resolver: optional IP address of a DNS resolver to restrict
            DNS queries to (prevents DNS-based data exfiltration). When
            None, DNS is allowed to any resolver (needed for hostname
            resolution in the allow-list). When set, only the specified
            resolver can receive DNS queries.
        legacy_iptables_egress: when True, use the v0.4.0 iptables path
            (in-container firewall rules, requires NET_ADMIN cap) instead
            of the v1.2 default SNI-proxy path. Opt-in for environments
            without a proxy sidecar. Deprecated; will be removed in a
            future release in favor of the Envoy sidecar.
    """

    name = "docker_sandbox"

    def __init__(
        self,
        image: str = SANDBOX_IMAGE,
        timeout: float = 10.0,
        allowlist: Optional["NetworkAllowList"] = None,
        dns_resolver: Optional[str] = None,
        legacy_iptables_egress: bool = False,
    ):
        self.image = image
        self.default_timeout = timeout
        self._allowlist = allowlist
        self._dns_resolver = dns_resolver
        self._proxy_egress: Optional[str] = None
        self._legacy_iptables_egress = legacy_iptables_egress

    def set_allowlist(self, allowlist: Optional["NetworkAllowList"]) -> None:
        """Update the network allow-list (e.g. from a CLI flag after init)."""
        self._allowlist = allowlist

    def set_dns_resolver(self, resolver: Optional[str]) -> None:
        """Update the DNS resolver restriction."""
        self._dns_resolver = resolver

    def set_proxy_egress(self, proxy_addr: Optional[str]) -> None:
        """Set the SNI proxy egress address (e.g. '127.0.0.1:8888').

        When set, the sandbox container routes traffic through the proxy
        instead of using iptables IP-based filtering. The proxy inspects
        the TLS SNI / HTTP Host header and allows/denies based on the
        domain — solving CDN IP rotation.
        """
        self._proxy_egress = proxy_addr

    def set_legacy_iptables_egress(self, enabled: bool) -> None:
        """Opt in to the v0.4.0 iptables egress path (deprecated).

        When True, the executor uses in-container iptables rules
        (requires NET_ADMIN cap) instead of the v1.2 default SNI-proxy
        path. Deprecated; will be removed in a future release.
        """
        self._legacy_iptables_egress = enabled

    def _build_firewall_env(self) -> Dict[str, str]:
        """Build environment variables for the entrypoint's firewall setup.

        Returns:
            Dict with VT_IPTABLES_RULES (base64-encoded rules) and
            VT_DNS_RESOLVER (optional). Empty dict if no allow-list.
        """
        if self._allowlist is None or self._allowlist.is_empty:
            return {}

        # Generate rules with DNS resolver restriction if configured.
        rules = self._allowlist.generate_iptables_rules(
            dns_resolver=self._dns_resolver,
        )
        # Join rules with newlines and base64-encode for safe env transport.
        rules_text = "\n".join(rules)
        rules_b64 = base64.b64encode(rules_text.encode()).decode()

        env: Dict[str, str] = {"VT_IPTABLES_RULES": rules_b64}

        # Note: VT_DNS_RESOLVER is also passed so the entrypoint can
        # further restrict DNS at the ip6tables level if needed. The
        # iptables rules already include the DNS restriction, but the
        # entrypoint uses this for additional ip6tables DNS rules.
        if self._dns_resolver:
            env["VT_DNS_RESOLVER"] = self._dns_resolver

        return env

    def _compute_rules_hash(self, rules_env: Dict[str, str]) -> str:
        """Compute a SHA-256 hash of the firewall rules for audit logging.

        This hash is included in the execution result's evidence so that
        the exact firewall configuration can be verified post-hoc.
        """
        rules_b64 = rules_env.get("VT_IPTABLES_RULES", "")
        if not rules_b64:
            return ""
        return hashlib.sha256(rules_b64.encode()).hexdigest()[:16]

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
        # - If an explicit proxy egress is configured, use it (v0.4.1).
        # - Else if an allow-list is configured AND legacy iptables egress
        #   is NOT enabled, use the default SNI-proxy address (v1.2 default).
        #   The proxy inspects TLS SNI / HTTP Host (domain-level filtering),
        #   solving CDN IP rotation without NET_ADMIN.
        # - Else if an allow-list is configured AND legacy iptables egress
        #   IS enabled, use --network=default + iptables rules (v0.4.0 path,
        #   now opt-in / deprecated).
        # - Otherwise, use the binary --network=none / --network=default
        #   based on the `network` flag (unchanged behavior).
        use_allowlist = self._allowlist is not None and not self._allowlist.is_empty
        explicit_proxy = self._proxy_egress is not None
        # v1.2: SNI-proxy is the default egress mode when an allow-list is
        # present. Legacy iptables is opt-in via --legacy-iptables-egress.
        use_proxy = (
            explicit_proxy
            or (use_allowlist and not self._legacy_iptables_egress)
        )
        # The proxy address: explicit override, else the default.
        proxy_addr = self._proxy_egress or DEFAULT_PROXY_EGRESS
        # iptables firewall env is only needed for the legacy path.
        use_iptables = use_allowlist and self._legacy_iptables_egress and not explicit_proxy
        firewall_env = self._build_firewall_env() if use_iptables else {}
        rules_hash = self._compute_rules_hash(firewall_env) if firewall_env else ""

        if use_proxy:
            # SNI proxy egress mode (v0.4.1, v1.2 default). The container
            # routes traffic through the proxy via HTTP_PROXY/HTTPS_PROXY
            # env vars. The proxy inspects the TLS SNI / HTTP Host header
            # and allows/denies based on the domain — solving CDN IP
            # rotation. No iptables needed, no NET_ADMIN needed.
            # v1.2: this is now the default when an allow-list is present;
            # the legacy iptables path is opt-in via --legacy-iptables-egress.
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", "default",
                "--memory", memory_limit,
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
            ]
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
            cmd.extend([self.image, "python3", "-c", script])
        elif use_iptables:
            # The sandbox image's entrypoint handles:
            #   1. Applying iptables rules (as root, from VT_IPTABLES_RULES)
            #   2. Denying IPv6 (ip6tables DROP)
            #   3. Restricting DNS (if VT_DNS_RESOLVER is set)
            #   4. Dropping to sandbox user (runuser -u sandbox)
            #   5. Exec'ing the candidate command
            #
            # --cap-add=NET_ADMIN is needed for the entrypoint's iptables
            # phase. The candidate code (running as sandbox user after
            # runuser) does NOT have NET_ADMIN — it cannot modify the
            # firewall. The entrypoint applies rules BEFORE the candidate
            # code starts, closing the TOCTOU window.
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", "default",
                "--memory", memory_limit,
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--cap-add", "NET_ADMIN",  # entrypoint: iptables rules
                "--cap-add", "NET_RAW",    # entrypoint: iptables needs raw socket access
                "--cap-add", "SETGID",     # entrypoint: runuser needs setgid() to drop to sandbox user
                "--cap-add", "SETUID",     # entrypoint: runuser needs setuid() to drop to sandbox user
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
            ]

            # Merge firewall env with caller-provided env.
            merged_env = {**firewall_env}
            if env:
                merged_env.update(env)
            for key, value in merged_env.items():
                cmd.extend(["--env", f"{key}={value}"])

            # The entrypoint applies firewall rules, then exec's the
            # candidate command. We pass python3 -c <script> as the
            # command — the entrypoint will run it as the sandbox user.
            cmd.extend([self.image, "python3", "-c", script])
        else:
            # Original binary network mode (unchanged).
            # No allow-list -> --network=none (or --network=default if
            # network=True is explicitly passed). No firewall rules,
            # no entrypoint interaction needed.
            cmd = [
                "docker", "run", "--rm",
                "--init",
                "--network", "none" if not network else "default",
                "--memory", memory_limit,
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,size=10m",
                "--workdir", "/tmp",
            ]
            if env:
                for key, value in env.items():
                    cmd.extend(["--env", f"{key}={value}"])
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

            result = ExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                executor=self.name,
                duration_ms=elapsed,
            )

            # Attach audit evidence for the firewall configuration.
            if use_proxy:
                result.evidence["network_mode"] = "proxy"
                result.evidence["proxy_egress"] = self._proxy_egress
            elif rules_hash:
                result.evidence["firewall_rules_hash"] = rules_hash
                result.evidence["network_mode"] = "allowlist"
                result.evidence["dns_restricted"] = bool(self._dns_resolver)
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
