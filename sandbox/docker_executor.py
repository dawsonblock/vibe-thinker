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
            filtering (v0.4.0). When set, the executor uses
            --network=default + iptables rules instead of --network=none.
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

    def set_allowlist(self, allowlist: Optional["NetworkAllowList"]) -> None:
        """Update the network allow-list (e.g. from a CLI flag after init)."""
        self._allowlist = allowlist

    def set_dns_resolver(self, resolver: Optional[str]) -> None:
        """Update the DNS resolver restriction."""
        self._dns_resolver = resolver

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
        # - If an allow-list is configured, use --network=default and
        #   pass firewall rules to the entrypoint (v0.4.0 hardened).
        # - Otherwise, use the binary --network=none / --network=default
        #   based on the `network` flag (unchanged behavior).
        use_allowlist = self._allowlist is not None and not self._allowlist.is_empty
        firewall_env = self._build_firewall_env() if use_allowlist else {}
        rules_hash = self._compute_rules_hash(firewall_env) if firewall_env else ""

        if use_allowlist:
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
            if rules_hash:
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
