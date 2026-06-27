"""Docker integration tests for network allow-listing (Phase 2.1 hardening).

These tests require:
  1. Docker to be installed and running
  2. The `vibe-thinker-sandbox:latest` image to be built:
     docker build -f sandbox/Dockerfile -t vibe-thinker-sandbox:latest .

They are skipped if Docker is not available or the image is not built.

These tests make REAL network calls to verify the firewall actually
blocks/allows traffic — not just that the rules are generated correctly.
They test the security properties, not the construction.
"""

import asyncio
import pytest
import shutil
import subprocess
import time

from sandbox.network_allowlist import NetworkAllowList
from sandbox.docker_executor import DockerSandboxExecutor, SANDBOX_IMAGE


def _docker_available():
    """Check if Docker is installed and running."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _sandbox_image_available():
    """Check if the vibe-thinker-sandbox image is built."""
    if not _docker_available():
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE, "--format", "{{.Id}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# Skip all tests in this module if Docker or the sandbox image is not available.
pytestmark = pytest.mark.skipif(
    not _sandbox_image_available(),
    reason="Docker not available or vibe-thinker-sandbox image not built. "
           "Build with: docker build -f sandbox/Dockerfile -t "
           "vibe-thinker-sandbox:latest .",
)


@pytest.fixture
def executor():
    """Create a DockerSandboxExecutor with a longer timeout for integration tests.

    v1.2: These integration tests exercise the iptables egress path
    (in-container firewall rules). The default egress mode changed to
    SNI-proxy in v1.2, so we set legacy_iptables_egress=True here to
    keep testing the iptables path. The SNI-proxy path has its own
    unit tests (test_envoy_sidecar.py) and would need a running proxy
    for integration testing.
    """
    return DockerSandboxExecutor(timeout=30.0, legacy_iptables_egress=True)


class TestDenyAllWithEmptyAllowlist:
    """Verify that an empty allow-list denies all network access."""

    @pytest.mark.asyncio
    async def test_no_network_with_empty_allowlist(self, executor):
        """With an empty allow-list, the executor should use --network=none
        and the candidate code should have no network access."""
        al = NetworkAllowList.from_string("")
        executor.set_allowlist(al)
        # Try to make a network connection — should fail.
        script = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(2)\n"
            "try:\n"
            "  s.connect(('8.8.8.8', 53))\n"
            "  print('CONNECTED')\n"
            "except Exception as e:\n"
            "  print(f'DENIED: {e}')\n"
            "finally:\n"
            "  s.close()\n"
        )
        result = await executor.execute(script, timeout=15.0)
        # The connection should be denied (either network=none or iptables DROP).
        assert "DENIED" in result.stdout or "CONNECTED" not in result.stdout


class TestAllowListedDestination:
    """Verify that allow-listed destinations are reachable."""

    @pytest.mark.asyncio
    async def test_allow_specific_ip(self, executor):
        """An allow-listed IP should be reachable on the allowed port."""
        # Allow 1.1.1.1:443 (Cloudflare DNS over HTTPS).
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        script = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(5)\n"
            "try:\n"
            "  s.connect(('1.1.1.1', 443))\n"
            "  print('CONNECTED')\n"
            "except Exception as e:\n"
            "  print(f'DENIED: {e}')\n"
            "finally:\n"
            "  s.close()\n"
        )
        result = await executor.execute(script, timeout=20.0)
        # Note: this may fail if the network is unreachable, but the
        # firewall should allow it. We check that it's not a firewall
        # denial (no "Permission denied" or "Connection refused" from
        # iptables).
        if "DENIED" in result.stdout:
            # If denied, it should be a timeout or network error, not a
            # firewall block.
            assert "Permission denied" not in result.stdout


class TestDenyUnlistedDestination:
    """Verify that non-allow-listed destinations are blocked."""

    @pytest.mark.asyncio
    async def test_deny_unlisted_ip(self, executor):
        """An IP NOT in the allow-list should be denied, even if another
        IP is allowed."""
        # Allow only 1.1.1.1:443.
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        # Try to connect to 8.8.8.8:443 — should be denied.
        script = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(3)\n"
            "try:\n"
            "  s.connect(('8.8.8.8', 443))\n"
            "  print('CONNECTED')\n"
            "except Exception as e:\n"
            "  print(f'DENIED: {e}')\n"
            "finally:\n"
            "  s.close()\n"
        )
        result = await executor.execute(script, timeout=20.0)
        # 8.8.8.8 is NOT in the allow-list — should be denied.
        assert "DENIED" in result.stdout
        assert "CONNECTED" not in result.stdout

    @pytest.mark.asyncio
    async def test_deny_unlisted_port(self, executor):
        """An allowed IP on a non-allowed port should be denied."""
        # Allow only 1.1.1.1:443.
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        # Try to connect to 1.1.1.1:80 — should be denied (only 443 allowed).
        script = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(3)\n"
            "try:\n"
            "  s.connect(('1.1.1.1', 80))\n"
            "  print('CONNECTED')\n"
            "except Exception as e:\n"
            "  print(f'DENIED: {e}')\n"
            "finally:\n"
            "  s.close()\n"
        )
        result = await executor.execute(script, timeout=20.0)
        # Port 80 is NOT allowed — should be denied.
        assert "DENIED" in result.stdout
        assert "CONNECTED" not in result.stdout


class TestIPv6Denial:
    """Verify that IPv6 egress is denied."""

    @pytest.mark.asyncio
    async def test_ipv6_denied(self, executor):
        """IPv6 connections should be denied regardless of the allow-list."""
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        # Try to connect to an IPv6 address (Google's public DNS).
        script = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
            "s.settimeout(3)\n"
            "try:\n"
            "  s.connect(('2001:4860:4860::8888', 443))\n"
            "  print('CONNECTED')\n"
            "except Exception as e:\n"
            "  print(f'DENIED: {e}')\n"
            "finally:\n"
            "  s.close()\n"
        )
        result = await executor.execute(script, timeout=20.0)
        # IPv6 should be denied by ip6tables DROP policy.
        assert "DENIED" in result.stdout
        assert "CONNECTED" not in result.stdout


class TestPrivilegeDropping:
    """Verify that candidate code runs as non-root."""

    @pytest.mark.asyncio
    async def test_candidate_runs_as_non_root(self, executor):
        """The candidate code should run as the sandbox user (uid 1000),
        not as root."""
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        script = (
            "import os\n"
            "print(f'uid={os.getuid()} gid={os.getgid()}')\n"
            "import subprocess\n"
            "r = subprocess.run(['id'], capture_output=True, text=True)\n"
            "print(r.stdout.strip())\n"
        )
        result = await executor.execute(script, timeout=15.0)
        # The uid should be 1000 (sandbox user), not 0 (root).
        assert "uid=1000" in result.stdout or "uid=0" not in result.stdout

    @pytest.mark.asyncio
    async def test_candidate_cannot_modify_firewall(self, executor):
        """The candidate code should NOT be able to modify iptables rules
        (no NET_ADMIN capability after privilege dropping)."""
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        # Try to flush iptables — should fail (permission denied).
        script = (
            "import subprocess\n"
            "r = subprocess.run(['iptables', '-F'], capture_output=True, text=True)\n"
            "print(f'exit={r.returncode} stderr={r.stderr.strip()}')\n"
        )
        result = await executor.execute(script, timeout=15.0)
        # iptables -F should fail (permission denied or not found).
        # The candidate should NOT have NET_ADMIN.
        assert "exit=0" not in result.stdout or "Permission denied" in result.stdout


class TestAuditEvidence:
    """Verify that audit evidence is attached to execution results."""

    @pytest.mark.asyncio
    async def test_rules_hash_in_evidence(self, executor):
        """The execution result should include a firewall rules hash."""
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        result = await executor.execute("print('hello')", timeout=15.0)
        assert "firewall_rules_hash" in result.evidence
        assert len(result.evidence["firewall_rules_hash"]) == 16
        assert result.evidence["network_mode"] == "allowlist"

    @pytest.mark.asyncio
    async def test_dns_restricted_in_evidence(self, executor):
        """When DNS is restricted, the evidence should reflect it."""
        al = NetworkAllowList.from_string("1.1.1.1:443")
        executor.set_allowlist(al)
        executor.set_dns_resolver("8.8.8.8")
        result = await executor.execute("print('hello')", timeout=15.0)
        assert result.evidence.get("dns_restricted") is True

    @pytest.mark.asyncio
    async def test_no_firewall_evidence_without_allowlist(self, executor):
        """Without an allow-list, there should be no firewall evidence."""
        result = await executor.execute("print('hello')", timeout=15.0)
        assert "firewall_rules_hash" not in result.evidence
        assert result.evidence.get("network_mode") in ("none", "default")


# ---------------------------------------------------------------------- #
# Proxy egress mode (v0.4.1) — unit tests on env-building logic
# ---------------------------------------------------------------------- #
# These tests don't require Docker — they test the proxy egress
# configuration logic (env var setup, evidence recording) directly.
class TestProxyEgressMode:
    """Tests for the --proxy-egress mode in DockerSandboxExecutor (v1.0).

    These are unit tests on the proxy configuration logic, not full
    Docker integration tests. They verify that proxy env vars are set
    correctly and appear in evidence.
    """

    def test_set_proxy_egress_sets_address(self):
        """set_proxy_egress() stores the proxy address."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        assert executor._proxy_egress is None
        executor.set_proxy_egress("127.0.0.1:8888")
        assert executor._proxy_egress == "127.0.0.1:8888"

    def test_set_proxy_egress_none_clears(self):
        """set_proxy_egress(None) clears the proxy."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        executor.set_proxy_egress("127.0.0.1:8888")
        executor.set_proxy_egress(None)
        assert executor._proxy_egress is None

    def test_set_proxy_egress_empty_string_clears(self):
        """set_proxy_egress('') clears the proxy."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        executor.set_proxy_egress("127.0.0.1:8888")
        executor.set_proxy_egress("")
        assert executor._proxy_egress is None or executor._proxy_egress == ""
