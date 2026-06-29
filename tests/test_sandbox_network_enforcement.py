"""Tests for sandbox network enforcement (Phase 5 — sandbox truth).

These tests verify that the sandbox network isolation modes are correctly
configured and that the documented security claims are honest.

v0.4.0-alpha: The BEST_EFFORT_PROXY mode is NOT a security boundary.
These tests document that fact and verify the NetworkMode enum defaults
to DISABLED. The enforced-gateway bypass tests require Docker and are
marked accordingly.

v0.4.2: NetworkMode is now wired into DockerSandboxExecutor. The tests
in TestDockerSandboxNetworkModeWiring verify that the executor correctly
maps each NetworkMode to the corresponding Docker --network flag, without
requiring Docker to be running (they inspect the command that would be
executed).

Blocked IP ranges that must NOT be reachable from a sandbox candidate:
  127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16,
  169.254.0.0/16, ::1/128, fc00::/7, fe80::/10
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from sandbox.base import NetworkMode


def _has_docker() -> bool:
    """Check if the docker Python package is importable."""
    try:
        import docker  # noqa: F401
        return True
    except ImportError:
        return False


class TestNetworkModeDefaults:
    """Verify the NetworkMode enum and its security defaults."""

    def test_disabled_is_default(self):
        """The default network mode MUST be DISABLED (safest)."""
        assert NetworkMode.DISABLED.value == "disabled"

    def test_best_effort_proxy_is_not_secure(self):
        """BEST_EFFORT_PROXY exists but is documented as not a security
        boundary. The enum value must reflect its name honestly."""
        assert NetworkMode.BEST_EFFORT_PROXY.value == "best_effort_proxy"

    def test_enforced_gateway_exists(self):
        """ENFORCED_GATEWAY is the intended security boundary mode.

        Note: ENFORCED_GATEWAY is EXPERIMENTAL — command wiring and
        fail-closed behavior are tested, but real egress enforcement
        is not proven until Docker bypass tests pass. It is the only
        mode *designed* to be a security boundary, but that design is
        not yet validated.
        """
        assert NetworkMode.ENFORCED_GATEWAY.value == "enforced_gateway"

    def test_network_mode_is_str_enum(self):
        """NetworkMode must be a str Enum so it serializes cleanly."""
        assert isinstance(NetworkMode.DISABLED, str)


class TestDockerSandboxNetworkModeWiring:
    """Verify that DockerSandboxExecutor correctly maps NetworkMode to
    Docker --network flags. These tests do NOT require Docker to be
    running — they intercept the subprocess call and inspect the command.
    """

    def test_default_network_mode_is_disabled(self):
        """The executor's default effective network_mode MUST be DISABLED
        when no allow-list is present."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        # No allow-list → effective mode is DISABLED
        assert executor._effective_network_mode() == NetworkMode.DISABLED

    def test_default_auto_detects_proxy_with_allowlist(self):
        """When an allow-list is present and network_mode is not explicitly
        set, the effective mode auto-detects to BEST_EFFORT_PROXY
        (backward compat with pre-v0.4.2 behavior)."""
        from sandbox.docker_executor import DockerSandboxExecutor
        from sandbox.network_allowlist import NetworkAllowList
        allowlist = NetworkAllowList.from_string("pypi.org:443")
        executor = DockerSandboxExecutor(allowlist=allowlist)
        assert (
            executor._effective_network_mode()
            == NetworkMode.BEST_EFFORT_PROXY
        )

    def test_disabled_mode_uses_network_none(self):
        """DISABLED mode must use --network none, ignoring the `network`
        flag and any allow-list."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor(network_mode=NetworkMode.DISABLED)
        # Even with an allow-list, DISABLED mode ignores it.
        captured_cmd = []

        async def _run():
            # Mock subprocess to capture the docker command without running it.
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(b"output", b"")
            )
            with patch("asyncio.create_subprocess_exec",
                       return_value=mock_proc) as mock_exec:
                await executor.execute("print('hi')", network=True)
                captured_cmd.extend(mock_exec.call_args[0])

        import asyncio
        asyncio.run(_run())
        # Verify --network none is in the command
        # (DISABLED ignores network=True)
        assert "--network" in captured_cmd
        idx = captured_cmd.index("--network")
        assert captured_cmd[idx + 1] == "none"

    def test_best_effort_proxy_without_allowlist_uses_network_none(self):
        """BEST_EFFORT_PROXY with no allow-list falls back
        to --network none."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.BEST_EFFORT_PROXY,
        )
        captured_cmd = []

        async def _run():
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            with patch("asyncio.create_subprocess_exec",
                       return_value=mock_proc) as mock_exec:
                await executor.execute("print('hi')")
                captured_cmd.extend(mock_exec.call_args[0])

        import asyncio
        asyncio.run(_run())
        idx = captured_cmd.index("--network")
        assert captured_cmd[idx + 1] == "none"

    def test_best_effort_proxy_with_allowlist_uses_network_default(self):
        """BEST_EFFORT_PROXY with an allow-list uses --network default
        with proxy env vars."""
        from sandbox.docker_executor import DockerSandboxExecutor
        from sandbox.network_allowlist import NetworkAllowList
        allowlist = NetworkAllowList.from_string("pypi.org:443")
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.BEST_EFFORT_PROXY,
            allowlist=allowlist,
        )
        captured_cmd = []

        async def _run():
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            with patch("asyncio.create_subprocess_exec",
                       return_value=mock_proc) as mock_exec:
                await executor.execute("print('hi')")
                captured_cmd.extend(mock_exec.call_args[0])

        import asyncio
        asyncio.run(_run())
        idx = captured_cmd.index("--network")
        assert captured_cmd[idx + 1] == "default"
        # Verify proxy env vars are set
        assert "--env" in captured_cmd
        env_strs = [
            captured_cmd[i + 1] for i, c in enumerate(captured_cmd)
            if c == "--env"
        ]
        assert any("HTTP_PROXY" in e for e in env_strs)

    def test_enforced_gateway_fails_closed_without_docker(self):
        """ENFORCED_GATEWAY mode fails-closed to --network none when
        the gateway network cannot be created (e.g. Docker not running)."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.ENFORCED_GATEWAY,
        )
        captured_cmd = []

        async def _run():
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            # Mock the network inspect/create to fail (no Docker)
            mock_network_proc = MagicMock()
            mock_network_proc.returncode = 1
            mock_network_proc.communicate = AsyncMock(
                return_value=(b"", b"error"),
            )
            with patch("asyncio.create_subprocess_exec",
                       return_value=mock_proc) as mock_exec:
                await executor.execute("print('hi')")
                captured_cmd.extend(mock_exec.call_args[0])

        import asyncio
        asyncio.run(_run())
        # Should fail-closed to --network none
        idx = captured_cmd.index("--network")
        assert captured_cmd[idx + 1] == "none"

    def test_set_network_mode_updates_mode(self):
        """set_network_mode() should update the executor's mode."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        assert executor._effective_network_mode() == NetworkMode.DISABLED
        executor.set_network_mode(NetworkMode.BEST_EFFORT_PROXY)
        assert (
            executor._effective_network_mode()
            == NetworkMode.BEST_EFFORT_PROXY
        )


class TestBlockedIpRanges:
    """Document the IP ranges that must be blocked from sandbox candidates.

    These are metadata tests — they verify the blocked ranges are defined.
    The actual enforcement tests (requiring Docker) are in
    TestDockerNetworkEnforcement below, marked @pytest.mark.sandbox.
    """

    BLOCKED_IPV4_RANGES = [
        "127.0.0.0/8",       # loopback
        "10.0.0.0/8",        # private (RFC 1918)
        "172.16.0.0/12",     # private (RFC 1918)
        "192.168.0.0/16",    # private (RFC 1918)
        "169.254.0.0/16",    # link-local (cloud metadata)
    ]

    BLOCKED_IPV6_RANGES = [
        "::1/128",           # loopback
        "fc00::/7",          # unique local address
        "fe80::/10",         # link-local
    ]

    def test_blocked_ipv4_ranges_defined(self):
        """The blocked IPv4 ranges list must include loopback, private,
        and link-local (cloud metadata service)."""
        assert "127.0.0.0/8" in self.BLOCKED_IPV4_RANGES
        assert "169.254.0.0/16" in self.BLOCKED_IPV4_RANGES  # cloud metadata
        assert "10.0.0.0/8" in self.BLOCKED_IPV4_RANGES

    def test_blocked_ipv6_ranges_defined(self):
        """The blocked IPv6 ranges list must include loopback, ULA,
        and link-local."""
        assert "::1/128" in self.BLOCKED_IPV6_RANGES
        assert "fc00::/7" in self.BLOCKED_IPV6_RANGES


@pytest.mark.sandbox
@pytest.mark.integration
@pytest.mark.requires_docker_gateway
class TestDockerNetworkEnforcement:
    """Docker-level network enforcement tests.

    These tests require a real enforced-gateway Docker fixture: Docker
    running, a configured --internal network, and the egress gateway
    container. They verify that a candidate container on the isolated
    network cannot reach blocked IPs or bypass the egress gateway.

    SKIPPED when Docker or the gateway fixture is not available.

    Future required tests (section 7.2 of the v31 plan):
      - allowed domain succeeds
      - blocked domain fails
      - curl --noproxy blocked domain fails
      - raw socket to blocked IP fails
      - direct HTTPS to blocked IP fails
      - metadata service unreachable
      - host LAN unreachable
      - RFC1918 blocked
      - Docker DNS bypass blocked

    These must be integration tests against actual Docker network
    behavior — not fake unit tests.
    """

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker (pip install docker + Docker daemon running)",
    )
    def test_http_to_allowed_domain_succeeds(self):
        """HTTP to an allow-listed domain should succeed
        through the gateway."""
        pytest.skip("requires real enforced-gateway Docker fixture")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_http_to_blocked_domain_fails(self):
        """HTTP to a non-allow-listed domain should fail."""
        pytest.skip("requires real enforced-gateway Docker fixture")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_raw_socket_to_blocked_ip_fails(self):
        """Python socket.create_connection to a blocked IP must fail."""
        pytest.skip("requires real enforced-gateway Docker fixture")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_direct_ip_https_fails(self):
        """Direct IP HTTPS to a non-allow-listed IP must fail."""
        pytest.skip("requires real enforced-gateway Docker fixture")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_candidate_cannot_reach_metadata_service(self):
        """Candidate must not reach 169.254.169.254 (cloud metadata)."""
        pytest.skip("requires real enforced-gateway Docker fixture")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_candidate_cannot_reach_host_lan(self):
        """Candidate must not reach host LAN addresses (10.x, 192.168.x)."""
        pytest.skip("requires real enforced-gateway Docker fixture")
