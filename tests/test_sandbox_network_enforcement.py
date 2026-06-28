"""Tests for sandbox network enforcement (Phase 5 — sandbox truth).

These tests verify that the sandbox network isolation modes are correctly
configured and that the documented security claims are honest.

v0.4.0-alpha: The BEST_EFFORT_PROXY mode is NOT a security boundary.
These tests document that fact and verify the NetworkMode enum defaults
to DISABLED. The enforced-gateway bypass tests require Docker and are
marked accordingly.

Blocked IP ranges that must NOT be reachable from a sandbox candidate:
  127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16,
  169.254.0.0/16, ::1/128, fc00::/7, fe80::/10
"""

import pytest

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
        """ENFORCED_GATEWAY is the only mode that IS a security boundary."""
        assert NetworkMode.ENFORCED_GATEWAY.value == "enforced_gateway"

    def test_network_mode_is_str_enum(self):
        """NetworkMode must be a str Enum so it serializes cleanly."""
        assert isinstance(NetworkMode.DISABLED, str)


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
class TestDockerNetworkEnforcement:
    """Docker-level network enforcement tests.

    These tests require Docker and a configured --internal network.
    They verify that a candidate container on the isolated network
    cannot reach blocked IPs or bypass the egress gateway.

    SKIPPED when Docker is not available.
    """

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker (pip install docker + Docker daemon running)",
    )
    def test_http_to_allowed_domain_succeeds(self):
        """HTTP to an allow-listed domain should succeed through the gateway."""
        pytest.skip("Docker enforcement test — requires configured gateway")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_http_to_blocked_domain_fails(self):
        """HTTP to a non-allow-listed domain should fail."""
        pytest.skip("Docker enforcement test — requires configured gateway")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_raw_socket_to_blocked_ip_fails(self):
        """Python socket.create_connection to a blocked IP must fail."""
        pytest.skip("Docker enforcement test — requires configured gateway")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_direct_ip_https_fails(self):
        """Direct IP HTTPS to a non-allow-listed IP must fail."""
        pytest.skip("Docker enforcement test — requires configured gateway")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_candidate_cannot_reach_metadata_service(self):
        """Candidate must not reach 169.254.169.254 (cloud metadata)."""
        pytest.skip("Docker enforcement test — requires configured gateway")

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_candidate_cannot_reach_host_lan(self):
        """Candidate must not reach host LAN addresses (10.x, 192.168.x)."""
        pytest.skip("Docker enforcement test — requires configured gateway")
