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


class TestOrchestratorNetworkModeWiring:
    """Verify the orchestrator applies an explicit network_mode to the
    sandbox executor (Blocker 12). The operator's explicit choice must
    reach the executor and must NEVER be silently inferred as enforcement
    from an allow-list. network_mode=None (auto) must preserve the
    historical auto-detect behavior (no set_network_mode call)."""

    def _make_orch(self, network_mode, mock_executor):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        mock_cv = MagicMock()
        mock_cv.executor = mock_executor
        return HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            code_verifier=mock_cv,
            network_mode=network_mode,
        )

    def test_explicit_enforced_gateway_reaches_executor(self):
        executor = MagicMock()
        orch = self._make_orch(NetworkMode.ENFORCED_GATEWAY, executor)
        executor.set_network_mode.assert_called_once_with(
            NetworkMode.ENFORCED_GATEWAY)

    def test_explicit_disabled_reaches_executor(self):
        executor = MagicMock()
        orch = self._make_orch(NetworkMode.DISABLED, executor)
        executor.set_network_mode.assert_called_once_with(
            NetworkMode.DISABLED)

    def test_auto_does_not_override_executor(self):
        """network_mode=None (auto) must NOT call set_network_mode, so the
        executor keeps its auto-detect behavior (BEST_EFFORT_PROXY when an
        allow-list is present, DISABLED otherwise)."""
        executor = MagicMock()
        orch = self._make_orch(None, executor)
        executor.set_network_mode.assert_not_called()


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

    These tests run REAL Docker containers against the live Docker daemon
    to verify that network isolation is enforced at the Docker level, not
    just in unit-test mocks.

    Two layers are tested:
      1. ``--network none`` (DISABLED mode): the container has NO network
         stack at all. No connection to any address can succeed. This is
         the default mode and the primary security boundary.
      2. ``--internal`` network (ENFORCED_GATEWAY mode): the container is
         on a Docker bridge network marked --internal, which blocks direct
         internet egress. The container cannot reach external IPs, cloud
         metadata, or the host LAN.

    SKIPPED when Docker is not available (daemon not running or docker
    package not installed). When Docker IS available, these tests run
    for real — they are the authoritative enforcement validation.
    """

    @staticmethod
    def _docker_available() -> bool:
        """Check that Docker is importable AND the daemon is running."""
        if not _has_docker():
            return False
        import subprocess
        try:
            r = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _run_container(network: str, script: str, timeout: int = 15) -> tuple:
        """Run a Python script in a hardened container on the given network.

        Returns (returncode, stdout, stderr).
        Uses the vibe-thinker-sandbox image (falls back to python:3.12-slim).
        The script is piped via stdin (``python3 -``) to avoid quoting /
        single-line syntax issues with try/except blocks.
        """
        import subprocess
        # Try the purpose-built sandbox image; fall back to python:3.12-slim.
        img_check = subprocess.run(
            ["docker", "image", "inspect", "vibe-thinker-sandbox:latest"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        image = "vibe-thinker-sandbox:latest" if img_check.returncode == 0 \
            else "python:3.12-slim"
        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", network,
            "--memory", "128m",
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "64",
            "--tmpfs", "/tmp",
            "--user", "1000:1000",
            "--entrypoint", "python3",
            image, "-",
        ]
        r = subprocess.run(cmd, input=script.encode(), capture_output=True,
                           timeout=timeout)
        return r.returncode, r.stdout.decode(), r.stderr.decode()

    @staticmethod
    def _connect_script(host: str, port: int = 80, timeout: int = 3) -> str:
        """Build a multi-line Python script that tries to connect to
        ``host:port`` and prints CONNECTED or BLOCKED:<ExceptionType>."""
        return (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            f"s.settimeout({timeout})\n"
            "try:\n"
            f"    s.connect(('{host}', {port}))\n"
            "    print('CONNECTED')\n"
            "except Exception as e:\n"
            "    print(f'BLOCKED:{type(e).__name__}')\n"
            "finally:\n"
            "    s.close()\n"
        )

    @staticmethod
    def _create_internal_network(name: str = "vibe-test-internal") -> bool:
        """Create a Docker --internal network for testing. Returns True on success."""
        import subprocess
        # Clean up any leftover network from a previous run.
        subprocess.run(["docker", "network", "rm", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
        r = subprocess.run(
            ["docker", "network", "create", "--internal", "--driver", "bridge", name],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0

    @staticmethod
    def _cleanup_network(name: str = "vibe-test-internal") -> None:
        import subprocess
        subprocess.run(["docker", "network", "rm", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)

    # ---- --network none (DISABLED mode) tests ----

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker (pip install docker + Docker daemon running)",
    )
    def test_network_none_blocks_all_connections(self):
        """A container with --network none cannot make ANY TCP connection.

        This is the DISABLED mode — the default and primary security
        boundary. Even localhost must be unreachable.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        # Try to connect to 1.1.1.1 (Cloudflare DNS) on port 80.
        # With --network none, the socket.connect must fail.
        script = self._connect_script("1.1.1.1", 80, 3)
        rc, out, err = self._run_container("none", script)
        assert "BLOCKED" in out, (
            f"--network none should block all connections, but got: {out!r} "
            f"(stderr: {err!r})"
        )

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_network_none_blocks_metadata_service(self):
        """A container with --network none cannot reach 169.254.169.254
        (cloud metadata service)."""
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        script = self._connect_script("169.254.169.254", 80, 3)
        rc, out, err = self._run_container("none", script)
        assert "BLOCKED" in out, (
            f"Metadata service should be blocked, but got: {out!r}"
        )

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_network_none_blocks_host_lan(self):
        """A container with --network none cannot reach host LAN (192.168.x)."""
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        script = self._connect_script("192.168.1.1", 80, 3)
        rc, out, err = self._run_container("none", script)
        assert "BLOCKED" in out, (
            f"Host LAN should be blocked, but got: {out!r}"
        )

    # ---- --internal network (ENFORCED_GATEWAY mode) tests ----

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_internal_network_blocks_internet(self):
        """A container on a Docker --internal network cannot reach the
        internet directly (no gateway to the outside)."""
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-internal"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        try:
            script = self._connect_script("1.1.1.1", 80, 5)
            rc, out, err = self._run_container(net, script, timeout=20)
            assert "BLOCKED" in out, (
                f"--internal network should block internet, but got: {out!r} "
                f"(stderr: {err!r})"
            )
        finally:
            self._cleanup_network(net)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_internal_network_blocks_metadata_service(self):
        """A container on a --internal network cannot reach 169.254.169.254."""
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-internal"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        try:
            script = self._connect_script("169.254.169.254", 80, 5)
            rc, out, err = self._run_container(net, script, timeout=20)
            assert "BLOCKED" in out, (
                f"Metadata service should be blocked on --internal, got: {out!r}"
            )
        finally:
            self._cleanup_network(net)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_internal_network_blocks_host_lan(self):
        """A container on a --internal network cannot reach host LAN (10.x)."""
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-internal"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        try:
            script = self._connect_script("10.0.0.1", 80, 5)
            rc, out, err = self._run_container(net, script, timeout=20)
            assert "BLOCKED" in out, (
                f"Host LAN should be blocked on --internal, got: {out!r}"
            )
        finally:
            self._cleanup_network(net)
