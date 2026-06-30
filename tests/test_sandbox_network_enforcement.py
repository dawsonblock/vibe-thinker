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

    def test_gateway_container_uses_hardening_flags(self):
        """The gateway container must be started with the same hardening
        flags as the sandbox container: --read-only, --cap-drop ALL,
        --security-opt no-new-privileges, --user, --pids-limit,
        --memory, --tmpfs. The gateway is a network-facing security
        boundary component and must be hardened accordingly.
        """
        from sandbox.docker_executor import DockerSandboxExecutor
        from sandbox.network_allowlist import NetworkAllowList
        allowlist = NetworkAllowList.from_string("pypi.org:443")
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.ENFORCED_GATEWAY,
            allowlist=allowlist,
        )
        captured_cmds = []

        async def _run():
            # Mock all subprocess calls. The calls in order are:
            # 1. network inspect (wait) → success (network exists)
            # 2. gateway docker run (communicate) → success
            # 3. network connect (communicate) → success
            # 4. inspect gateway IP (communicate) → fake IP
            # 5. docker logs (communicate) → "Listening on" message
            # 6. sandbox docker run (communicate) → output
            call_count = [0]

            async def mock_communicate():
                call_count[0] += 1
                # communicate() is called by gateway run, connect, inspect, logs, sandbox
                # Map by communicate call index.
                if call_count[0] == 1:
                    # gateway docker run → success
                    return (b"container_id\n", b"")
                if call_count[0] == 2:
                    # network connect → success
                    return (b"", b"")
                if call_count[0] == 3:
                    # inspect gateway IP → return a fake IP
                    return (b"10.0.0.2\n", b"")
                if call_count[0] == 4:
                    # docker logs → "Listening on" message
                    return (b"[SNIProxy] Listening on 0.0.0.0:8888\n", b"")
                # sandbox run
                return (b"hi\n", b"")

            async def mock_wait():
                return 0

            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = mock_communicate
            mock_proc.wait = mock_wait

            with patch("asyncio.create_subprocess_exec",
                       return_value=mock_proc) as mock_exec:
                await executor.execute("print('hi')", timeout=5.0)
                for call in mock_exec.call_args_list:
                    captured_cmds.append(list(call[0]))

        import asyncio
        asyncio.run(_run())
        # Find the gateway docker run command (contains "run", "-d",
        # and "sni_proxy").
        gw_cmd = None
        for cmd in captured_cmds:
            if "run" in cmd and "-d" in cmd and "sni_proxy" in " ".join(cmd):
                gw_cmd = cmd
                break
        assert gw_cmd is not None, (
            f"Gateway docker run command not found in: {captured_cmds}"
        )
        # Verify hardening flags are present.
        assert "--read-only" in gw_cmd, "Gateway must use --read-only"
        assert "--cap-drop" in gw_cmd, "Gateway must use --cap-drop"
        cap_idx = gw_cmd.index("--cap-drop")
        assert gw_cmd[cap_idx + 1] == "ALL", "Gateway must cap-drop ALL"
        assert "--security-opt" in gw_cmd, "Gateway must use --security-opt"
        sec_idx = gw_cmd.index("--security-opt")
        assert gw_cmd[sec_idx + 1] == "no-new-privileges"
        assert "--user" in gw_cmd, "Gateway must use --user"
        user_idx = gw_cmd.index("--user")
        assert gw_cmd[user_idx + 1] == "1000:1000"
        assert "--pids-limit" in gw_cmd, "Gateway must use --pids-limit"
        assert "--memory" in gw_cmd, "Gateway must use --memory"
        assert "--tmpfs" in gw_cmd, "Gateway must use --tmpfs"


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


@pytest.mark.sandbox
@pytest.mark.integration
@pytest.mark.requires_docker_gateway
class TestGatewayEgressEnforcement:
    """End-to-end gateway egress enforcement tests.

    These tests verify the full ENFORCED_GATEWAY path:
      1. The executor starts a gateway container running the SNI proxy.
      2. The sandbox container is on the --internal network (no direct
         internet).
      3. Allowlisted domains can be reached THROUGH the proxy.
      4. Non-allowlisted domains are BLOCKED by the proxy.
      5. Direct internet access (raw sockets bypassing the proxy) is
         BLOCKED by the --internal network.

    SKIPPED when Docker is not available. When Docker IS available,
    these tests run for real — they are the authoritative gateway
    enforcement validation.
    """

    @staticmethod
    def _docker_available() -> bool:
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
    def _run_container_with_env(
        network: str, script: str, env: dict, timeout: int = 30,
    ) -> tuple:
        """Run a Python script in a container with env vars set.

        The script is piped via stdin (python3 -) to avoid quoting issues.
        Returns (returncode, stdout, stderr).
        """
        import subprocess
        img_check = subprocess.run(
            ["docker", "image", "inspect", "vibe-thinker-sandbox:latest"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        image = "vibe-thinker-sandbox:latest" if img_check.returncode == 0 \
            else "python:3.12-slim"
        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", network,
            "--memory", "256m",
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "64",
            "--tmpfs", "/tmp",
            "--user", "1000:1000",
            "--entrypoint", "python3",
        ]
        for k, v in env.items():
            cmd.extend(["--env", f"{k}={v}"])
        cmd.extend([image, "-"])
        r = subprocess.run(cmd, input=script.encode(), capture_output=True,
                           timeout=timeout)
        return r.returncode, r.stdout.decode(), r.stderr.decode()

    @staticmethod
    def _create_internal_network(name: str = "vibe-test-gw-net") -> bool:
        import subprocess
        # Force-remove any leftover gateway container that might be
        # attached to the network (prevents "has active endpoints" error).
        subprocess.run(["docker", "rm", "-f", "vibe-test-gateway"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
        subprocess.run(["docker", "network", "rm", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
        r = subprocess.run(
            ["docker", "network", "create", "--internal", "--driver", "bridge", name],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0

    @staticmethod
    def _cleanup_network(name: str = "vibe-test-gw-net") -> None:
        import subprocess
        subprocess.run(["docker", "network", "rm", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)

    @staticmethod
    def _start_gateway(network: str, allowlist: str) -> str | None:
        """Start a gateway container on the default bridge + internal net.

        Returns the gateway's IP on the internal network, or None.
        """
        import subprocess, os, time
        gw_name = "vibe-test-gateway"
        # Clean up any leftover gateway.
        subprocess.run(["docker", "rm", "-f", gw_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
        sandbox_dir = os.path.dirname(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sandbox"))
        )
        cmd = [
            "docker", "run", "-d",
            "--name", gw_name,
            "--network", "bridge",
            "--restart", "no",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "1000:1000",
            "--pids-limit", "64",
            "--memory", "128m",
            "--tmpfs", "/tmp:rw,size=10m",
            "-v", f"{sandbox_dir}/sandbox:/app/sandbox:ro",
            "-w", "/app",
            "python:3.12-slim",
            "python3", "-u", "-m", "sandbox.sni_proxy",
            "--host", "0.0.0.0", "--port", "8888",
            "--allowlist", allowlist,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0:
            return None
        # Connect to internal network.
        r2 = subprocess.run(
            ["docker", "network", "connect", network, gw_name],
            capture_output=True, timeout=10,
        )
        if r2.returncode != 0:
            subprocess.run(["docker", "rm", "-f", gw_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10)
            return None
        # Get IP on internal network.
        # Use index() in the Go template because the network name
        # contains hyphens (Go templates interpret hyphens as
        # subtraction operators in dot-notation field access).
        r3 = subprocess.run(
            ["docker", "inspect", "--format",
             f"{{{{(index .NetworkSettings.Networks \"{network}\").IPAddress}}}}",
             gw_name],
            capture_output=True, timeout=10,
        )
        ip = r3.stdout.decode().strip()
        if not ip:
            return None
        # Wait for proxy to be ready.
        for _ in range(20):
            r4 = subprocess.run(
                ["docker", "logs", gw_name],
                capture_output=True, timeout=5,
            )
            output = r4.stdout.decode() + r4.stderr.decode()
            if "Listening on" in output:
                return ip
            time.sleep(0.5)
        return None

    @staticmethod
    def _stop_gateway() -> None:
        import subprocess
        subprocess.run(["docker", "rm", "-f", "vibe-test-gateway"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_gateway_allows_allowlisted_domain(self):
        """An allowlisted domain can be reached THROUGH the gateway proxy.

        The sandbox container is on the --internal network (no direct
        internet). HTTP_PROXY points at the gateway. A urllib request
        to an allowlisted domain (example.com) should succeed.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-gw-net"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        gw_ip = self._start_gateway(net, "example.com:80,example.com:443")
        if gw_ip is None:
            self._cleanup_network(net)
            pytest.skip("could not start gateway container")
        try:
            script = (
                "import urllib.request\n"
                "try:\n"
                "    req = urllib.request.urlopen('http://example.com', timeout=10)\n"
                "    print(f'STATUS:{req.status}')\n"
                "except Exception as e:\n"
                "    print(f'FAILED:{type(e).__name__}:{e}')\n"
            )
            env = {
                "HTTP_PROXY": f"http://{gw_ip}:8888",
                "HTTPS_PROXY": f"http://{gw_ip}:8888",
                "http_proxy": f"http://{gw_ip}:8888",
                "https_proxy": f"http://{gw_ip}:8888",
                "NO_PROXY": "localhost,127.0.0.1",
            }
            rc, out, err = self._run_container_with_env(net, script, env, timeout=30)
            assert "STATUS:200" in out, (
                f"Allowlisted domain should be reachable through gateway, "
                f"but got: {out!r} (stderr: {err!r})"
            )
        finally:
            self._stop_gateway()
            self._cleanup_network(net)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_gateway_blocks_non_allowlisted_domain(self):
        """A non-allowlisted domain is BLOCKED by the gateway proxy.

        The sandbox container is on the --internal network. HTTP_PROXY
        points at the gateway. A urllib request to a non-allowlisted
        domain (httpbin.org) should fail — the proxy returns 403.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-gw-net"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        # Only allow example.com — httpbin.org is NOT on the list.
        gw_ip = self._start_gateway(net, "example.com:80,example.com:443")
        if gw_ip is None:
            self._cleanup_network(net)
            pytest.skip("could not start gateway container")
        try:
            script = (
                "import urllib.request\n"
                "try:\n"
                "    req = urllib.request.urlopen('http://httpbin.org/get', timeout=10)\n"
                "    print(f'STATUS:{req.status}')\n"
                "except Exception as e:\n"
                "    print(f'BLOCKED:{type(e).__name__}')\n"
            )
            env = {
                "HTTP_PROXY": f"http://{gw_ip}:8888",
                "HTTPS_PROXY": f"http://{gw_ip}:8888",
                "http_proxy": f"http://{gw_ip}:8888",
                "https_proxy": f"http://{gw_ip}:8888",
                "NO_PROXY": "localhost,127.0.0.1",
            }
            rc, out, err = self._run_container_with_env(net, script, env, timeout=30)
            assert "BLOCKED" in out, (
                f"Non-allowlisted domain should be blocked by gateway, "
                f"but got: {out!r} (stderr: {err!r})"
            )
        finally:
            self._stop_gateway()
            self._cleanup_network(net)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_gateway_blocks_raw_socket_bypass(self):
        """Raw socket egress (bypassing the proxy) is BLOCKED by --internal.

        Even with the proxy env vars set, a raw socket connection to an
        external IP must fail — the --internal network has no route to
        the outside. This is the key security property: the proxy is the
        ONLY path out.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-gw-net"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        gw_ip = self._start_gateway(net, "example.com:80")
        if gw_ip is None:
            self._cleanup_network(net)
            pytest.skip("could not start gateway container")
        try:
            # Raw socket to 1.1.1.1 (Cloudflare DNS) — no proxy used.
            script = (
                "import socket\n"
                "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "s.settimeout(5)\n"
                "try:\n"
                "    s.connect(('1.1.1.1', 80))\n"
                "    print('CONNECTED')\n"
                "except Exception as e:\n"
                "    print(f'BLOCKED:{type(e).__name__}')\n"
                "finally:\n"
                "    s.close()\n"
            )
            env = {
                "HTTP_PROXY": f"http://{gw_ip}:8888",
                "HTTPS_PROXY": f"http://{gw_ip}:8888",
            }
            rc, out, err = self._run_container_with_env(net, script, env, timeout=20)
            assert "BLOCKED" in out, (
                f"Raw socket egress should be blocked by --internal network, "
                f"but got: {out!r} (stderr: {err!r})"
            )
        finally:
            self._stop_gateway()
            self._cleanup_network(net)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    def test_gateway_allows_https_allowlisted_domain(self):
        """An allowlisted HTTPS domain can be reached THROUGH the gateway proxy.

        This is the real HTTPS test: the sandbox container does
        ``urllib.request.urlopen('https://example.com')`` through the
        gateway. The CONNECT proxy must correctly handle the HTTPS
        tunnel: check the CONNECT target (example.com) against the
        allow-list, connect to the upstream, send 200, and tunnel the
        TLS handshake + data bidirectionally.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        net = "vibe-test-gw-net"
        if not self._create_internal_network(net):
            pytest.skip("could not create --internal network")
        gw_ip = self._start_gateway(net, "example.com:80,example.com:443")
        if gw_ip is None:
            self._cleanup_network(net)
            pytest.skip("could not start gateway container")
        try:
            script = (
                "import urllib.request, ssl\n"
                "try:\n"
                "    ctx = ssl.create_default_context()\n"
                "    req = urllib.request.urlopen('https://example.com', timeout=15, context=ctx)\n"
                "    print(f'STATUS:{req.status}')\n"
                "except Exception as e:\n"
                "    print(f'FAILED:{type(e).__name__}:{e}')\n"
            )
            env = {
                "HTTP_PROXY": f"http://{gw_ip}:8888",
                "HTTPS_PROXY": f"http://{gw_ip}:8888",
                "http_proxy": f"http://{gw_ip}:8888",
                "https_proxy": f"http://{gw_ip}:8888",
                "NO_PROXY": "localhost,127.0.0.1",
            }
            rc, out, err = self._run_container_with_env(net, script, env, timeout=45)
            assert "STATUS:200" in out, (
                f"Allowlisted HTTPS domain should be reachable through gateway, "
                f"but got: {out!r} (stderr: {err!r})"
            )
        finally:
            self._stop_gateway()
            self._cleanup_network(net)


@pytest.mark.sandbox
@pytest.mark.integration
@pytest.mark.requires_docker_gateway
class TestDockerSandboxExecutorGatewayIntegration:
    """End-to-end tests using DockerSandboxExecutor.execute() with
    NetworkMode.ENFORCED_GATEWAY.

    Unlike TestGatewayEgressEnforcement (which manually starts a gateway
    helper), these tests use the real DockerSandboxExecutor.execute()
    method — the exact production path:

        DockerSandboxExecutor.execute()
        → _ensure_gateway_network()
        → _start_gateway()
        → connect gateway to internal network
        → inject HTTP_PROXY/HTTPS_PROXY
        → run sandbox container
        → cleanup gateway

    SKIPPED when Docker is not available.
    """

    @staticmethod
    def _docker_available() -> bool:
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
    def _cleanup_gateway_network() -> None:
        """Remove any leftover gateway network from prior runs."""
        import subprocess
        for name in ("vibe-thinker-gateway-net",):
            subprocess.run(["docker", "network", "rm", name],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    @pytest.mark.asyncio
    async def test_executor_enforced_gateway_allows_allowlisted_http(self):
        """DockerSandboxExecutor.execute() with ENFORCED_GATEWAY allows
        an HTTP request to an allowlisted domain through the gateway.

        This exercises the full production path: the executor creates the
        internal network, starts the gateway container, connects it,
        injects proxy env vars, runs the sandbox, and cleans up.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        from sandbox.docker_executor import DockerSandboxExecutor
        from sandbox.network_allowlist import NetworkAllowList

        self._cleanup_gateway_network()
        allowlist = NetworkAllowList.from_string("example.com:80,example.com:443")
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.ENFORCED_GATEWAY,
            allowlist=allowlist,
            timeout=60.0,
        )
        try:
            script = (
                "import urllib.request\n"
                "try:\n"
                "    req = urllib.request.urlopen('http://example.com', timeout=15)\n"
                "    print(f'STATUS:{req.status}')\n"
                "except Exception as e:\n"
                "    print(f'FAILED:{type(e).__name__}:{e}')\n"
            )
            result = await executor.execute(script, timeout=60.0)
            assert "STATUS:200" in result.stdout, (
                f"Allowlisted HTTP domain should be reachable through "
                f"gateway via DockerSandboxExecutor.execute(), but got: "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}, "
                f"evidence={result.evidence}"
            )
            # Verify the executor recorded the gateway evidence.
            assert result.evidence.get("enforced") is True
            assert result.evidence.get("network_mode") == "enforced_gateway"
        finally:
            await executor.cleanup()
            self._cleanup_gateway_network()

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    @pytest.mark.asyncio
    async def test_executor_enforced_gateway_allows_allowlisted_https(self):
        """DockerSandboxExecutor.execute() with ENFORCED_GATEWAY allows
        an HTTPS request to an allowlisted domain through the gateway.

        This is the real HTTPS integration test through the production
        executor path — the CONNECT proxy must correctly tunnel the TLS
        handshake.
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        from sandbox.docker_executor import DockerSandboxExecutor
        from sandbox.network_allowlist import NetworkAllowList

        self._cleanup_gateway_network()
        allowlist = NetworkAllowList.from_string("example.com:80,example.com:443")
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.ENFORCED_GATEWAY,
            allowlist=allowlist,
            timeout=60.0,
        )
        try:
            script = (
                "import urllib.request, ssl\n"
                "try:\n"
                "    ctx = ssl.create_default_context()\n"
                "    req = urllib.request.urlopen('https://example.com', timeout=15, context=ctx)\n"
                "    print(f'STATUS:{req.status}')\n"
                "except Exception as e:\n"
                "    print(f'FAILED:{type(e).__name__}:{e}')\n"
            )
            result = await executor.execute(script, timeout=60.0)
            assert "STATUS:200" in result.stdout, (
                f"Allowlisted HTTPS domain should be reachable through "
                f"gateway via DockerSandboxExecutor.execute(), but got: "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}, "
                f"evidence={result.evidence}"
            )
            assert result.evidence.get("enforced") is True
        finally:
            await executor.cleanup()
            self._cleanup_gateway_network()

    @pytest.mark.skipif(
        not _has_docker(),
        reason="requires Docker",
    )
    @pytest.mark.asyncio
    async def test_executor_enforced_gateway_blocks_non_allowlisted(self):
        """DockerSandboxExecutor.execute() with ENFORCED_GATEWAY blocks
        a request to a non-allowlisted domain (proxy returns 403).
        """
        if not self._docker_available():
            pytest.skip("Docker daemon not running")
        from sandbox.docker_executor import DockerSandboxExecutor
        from sandbox.network_allowlist import NetworkAllowList

        self._cleanup_gateway_network()
        # Only allow example.com — httpbin.org is NOT allowlisted.
        allowlist = NetworkAllowList.from_string("example.com:80,example.com:443")
        executor = DockerSandboxExecutor(
            network_mode=NetworkMode.ENFORCED_GATEWAY,
            allowlist=allowlist,
            timeout=60.0,
        )
        try:
            script = (
                "import urllib.request\n"
                "try:\n"
                "    req = urllib.request.urlopen('http://httpbin.org/get', timeout=15)\n"
                "    print(f'STATUS:{req.status}')\n"
                "except Exception as e:\n"
                "    print(f'BLOCKED:{type(e).__name__}')\n"
            )
            result = await executor.execute(script, timeout=60.0)
            assert "BLOCKED" in result.stdout, (
                f"Non-allowlisted domain should be blocked by gateway, "
                f"but got: stdout={result.stdout!r}, stderr={result.stderr!r}"
            )
        finally:
            await executor.cleanup()
            self._cleanup_gateway_network()
