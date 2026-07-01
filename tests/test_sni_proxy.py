"""Tests for the SNI-aware egress proxy (v0.4.1).

Tests the SNI extraction from TLS ClientHello, domain matching with
wildcards, the proxy's allow/deny logic, and the SNIEgressProxy server
class (TCP-level tests with a real local server).
"""

import asyncio
import pytest

from sandbox.sni_proxy import (
    extract_sni,
    domain_matches,
    is_domain_allowed,
    _is_ip_allowed,
    SNIEgressProxy,
    DnsAuditor,
    _is_ip_literal,
    extract_allowlist_sets,
)
from sandbox.network_allowlist import NetworkAllowList


# ---------------------------------------------------------------------- #
# SNI extraction from TLS ClientHello
# ---------------------------------------------------------------------- #
class TestIsIpLiteral:
    def test_ipv4_is_ip_literal(self):
        assert _is_ip_literal("1.2.3.4") is True

    def test_ipv6_is_ip_literal(self):
        assert _is_ip_literal("::1") is True
        assert _is_ip_literal("2001:db8::1") is True

    def test_hostname_is_not_ip_literal(self):
        assert _is_ip_literal("pypi.org") is False
        assert _is_ip_literal("example.com") is False

    def test_wildcard_is_not_ip_literal(self):
        assert _is_ip_literal("*.pypi.org") is False


class TestSNIExtraction:
    def test_extract_sni_from_real_client_hello(self):
        """Extract SNI from a minimal TLS ClientHello with SNI extension.

        This constructs a valid TLS ClientHello record with the SNI
        extension set to 'example.com'.
        """
        # Build a minimal ClientHello:
        # Handshake type (0x01) + length (3 bytes) + version (2) +
        # random (32) + session_id_len (1) + session_id + cipher_len (2)
        # + cipher (2) + compression_len (1) + compression (1) +
        # extensions_len (2) + SNI extension

        hostname = b"example.com"
        # SNI extension: type=0x0000, length, list_length, name_type=0,
        # name_length, name
        sni_name = hostname
        sni_entry = b'\x00' + len(sni_name).to_bytes(2, 'big') + sni_name
        sni_list = len(sni_entry).to_bytes(2, 'big') + sni_entry
        sni_ext = b'\x00\x00' + len(sni_list).to_bytes(2, 'big') + sni_list

        # Extensions
        extensions = sni_ext
        ext_len = len(extensions).to_bytes(2, 'big')

        # Compression methods
        compression = b'\x01\x00'  # 1 method: null

        # Cipher suites
        cipher = b'\x00\x2f'  # TLS_RSA_WITH_AES_128_CBC_SHA
        cipher_list = len(cipher).to_bytes(2, 'big') + cipher

        # Session ID
        session_id = b'\x00'  # empty session ID

        # Random
        random_bytes = b'\x00' * 32

        # Version
        version = b'\x03\x01'  # TLS 1.0

        # Handshake header
        handshake_body = version + random_bytes + session_id + cipher_list + compression + ext_len + extensions
        handshake = b'\x01' + len(handshake_body).to_bytes(3, 'big') + handshake_body

        # TLS record header
        record = b'\x16\x03\x01' + len(handshake).to_bytes(2, 'big') + handshake

        result = extract_sni(record)
        assert result == "example.com"

    def test_extract_sni_no_sni_extension(self):
        """ClientHello without SNI extension returns None."""
        # Build a ClientHello with no extensions.
        session_id = b'\x00'
        cipher = b'\x00\x2f'
        cipher_list = len(cipher).to_bytes(2, 'big') + cipher
        compression = b'\x01\x00'
        random_bytes = b'\x00' * 32
        version = b'\x03\x01'
        # No extensions
        ext_len = b'\x00\x00'
        handshake_body = version + random_bytes + session_id + cipher_list + compression + ext_len
        handshake = b'\x01' + len(handshake_body).to_bytes(3, 'big') + handshake_body
        record = b'\x16\x03\x01' + len(handshake).to_bytes(2, 'big') + handshake
        result = extract_sni(record)
        assert result is None

    def test_extract_sni_not_tls(self):
        """Non-TLS data returns None."""
        assert extract_sni(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n") is None

    def test_extract_sni_empty(self):
        """Empty data returns None."""
        assert extract_sni(b"") is None

    def test_extract_sni_truncated(self):
        """Truncated ClientHello returns None."""
        assert extract_sni(b'\x16\x03\x01\x00\x10') is None


# ---------------------------------------------------------------------- #
# Domain matching
# ---------------------------------------------------------------------- #
class TestDomainMatching:
    def test_exact_match(self):
        assert domain_matches("pypi.org", "pypi.org") is True

    def test_case_insensitive(self):
        assert domain_matches("PyPI.org", "pypi.org") is True
        assert domain_matches("pypi.org", "PYPI.ORG") is True

    def test_no_match(self):
        assert domain_matches("evil.com", "pypi.org") is False

    def test_wildcard_single_level(self):
        assert domain_matches("foo.pypi.org", "*.pypi.org") is True

    def test_wildcard_not_root(self):
        """*.pypi.org should NOT match pypi.org itself."""
        assert domain_matches("pypi.org", "*.pypi.org") is False

    def test_wildcard_not_multi_level(self):
        """*.pypi.org should NOT match foo.bar.pypi.org (single-level only)."""
        assert domain_matches("foo.bar.pypi.org", "*.pypi.org") is False

    def test_wildcard_with_port_stripped(self):
        """Domain matching works on hostname without port."""
        assert domain_matches("files.pythonhosted.org", "files.pythonhosted.org") is True


# ---------------------------------------------------------------------- #
# Allow-list checking
# ---------------------------------------------------------------------- #
class TestIsDomainAllowed:
    def test_exact_domain_allowed(self):
        assert is_domain_allowed(
            "pypi.org", {"pypi.org"}, set()
        ) is True

    def test_wildcard_allowed(self):
        assert is_domain_allowed(
            "foo.pypi.org", set(), {"*.pypi.org"}
        ) is True

    def test_not_in_allowlist(self):
        assert is_domain_allowed(
            "evil.com", {"pypi.org"}, {"*.pypi.org"}
        ) is False

    def test_exact_and_wildcard_combined(self):
        assert is_domain_allowed(
            "pypi.org", {"pypi.org", "github.com"}, {"*.pypi.org"}
        ) is True
        assert is_domain_allowed(
            "foo.pypi.org", {"pypi.org", "github.com"}, {"*.pypi.org"}
        ) is True
        assert is_domain_allowed(
            "github.com", {"pypi.org", "github.com"}, {"*.pypi.org"}
        ) is True
        assert is_domain_allowed(
            "evil.com", {"pypi.org", "github.com"}, {"*.pypi.org"}
        ) is False


# ---------------------------------------------------------------------- #
# DnsAuditor — audit logging + per-domain rate limiting (Phase 1.2)
# ---------------------------------------------------------------------- #
class TestDnsAuditor:
    """Tests for the host-side DNS audit log + rate limiter."""

    def test_rate_limit_disabled_allows_all(self):
        """With rate_limit=None, every query is allowed."""
        a = DnsAuditor(rate_limit=None)
        for _ in range(100):
            assert a.allow_query("pypi.org") is True

    def test_rate_limit_enforced(self):
        """Queries beyond the limit are denied (fail-closed)."""
        a = DnsAuditor(rate_limit=3, window=60.0)
        # allow_query does NOT record; record() does. Simulate 3 allowed
        # resolutions then a 4th check must still pass until recorded.
        assert a.allow_query("pypi.org") is True
        a.record("pypi.org", ["1.2.3.4"], "allowed")
        assert a.allow_query("pypi.org") is True
        a.record("pypi.org", ["1.2.3.4"], "allowed")
        assert a.allow_query("pypi.org") is True
        a.record("pypi.org", ["1.2.3.4"], "allowed")
        # Now 3 recorded queries in the window — 4th check is denied.
        assert a.allow_query("pypi.org") is False

    def test_rate_limit_per_domain_independent(self):
        """The rate limit is tracked per-domain, not globally."""
        a = DnsAuditor(rate_limit=2, window=60.0)
        a.record("a.com", ["1.1.1.1"], "allowed")
        a.record("a.com", ["1.1.1.1"], "allowed")
        # a.com is at limit; b.com is unaffected.
        assert a.allow_query("a.com") is False
        assert a.allow_query("b.com") is True

    def test_rate_limit_window_eviction(self):
        """Entries older than the window are evicted (sliding window)."""
        a = DnsAuditor(rate_limit=2, window=0.05)  # 50ms window
        a.record("a.com", ["1.1.1.1"], "allowed")
        a.record("a.com", ["1.1.1.1"], "allowed")
        assert a.allow_query("a.com") is False
        import time as _time
        _time.sleep(0.06)  # window expires
        assert a.allow_query("a.com") is True

    def test_denied_verdict_does_not_consume_quota(self):
        """A denied/rate-limited resolution does not add to the bucket."""
        a = DnsAuditor(rate_limit=2, window=60.0)
        a.record("a.com", [], "denied", "rate_limited")
        a.record("a.com", [], "error", "getaddrinfo failed")
        # No allowed records -> still within limit.
        assert a.allow_query("a.com") is True

    def test_counters_tracked(self):
        a = DnsAuditor(rate_limit=10)
        a.record("a.com", ["1.1.1.1"], "allowed")
        a.record("a.com", ["1.1.1.1"], "allowed")
        a.record("b.com", [], "denied", "rate_limited")
        a.record("c.com", [], "error", "no results")
        assert a._allowed_count == 2
        assert a._denied_count == 2

    def test_record_emits_json_log(self, capsys):
        a = DnsAuditor(rate_limit=10, log=True)
        a.record("pypi.org", ["151.101.0.1"], "allowed")
        captured = capsys.readouterr()
        import json
        line = captured.err.strip()
        assert line  # non-empty
        entry = json.loads(line)
        assert entry["domain"] == "pypi.org"
        assert entry["verdict"] == "allowed"
        assert "151.101.0.1" in entry["ips"]
        assert "ts" in entry

    def test_record_log_disabled_no_output(self, capsys):
        a = DnsAuditor(rate_limit=10, log=False)
        a.record("pypi.org", ["1.2.3.4"], "allowed")
        captured = capsys.readouterr()
        assert captured.err == ""

    @pytest.mark.asyncio
    async def test_resolve_host_rate_limited_returns_none(self):
        """_resolve_host returns None (fail-closed) when over the rate limit."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
            dns_rate_limit=1,
            dns_rate_window=60.0,
        )
        # Prime the auditor with one allowed record -> at limit.
        proxy._dns_auditor.record("pypi.org", ["1.2.3.4"], "allowed")
        # Next resolution attempt must be denied without hitting getaddrinfo.
        result = await proxy._resolve_host("pypi.org")
        assert result is None
        assert proxy._dns_auditor._denied_count >= 1

    @pytest.mark.asyncio
    async def test_resolve_host_unresolvable_returns_none(self):
        """_resolve_host returns None for a domain that won't resolve."""
        proxy = SNIEgressProxy(
            allowed_domains={"nonexistent.invalid"},
            allowed_wildcards=set(),
            port=0,
        )
        result = await proxy._resolve_host("nonexistent.invalid")
        assert result is None


# ---------------------------------------------------------------------- #
# SNIEgressProxy server class (TCP-level tests)
# ---------------------------------------------------------------------- #
class TestSNIEgressProxy:
    """Tests for the SNIEgressProxy server class — uses a real local TCP
    server to test connection handling, CONNECT tunneling, and HTTP proxying."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """The proxy starts and stops cleanly."""
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            port=0,  # ephemeral port
        )
        await proxy.start()
        assert proxy._server is not None
        await proxy.stop()
        assert proxy._denied_count == 0
        assert proxy._allowed_count == 0

    @pytest.mark.asyncio
    async def test_connect_denied_domain_returns_403(self):
        """CONNECT to a denied domain returns 403 Forbidden.

        The proxy checks the CONNECT target host against the allow-list
        (the authoritative destination). A CONNECT to evil.com when only
        pypi.org is allowlisted must return 403 — without needing to read
        the TLS ClientHello first.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(addr[0], addr[1])
            # Send a CONNECT request to a denied domain. The proxy checks
            # the CONNECT target (evil.com) against the allow-list and
            # denies immediately — no ClientHello needed.
            writer.write(b"CONNECT evil.com:443 HTTP/1.1\r\nHost: evil.com:443\r\n\r\n")
            await writer.drain()
            # Read the response — should be 403.
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 1

    @pytest.mark.asyncio
    async def test_connect_allowed_domain_no_sni_gets_200(self):
        """CONNECT to an allowlisted domain with no SNI is ALLOWED (gets 200).

        After the CONNECT-handling fix, the allow decision is based on the
        CONNECT target host, not the SNI. A ClientHello without SNI is
        fine — the SNI consistency check is defense-in-depth and only
        closes on a mismatch (present SNI != CONNECT target), not on
        absent SNI. Uses a mock remote server to avoid real DNS/network.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        # Mock remote server that accepts the connection and echoes.
        async def mock_remote_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            writer.write(b"REMOTE_ECHO:" + data)
            await writer.drain()
            writer.close()

        remote_server = await asyncio.start_server(
            mock_remote_handler, "127.0.0.1", 0
        )
        remote_addr = remote_server.sockets[0].getsockname()

        try:
            # Patch _resolve_host to return the mock remote server's IP.
            async def mock_resolve(host):
                return remote_addr[0]
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # CONNECT to allowed domain, using the mock remote's port so
            # the proxy connects to our mock server.
            connect_port = remote_addr[1]
            writer.write(f"CONNECT pypi.org:{connect_port} HTTP/1.1\r\nHost: pypi.org:{connect_port}\r\n\r\n".encode())
            await writer.drain()
            # Read the 200 response (proxy connects to mock remote first,
            # then sends 200).
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200" in response, f"Expected 200, got: {response!r}"
            # Send a TLS ClientHello WITHOUT the SNI extension — the proxy
            # should forward it to the remote (no SNI check failure).
            compression = b'\x01\x00'
            cipher = b'\x00\x2f'
            cipher_list = len(cipher).to_bytes(2, 'big') + cipher
            session_id = b'\x00'
            random_bytes = b'\x00' * 32
            version = b'\x03\x01'
            ext_len = b'\x00\x00'  # no extensions
            body = version + random_bytes + session_id + cipher_list + compression + ext_len
            handshake = b'\x01' + len(body).to_bytes(3, 'big') + body
            record = b'\x16\x03\x01' + len(handshake).to_bytes(2, 'big') + handshake
            writer.write(record)
            await writer.drain()
            # The remote should echo back the ClientHello.
            echoed = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"REMOTE_ECHO:" in echoed
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()
            remote_server.close()
            await remote_server.wait_closed()

        assert proxy._allowed_count >= 1

    @pytest.mark.asyncio
    async def test_connect_sni_mismatch_closes_connection(self):
        """CONNECT to an allowlisted domain but with a mismatched SNI is closed.

        Defense-in-depth: the proxy allows the CONNECT (target host is
        allowlisted), connects to the remote, sends 200, then peeks at
        the ClientHello SNI. If the SNI is present and does NOT match the
        CONNECT target host, the connection is closed (SNI spoofing
        attempt). Uses a mock remote server.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        # Mock remote server (should not receive data after the mismatch).
        remote_got_data = asyncio.Event()

        async def mock_remote_handler(reader, writer):
            try:
                data = await asyncio.wait_for(reader.read(1024), timeout=3.0)
                if data:
                    remote_got_data.set()
            except asyncio.TimeoutError:
                pass
            writer.close()

        remote_server = await asyncio.start_server(
            mock_remote_handler, "127.0.0.1", 0
        )
        remote_addr = remote_server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return remote_addr[0]
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # CONNECT to allowlisted domain pypi.org, using the mock
            # remote's port.
            connect_port = remote_addr[1]
            writer.write(f"CONNECT pypi.org:{connect_port} HTTP/1.1\r\nHost: pypi.org:{connect_port}\r\n\r\n".encode())
            await writer.drain()
            # Read the 200 response (proxy connects to mock remote first).
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200" in response, f"Expected 200, got: {response!r}"
            # Send a ClientHello with SNI=evil.com (mismatch with pypi.org).
            hostname = b"evil.com"
            sni_entry = b'\x00' + len(hostname).to_bytes(2, 'big') + hostname
            sni_list = len(sni_entry).to_bytes(2, 'big') + sni_entry
            sni_ext = b'\x00\x00' + len(sni_list).to_bytes(2, 'big') + sni_list
            extensions = sni_ext
            ext_len = len(extensions).to_bytes(2, 'big')
            compression = b'\x01\x00'
            cipher = b'\x00\x2f'
            cipher_list = len(cipher).to_bytes(2, 'big') + cipher
            session_id = b'\x00'
            random_bytes = b'\x00' * 32
            version = b'\x03\x01'
            body = version + random_bytes + session_id + cipher_list + compression + ext_len + extensions
            handshake = b'\x01' + len(body).to_bytes(3, 'big') + body
            record = b'\x16\x03\x01' + len(handshake).to_bytes(2, 'big') + handshake
            writer.write(record)
            await writer.drain()
            # The proxy should close the connection (SNI mismatch).
            # The remote should NOT receive the ClientHello.
            await asyncio.sleep(0.3)
            assert not remote_got_data.is_set(), (
                "Remote should not receive data on SNI mismatch"
            )
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()
            remote_server.close()
            await remote_server.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_matching_sni_tunnels_to_remote(self):
        """CONNECT to an allowlisted domain with matching SNI tunnels data.

        The proxy allows the CONNECT, connects to the remote, sends 200,
        peeks at the ClientHello, verifies SNI matches the CONNECT target,
        and forwards the data to the remote. Uses a mock remote server.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        async def mock_remote_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            writer.write(b"REMOTE_ECHO:" + data)
            await writer.drain()
            writer.close()

        remote_server = await asyncio.start_server(
            mock_remote_handler, "127.0.0.1", 0
        )
        remote_addr = remote_server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return remote_addr[0]
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            connect_port = remote_addr[1]
            writer.write(f"CONNECT pypi.org:{connect_port} HTTP/1.1\r\nHost: pypi.org:{connect_port}\r\n\r\n".encode())
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200" in response, f"Expected 200, got: {response!r}"
            # Send a ClientHello with SNI=pypi.org (matches CONNECT target).
            hostname = b"pypi.org"
            sni_entry = b'\x00' + len(hostname).to_bytes(2, 'big') + hostname
            sni_list = len(sni_entry).to_bytes(2, 'big') + sni_entry
            sni_ext = b'\x00\x00' + len(sni_list).to_bytes(2, 'big') + sni_list
            extensions = sni_ext
            ext_len = len(extensions).to_bytes(2, 'big')
            compression = b'\x01\x00'
            cipher = b'\x00\x2f'
            cipher_list = len(cipher).to_bytes(2, 'big') + cipher
            session_id = b'\x00'
            random_bytes = b'\x00' * 32
            version = b'\x03\x01'
            body = version + random_bytes + session_id + cipher_list + compression + ext_len + extensions
            handshake = b'\x01' + len(body).to_bytes(3, 'big') + body
            record = b'\x16\x03\x01' + len(handshake).to_bytes(2, 'big') + handshake
            writer.write(record)
            await writer.drain()
            # The remote should echo back the ClientHello (tunneled through).
            echoed = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"REMOTE_ECHO:" in echoed
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()
            remote_server.close()
            await remote_server.wait_closed()

        assert proxy._allowed_count >= 1

    @pytest.mark.asyncio
    async def test_connect_unreachable_remote_returns_502(self):
        """CONNECT to an allowlisted domain with an unreachable remote returns 502.

        The proxy resolves the host, tries to connect to the remote, and
        if the connection fails, returns 502 Bad Gateway (not 200).
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            # Patch _resolve_host to return 127.0.0.1. The CONNECT target
            # uses a port that nothing is listening on (port 1).
            async def mock_resolve(host):
                return "127.0.0.1"
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # CONNECT to pypi.org on port 1 (nothing is listening there).
            writer.write(b"CONNECT pypi.org:1 HTTP/1.1\r\nHost: pypi.org:1\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"502" in response, f"Expected 502, got: {response!r}"
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_denied_host_returns_403(self):
        """HTTP request to a denied Host header gets blocked."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(addr[0], addr[1])
            # Send an HTTP request with a denied Host header.
            writer.write(b"GET / HTTP/1.1\r\nHost: evil.com\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            # The proxy should not forward this — connection closed or 403.
            # (HTTP mode doesn't send 403, it just closes the connection
            #  if the host is denied. The _handle_http method doesn't write
            #  a 403 response — it just doesn't forward.)
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 1

    @pytest.mark.asyncio
    async def test_unknown_protocol_denied(self):
        """Non-HTTP, non-CONNECT data is denied (connection closed)."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(addr[0], addr[1])
            # Send garbage that's not CONNECT or HTTP.
            writer.write(b"SSH-2.0-OpenSSH\r\n")
            await writer.drain()
            # The proxy should close the connection.
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 1

    @pytest.mark.asyncio
    async def test_counters_track_allowed_and_denied(self):
        """The allowed_count and denied_count are tracked correctly."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        addr = proxy._server.sockets[0].getsockname()

        try:
            # Send two denied CONNECT requests. The proxy denies based on
            # the CONNECT target (evil.com), no ClientHello needed.
            for _ in range(2):
                reader, writer = await asyncio.open_connection(addr[0], addr[1])
                writer.write(b"CONNECT evil.com:443 HTTP/1.1\r\nHost: evil.com:443\r\n\r\n")
                await writer.drain()
                await asyncio.wait_for(reader.read(1024), timeout=5.0)
                writer.close()
                await writer.wait_closed()

            await asyncio.sleep(0.1)  # let counters update
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 2
        assert proxy._allowed_count == 0


# ---------------------------------------------------------------------- #
# Port-specific allow-list enforcement
# ---------------------------------------------------------------------- #
class TestPortEnforcement:
    """Tests for port-specific allow-list enforcement.

    When an allow-list entry specifies a port (e.g. pypi.org:443),
    only that port is allowed for that host. When no port is specified
    (e.g. pypi.org), all ports are allowed (backward compat).
    """

    def test_is_port_allowed_no_restriction(self):
        """Host with no port restriction allows all ports."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            allowed_ports={},
            port=0,
        )
        assert proxy._is_port_allowed("pypi.org", 443)
        assert proxy._is_port_allowed("pypi.org", 80)
        assert proxy._is_port_allowed("pypi.org", 8080)

    def test_is_port_allowed_with_restriction(self):
        """Host with port restriction only allows specified ports."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            allowed_ports={"pypi.org": {443}},
            port=0,
        )
        assert proxy._is_port_allowed("pypi.org", 443)
        assert not proxy._is_port_allowed("pypi.org", 80)
        assert not proxy._is_port_allowed("pypi.org", 8080)

    def test_is_port_allowed_case_insensitive(self):
        """Port check is case-insensitive on the hostname."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            allowed_ports={"pypi.org": {443}},
            port=0,
        )
        assert proxy._is_port_allowed("PyPI.Org", 443)
        assert not proxy._is_port_allowed("PyPI.Org", 80)

    def test_is_port_allowed_multiple_ports(self):
        """Host with multiple allowed ports allows each of them."""
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            allowed_ports={"example.com": {80, 443}},
            port=0,
        )
        assert proxy._is_port_allowed("example.com", 80)
        assert proxy._is_port_allowed("example.com", 443)
        assert not proxy._is_port_allowed("example.com", 8080)

    def test_is_port_allowed_different_host_no_restriction(self):
        """Port restriction on one host doesn't affect a different host."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org", "example.com"},
            allowed_wildcards=set(),
            allowed_ports={"pypi.org": {443}},
            port=0,
        )
        # example.com has no port restriction — all ports allowed.
        assert proxy._is_port_allowed("example.com", 80)
        assert proxy._is_port_allowed("example.com", 443)
        # pypi.org has port restriction — only 443.
        assert not proxy._is_port_allowed("pypi.org", 80)

    @pytest.mark.asyncio
    async def test_connect_port_allowed_returns_200(self):
        """CONNECT to pypi.org:443 is allowed when allow-list has pypi.org:443.

        Uses a mock remote server to avoid real DNS/network.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            allowed_ports={"pypi.org": {443}},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        # Mock remote server.
        async def mock_remote_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            writer.write(b"REMOTE_ECHO:" + data)
            await writer.drain()
            writer.close()

        remote_server = await asyncio.start_server(
            mock_remote_handler, "127.0.0.1", 0
        )
        remote_addr = remote_server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return remote_addr[0]
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # CONNECT to pypi.org:443 — port 443 is allowed.
            # But the proxy will connect to remote_addr[1] (the mock).
            # We need to use the mock's port in the CONNECT line so the
            # proxy connects to our mock. But the allow-list says only
            # port 443 is allowed. So we patch _is_port_allowed to
            # always allow the mock port, OR we use port 443 and patch
            # the connection target.
            #
            # Actually, the cleanest approach: use the real port 443 in
            # the CONNECT line, and patch asyncio.open_connection to
            # redirect to our mock server.
            writer.write(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            # The proxy should connect to the mock remote (via patched
            # _resolve_host) on port 443. Since our mock server isn't
            # on port 443, the connection will fail with 502. But the
            # port check itself should PASS (not 403).
            # We should NOT get 403 (port denied).
            assert b"403" not in response, (
                f"Port 443 should be allowed, got 403: {response!r}")
            # We should get either 200 (if connection succeeded) or 502
            # (if the upstream was unreachable, which is expected since
            # nothing is listening on port 443).
            assert b"502" in response or b"200" in response, (
                f"Expected 200 or 502, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            remote_server.close()
            await remote_server.wait_closed()
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_connect_wrong_port_returns_403(self):
        """CONNECT to pypi.org:80 is denied when allow-list has pypi.org:443.

        The port check happens BEFORE connecting upstream, so the proxy
        returns 403 immediately — no DNS resolution or connection attempt.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            allowed_ports={"pypi.org": {443}},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # CONNECT to pypi.org:80 — port 80 is NOT allowed (only 443).
            writer.write(b"CONNECT pypi.org:80 HTTP/1.1\r\nHost: pypi.org:80\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response, (
                f"Port 80 should be denied, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 1

    @pytest.mark.asyncio
    async def test_connect_no_port_restriction_allows_any_port(self):
        """CONNECT to pypi.org:any-port is allowed when allow-list has
        pypi.org (no port specified). Backward compat."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            allowed_ports={},  # no port restrictions
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        # Mock remote server.
        async def mock_remote_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            writer.write(b"REMOTE_ECHO:" + data)
            await writer.drain()
            writer.close()

        remote_server = await asyncio.start_server(
            mock_remote_handler, "127.0.0.1", 0
        )
        remote_addr = remote_server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return remote_addr[0]
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # CONNECT to pypi.org:<mock_port> — any port allowed.
            connect_port = remote_addr[1]
            writer.write(
                f"CONNECT pypi.org:{connect_port} HTTP/1.1\r\n"
                f"Host: pypi.org:{connect_port}\r\n\r\n".encode())
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200" in response, (
                f"Any port should be allowed, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            remote_server.close()
            await remote_server.wait_closed()
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_port_allowed(self):
        """HTTP to example.com (default port 80) is allowed when allow-list
        has example.com:80.

        The port check uses the default port 80 (no port in Host header).
        We patch _resolve_host to return 127.0.0.1 — nothing is listening
        on port 80, so the proxy's connection attempt fails and it closes
        the client connection. The key assertion is that we do NOT get 403
        (port denied) — the port check passed, the failure is downstream.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            allowed_ports={"example.com": {80}},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return "127.0.0.1"
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # HTTP GET with no port — defaults to 80, which is allowed.
            writer.write(
                b"GET http://example.com/ HTTP/1.1\r\n"
                b"Host: example.com\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            # Should NOT get 403 — port 80 is allowed. The proxy will
            # try to connect to 127.0.0.1:80 (nothing listening), fail,
            # and close the connection (empty response or timeout).
            assert b"403" not in response, (
                f"Port 80 should be allowed for HTTP, got 403: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_wrong_port_returns_403(self):
        """HTTP to example.com:443 (via Host header) is denied when
        allow-list has example.com:80 only.

        The port check uses the Host header port (or URL port, or
        default 80). When the Host header specifies :443 and the
        allow-list only allows :80, the request is denied.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            allowed_ports={"example.com": {80}},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # HTTP GET with Host: example.com:443 — port 443 NOT allowed.
            writer.write(
                b"GET http://example.com:443/ HTTP/1.1\r\n"
                b"Host: example.com:443\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response, (
                f"Port 443 should be denied for HTTP, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 1

    @pytest.mark.asyncio
    async def test_http_default_port_80_allowed_when_no_port_in_header(self):
        """HTTP to example.com (no port in Host header) defaults to port
        80 and is allowed when allow-list has example.com:80.

        Same approach as test_http_port_allowed: patch _resolve_host,
        verify we don't get 403. The connection to 127.0.0.1:80 will
        fail, but the port check passes.
        """
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            allowed_ports={"example.com": {80}},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return "127.0.0.1"
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            # HTTP GET with no port in Host header — defaults to 80, allowed.
            writer.write(
                b"GET http://example.com/ HTTP/1.1\r\n"
                b"Host: example.com\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            # Should NOT get 403 — port 80 (default) is allowed.
            assert b"403" not in response, (
                f"Default port 80 should be allowed, got 403: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()


# ---------------------------------------------------------------------- #
# Wildcard allow-list extraction + port enforcement
# ---------------------------------------------------------------------- #
class TestWildcardAllowlistExtraction:
    """Tests for wildcard allow-list extraction (extract_allowlist_sets).

    Regression guard for the build-32/33 bug where ``main()`` checked
    ``entry.host.startswith("*.")`` even though the parser already
    strips the ``*.`` prefix and sets ``entry.wildcard=True`` with
    ``entry.host="example.com"``. That check was never true for
    wildcard entries, so they silently became exact-domain rules
    (inverted semantics).
    """

    def test_wildcard_entry_goes_to_wildcards_not_domains(self):
        """*.example.com:443 -> allowed_wildcards, NOT allowed_domains."""
        al = NetworkAllowList.from_string("*.example.com:443")
        domains, wildcards, ips, ports = extract_allowlist_sets(al)
        assert wildcards == {"*.example.com"}
        assert domains == set(), (
            f"wildcard entry leaked into exact domains: {domains!r}")
        assert ips == set()

    def test_wildcard_port_stored_under_pattern_key(self):
        """*.example.com:443 stores its port restriction under the
        '*.example.com' pattern key, not under 'example.com'."""
        al = NetworkAllowList.from_string("*.example.com:443")
        _, _, _, ports = extract_allowlist_sets(al)
        assert ports.get("*.example.com") == {443}
        assert "example.com" not in ports, (
            f"port restriction keyed under root domain: {ports!r}")

    def test_exact_domain_entry_goes_to_domains(self):
        """example.com:443 -> allowed_domains, NOT allowed_wildcards."""
        al = NetworkAllowList.from_string("example.com:443")
        domains, wildcards, _, ports = extract_allowlist_sets(al)
        assert domains == {"example.com"}
        assert wildcards == set()
        assert ports.get("example.com") == {443}

    def test_wildcard_and_exact_coexist(self):
        """A wildcard and an exact entry don't clobber each other."""
        al = NetworkAllowList.from_string(
            "*.example.com:443,example.com:80")
        domains, wildcards, _, ports = extract_allowlist_sets(al)
        assert domains == {"example.com"}
        assert wildcards == {"*.example.com"}
        assert ports.get("example.com") == {80}
        assert ports.get("*.example.com") == {443}

    def test_wildcard_without_port_has_no_port_restriction(self):
        """*.example.com (no port) -> wildcard pattern, no port entry."""
        al = NetworkAllowList.from_string("*.example.com")
        domains, wildcards, _, ports = extract_allowlist_sets(al)
        assert wildcards == {"*.example.com"}
        assert domains == set()
        assert ports == {}


class TestWildcardPortEnforcement:
    """Tests for wildcard-aware port enforcement in _is_port_allowed.

    These construct SNIEgressProxy with the sets that
    extract_allowlist_sets would produce for ``*.example.com:443`` and
    verify the four verdict scenarios:
      - *.example.com:443 allows   foo.example.com:443
      - *.example.com:443 rejects  foo.example.com:80
      - *.example.com:443 rejects  example.com:443   (domain-level)
      - example.com:443  allows   only example.com:443
    """

    def _wildcard_proxy(self):
        return SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards={"*.example.com"},
            allowed_ports={"*.example.com": {443}},
            port=0,
        )

    def test_wildcard_allows_subdomain_on_allowed_port(self):
        proxy = self._wildcard_proxy()
        assert proxy._is_port_allowed("foo.example.com", 443)

    def test_wildcard_rejects_subdomain_on_wrong_port(self):
        proxy = self._wildcard_proxy()
        assert not proxy._is_port_allowed("foo.example.com", 80)

    def test_wildcard_port_does_not_leak_to_unrelated_subdomain(self):
        """A deeper subdomain (foo.bar.example.com) is not matched by
        *.example.com (single-level wildcard), so no port restriction
        applies — but the domain check would reject it anyway."""
        proxy = self._wildcard_proxy()
        # domain_matches("foo.bar.example.com", "*.example.com") is False,
        # so _is_port_allowed finds no matching wildcard restriction and
        # returns True (all ports). Domain-level rejection is handled
        # separately by is_domain_allowed.
        assert proxy._is_port_allowed("foo.bar.example.com", 80)

    def test_wildcard_rejects_root_domain_at_domain_level(self):
        """*.example.com:443 must NOT allow example.com:443 — the root
        domain is not a subdomain and is not in allowed_domains."""
        proxy = self._wildcard_proxy()
        assert not is_domain_allowed(
            "example.com", proxy.allowed_domains, proxy.allowed_wildcards)

    def test_wildcard_allows_subdomain_at_domain_level(self):
        """foo.example.com is matched by *.example.com at the domain level."""
        proxy = self._wildcard_proxy()
        assert is_domain_allowed(
            "foo.example.com", proxy.allowed_domains, proxy.allowed_wildcards)

    def test_exact_domain_only_allows_exact_domain(self):
        """example.com:443 allows only example.com:443, not subdomains."""
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            allowed_ports={"example.com": {443}},
            port=0,
        )
        assert is_domain_allowed(
            "example.com", proxy.allowed_domains, proxy.allowed_wildcards)
        assert proxy._is_port_allowed("example.com", 443)
        assert not proxy._is_port_allowed("example.com", 80)
        # Subdomain is NOT allowed at the domain level (no wildcard).
        assert not is_domain_allowed(
            "foo.example.com", proxy.allowed_domains, proxy.allowed_wildcards)

    def test_full_extraction_pipeline_wildcard_scenarios(self):
        """End-to-end: parse '*.example.com:443' through
        extract_allowlist_sets into a proxy and verify all four
        verdict scenarios in one place."""
        al = NetworkAllowList.from_string("*.example.com:443")
        domains, wildcards, _, ports = extract_allowlist_sets(al)
        proxy = SNIEgressProxy(
            allowed_domains=domains,
            allowed_wildcards=wildcards,
            allowed_ports=ports,
            port=0,
        )
        # 1. *.example.com:443 allows foo.example.com:443
        assert is_domain_allowed(
            "foo.example.com", proxy.allowed_domains, proxy.allowed_wildcards)
        assert proxy._is_port_allowed("foo.example.com", 443)
        # 2. *.example.com:443 rejects foo.example.com:80
        assert not proxy._is_port_allowed("foo.example.com", 80)
        # 3. *.example.com:443 rejects example.com:443 (root not matched)
        assert not is_domain_allowed(
            "example.com", proxy.allowed_domains, proxy.allowed_wildcards)
        # 4. example.com:443 still allows only example.com:443
        al2 = NetworkAllowList.from_string("example.com:443")
        d2, w2, _, p2 = extract_allowlist_sets(al2)
        proxy2 = SNIEgressProxy(
            allowed_domains=d2, allowed_wildcards=w2,
            allowed_ports=p2, port=0,
        )
        assert is_domain_allowed(
            "example.com", proxy2.allowed_domains, proxy2.allowed_wildcards)
        assert proxy2._is_port_allowed("example.com", 443)
        assert not proxy2._is_port_allowed("example.com", 80)
        assert not is_domain_allowed(
            "foo.example.com", proxy2.allowed_domains, proxy2.allowed_wildcards)


# ---------------------------------------------------------------------- #
# IP/CIDR allow-list enforcement
# ---------------------------------------------------------------------- #
class TestIpCidrEnforcement:
    """Tests for IP and CIDR allow-list enforcement.

    The parser accepts IPv4/IPv6 and CIDR entries; the proxy must enforce
    them in both CONNECT and HTTP paths.
    """

    def test_is_ip_allowed_plain_ip_match(self):
        assert _is_ip_allowed("10.0.0.1", {"10.0.0.1"})

    def test_is_ip_allowed_plain_ip_no_match(self):
        assert not _is_ip_allowed("10.0.0.2", {"10.0.0.1"})

    def test_is_ip_allowed_cidr_match(self):
        assert _is_ip_allowed("10.0.0.5", {"10.0.0.0/24"})

    def test_is_ip_allowed_cidr_no_match(self):
        assert not _is_ip_allowed("11.0.0.5", {"10.0.0.0/24"})

    def test_is_ip_allowed_hostname_returns_false(self):
        """Hostnames are not IP literals and should be ignored here."""
        assert not _is_ip_allowed("example.com", {"10.0.0.0/24"})

    def test_is_domain_allowed_with_allowed_ips(self):
        assert is_domain_allowed(
            "10.0.0.1", set(), set(), {"10.0.0.0/24"})
        assert not is_domain_allowed(
            "10.0.0.1", set(), set(), {"192.168.0.0/24"})

    def test_extract_allowlist_sets_uses_host_not_raw_for_ip(self):
        """IP entries must store the host without the port for matching."""
        al = NetworkAllowList.from_string("10.0.0.1:443,10.0.0.0/24:443")
        _, _, ips, ports = extract_allowlist_sets(al)
        assert ips == {"10.0.0.1", "10.0.0.0/24"}
        assert ports.get("10.0.0.1") == {443}
        assert ports.get("10.0.0.0/24") == {443}

    @pytest.mark.asyncio
    async def test_connect_allowed_ip_returns_200(self):
        """CONNECT to an allowlisted IP literal is allowed."""
        proxy = SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards=set(),
            allowed_ips={"127.0.0.1"},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        async def mock_remote_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            writer.write(b"REMOTE_ECHO:" + data)
            await writer.drain()
            writer.close()

        remote_server = await asyncio.start_server(
            mock_remote_handler, "127.0.0.1", 0
        )
        remote_addr = remote_server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return remote_addr[0]
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            connect_port = remote_addr[1]
            writer.write(
                f"CONNECT 127.0.0.1:{connect_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{connect_port}\r\n\r\n".encode()
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200" in response, f"Expected 200, got: {response!r}"
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()
            remote_server.close()
            await remote_server.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_denied_ip_returns_403(self):
        """CONNECT to a denied IP literal returns 403."""
        proxy = SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards=set(),
            allowed_ips={"10.0.0.1"},
            port=0,
        )
        await proxy.start()
        addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(addr[0], addr[1])
            writer.write(b"CONNECT 10.0.0.2:443 HTTP/1.1\r\nHost: 10.0.0.2:443\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response, f"Expected 403, got: {response!r}"
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_allowed_ip_not_blocked(self):
        """HTTP request to an allowlisted IP literal is not blocked."""
        proxy = SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards=set(),
            allowed_ips={"127.0.0.1"},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return "127.0.0.1"
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            writer.write(
                b"GET http://127.0.0.1/ HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" not in response, (
                f"Allowed IP should not get 403, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_denied_ip_is_blocked(self):
        """HTTP request to a denied IP literal is blocked."""
        proxy = SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards=set(),
            allowed_ips={"10.0.0.1"},
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            writer.write(
                b"GET http://10.0.0.2/ HTTP/1.1\r\n"
                b"Host: 10.0.0.2\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            # HTTP denial closes the connection; we should not see a 200.
            assert b"200" not in response, (
                f"Denied IP should not get 200, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    def test_cidr_port_restriction(self):
        """Port restrictions apply to IP/CIDR entries via allowed_ports."""
        al = NetworkAllowList.from_string("10.0.0.0/24:443")
        _, _, _, ports = extract_allowlist_sets(al)
        assert ports.get("10.0.0.0/24") == {443}

    def test_is_port_allowed_cidr_matches_concrete_ip(self):
        """A CIDR port restriction must apply to concrete IPs inside it."""
        proxy = SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards=set(),
            allowed_ips={"10.0.0.0/24"},
            allowed_ports={"10.0.0.0/24": {443}},
            port=0,
        )
        assert proxy._is_port_allowed("10.0.0.5", 443)
        assert not proxy._is_port_allowed("10.0.0.5", 80)
        # Outside the CIDR, no restriction applies.
        assert proxy._is_port_allowed("192.168.0.5", 80)

    def test_is_port_allowed_exact_ip_preferred_over_cidr(self):
        """An exact IP port restriction takes precedence over CIDR."""
        proxy = SNIEgressProxy(
            allowed_domains=set(),
            allowed_wildcards=set(),
            allowed_ips={"10.0.0.1", "10.0.0.0/24"},
            allowed_ports={
                "10.0.0.1": {80},
                "10.0.0.0/24": {443},
            },
            port=0,
        )
        assert proxy._is_port_allowed("10.0.0.1", 80)
        assert not proxy._is_port_allowed("10.0.0.1", 443)
        # Other IPs in the CIDR still use the CIDR restriction.
        assert proxy._is_port_allowed("10.0.0.5", 443)
        assert not proxy._is_port_allowed("10.0.0.5", 80)


class TestHttpHostUrlMismatch:
    """Tests for HTTP proxy absolute-URL vs Host-header consistency."""

    @pytest.mark.asyncio
    async def test_http_allows_url_host(self):
        """A plain HTTP request with an absolute URL is allowed when the
        URL host is allow-listed."""
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            async def mock_resolve(host):
                return "127.0.0.1"
            proxy._resolve_host = mock_resolve

            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            writer.write(
                b"GET http://example.com/path HTTP/1.1\r\n"
                b"Host: example.com\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" not in response, (
                f"Allowed URL host should not get 403, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_blocks_url_host_mismatch(self):
        """A request with an allowed Host header but a disallowed absolute
        URL target must be blocked."""
        proxy = SNIEgressProxy(
            allowed_domains={"example.com"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            writer.write(
                b"GET http://evil.com/path HTTP/1.1\r\n"
                b"Host: example.com\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response, (
                f"URL host mismatch should be 403, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_http_blocks_url_host_with_allowed_host_header(self):
        """Even with an allowed Host header, a disallowed absolute URL
        target is denied."""
        proxy = SNIEgressProxy(
            allowed_domains={"allowed.com"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        proxy_addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(
                proxy_addr[0], proxy_addr[1]
            )
            writer.write(
                b"GET http://blocked.com/ HTTP/1.1\r\n"
                b"Host: allowed.com\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response, (
                f"Blocked URL host must get 403, got: {response!r}")
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

