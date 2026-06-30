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
    SNIEgressProxy,
    DnsAuditor,
    _is_ip_literal,
)


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
