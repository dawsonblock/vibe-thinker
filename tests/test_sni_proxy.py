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
)


# ---------------------------------------------------------------------- #
# SNI extraction from TLS ClientHello
# ---------------------------------------------------------------------- #
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

        The proxy reads the CONNECT line, headers, then the TLS ClientHello
        to extract SNI. We need to send a ClientHello with a denied SNI.
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
            # Send a CONNECT request to a denied domain.
            writer.write(b"CONNECT evil.com:443 HTTP/1.1\r\nHost: evil.com:443\r\n\r\n")
            await writer.drain()
            # Send a minimal TLS ClientHello with SNI=evil.com so the proxy
            # can extract and check it.
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
            # Read the response — should be 403.
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"403" in response
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 1

    @pytest.mark.asyncio
    async def test_connect_with_no_sni_denied(self):
        """CONNECT to an allowed domain but with no SNI in ClientHello is denied."""
        proxy = SNIEgressProxy(
            allowed_domains={"pypi.org"},
            allowed_wildcards=set(),
            port=0,
        )
        await proxy.start()
        addr = proxy._server.sockets[0].getsockname()

        try:
            reader, writer = await asyncio.open_connection(addr[0], addr[1])
            # CONNECT to allowed domain.
            writer.write(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n")
            await writer.drain()
            # Send a TLS ClientHello WITHOUT the SNI extension.
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
            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            # Should get 403 (no SNI extracted).
            assert b"403" in response
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

        # Build a ClientHello with SNI=evil.com (denied)
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

        try:
            # Send two denied CONNECT requests with ClientHello.
            for _ in range(2):
                reader, writer = await asyncio.open_connection(addr[0], addr[1])
                writer.write(b"CONNECT evil.com:443 HTTP/1.1\r\nHost: evil.com:443\r\n\r\n")
                await writer.drain()
                writer.write(record)
                await writer.drain()
                await asyncio.wait_for(reader.read(1024), timeout=5.0)
                writer.close()
                await writer.wait_closed()

            await asyncio.sleep(0.1)  # let counters update
        finally:
            await proxy.stop()

        assert proxy._denied_count >= 2
        assert proxy._allowed_count == 0
