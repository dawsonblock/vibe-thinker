"""Tests for the SNI-aware egress proxy (v0.4.1).

Tests the SNI extraction from TLS ClientHello, domain matching with
wildcards, and the proxy's allow/deny logic. Does NOT test the actual
TCP tunneling (that requires a live server — covered by integration
tests).
"""

import pytest

from sandbox.sni_proxy import (
    extract_sni,
    domain_matches,
    is_domain_allowed,
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
