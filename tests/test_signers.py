"""Tests for the pluggable signer abstraction (signers.py).

Covers:
  - HmacSigner (stdlib, default since v0.3.8)
  - Ed25519Signer (optional cryptography package, v0.3.9)
  - make_signer factory
  - Integration with BiTemporalAuditLog (Ed25519 signing + verification)
  - Tamper detection with Ed25519 (asymmetric — public key can't forge)
"""

import json
import os
import tempfile

import pytest

from signers import HmacSigner, Ed25519Signer, make_signer, Signer


# Skip Ed25519 tests if the cryptography package is not installed.
cryptography_available = True
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: F401
except ImportError:
    cryptography_available = False

skip_no_cryptography = pytest.mark.skipif(
    not cryptography_available,
    reason="cryptography package not installed — Ed25519 tests skipped",
)


class TestHmacSigner:
    """Tests for the stdlib HMAC-SHA256 signer."""

    def test_sign_produces_correct_prefix(self):
        s = HmacSigner("secret-key")
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig = s.sign(entry)
        assert sig.startswith("hmac-sha256:")

    def test_verify_accepts_valid_signature(self):
        s = HmacSigner("secret-key")
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig = s.sign(entry)
        assert s.verify(entry, sig)

    def test_verify_rejects_tampered_entry(self):
        s = HmacSigner("secret-key")
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig = s.sign(entry)
        tampered = dict(entry)
        tampered["status"] = "tampered"
        assert not s.verify(tampered, sig)

    def test_verify_rejects_wrong_scheme(self):
        s = HmacSigner("secret-key")
        entry = {"record_id": "r1"}
        assert not s.verify(entry, "ed25519:abc123")

    def test_str_key_and_bytes_key_equivalent(self):
        """A str key and its bytes equivalent produce the same signature."""
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig_str = HmacSigner("mykey").sign(entry)
        sig_bytes = HmacSigner(b"mykey").sign(entry)
        assert sig_str == sig_bytes

    def test_signer_protocol_conformance(self):
        s = HmacSigner("key")
        assert isinstance(s, Signer)


@skip_no_cryptography
class TestEd25519Signer:
    """Tests for the Ed25519 asymmetric signer."""

    def test_generate_and_sign(self):
        s = Ed25519Signer.generate()
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig = s.sign(entry)
        assert sig.startswith("ed25519:")
        assert s.verify(entry, sig)

    def test_tamper_detection(self):
        s = Ed25519Signer.generate()
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig = s.sign(entry)
        tampered = dict(entry)
        tampered["status"] = "tampered"
        assert not s.verify(tampered, sig)

    def test_public_key_cannot_forge(self):
        """The verify-only signer (public key) cannot produce signatures."""
        s = Ed25519Signer.generate()
        pub_only = Ed25519Signer.from_public_key_hex(s.public_key_hex)
        entry = {"record_id": "r1"}
        # Verify-only signer should refuse to sign
        with pytest.raises(PermissionError):
            pub_only.sign(entry)
        # But it CAN verify signatures from the private-key holder
        sig = s.sign(entry)
        assert pub_only.verify(entry, sig)

    def test_persist_and_reload_private_key(self):
        s1 = Ed25519Signer.generate()
        priv_hex = s1.private_key_hex
        s2 = Ed25519Signer.from_private_key_hex(priv_hex)
        entry = {"record_id": "r1", "status": "pending", "record_hash": "x"}
        sig = s1.sign(entry)
        # Reloaded signer should verify the original signature
        assert s2.verify(entry, sig)
        # And produce identical signatures (deterministic Ed25519)
        assert s2.sign(entry) == sig

    def test_public_key_hex_is_consistent(self):
        s = Ed25519Signer.generate()
        pub_hex_1 = s.public_key_hex
        pub_hex_2 = s.public_key_hex
        assert pub_hex_1 == pub_hex_2
        # Verify-only signer loaded from the public key hex should verify
        pub_only = Ed25519Signer.from_public_key_hex(pub_hex_1)
        entry = {"record_id": "r1"}
        assert pub_only.verify(entry, s.sign(entry))

    def test_wrong_public_key_fails_verification(self):
        s1 = Ed25519Signer.generate()
        s2 = Ed25519Signer.generate()  # different keypair
        entry = {"record_id": "r1"}
        sig = s1.sign(entry)
        # s2's public key should NOT verify s1's signature
        pub2 = Ed25519Signer.from_public_key_hex(s2.public_key_hex)
        assert not pub2.verify(entry, sig)

    def test_malformed_signature_rejected(self):
        s = Ed25519Signer.generate()
        entry = {"record_id": "r1"}
        assert not s.verify(entry, "ed25519:not-hex!")
        assert not s.verify(entry, "ed25519:")
        assert not s.verify(entry, "wrong-scheme:abc")

    def test_signer_protocol_conformance(self):
        s = Ed25519Signer.generate()
        assert isinstance(s, Signer)


class TestMakeSignerFactory:
    """Tests for the make_signer factory function."""

    def test_none_returns_none(self):
        assert make_signer() is None

    def test_hmac_key_returns_hmacsigner(self):
        s = make_signer(signing_key="secret")
        assert isinstance(s, HmacSigner)

    @skip_no_cryptography
    def test_ed25519_private_key_takes_precedence(self):
        signer = Ed25519Signer.generate()
        s = make_signer(
            signing_key="secret",
            ed25519_private_key_hex=signer.private_key_hex,
        )
        assert isinstance(s, Ed25519Signer)

    @skip_no_cryptography
    def test_ed25519_public_key_returns_verify_only(self):
        signer = Ed25519Signer.generate()
        s = make_signer(ed25519_public_key_hex=signer.public_key_hex)
        assert isinstance(s, Ed25519Signer)
        with pytest.raises(PermissionError):
            s.sign({"record_id": "r1"})


@skip_no_cryptography
class TestEd25519BitemporalIntegration:
    """End-to-end: BiTemporalAuditLog with Ed25519 signing."""

    @pytest.fixture
    def log_path(self):
        path = tempfile.mktemp(suffix=".jsonl")
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_ed25519_signed_entry_has_signature(self, log_path):
        from bitemporal_log import BiTemporalAuditLog
        signer = Ed25519Signer.generate()
        log = BiTemporalAuditLog(log_path, signer=signer)

        class FakeJob:
            job_id = "j1"
            status = type("S", (), {"value": "pending"})()
            query = "q"
            priority = 5
            force_route = None

        log.record(FakeJob(), "submitted")
        entry = log.read_all()[0]
        assert entry["signature"].startswith("ed25519:")

    def test_ed25519_chain_verifies(self, log_path):
        from bitemporal_log import BiTemporalAuditLog
        signer = Ed25519Signer.generate()
        log = BiTemporalAuditLog(log_path, signer=signer)

        class FakeJob:
            job_id = "j1"
            status = type("S", (), {"value": "pending"})()
            query = "q"
            priority = 5
            force_route = None

        for evt in ["submitted", "started", "completed"]:
            log.record(FakeJob(), evt)
        ok, errors = log.verify_chain(strict=True)
        assert ok, f"Ed25519 chain should verify: {errors}"

    def test_ed25519_detects_tampering(self, log_path):
        from bitemporal_log import BiTemporalAuditLog, _hash_entry
        signer = Ed25519Signer.generate()
        log = BiTemporalAuditLog(log_path, signer=signer)

        class FakeJob:
            job_id = "j1"
            status = type("S", (), {"value": "pending"})()
            query = "q"
            priority = 5
            force_route = None

        log.record(FakeJob(), "submitted")
        # Tamper: change status AND recompute record_hash (simulating an
        # attacker who fixes the hash chain). Ed25519 should still catch it.
        with open(log_path) as f:
            lines = f.readlines()
        entry = json.loads(lines[0])
        entry["status"] = "tampered"
        entry["record_hash"] = _hash_entry(entry)
        lines[0] = json.dumps(entry) + "\n"
        with open(log_path, "w") as f:
            f.writelines(lines)
        ok, errors = log.verify_chain(strict=True)
        assert not ok
        assert any("signature mismatch" in e for e in errors)

    def test_ed25519_via_private_key_hex_param(self, log_path):
        """The constructor accepts ed25519_private_key_hex directly."""
        from bitemporal_log import BiTemporalAuditLog
        signer = Ed25519Signer.generate()
        log = BiTemporalAuditLog(
            log_path, ed25519_private_key_hex=signer.private_key_hex
        )

        class FakeJob:
            job_id = "j1"
            status = type("S", (), {"value": "pending"})()
            query = "q"
            priority = 5
            force_route = None

        log.record(FakeJob(), "submitted")
        entry = log.read_all()[0]
        assert entry["signature"].startswith("ed25519:")
        ok, _ = log.verify_chain(strict=True)
        assert ok

    def test_ed25519_verify_only_via_public_key_hex(self, log_path):
        """A verify-only node (public key) can verify but not sign."""
        from bitemporal_log import BiTemporalAuditLog
        signer = Ed25519Signer.generate()
        # Write with the private key
        log = BiTemporalAuditLog(
            log_path, ed25519_private_key_hex=signer.private_key_hex
        )

        class FakeJob:
            job_id = "j1"
            status = type("S", (), {"value": "pending"})()
            query = "q"
            priority = 5
            force_route = None

        log.record(FakeJob(), "submitted")
        # Verify with a public-key-only log (no signing capability)
        verify_log = BiTemporalAuditLog(
            log_path, ed25519_public_key_hex=signer.public_key_hex
        )
        ok, errors = verify_log.verify_chain(strict=True)
        assert ok, f"Public-key verify should pass: {errors}"
