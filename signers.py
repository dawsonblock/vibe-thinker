"""
Pluggable signers for the bi-temporal audit log.

The audit log needs cryptographic proof of authorship so that an attacker
who tampers with the log file cannot simply recompute the hash chain.
The signer abstraction lets the log use different signature schemes
without coupling the log code to a specific crypto library.

Available signers:
  - ``HmacSigner``    : HMAC-SHA256 with a shared secret key (stdlib only,
                        the default since v0.3.8). Tamper-proof as long as
                        the key is not leaked, but anyone holding the key
                        can forge entries.
  - ``Ed25519Signer`` : Ed25519 public-key signatures via the optional
                        ``cryptography`` package. Each node has a private
                        key; the public key is published. An attacker
                        cannot forge signatures without the private key,
                        even if they see the public key. This is SLSA L2
                        compliant asymmetric provenance — the integration
                        plan's Phase 1.1 goal.

The ``cryptography`` package is an OPTIONAL dependency. When it is not
installed, ``Ed25519Signer`` cannot be constructed (raises
``ImportError`` with a helpful message). The default behavior of the
audit log (HMAC-SHA256, stdlib only) is unchanged.

Key generation:
  >>> from signers import Ed25519Signer
  >>> signer = Ed25519Signer.generate()  # new random keypair
  >>> signer.public_key_hex  # publish this so verifiers can check
  'a1b2c3...'
  >>> signer.private_key_hex  # keep this secret, persist to disk

Loading a persisted keypair:
  >>> signer = Ed25519Signer.from_private_key_hex(private_key_hex)

Integration plan reference: Phase 1.1 — "Secure the Bi-Temporal Audit
Log" with Ed25519 signatures from ruvnet/agent-harness-generator. This
module provides the same cryptographic property (asymmetric Ed25519
signatures) using the well-audited ``cryptography`` package instead of
the ruvnet crate, so vibe-thinker remains stdlib-by-default with an
optional, widely-vetted crypto dep.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class Signer(Protocol):
    """Protocol for audit-log entry signers.

    A signer produces a signature string for a log entry's content and
    verifies that a signature is valid. The signature covers the entry
    content EXCLUDING the ``record_hash`` and ``signature`` fields
    (those are derived from the content and cannot be part of it).

    The signature string format is ``"<scheme>:<hex>"`` (e.g.
    ``"hmac-sha256:abc..."`` or ``"ed25519:abc..."``). The scheme prefix
    lets ``verify_chain`` dispatch to the correct verifier even when a
    log contains entries signed by different schemes (e.g. during a
    migration from HMAC to Ed25519).
    """

    @property
    def scheme(self) -> str:
        """The scheme prefix (e.g. 'hmac-sha256', 'ed25519')."""
        ...

    def sign(self, entry: Dict[str, Any]) -> str:
        """Produce a signature string for the entry's content.

        Args:
            entry: the log entry dict. The ``record_hash`` and
                ``signature`` fields are excluded from the signed
                content.

        Returns:
            A signature string of the form ``"<scheme>:<hex>"``.
        """
        ...

    def verify(self, entry: Dict[str, Any], signature: str) -> bool:
        """Verify that ``signature`` is valid for ``entry``'s content.

        Args:
            entry: the log entry dict.
            signature: the signature string to check.

        Returns:
            True if the signature is valid, False otherwise.
        """
        ...


def _signed_content(entry: Dict[str, Any]) -> bytes:
    """Serialize the entry content (excluding record_hash and signature)
    to canonical bytes for signing/verification.

    This MUST match the content used by ``bitemporal_log._hash_entry`` and
    ``bitemporal_log._sign_entry`` so that signatures cover the same
    payload as the hash chain.
    """
    import json
    content = {k: v for k, v in entry.items()
               if k not in ("record_hash", "signature")}
    raw = json.dumps(content, sort_keys=True, default=str)
    return raw.encode("utf-8")


class HmacSigner:
    """HMAC-SHA256 signer with a shared secret key (stdlib only).

    This is the default signer. It is tamper-proof (an attacker cannot
    forge signatures without the key) but symmetric: anyone who can
    verify can also forge. For asymmetric provenance, use
    :class:`Ed25519Signer`.

    Args:
        key: the secret key (bytes or str). If str, encoded as UTF-8.
    """

    scheme = "hmac-sha256"

    def __init__(self, key):
        if not isinstance(key, (bytes, bytearray)):
            key = key.encode("utf-8")
        self._key = bytes(key)

    def sign(self, entry: Dict[str, Any]) -> str:
        raw = _signed_content(entry)
        digest = hmac.new(self._key, raw, hashlib.sha256).hexdigest()
        return f"{self.scheme}:{digest}"

    def verify(self, entry: Dict[str, Any], signature: str) -> bool:
        if not isinstance(signature, str) or not signature.startswith(f"{self.scheme}:"):
            return False
        expected = self.sign(entry)
        return hmac.compare_digest(signature, expected)


class Ed25519Signer:
    """Ed25519 asymmetric signer via the optional ``cryptography`` package.

    Ed25519 is a fast, secure elliptic-curve signature scheme. Unlike
    HMAC, the verification key (public key) cannot forge signatures —
    only the holder of the private key can sign. This provides SLSA L2
    compliant provenance: the log entries are mathematically tied to a
    specific node's private key.

    The ``cryptography`` package is required:
        pip install cryptography

    Args:
        private_key: an ``Ed25519PrivateKey`` object from ``cryptography``.
        public_key:  the corresponding ``Ed25519PublicKey``. If omitted,
            derived from the private key.

    Class methods:
        generate(): create a new random keypair.
        from_private_key_hex(hex): load from a persisted private key.
        from_public_key_hex(hex): load a verify-only signer (cannot sign).

    Attributes:
        public_key_hex:  the public key as hex (publish this for verifiers).
        private_key_hex: the private key as hex (persist this; keep secret).
    """

    scheme = "ed25519"

    def __init__(self, private_key=None, public_key=None):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey, Ed25519PublicKey,
            )
        except ImportError as e:
            raise ImportError(
                "Ed25519Signer requires the 'cryptography' package: "
                "pip install cryptography"
            ) from e

        self._Ed25519PrivateKey = Ed25519PrivateKey
        self._Ed25519PublicKey = Ed25519PublicKey

        if private_key is not None:
            self._private_key = private_key
            self._public_key = private_key.public_key()
            self._can_sign = True
        elif public_key is not None:
            self._private_key = None
            self._public_key = public_key
            self._can_sign = False
        else:
            raise ValueError("Either private_key or public_key must be provided")

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        """Generate a new random Ed25519 keypair."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as e:
            raise ImportError(
                "Ed25519Signer requires the 'cryptography' package: "
                "pip install cryptography"
            ) from e
        return cls(private_key=Ed25519PrivateKey.generate())

    @classmethod
    def from_private_key_hex(cls, hex_key: str) -> "Ed25519Signer":
        """Load a signer from a persisted private key (hex string)."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as e:
            raise ImportError(
                "Ed25519Signer requires the 'cryptography' package: "
                "pip install cryptography"
            ) from e
        raw = bytes.fromhex(hex_key)
        return cls(private_key=Ed25519PrivateKey.from_private_bytes(raw))

    @classmethod
    def from_public_key_hex(cls, hex_key: str) -> "Ed25519Signer":
        """Load a verify-only signer from a public key (hex string).

        The returned signer cannot produce signatures (``sign()`` raises
        ``PermissionError``). Use this for verification-only nodes that
        do not write to the log.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
        except ImportError as e:
            raise ImportError(
                "Ed25519Signer requires the 'cryptography' package: "
                "pip install cryptography"
            ) from e
        raw = bytes.fromhex(hex_key)
        return cls(public_key=Ed25519PublicKey.from_public_bytes(raw))

    @property
    def public_key_hex(self) -> str:
        return self._public_key.public_bytes(
            encoding=self._get_encoding(),
            format=self._get_public_format(),
        ).hex()

    @property
    def private_key_hex(self) -> Optional[str]:
        if self._private_key is None:
            return None
        return self._private_key.private_bytes(
            encoding=self._get_encoding(),
            format=self._get_private_format(),
            encryption_algorithm=self._get_no_encryption(),
        ).hex()

    def _get_encoding(self):
        from cryptography.hazmat.primitives import serialization
        return serialization.Encoding.Raw

    def _get_public_format(self):
        from cryptography.hazmat.primitives import serialization
        return serialization.PublicFormat.Raw

    def _get_private_format(self):
        from cryptography.hazmat.primitives import serialization
        return serialization.PrivateFormat.Raw

    def _get_no_encryption(self):
        from cryptography.hazmat.primitives import serialization
        return serialization.NoEncryption()

    def sign(self, entry: Dict[str, Any]) -> str:
        if not self._can_sign:
            raise PermissionError(
                "This Ed25519Signer was loaded from a public key and "
                "cannot sign entries. Load from a private key to sign."
            )
        raw = _signed_content(entry)
        sig_bytes = self._private_key.sign(raw)
        return f"{self.scheme}:{sig_bytes.hex()}"

    def verify(self, entry: Dict[str, Any], signature: str) -> bool:
        if not isinstance(signature, str) or not signature.startswith(f"{self.scheme}:"):
            return False
        sig_hex = signature[len(f"{self.scheme}:"):]
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            return False
        raw = _signed_content(entry)
        try:
            self._public_key.verify(sig_bytes, raw)
            return True
        except Exception:
            return False


def make_signer(signing_key=None, ed25519_private_key_hex: Optional[str] = None,
                ed25519_public_key_hex: Optional[str] = None) -> Optional[Signer]:
    """Factory: build the appropriate signer from config.

    Precedence:
      1. ed25519_private_key_hex  -> Ed25519Signer (sign + verify)
      2. ed25519_public_key_hex   -> Ed25519Signer (verify only)
      3. signing_key              -> HmacSigner (sign + verify, stdlib)
      4. None                     -> None (no signing, tamper-evident only)

    This lets the audit log accept any of the three config styles without
    knowing about the concrete signer classes.

    Args:
        signing_key: HMAC shared secret (bytes/str) — the v0.3.8 default.
        ed25519_private_key_hex: hex-encoded Ed25519 private key — enables
            asymmetric signing (Phase 1.1 of the integration plan).
        ed25519_public_key_hex: hex-encoded Ed25519 public key — verify-only
            mode for nodes that read but don't write the log.

    Returns:
        A Signer instance, or None if no key was provided.
    """
    if ed25519_private_key_hex:
        return Ed25519Signer.from_private_key_hex(ed25519_private_key_hex)
    if ed25519_public_key_hex:
        return Ed25519Signer.from_public_key_hex(ed25519_public_key_hex)
    if signing_key is not None:
        return HmacSigner(signing_key)
    return None
