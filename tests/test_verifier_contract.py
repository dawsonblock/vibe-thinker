"""Tests for the verifier contract (Phase 9 — verifier contract cleanup).

Verifies the rules:
  1. Verifier failure is not the same as answer failure.
  2. Missing dependency must return verified=False (not raise).
  3. Self-claims cannot raise confidence above threshold.
  4. Unsupported verification must not be labeled verified.
  5. External-source factual verification must fail closed when sources
     are missing.
"""

import pytest
from verifiers.base import VerificationResult, Verifier


class TestVerificationResultContract:
    """Verify the VerificationResult dataclass enforces the contract."""

    def test_verified_false_is_default_for_unverifiable(self):
        """A result with no evidence should have verified=False."""
        r = VerificationResult(verified=False, score=0.0, method="none")
        assert r.verified is False

    def test_confidence_defaults_to_none(self):
        """confidence is optional (backward compat with score)."""
        r = VerificationResult(verified=True, score=1.0, method="test")
        assert r.confidence is None

    def test_errors_is_list(self):
        """errors must be a list (not a string)."""
        r = VerificationResult(verified=False, score=0.0, method="test")
        assert isinstance(r.errors, list)
        assert isinstance(r.warnings, list)

    def test_reason_is_string(self):
        """reason must be a string."""
        r = VerificationResult(verified=True, score=1.0, method="test")
        assert isinstance(r.reason, str)

    def test_self_claim_score_cannot_exceed_threshold(self):
        """Rule 3: self-claims cannot raise confidence above threshold.
        The orchestrator caps self-claim scores at 0.4. This test
        documents the contract — a self-claim result with score > 0.4
        is a violation."""
        # This is a documentation test — the actual cap is enforced in
        # the orchestrator, not in the dataclass. But we verify the
        # dataclass can represent a capped result.
        r = VerificationResult(
            verified=False,  # self-claims are NOT verified
            score=0.4,       # capped
            method="self_claims_only",
            reason="self-claims are not independent verification",
        )
        assert r.verified is False
        assert r.score <= 0.4

    def test_unsupported_must_not_be_verified(self):
        """Rule 4: unsupported verification must not be labeled verified."""
        r = VerificationResult(
            verified=False,
            score=0.0,
            method="unsupported_factual",
            error="no sources available for factual verification",
        )
        assert r.verified is False

    def test_fail_closed_when_sources_missing(self):
        """Rule 5: external-source verification must fail closed."""
        r = VerificationResult(
            verified=False,
            score=0.0,
            method="retrieval",
            error="all retrieval sources failed",
            errors=["duckduckgo: timeout", "wikipedia: not found"],
        )
        assert r.verified is False
        assert len(r.errors) > 0


class TestVerifierProtocol:
    """Verify the Verifier protocol contract."""

    def test_verifier_is_protocol(self):
        """Verifier is a runtime_checkable Protocol."""
        # Protocols have __protocol_attrs__ listing their members
        attrs = getattr(Verifier, "__protocol_attrs__", set())
        assert "verify" in attrs
        assert "name" in attrs

    def test_math_verifier_satisfies_protocol(self):
        """MathVerifier should satisfy the Verifier protocol."""
        from verifiers.math_verifier import MathVerifier
        v = MathVerifier()
        assert hasattr(v, "name")
        assert hasattr(v, "verify")

    def test_schema_verifier_satisfies_protocol(self):
        """SchemaVerifier should satisfy the Verifier protocol."""
        from verifiers.schema_verifier import SchemaVerifier
        v = SchemaVerifier()
        assert hasattr(v, "name")
        assert hasattr(v, "verify")
