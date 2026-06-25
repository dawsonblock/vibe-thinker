"""Pytest tests for the separated confidence scoring."""

import pytest

from scoring import ConfidenceBreakdown, SELF_CLAIMS_ONLY_CAP, compute_confidence


class TestComputeConfidence:
    def test_deterministic_verified_blends_scores(self):
        result = compute_confidence(
            model_score=0.82,
            claim_consistency=0.74,
            deterministic_verification=1.0,
            verification_method="python_eval",
        )
        assert result.verified is True
        # 1.0 * 0.7 + 0.74 * 0.3 = 0.7 + 0.222 = 0.922
        assert abs(result.final_score - 0.922) < 1e-4

    def test_no_deterministic_capped_at_self_claims_only(self):
        result = compute_confidence(
            model_score=0.95,
            claim_consistency=0.90,
            deterministic_verification=None,
            verification_method="self_claims_only",
        )
        assert result.verified is False
        assert result.final_score == SELF_CLAIMS_ONLY_CAP  # 0.65
        assert result.verification_method == "self_claims_only"

    def test_self_claims_only_never_exceeds_cap(self):
        """Even with perfect self-verification, cap at 0.65."""
        result = compute_confidence(
            model_score=1.0,
            claim_consistency=1.0,
            deterministic_verification=None,
            verification_method="self_claims_only",
        )
        assert result.final_score == SELF_CLAIMS_ONLY_CAP
        assert result.verified is False

    def test_deterministic_refuted_scores_zero(self):
        result = compute_confidence(
            model_score=0.95,
            claim_consistency=0.90,
            deterministic_verification=0.0,
            verification_method="python_eval",
        )
        assert result.verified is False
        assert result.final_score == 0.0

    def test_low_claim_consistency_without_deterministic(self):
        result = compute_confidence(
            model_score=0.5,
            claim_consistency=0.3,
            deterministic_verification=None,
        )
        assert result.final_score == 0.3  # min(0.3, 0.65) = 0.3

    def test_to_dict_has_all_fields(self):
        result = compute_confidence(
            model_score=0.82,
            claim_consistency=0.74,
            deterministic_verification=1.0,
            verification_method="unit_tests",
        )
        d = result.to_dict()
        assert "model_confidence" in d
        assert "claim_consistency" in d
        assert "deterministic_verification" in d
        assert "final_score" in d
        assert "verified" in d
        assert "verification_method" in d

    def test_self_claims_only_cap_value(self):
        assert SELF_CLAIMS_ONLY_CAP == 0.65

    def test_partial_deterministic_verification(self):
        """Deterministic verification between 0 and 1 blends normally."""
        result = compute_confidence(
            model_score=0.8,
            claim_consistency=0.8,
            deterministic_verification=0.5,
            verification_method="numeric_comparison",
        )
        # 0.5 * 0.7 + 0.8 * 0.3 = 0.35 + 0.24 = 0.59
        assert abs(result.final_score - 0.59) < 1e-4
        # compute_confidence.verified is intentionally strict (>= 1.0).
        # Callers use v_result.verified for actual verification status,
        # not confidence.verified. Partial scores blend normally but
        # don't set the verified flag in ConfidenceBreakdown.
        assert result.verified is False
