"""Deterministic verifier adapters for the vibe-thinker control plane.

A verifier provides independent (non-model-self-checking) evidence that an
answer is correct. The base protocol and result model live in ``base.py``.

Available verifiers:
  - MathVerifier:    numeric answer extraction + comparison
  - CodeVerifier:    bounded subprocess execution of Python snippets / unit tests
  - FactualVerifier: honest "unsupported" placeholder (no fake factual verification)
"""

from verifiers.base import VerificationResult, Verifier
from verifiers.math_verifier import MathVerifier
from verifiers.code_verifier import CodeVerifier
from verifiers.factual_verifier import FactualVerifier

__all__ = [
    "VerificationResult",
    "Verifier",
    "MathVerifier",
    "CodeVerifier",
    "FactualVerifier",
]
