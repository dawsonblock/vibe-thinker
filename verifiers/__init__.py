"""Deterministic verifier adapters for the vibe-thinker control plane.

A verifier provides independent (non-model-self-checking) evidence that an
answer is correct. The base protocol and result model live in ``base.py``.

Available verifiers:
  - MathVerifier:    numeric answer extraction + comparison
  - CodeVerifier:    bounded subprocess execution of Python snippets / unit tests
  - FactualVerifier: NLI judge against retrieved sources (fail-closed)
  - SchemaVerifier:  JSON/YAML/regex structural conformance (v0.4.0)
  - LogicVerifier:   Z3/SMT constraint satisfaction (v0.4.0, optional z3-solver)
"""

from verifiers.base import VerificationResult, Verifier
from verifiers.math_verifier import MathVerifier
from verifiers.code_verifier import CodeVerifier
from verifiers.factual_verifier import FactualVerifier
from verifiers.schema_verifier import SchemaVerifier
from verifiers.logic_verifier import LogicVerifier

__all__ = [
    "VerificationResult",
    "Verifier",
    "MathVerifier",
    "CodeVerifier",
    "FactualVerifier",
    "SchemaVerifier",
    "LogicVerifier",
]
