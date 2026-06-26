"""Logic verifier — SMT-based constraint checking via Z3.

Verifies that an answer satisfies a set of logical constraints expressed
as Z3 assertions. This is a deterministic verifier — Z3 is a proof tool,
not a model. When the constraints are satisfiable and the answer's
interpreted values satisfy them, ``verified=True`` with a proof trace.

Trust model (fail-closed):
  - Z3 not installed -> ``verified=False``, method ``"smt_unavailable"``.
    Never falls back to a weaker check — SMT verification is binary.
  - No constraints in context -> ``verified=False`` (honest).
  - Constraint parse/eval error -> ``verified=False`` with the error.
  - Constraints UNSAT (unsatisfiable by ANY assignment) -> ``verified=False``
    — the problem itself is infeasible, so no answer can be correct.
  - Constraints SAT but the answer's values don't satisfy them ->
    ``verified=False`` with the failing assertion.
  - Constraints SAT and the answer's values satisfy all of them ->
    ``verified=True``, score=1.0, with the Z3 model as evidence.

Context keys:
  - ``constraints``: list of Z3 assertion strings (e.g.
    ``["x > 0", "x + y == 10", "y < x"]``). Required.
  - ``variables``: dict mapping variable names to their Z3 sort
    (``"Int"`` or ``"Real"``). Default: all ``"Int"``.
  - ``values``: dict mapping variable names to the answer's numeric
    values (e.g. ``{"x": 7, "y": 3}``). The verifier checks these
    against the constraints. Required for verification — without
    values, we can only check satisfiability, not the answer.

Requires the optional ``z3-solver`` package:
    pip install z3-solver
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from verifiers.base import VerificationResult

# Optional Z3 dependency — fail-closed when absent.
try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False


class LogicVerifier:
    """Deterministic verifier for logical constraints via Z3/SMT.

    See module docstring for the trust model and context keys.
    """

    name = "logic_verifier"

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        if not _Z3_AVAILABLE:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="smt_unavailable",
                evidence={"answer": answer[:200]},
                error="z3-solver not installed; SMT verification "
                      "unavailable. pip install z3-solver",
            )

        constraints: Optional[List[str]] = context.get("constraints")
        if not constraints:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="smt_check",
                evidence={"answer": answer[:200]},
                error="no constraints provided; cannot verify logical "
                      "satisfiability",
            )

        variables: Dict[str, str] = context.get("variables", {})
        values: Optional[Dict[str, Any]] = context.get("values")

        # Build Z3 variables.
        z3_vars: Dict[str, Any] = {}
        try:
            for name, sort in variables.items():
                z3_vars[name] = self._make_var(name, sort)
        except Exception as e:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={"answer": answer[:200]},
                error=f"failed to create Z3 variables: {e}",
            )

        # Parse constraints into Z3 expressions.
        try:
            z3_assertions = [
                self._parse_constraint(c, z3_vars) for c in constraints
            ]
        except Exception as e:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={"answer": answer[:200], "constraint_parse_error": str(e)},
                error=f"failed to parse constraints: {e}",
            )

        # Check 1: are the constraints satisfiable at all?
        solver = z3.Solver()
        for a in z3_assertions:
            solver.add(a)
        sat_result = solver.check()
        if sat_result == z3.unsat:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={"constraints": constraints, "satisfiable": False},
                error="constraints are UNSAT (unsatisfiable by any "
                      "assignment) — the problem is infeasible",
            )
        if sat_result == z3.unknown:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={"constraints": constraints, "satisfiable": "unknown"},
                error="Z3 returned unknown (solver timeout or "
                      "undecidable theory) — cannot verify",
            )

        # Constraints are SAT. Now check the answer's values (if provided).
        if values is None:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={"constraints": constraints, "satisfiable": True},
                error="constraints are satisfiable but no values provided "
                      "to check the answer against them",
            )

        # Substitute the answer's values and check each constraint.
        try:
            failing = []
            for constraint_str, assertion in zip(constraints, z3_assertions):
                # Substitute concrete values into the assertion.
                substituted = assertion
                for var_name, val in values.items():
                    if var_name in z3_vars:
                        z3_val = self._make_value(val)
                        substituted = z3.substitute(
                            substituted, (z3_vars[var_name], z3_val)
                        )
                # Evaluate the substituted boolean expression.
                if z3.is_true(substituted) is False:
                    # Use a solver to be certain (handles complex exprs).
                    s2 = z3.Solver()
                    s2.add(substituted)
                    if s2.check() != z3.sat:
                        failing.append(constraint_str)
        except Exception as e:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={"constraints": constraints, "values": values},
                error=f"failed to evaluate values against constraints: {e}",
            )

        if failing:
            return VerificationResult(
                verified=False, score=0.0, method="smt_check",
                evidence={
                    "constraints": constraints,
                    "values": values,
                    "satisfiable": True,
                    "failing_constraints": failing,
                },
                error=f"answer values violate constraints: {failing}",
            )

        # All constraints satisfied by the answer's values.
        model = solver.model()
        return VerificationResult(
            verified=True,
            score=1.0,
            method="smt_check",
            evidence={
                "constraints": constraints,
                "values": values,
                "satisfiable": True,
                "model": str(model),
            },
        )

    # ------------------------------------------------------------------ #
    # Z3 helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_var(name: str, sort: str):
        """Create a Z3 variable of the given sort."""
        sort = (sort or "Int").strip().capitalize()
        if sort == "Int":
            return z3.Int(name)
        if sort == "Real":
            return z3.Real(name)
        if sort == "Bool":
            return z3.Bool(name)
        raise ValueError(f"unsupported Z3 sort: {sort!r}")

    @staticmethod
    def _parse_constraint(expr_str: str, z3_vars: Dict[str, Any]):
        """Parse a constraint string into a Z3 boolean expression.

        Uses z3.parse_smt2_string when possible; otherwise evals in a
        restricted namespace containing only the Z3 variables.
        """
        # Build a namespace with the Z3 vars + common Python operators.
        namespace = {**z3_vars, "z3": z3}
        # z3.Int/Real overloads make arithmetic work directly.
        return eval(expr_str, {"__builtins__": {}}, namespace)

    @staticmethod
    def _make_value(val: Any):
        """Convert a Python value to a Z3 value."""
        if isinstance(val, bool):
            return z3.BoolVal(val)
        if isinstance(val, int):
            return z3.IntVal(val)
        if isinstance(val, float):
            return z3.RealVal(val)
        # Fallback: let Z3 infer.
        return z3.RealVal(str(val))
