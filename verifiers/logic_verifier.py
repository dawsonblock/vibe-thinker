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

import ast
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
    def validate_constraints(
        constraints: List[str], variables: Dict[str, str]
    ) -> Optional[str]:
        """Validate that constraints parse as Z3 expressions.

        Used by the translation retry loop (Phase 3.2) to check whether
        a generalist-produced constraint set is syntactically valid Z3
        BEFORE running the full verification. This separates "bad
        translation" (the generalist wrote invalid Z3 syntax) from "bad
        answer" (the answer's values violate valid constraints).

        Args:
            constraints: list of Z3 assertion strings, or None/empty.
            variables: dict mapping variable names to Z3 sorts, or None.

        Returns:
            None if all constraints parse successfully, or an error
            message string describing the first parse failure (suitable
            for feeding back to the generalist for a retry).
        """
        if not _Z3_AVAILABLE:
            return "z3-solver not installed; cannot validate constraints"
        if constraints is None:
            constraints = []
        if variables is None:
            variables = {}
        # Build Z3 variables.
        z3_vars: Dict[str, Any] = {}
        try:
            for name, sort in variables.items():
                z3_vars[name] = LogicVerifier._make_var(name, sort)
        except Exception as e:
            return f"failed to create Z3 variable: {e}"
        # Parse each constraint.
        for i, c in enumerate(constraints):
            try:
                LogicVerifier._parse_constraint(c, z3_vars)
            except Exception as e:
                return (
                    f"constraint {i} ({c!r}) failed to parse: {e}. "
                    f"Use only Z3 Python syntax with the declared "
                    f"variables: {list(z3_vars.keys())}. Supported ops: "
                    f"arithmetic (+, -, *, /), comparison (==, !=, <, >, "
                    f"<=, >=), And(), Or(), Not(), Implies(), If()."
                )
        return None

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

        Uses a safe AST-based evaluator instead of eval() to prevent
        code injection via Python introspection (e.g. accessing builtins
        through ``().__class__.__base__.__subclasses__()``).

        Only allows:
          - Names referencing Z3 variables or whitelisted Z3 functions
          - Arithmetic (+, -, *, /, unary -)
          - Comparisons (==, !=, <, >, <=, >=)
          - Boolean ops (and, or, not)
          - Calls to whitelisted Z3 functions (And, Or, Not, Implies, etc.)
          - Integer, float, and boolean literals

        Attribute access (``.``), subscripts (``[]``), comprehensions,
        lambdas, imports, and any other constructs are rejected.
        """
        namespace = {
            **z3_vars,
            "And": z3.And,
            "Or": z3.Or,
            "Not": z3.Not,
            "Implies": z3.Implies,
            "If": z3.If,
            "Xor": z3.Xor,
            "True": z3.BoolVal(True),
            "False": z3.BoolVal(False),
        }
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"invalid constraint syntax: {e}")

        evaluator = _SafeZ3Evaluator(namespace)
        return evaluator.visit(tree.body)

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


class _SafeZ3Evaluator(ast.NodeVisitor):
    """Safe AST evaluator for Z3 constraint strings.

    Replaces ``eval()`` with a whitelist-based approach that only allows
    arithmetic, comparisons, boolean ops, and calls to predefined Z3
    functions. This prevents code injection through Python's object
    introspection (e.g. ``().__class__.__base__.__subclasses__()``).
    """

    # AST node types that are allowed (everything else raises).
    _ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.Name,
        ast.Constant,
        ast.Call,
        ast.Load,
    )

    # Binary operators supported.
    _BIN_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }

    # Unary operators. Note: `Not` is handled separately in visit_UnaryOp
    # because Z3 BoolRef requires z3.Not() rather than Python's `~`.
    _UNARY_OPS = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    # Comparison operators.
    _CMP_OPS = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
    }

    # Boolean operators.
    _BOOL_OPS = {
        ast.And: all,
        ast.Or: any,
    }

    def __init__(self, namespace: Dict[str, Any]):
        self._ns = namespace

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(
                f"disallowed AST node: {type(node).__name__} — "
                f"only arithmetic, comparisons, boolean ops, and "
                f"calls to Z3 functions are allowed"
            )
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(
            f"only int, float, and bool literals are allowed, "
            f"got {type(node.value).__name__}"
        )

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._ns:
            raise ValueError(
                f"undefined name: {node.id!r} — only declared Z3 "
                f"variables and whitelisted Z3 functions are allowed"
            )
        return self._ns[node.id]

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_fn = self._BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
        return op_fn(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            # Z3 BoolRef requires z3.Not(); Python `~` doesn't work.
            try:
                import z3 as _z3
                if isinstance(operand, _z3.BoolRef):
                    return _z3.Not(operand)
            except Exception:
                pass
            return not operand
        op_fn = self._UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op_fn(operand)

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_fn = self._CMP_OPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"unsupported comparison: {type(op).__name__}")
            left = op_fn(left, right)
        return left

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        # For Z3, And/Or are functions that accept symbolic BoolRefs.
        # We use the Z3 functions directly from the namespace when available,
        # otherwise fall back to Python all()/any().
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            # Use z3.And if any value is a Z3 expression, else Python all.
            try:
                import z3 as _z3
                if any(isinstance(v, _z3.BoolRef) for v in values):
                    return _z3.And(*values)
            except Exception:
                pass
            return all(values)
        elif isinstance(node.op, ast.Or):
            try:
                import z3 as _z3
                if any(isinstance(v, _z3.BoolRef) for v in values):
                    return _z3.Or(*values)
            except Exception:
                pass
            return any(values)
        raise ValueError(f"unsupported boolean operator: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        # Keyword args are not allowed (keeps it simple + safe).
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in constraints")
        return func(*args)
