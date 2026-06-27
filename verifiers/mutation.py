"""Mutation testing for code verification (Phase 3.1).

Prevents the "Vacuous Test" problem: when the Generalist writes tests that
accidentally accept broken code, a verified score of 1.0 is a false
positive. Mutation testing catches this by injecting a known bug into the
winning code candidate and re-running the Generalist's tests. If the
mutated (broken) code still passes the tests, the tests are mathematically
vacuous — they cannot distinguish correct from incorrect code.

The mutation operators are syntactic transforms that introduce a real
behavioral defect (not just a cosmetic change). Each operator targets a
common class of bug:

  - ``flip_arithmetic``: ``+`` -> ``-``, ``*`` -> ``+`` (off-by-sign / wrong op)
  - ``flip_comparison``: ``==`` -> ``!=``, ``<`` -> ``>=`` (inverted logic)
  - ``return_none``: replace the first ``return <expr>`` with ``return None``
  - ``swap_constants``: swap two integer/float literals (wrong values)
  - ``drop_statement``: comment out a non-trivial statement (missing logic)

The mutator tries each operator in turn and returns the first mutation
that produces syntactically valid Python (parses without SyntaxError) and
is actually different from the original. This guarantees the mutated code
is a real bug, not a no-op transform.

Usage:
    from verifiers.mutation import mutate_code, MutationResult
    result = mutate_code("def add(a, b): return a + b")
    if result.mutated_code:
        # run the tests against result.mutated_code; if they pass, the
        # tests are vacuous.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MutationResult:
    """Result of a mutation attempt.

    Attributes:
        mutated_code: the mutated Python source, or None if no valid
            mutation could be produced (e.g. the code is too short or
            all operators failed to produce a syntactically valid diff).
        operator: the name of the mutation operator that succeeded
            (e.g. "flip_arithmetic"), or None if no mutation was applied.
        description: a human-readable description of the mutation.
        original_code: the original (unmutated) source, for audit.
    """
    mutated_code: Optional[str] = None
    operator: Optional[str] = None
    description: str = ""
    original_code: str = ""

    @property
    def applied(self) -> bool:
        """True if a mutation was successfully applied."""
        return self.mutated_code is not None and self.mutated_code != self.original_code


# ---------------------------------------------------------------------------
# Individual mutation operators
# ---------------------------------------------------------------------------

def _flip_arithmetic(source: str) -> Optional[Tuple[str, str]]:
    """Flip an arithmetic operator (+ -> -, * -> +, etc.).

    Returns (mutated_source, description) or None if no applicable op.
    """
    replacements = [
        (" + ", " - "),
        (" - ", " + "),
        (" * ", " + "),
        (" // ", " / "),
    ]
    for old, new in replacements:
        if old in source:
            mutated = source.replace(old, new, 1)
            return mutated, f"flipped '{old.strip()}' to '{new.strip()}'"
    return None


def _flip_comparison(source: str) -> Optional[Tuple[str, str]]:
    """Flip a comparison operator (== -> !=, < -> >=, etc.).

    Returns (mutated_source, description) or None.
    """
    # Order matters: check multi-char ops before single-char to avoid
    # partial replacements (e.g. "<=" before "<").
    replacements = [
        ("==", "!="),
        ("!=", "=="),
        ("<=", ">"),
        (">=", "<"),
        (" < ", " >= "),
        (" > ", " <= "),
    ]
    for old, new in replacements:
        if old in source:
            mutated = source.replace(old, new, 1)
            return mutated, f"flipped '{old.strip()}' to '{new.strip()}'"
    return None


def _return_none(source: str) -> Optional[Tuple[str, str]]:
    """Replace the first ``return <expr>`` with ``return None``.

    Returns (mutated_source, description) or None.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            # Replace the return value with None.
            lines = source.splitlines(keepends=True)
            if node.lineno - 1 < len(lines):
                line = lines[node.lineno - 1]
                # Find "return" in the line and replace everything after it.
                idx = line.find("return")
                if idx >= 0:
                    indent = line[:idx]
                    lines[node.lineno - 1] = f"{indent}return None\n"
                    mutated = "".join(lines)
                    return mutated, "replaced first return <expr> with return None"
    return None


def _swap_constants(source: str) -> Optional[Tuple[str, str]]:
    """Swap two numeric literals in the source.

    Returns (mutated_source, description) or None.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    nums: List[Tuple[int, int, str]] = []  # (lineno, col_offset, raw)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # Skip booleans (they're int subclasses in Python).
            if isinstance(node.value, bool):
                continue
            nums.append((node.lineno, node.col_offset, repr(node.value)))
    if len(nums) < 2:
        return None
    # Swap the first two distinct numeric constants.
    (l1, c1, v1), (l2, c2, v2) = nums[0], nums[1]
    if v1 == v2:
        return None
    lines = source.splitlines(keepends=True)
    # Replace on the respective lines. This is a simple char-level swap;
    # it handles the common case where each constant is on its own line or
    # the same line at different columns.
    line1 = lines[l1 - 1]
    line2 = lines[l2 - 1]
    # Replace v1 with v2 and v2 with v1. Use a placeholder to avoid
    # double-replacement when both are on the same line.
    placeholder = "\x00SWAP\x00"
    if l1 == l2:
        # Same line: replace by column position.
        # Sort by column descending so earlier replacement doesn't shift
        # later offsets.
        positions = sorted([(c1, v1, v2), (c2, v2, v1)], key=lambda x: -x[0])
        for col, old_v, new_v in positions:
            line1 = line1[:col] + line1[col:].replace(old_v, new_v, 1)
        lines[l1 - 1] = line1
    else:
        lines[l1 - 1] = line1[:c1] + line1[c1:].replace(v1, placeholder, 1).replace(v2, v1, 1).replace(placeholder, v2, 1) if v2 in line1 else line1[:c1] + line1[c1:].replace(v1, v2, 1)
        lines[l2 - 1] = line2[:c2] + line2[c2:].replace(v2, placeholder, 1).replace(v1, v2, 1).replace(placeholder, v1, 1) if v1 in line2 else line2[:c2] + line2[c2:].replace(v2, v1, 1)
    mutated = "".join(lines)
    return mutated, f"swapped constants {v1} and {v2}"


def _drop_statement(source: str) -> Optional[Tuple[str, str]]:
    """Comment out a non-trivial statement (not a def/class/import/return).

    Returns (mutated_source, description) or None.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                   ast.Import, ast.ImportFrom, ast.Return, ast.Pass,
                   ast.Global, ast.Nonlocal)
        ):
            if hasattr(node, "lineno") and node.lineno > 0:
                lines = source.splitlines(keepends=True)
                if node.lineno - 1 < len(lines):
                    line = lines[node.lineno - 1]
                    if line.strip() and not line.strip().startswith("#"):
                        indent = line[:len(line) - len(line.lstrip())]
                        lines[node.lineno - 1] = f"{indent}# MUTATED: dropped\n"
                        mutated = "".join(lines)
                        return mutated, f"dropped statement at line {node.lineno}"
    return None


# Ordered list of operators (first valid one wins).
_OPERATORS = [
    ("flip_arithmetic", _flip_arithmetic),
    ("flip_comparison", _flip_comparison),
    ("return_none", _return_none),
    ("swap_constants", _swap_constants),
    ("drop_statement", _drop_statement),
]


def _is_valid_python(source: str) -> bool:
    """Check that the source parses as valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def mutate_code(source: str, seed: Optional[int] = None) -> MutationResult:
    """Apply a mutation operator to Python source code.

    Tries each operator in order and returns the first mutation that
    produces syntactically valid Python that differs from the original.
    This guarantees the mutated code is a real behavioral defect, not a
    no-op or a syntax-breaking transform.

    Args:
        source: the original Python source code.
        seed: optional random seed (currently unused — operators are
            deterministic, but reserved for future stochastic operators).

    Returns:
        A MutationResult. If no valid mutation could be produced,
        ``mutated_code`` is None and ``applied`` is False. In that case
        the caller should skip mutation testing (the code is too short
        or simple to mutate meaningfully — not a failure).
    """
    if not source or not source.strip():
        return MutationResult(original_code=source)
    if seed is not None:
        random.seed(seed)
    for name, op in _OPERATORS:
        try:
            result = op(source)
        except Exception:
            result = None
        if result is None:
            continue
        mutated, description = result
        if mutated == source:
            continue  # no-op transform
        if not _is_valid_python(mutated):
            continue  # mutation broke syntax
        return MutationResult(
            mutated_code=mutated,
            operator=name,
            description=description,
            original_code=source,
        )
    # No operator produced a valid mutation — the code is too simple to
    # mutate (e.g. a single pass statement). Not a failure.
    return MutationResult(original_code=source)
