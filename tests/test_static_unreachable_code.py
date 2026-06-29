"""Static anti-regression check: unreachable code after terminal statements.

This catches the root cause of the ``_run_clr_with_cache`` regression: the
method body was orphaned as dead code (a bare string-literal "docstring"
plus an ``if`` block) placed AFTER a ``return`` inside another method. The
class compiled, the unreachable string was a harmless no-op, and the only
symptom was that ``run()`` called a method that did not exist (caught
separately by test_static_missing_self_methods.py).

Ruff does NOT flag a bare string literal after ``return`` (a string
expression is valid Python), so this AST check is the real gate for that
class of mistake.

Approach (AST, no execution):
  For every function/method body (and every nested block body: if/elif/else,
  for/else, while/else, with, try/except/else/finally), scan the statement
  list linearly. Once a terminal statement is seen (``return`` / ``raise`` /
  ``break`` / ``continue``), every subsequent sibling statement in THAT SAME
  body is unreachable and is reported.

  A terminal inside a nested block (e.g. ``return`` inside an ``if``) only
  makes the rest of that ``if`` body unreachable — it does NOT affect the
  siblings of the ``if`` (the branch may not be taken). This avoids false
  positives.

This is a strict gate: any unreachable statement fails the test with a
precise file:line listing.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_EXCLUDE_PARTS = frozenset({
    "tests", "build", "rust", "__pycache__", ".venv", ".venv-core",
    "dist", ".git", "examples", "docs", ".pytest_cache", ".mypy_cache",
    ".ruff_cache",
})

# Statements that terminate control flow within their enclosing block.
_TERMINAL = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _iter_production_files():
    for p in ROOT.rglob("*.py"):
        if any(part in _EXCLUDE_PARTS for part in p.parts):
            continue
        yield p


def _statement_lists(node):
    """Yield every ordered statement-list (body) within `node` where
    unreachable-after-terminal detection should run.

    Covers: the node's own body, plus the bodies of nested control-flow
    blocks (if/elif/else, for/else, while/else, with, try/except/else/
    finally). Each yielded list is checked independently so a terminal in
    one branch does not flag siblings of that branch.
    """
    # Direct children statement lists of common block nodes.
    bodies = []
    for attr in ("body", "orelse", "finalbody"):
        val = getattr(node, attr, None)
        if isinstance(val, list):
            bodies.append(val)
    # except handlers each have their own body.
    handlers = getattr(node, "handlers", None)
    if handlers:
        for h in handlers:
            if isinstance(getattr(h, "body", None), list):
                bodies.append(h.body)
    return bodies


def _scan_function(func_node, path):
    """Report unreachable statements within one function/method."""
    violations = []

    def _check_body(body):
        terminal_seen_line = None
        for stmt in body:
            if terminal_seen_line is not None:
                # Anything after a terminal in this same block is dead.
                violations.append((stmt.lineno, type(stmt).__name__))
                # Recurse into the dead stmt's sub-blocks too (report all).
                for sub in _statement_lists(stmt):
                    _check_body(sub)
                continue
            if isinstance(stmt, _TERMINAL):
                terminal_seen_line = stmt.lineno
            # Recurse into nested blocks of this (reachable) statement.
            for sub in _statement_lists(stmt):
                _check_body(sub)

    _check_body(func_node.body)
    return violations


def _scan_file(path):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for lineno, kind in _scan_function(node, path):
                out.append((lineno, kind, node.name))
    return out


def test_no_unreachable_code_after_terminal():
    """No statement may follow a return/raise/break/continue in the same
    block — that is unreachable dead code (the orphaned-method-body bug).
    """
    all_violations = []
    for path in _iter_production_files():
        try:
            violations = _scan_file(path)
        except SyntaxError as e:
            all_violations.append(
                (f"<file:{path.relative_to(ROOT)}>",
                 f"SyntaxError: {e}", e.lineno or 0))
            continue
        for lineno, kind, func in violations:
            all_violations.append(
                (f"{path.relative_to(ROOT)}::{func}()", kind, lineno))

    if all_violations:
        lines = "\n".join(
            f"  {loc}: unreachable {kind} at line {line} "
            f"(follows a return/raise/break/continue in the same block)"
            for loc, kind, line in all_violations
        )
        pytest.fail(
            f"{len(all_violations)} unreachable statement(s) found — dead "
            f"code after a terminal statement (the orphaned-method-body "
            f"regression class):\n{lines}")
