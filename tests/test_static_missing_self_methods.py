"""Static anti-regression check: catch missing private methods before runtime.

This test would have failed on the build that shipped with
``HybridReasoningOrchestrator.run`` calling ``self._run_clr_with_cache`` while
the method body was orphaned as dead code (a bare string literal after a
``return``) inside another method — so the class compiled but ``run()``
crashed with AttributeError at runtime, and the existing tests masked it by
monkeypatching the method onto instances.

Approach (AST-based, no execution):
  1. Parse every production .py file (tests/, build/, rust/, venvs, docs/,
     examples/ are excluded).
  2. For each class, build the set of "known" callable names:
       - methods defined directly on the class (def/async def),
       - methods inherited from base classes defined in the SAME file
         (cross-module bases can't be resolved statically),
       - instance attributes assigned via ``self.<name> = ...`` anywhere in
         the class's methods (callable instance attributes such as
         ``self._clock = time.monotonic`` are a legitimate, common pattern
         and must not be reported as missing methods).
  3. For every ``self.<name>(...)`` call inside a class method, flag it if
     ``<name>`` is not known and not in the explicit allowlist below.

The allowlist is intentionally tiny: a new entry means a method is genuinely
supplied dynamically (e.g. injected via setattr at runtime) and cannot be
seen statically. Prefer defining the method for real over allowlisting.

This is a strict gate: any violation fails the test with a precise
file:Class:line listing so the missing method is fixed before runtime.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Directories that are not production Python source we want to gate.
_EXCLUDE_PARTS = frozenset({
    "tests", "build", "rust", "__pycache__", ".venv", ".venv-core",
    "dist", ".git", "examples", "docs", ".pytest_cache", ".mypy_cache",
    ".ruff_cache",
})


def _is_excluded_part(part: str) -> bool:
    """True if a path component is not production source we want to gate.

    Covers the explicit set above plus ANY ``.venv*`` directory (e.g.
    ``.venv-local``, ``.venv-docker``, ``.venv-embeddings``,
    ``.venv-federation``) so per-profile gate venvs created in the project
    root are never scanned as production source.
    """
    return part in _EXCLUDE_PARTS or part.startswith(".venv")

# Methods that are genuinely dynamic (set via setattr / injection at runtime
# and never visible to a static scan). Add here ONLY if a flagged call is
# legitimately supplied outside the class body. Empty by design — the current
# codebase needs no entries.
_ALLOWLIST: frozenset[str] = frozenset()


def _iter_production_files():
    for p in ROOT.rglob("*.py"):
        if any(_is_excluded_part(part) for part in p.parts):
            continue
        yield p


def _class_info(node: ast.ClassDef):
    """Return (methods, self_attrs, self_calls, base_names) for a ClassDef."""
    methods: set[str] = set()
    self_attrs: set[str] = set()
    self_calls: list[tuple[str, int]] = []

    def _record_self_target(target):
        if (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            self_attrs.add(target.attr)

    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        methods.add(item.name)
        # Walk the whole method body (including nested closures that close
        # over `self`), but skip nested ClassDefs — their `self` is a
        # different instance.
        for sub in ast.walk(item):
            if isinstance(sub, ast.ClassDef):
                # Don't attribute nested-class self-calls to this class.
                continue
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "self"):
                self_calls.append((sub.func.attr, sub.lineno))
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    _record_self_target(t)
            if isinstance(sub, ast.AnnAssign):
                _record_self_target(sub.target)
            # Augmented assignments like self.x += 1 also establish the attr.
            if isinstance(sub, ast.AugAssign):
                _record_self_target(sub.target)

    base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
    return methods, self_attrs, self_calls, base_names


def _scan_file(path: pathlib.Path):
    """Return a list of (class, call_name, lineno) violations in one file."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    classes = {
        n.name: _class_info(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    violations = []
    for cname, (methods, attrs, calls, bases) in classes.items():
        known = set(methods) | set(attrs)
        for b in bases:
            if b in classes:
                bmethods, battrs, _, _ = classes[b]
                known |= bmethods | battrs
        for call_name, lineno in calls:
            if call_name in known or call_name in _ALLOWLIST:
                continue
            violations.append((cname, call_name, lineno))
    return violations


def test_no_missing_self_method_calls():
    """Every self.<name>(...) call must resolve to a defined method or a
    self-assigned callable attribute on the class (or a same-file base).

    Fails with a precise listing if any production class calls a private
    method it does not define — the exact signature of the
    _run_clr_with_cache regression.
    """
    all_violations = []
    for path in _iter_production_files():
        try:
            violations = _scan_file(path)
        except SyntaxError as e:
            # A syntax error in gated production source is itself a failure.
            all_violations.append(
                (f"<file:{path.relative_to(ROOT)}>",
                 f"SyntaxError: {e}", e.lineno or 0))
            continue
        for cname, call_name, lineno in violations:
            all_violations.append(
                (f"{path.relative_to(ROOT)}::{cname}", call_name, lineno))

    if all_violations:
        lines = "\n".join(
            f"  {loc}: self.{name}() called at line {line} "
            f"but not defined on the class (or allowlisted)"
            for loc, name, line in all_violations
        )
        pytest.fail(
            f"{len(all_violations)} unresolved self.<method>() call(s) "
            f"found — production code calls a method the class does not "
            f"define:\n{lines}\n"
            f"Define the method, or add it to _ALLOWLIST if it is genuinely "
            f"dynamic.")
