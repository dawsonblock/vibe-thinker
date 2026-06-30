#!/usr/bin/env python3
"""Build a clean source distribution ZIP for vibe-thinker.

Copies the full project source (Python modules, tests, Docker, CI, Rust
probes, ruvllm_py, docs, configs) into a staging directory, excludes all
runtime/build junk (__pycache__, .pyc, .pytest_cache, target/, vendor/,
dist/, cache JSON, audit logs), runs compileall + the *core* pytest
marker filter as validation (NOT full pytest — optional
web/federation/embeddings tests would error without their extras), then
creates the ZIP. The output filename is derived from the version in
``pyproject.toml`` so it is never stale.

The pytest step uses no --timeout flags so it works with plain pytest
(no pytest-timeout required). A subprocess timeout provides the outer
guard.

Mode flags:
    (default)          Run compile + core tests in the current Python
                       environment. Fails if test deps (pytest) are
                       missing — install with: pip install -e '.[dev,test]'
    --use-current-env  Explicit alias for the default: run tests in the
                       current environment.
    --self-contained   Create a temporary venv, install .[dev,test], run
                       core tests there, then build the ZIP. Best for
                       release use — proves the repo works from clean.
    --no-tests         Skip the pytest gate entirely (compileall only).
    --tests            Force the pytest gate; fail if pytest is absent.

Usage:
    python scripts/build_clean_zip.py
    python scripts/build_clean_zip.py --no-tests
    python scripts/build_clean_zip.py --use-current-env
    python scripts/build_clean_zip.py --self-contained

Output:
    dist/vibe-thinker-v<version>.zip   (e.g. dist/vibe-thinker-v0.4.6a2.zip)

Exit code is nonzero if compileall fails, or if the core pytest gate
fails. If pytest is absent and tests are not skipped (--no-tests), the
build fails with a clear remediation message.
"""

import compileall
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")

# Marker filter identical to scripts/test_local.sh (the broad local gate)
# — only core deps needed. scripts/test_core.sh is now the FAST gate
# (curated subset); this build-time gate keeps the broad marker filter
# for full pre-release coverage. Full pytest is intentionally NOT run
# here: optional-subsystem tests (web/federation/embeddings/sandbox/nli
# /logic/integration) require extra dependencies and would error, not
# skip, when run directly from the staging dir without those extras.
CORE_MARKERS = (
    "not logic and not embeddings and not federation and not web "
    "and not sandbox and not nli and not integration"
)


def get_project_version() -> str:
    """Read the project version from pyproject.toml.

    Uses tomllib (stdlib on py>=3.11) so the ZIP filename always matches
    the declared version — no hardcoded versioned names to go stale.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py>=3.11 has tomllib
        import tomli as tomllib  # type: ignore[no-redef]
    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


VERSION = get_project_version()
ZIP_NAME = f"vibe-thinker-v{VERSION}.zip"

# Directories to include (full source snapshot). Everything else in the
# project root is excluded. This MUST cover all project content so the
# ZIP is a complete, self-contained source release — not a Python-only
# runtime subset.
INCLUDE_DIRS = [
    "verifiers", "sandbox", "tests", "examples", "scripts",
    "web", "profiles", "docs",
    ".github", "docker", "ruvllm_py", "rust",
]

# Non-.py root files shipped verbatim.
_INCLUDE_NONPY_FILES = [
    "README.md", "LICENSE", "pyproject.toml", "AGENTS.md", "CHANGELOG.md",
    "requirements.txt", "requirements-core.txt", "requirements-dev.txt",
    "requirements-embeddings.txt", "requirements-federation.txt",
    "requirements-sandbox.txt", "requirements-models.txt",
    "requirements-legacy-full.txt",
    ".env.example", ".gitignore", ".dockerignore",
    "Dockerfile", "docker-compose.yml",
]


def _discover_root_py_modules() -> list:
    """Auto-discover every top-level ``*.py`` module in the project root.

    The pyproject ``[tool.setuptools] py-modules`` list is the source of
    truth for what gets installed, and a stale hardcoded list here
    previously omitted modules (e.g. ``format_enforcer``), so the
    extracted ZIP could not import the package. Discovering dynamically
    keeps the ZIP a complete source snapshot that can never silently drop
    a module again.
    """
    import glob
    return [
        os.path.basename(p)
        for p in glob.glob(os.path.join(PROJECT_ROOT, "*.py"))
    ]


INCLUDE_FILES = _discover_root_py_modules() + _INCLUDE_NONPY_FILES

# Patterns to EXCLUDE from the staging dir (runtime/build junk).
# - __pycache__, .pyc, .pyo, .pytest_cache: Python runtime caches
# - target, vendor: Rust/Cargo build artifacts (can be multi-GB)
# - dist: built wheels/sdists (the ZIP itself goes there)
# - route_cache.json etc.: runtime cache/audit files
# - .DS_Store: macOS metadata
# - .git (exact): Git metadata — NOT .github (which is CI config we ship)
EXCLUDE_PATTERNS = [
    "__pycache__", ".pyc", ".pyo", ".pytest_cache",
    "target", "vendor", "dist",
    "route_cache.json", "clr_result_cache.json", "clr_trace.json",
    "rfsn_jobs.jsonl", "rfsn_jobs_bitemporal.jsonl",
    "orchestrator_memory.jsonl",
    ".DS_Store",
]
# Directory/file names to exclude ONLY on exact match (not substring).
# This prevents ".git" from matching ".github".
EXCLUDE_EXACT = {".git"}


def should_exclude(name: str) -> bool:
    base = os.path.basename(name)
    if base in EXCLUDE_EXACT:
        return True
    for pat in EXCLUDE_PATTERNS:
        if pat in name:
            return True
    return False


def copy_tree_clean(src: str, dst: str) -> None:
    """Copy a directory tree, excluding junk files."""
    for root, dirs, files in os.walk(src):
        # Filter out excluded dirs in-place
        dirs[:] = [d for d in dirs if not should_exclude(d)]
        for fname in files:
            if should_exclude(fname):
                continue
            rel = os.path.relpath(os.path.join(root, fname), src)
            dst_path = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(os.path.join(root, fname), dst_path)
            # Ensure shell scripts are executable in the staging dir.
            if fname.endswith(".sh"):
                os.chmod(dst_path, 0o755)


def _pytest_available() -> bool:
    """Check whether pytest is importable in the current Python."""
    try:
        import importlib.util
        return importlib.util.find_spec("pytest") is not None
    except Exception:
        return False


def require_test_deps() -> None:
    """Fail with a clear message if pytest is not installed.

    Called before running the pytest gate in current-env mode so the
    error is actionable instead of a silent skip.
    """
    if not _pytest_available():
        raise SystemExit(
            "Missing test dependency: pytest\n"
            "Install with: pip install -e '.[dev,test]'\n"
            "Or run with --no-tests to skip the pytest gate."
        )


def _parse_args() -> tuple:
    """Parse command-line flags.

    Returns (no_tests, force_tests, self_contained).

    --no-tests        : skip the pytest gate (compileall only).
    --tests           : force the pytest gate; fail if pytest is absent.
    --use-current-env : explicit alias for the default (run tests in the
                        current environment).
    --self-contained  : create a temp venv, install .[dev,test], run core
                        tests there, then build the ZIP.
    """
    no_tests = False
    force_tests = False
    self_contained = False
    args = sys.argv[1:]
    for arg in args:
        if arg == "--no-tests":
            no_tests = True
        elif arg == "--tests":
            force_tests = True
        elif arg == "--use-current-env":
            pass  # explicit alias for default; no flag needed
        elif arg == "--self-contained":
            self_contained = True
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            print("Usage: python scripts/build_clean_zip.py "
                  "[--no-tests|--use-current-env|--self-contained]",
                  file=sys.stderr)
            sys.exit(2)
    return no_tests, force_tests, self_contained


def _run_self_contained_tests(staging: str) -> int:
    """Create a temp venv, install .[dev,test], run core tests.

    Returns 0 on success, nonzero on failure. The temp venv is cleaned
    up by the caller.
    """
    venv_dir = tempfile.mkdtemp(prefix="vibe_zip_venv_")
    try:
        venv_python = os.path.join(venv_dir, "bin", "python")
        print("  Creating isolated venv for self-contained test...")
        subprocess.run(
            [sys.executable, "-m", "venv", venv_dir],
            check=True, capture_output=True,
        )
        subprocess.run(
            [venv_python, "-m", "pip", "install", "-q", "--upgrade",
             "pip", "setuptools", "wheel"],
            check=True, capture_output=True,
        )
        print("  Installing .[dev,test] in isolated venv...")
        subprocess.run(
            [venv_python, "-m", "pip", "install", "-q", "-e",
             ".[dev,test]"],
            check=True, capture_output=True, cwd=PROJECT_ROOT,
        )
        print("  Running core pytest gate in isolated venv...")
        try:
            result = subprocess.run(
                [venv_python, "-m", "pytest",
                 staging + "/tests", "-q",
                 "-m", CORE_MARKERS],
                capture_output=True, text=True, cwd=staging,
                timeout=300, check=False,
            )
        except subprocess.TimeoutExpired:
            print("  pytest: TIMED OUT (300s outer limit)",
                  file=sys.stderr)
            return 1
        if result.returncode == 0:
            print("  pytest: OK (self-contained)")
            return 0
        else:
            print("  pytest: FAILED (self-contained)")
            print(result.stdout[-800:])
            print(result.stderr[-800:])
            return 1
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def main() -> int:
    no_tests, force_tests, self_contained = _parse_args()
    staging = tempfile.mkdtemp(prefix="vibe_build_")
    try:
        # Copy included dirs
        for d in INCLUDE_DIRS:
            src = os.path.join(PROJECT_ROOT, d)
            if os.path.isdir(src):
                copy_tree_clean(src, os.path.join(staging, d))

        # Copy included files
        for f in INCLUDE_FILES:
            src = os.path.join(PROJECT_ROOT, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(staging, f))

        # --- Validation: compileall (mandatory) ---
        print("Running compileall...")
        if compileall.compile_dir(staging, quiet=1):
            print("  compileall: OK")
        else:
            print("  compileall: FAILED")
            return 1
        # Clean up __pycache__ created by compileall
        for root, dirs, _ in os.walk(staging):
            for d in dirs:
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)

        # --- Validation: broad core-marker pytest gate ---
        # Run the broad core marker filter (same as scripts/test_local.sh).
        # No --timeout flags are used so this works with plain pytest
        # (pytest-timeout is not required). A subprocess timeout provides
        # the outer guard.
        #
        # Behavior:
        #   --no-tests        : skip the pytest gate entirely.
        #   --self-contained  : temp venv + install .[dev,test] + test.
        #   --tests           : force the gate; fail if pytest is absent.
        #   (default)         : run in current env; fail if pytest absent.
        if no_tests:
            print("Running core pytest gate...")
            print("  pytest: SKIPPED (--no-tests)")
        elif self_contained:
            print("Running core pytest gate (self-contained)...")
            rc = _run_self_contained_tests(staging)
            if rc != 0:
                return rc
        elif force_tests and not _pytest_available():
            print("Running core pytest gate...")
            print("  pytest: FAILED (--tests requested but pytest not "
                  "installed)")
            return 1
        elif not _pytest_available():
            # Default/current-env mode: fail if deps are missing instead
            # of silently skipping — the operator should know.
            print("Running core pytest gate...")
            require_test_deps()  # raises SystemExit with remediation
        else:
            print("Running core pytest gate...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest",
                     staging + "/tests", "-q",
                     "-m", CORE_MARKERS],
                    capture_output=True, text=True, cwd=staging,
                    timeout=300, check=False,
                )
            except subprocess.TimeoutExpired:
                print("  pytest: TIMED OUT (300s outer limit)",
                      file=sys.stderr)
                return 1
            if result.returncode == 0:
                print("  pytest: OK")
            else:
                print("  pytest: FAILED")
                print(result.stdout[-800:])
                print(result.stderr[-800:])
                return 1

        # --- Create ZIP ---
        # Wrap all files under a single top-level directory named
        # ``vibe-thinker-v<version>/`` so extraction produces one
        # self-contained repo dir (the standard release-ZIP convention).
        # test_zip_release.sh locates that dir with
        # ``find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d | head -n 1``.
        top_dir = f"vibe-thinker-v{VERSION}"
        os.makedirs(DIST_DIR, exist_ok=True)
        zip_path = os.path.join(DIST_DIR, ZIP_NAME)
        if os.path.exists(zip_path):
            os.unlink(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(staging):
                dirs[:] = [d for d in dirs if not should_exclude(d)]
                for fname in files:
                    if should_exclude(fname):
                        continue
                    fpath = os.path.join(root, fname)
                    arcname = os.path.join(
                        top_dir, os.path.relpath(fpath, staging))
                    # zf.write() builds the ZipInfo via
                    # ZipInfo.from_file internally, which copies the
                    # file's mode into external_attr. copy_tree_clean
                    # already chmods .sh files to 0o755 in staging.
                    zf.write(fpath, arcname)

        print(f"\nClean ZIP created: {zip_path}")
        print(f"  Size: {os.path.getsize(zip_path)} bytes")

        # Verify no junk in the ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            bad = [n for n in names
                   if "__pycache__" in n or n.endswith(".pyc")]
            if bad:
                print(f"  ERROR: ZIP contains junk files: {bad}")
                return 1
            print("  Verified: no __pycache__ or .pyc files in ZIP")
            print(f"  Entries: {len(names)}")

            # Verify every .sh file in the ZIP has the executable bit
            # set in external_attr. ZIP extraction tools that honor
            # external_attr will then preserve the +x permission; tools
            # that ignore it should be caught by test_zip_release.sh.
            non_exec_sh = []
            for info in zf.infolist():
                if not info.filename.endswith(".sh"):
                    continue
                # external_attr is the high 16 bits of the UNIX st_mode.
                # 0o755 -> 0o755 << 16 = 0o7550000.
                mode = (info.external_attr >> 16) & 0o7777
                if mode & 0o100 == 0:  # owner-exec bit not set
                    non_exec_sh.append((info.filename, oct(mode)))
            if non_exec_sh:
                print("  ERROR: .sh files missing executable bit in ZIP:")
                for name, mode in non_exec_sh:
                    print(f"    {name} (mode={mode})")
                return 1
            sh_count = sum(1 for n in names if n.endswith(".sh"))
            print(f"  Verified: all {sh_count} .sh files have +x bit in ZIP")

        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
