#!/usr/bin/env python3
"""Build a clean distribution ZIP for vibe-thinker.

Copies only source/docs/tests/config files into a staging directory,
excludes all runtime junk (__pycache__, .pyc, cache JSON, audit logs),
runs compileall + the *core* pytest marker filter as validation (NOT
full pytest — optional web/federation/embeddings tests would error
without their extras), then creates the ZIP. The output filename is
derived from the version in ``pyproject.toml`` so it is never stale.

Usage:
    python scripts/build_clean_zip.py

Output:
    dist/vibe-thinker-v<version>.zip   (e.g. dist/vibe-thinker-v0.4.3a0.zip)

Exit code is nonzero if compileall or the core pytest gate fails.
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

# Marker filter identical to scripts/test_core.sh — only core deps needed.
# Full pytest is intentionally NOT run here: optional-subsystem tests
# (web/federation/embeddings/sandbox/nli/logic/integration) require extra
# dependencies and would error, not skip, when run directly from the
# staging dir without those extras installed.
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

# Files/dirs to include (everything else is excluded)
INCLUDE_DIRS = [
    "verifiers", "sandbox", "tests", "examples", "scripts",
    "web", "profiles", "docs",
]

# Non-.py root files shipped verbatim.
_INCLUDE_NONPY_FILES = [
    "README.md", "LICENSE", "pyproject.toml", "AGENTS.md", "CHANGELOG.md",
    "requirements.txt", "requirements-core.txt", "requirements-dev.txt",
    "requirements-embeddings.txt", "requirements-federation.txt",
    "requirements-sandbox.txt", "requirements-models.txt",
    "requirements-legacy-full.txt",
    ".env.example", ".gitignore",
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
# Also include tests/__init__.py, verifiers/__init__.py etc.
# (handled by dir copy)

# Patterns to EXCLUDE from the staging dir (runtime junk)
EXCLUDE_PATTERNS = [
    "__pycache__", ".pyc", ".pyo", ".pytest_cache",
    "route_cache.json", "clr_result_cache.json", "clr_trace.json",
    "rfsn_jobs.jsonl", "rfsn_jobs_bitemporal.jsonl",
    "orchestrator_memory.jsonl",
    ".DS_Store", ".git",
]


def should_exclude(name: str) -> bool:
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
            # shutil.copy2 preserves source permissions, but if the
            # source lost its +x bit (e.g. via git on a filesystem that
            # doesn't track exec bits), force it here so the ZIP
            # preserves it.
            if fname.endswith(".sh"):
                os.chmod(dst_path, 0o755)


def main() -> int:
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

        # --- Validation: compileall ---
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

        # --- Validation: core pytest gate ---
        # Run ONLY the core marker filter (same as scripts/test_core.sh).
        # Full pytest is not run: optional-subsystem tests need extra deps
        # that are not installed in the staging dir and would error.
        print("Running core pytest gate...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", staging + "/tests", "-q",
                 "--timeout=60", "--timeout-method=thread",
                 "-m", CORE_MARKERS],
                capture_output=True, text=True, cwd=staging,
                timeout=180, check=False,
            )
        except subprocess.TimeoutExpired:
            print("  pytest: TIMED OUT (180s outer limit)", file=sys.stderr)
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
                    # Preserve Unix file permissions so executable
                    # scripts (e.g. scripts/*.sh) remain executable after
                    # extraction. copy_tree_clean already chmods .sh files
                    # to 0o755 in staging; zf.write() builds the ZipInfo
                    # via ZipInfo.from_file internally, which copies the
                    # file's mode into external_attr. (The previous code
                    # built a ZipInfo manually and then passed it to
                    # zf.write() as the *filename* argument, which raised
                    # TypeError at runtime.)
                    zf.write(fpath, arcname)

        print(f"\nClean ZIP created: {zip_path}")
        print(f"  Size: {os.path.getsize(zip_path)} bytes")

        # Verify no junk in the ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = [n for n in zf.namelist()
                   if "__pycache__" in n or n.endswith(".pyc")]
            if bad:
                print(f"  ERROR: ZIP contains junk files: {bad}")
                return 1
            print("  Verified: no __pycache__ or .pyc files in ZIP")
            print(f"  Entries: {len(zf.namelist())}")

        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
