#!/usr/bin/env python3
"""Build a clean distribution ZIP for vibe-thinker.

Copies only source/docs/tests/config files into a staging directory,
excludes all runtime junk (__pycache__, .pyc, cache JSON, audit logs),
runs compileall + pytest as validation, then creates the ZIP.

Usage:
    python scripts/build_clean_zip.py

Output:
    dist/vibe-thinker-v0.3.zip

Exit code is nonzero if compileall or pytest fails.
"""

import compileall
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "0.3.1"
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
ZIP_NAME = f"vibe-thinker-v{VERSION}.zip"

# Files/dirs to include (everything else is excluded)
INCLUDE_DIRS = ["verifiers", "sandbox", "tests", "examples", "scripts"]
INCLUDE_FILES = [
    "vibe_clr.py", "vibe_clr_async.py", "hybrid_orchestrator.py",
    "persistent_cache.py", "rfsn_job_queue.py", "bitemporal_log.py",
    "rfsn_cli.py", "scoring.py", "math_solver.py", "demo.py", "test_demo.py",
    "test_full_stack.py", "test_clr.py",
    "README.md", "LICENSE", "pyproject.toml", "requirements.txt",
    ".env.example", ".gitignore",
]
# Also include tests/__init__.py, verifiers/__init__.py etc. (handled by dir copy)

# Patterns to EXCLUDE from the staging dir (runtime junk)
EXCLUDE_PATTERNS = [
    "__pycache__", ".pyc", ".pyo", ".pytest_cache",
    "route_cache.json", "clr_result_cache.json", "clr_trace.json",
    "rfsn_jobs.jsonl", "rfsn_jobs_bitemporal.jsonl", "orchestrator_memory.jsonl",
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

        # --- Validation: pytest ---
        # Use --timeout=60 for per-test timeout (requires pytest-timeout)
        # and subprocess timeout=120 as a hard outer limit.
        print("Running pytest...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", staging + "/tests", "-q",
                 "--timeout=60", "--timeout-method=thread"],
                capture_output=True, text=True, cwd=staging,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print("  pytest: TIMED OUT (120s outer limit)", file=sys.stderr)
            return 1
        if result.returncode == 0:
            print("  pytest: OK")
        else:
            print("  pytest: FAILED")
            print(result.stdout[-500:])
            print(result.stderr[-500:])
            return 1

        # --- Create ZIP ---
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
                    arcname = os.path.relpath(fpath, staging)
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
            print(f"  Verified: no __pycache__ or .pyc files in ZIP")
            print(f"  Entries: {len(zf.namelist())}")

        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
