"""Release-ZIP hygiene guard — automated pytest check for fix #2.

The shell-level guard lives in ``scripts/test_zip_release.sh`` (rejects
``__pycache__``/``.pytest_cache``/``*.pyc``/``*.egg-info``/``build`` in an
extracted archive). This test mirrors that guard at the pytest level so
the "no __pycache__ / .pyc artifacts in release archives" rule is
enforced inside the test suite too, not only by a manual shell command.

It scans every ``dist/vibe-thinker-v*.zip`` present in the project's
``dist/`` directory. If no release ZIP exists, the test is skipped (the
guard only applies once an archive has been built). When a ZIP exists,
every entry is checked against the same junk patterns the clean builder
excludes and the ZIP self-test rejects.

This is the automated guard the build audit asked for under fix #2:
"Remove all __pycache__ / .pyc artifacts from release archives."
"""

import os
import zipfile

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIST_DIR = os.path.join(_PROJECT_ROOT, "dist")

# Must match EXCLUDE_PATTERNS in scripts/build_clean_zip.py and the junk
# check in scripts/test_zip_release.sh. Keep these in sync.
_JUNK_TOKENS = (
    "__pycache__", ".pyc", ".pyo", ".pytest_cache",
    ".egg-info", "/build/", "\\build\\",
    # Isolated venvs created by the gate scripts / self-contained builds.
    # A release archive must never carry a virtualenv.
    ".venv",
    # Tool caches that bloat archives and are never source.
    ".mypy_cache", ".ruff_cache",
)
# .DS_Store is macOS metadata, not Python junk, but the clean builder
# excludes it too — flag it so a release archive never carries it.
_EXTRA_JUNK = (".DS_Store",)
# Hidden directories that are never legitimate release content. The clean
# builder intentionally ships .github (CI config), .env.example,
# .gitignore, and .dockerignore — those are NOT junk and are excluded
# from this list. Anything else starting with a dot and looking like an
# editor/tool private dir is rejected.
_JUNK_HIDDEN_DIRS = (".idea", ".vscode", ".git/")


def _release_zips():
    if not os.path.isdir(_DIST_DIR):
        return []
    return sorted(
        os.path.join(_DIST_DIR, n)
        for n in os.listdir(_DIST_DIR)
        if n.startswith("vibe-thinker-v") and n.endswith(".zip")
    )


def _is_junk(name: str) -> bool:
    # Normalize separators so /build/ matches both Unix and Windows paths.
    norm = name.replace("\\", "/")
    for tok in _JUNK_TOKENS:
        if tok in norm:
            return True
    base = os.path.basename(name)
    if base in _EXTRA_JUNK:
        return True
    # Reject hidden junk directories as exact path components — but NOT
    # .github (legitimate CI config we ship) which would be falsely matched
    # by a naive ".git" substring test.
    segments = [s for s in norm.split("/") if s]
    for hidden in _JUNK_HIDDEN_DIRS:
        if hidden.rstrip("/") in segments:
            return True
    return False


@pytest.mark.parametrize("zip_path", _release_zips())
def test_release_zip_has_no_junk(zip_path):
    if not os.path.exists(zip_path):
        pytest.skip(f"no release ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        junk = [n for n in z.namelist() if _is_junk(n)]
    assert not junk, (
        f"{os.path.basename(zip_path)} contains junk entries that the "
        f"clean builder should have excluded: {junk}. Rebuild with "
        f"./scripts/release_zip.sh (build_clean_zip.py --self-contained)."
    )


@pytest.mark.parametrize("zip_path", _release_zips())
def test_release_zip_scripts_are_executable(zip_path):
    """Every shipped .sh must have the +x bit set in external_attr.

    Mirrors the check in scripts/build_clean_zip.py and test_zip_release.sh
    so extraction tools that honor external_attr produce runnable scripts.
    """
    if not os.path.exists(zip_path):
        pytest.skip(f"no release ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        non_exec = [
            n for n in z.namelist()
            if n.endswith(".sh")
            and not (z.getinfo(n).external_attr >> 16 & 0o111)
        ]
    assert not non_exec, (
        f"{os.path.basename(zip_path)} ships .sh scripts without the +x "
        f"bit: {non_exec}."
    )
