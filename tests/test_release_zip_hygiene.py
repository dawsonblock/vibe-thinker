"""Release-ZIP and proof-bundle hygiene guard.

Two artifact types are produced by ``scripts/build_clean_zip.py``:

  - **Release ZIP** (``vibe-thinker-v<version>.zip``): clean source
    release. Must NOT contain ``gate_results/`` (proof logs are not
    source). Must not contain any junk (``__pycache__``, ``.pyc``,
    ``.venv``, ``.DS_Store``, editor folders, etc.).

  - **Proof bundle ZIP** (``vibe-thinker-v<version>-proof-bundle.zip``):
    evidence archive. MUST contain ``gate_results/``. Must not contain
    any junk (same junk list as the release ZIP).

This test enforces the freeze plan's hygiene rules at the pytest level so
the "no junk in release archives" and "proof bundle has proof logs" rules
are checked inside the test suite, not only by manual shell commands.

The shell-level guard lives in ``scripts/test_zip_release.sh`` (rejects
``__pycache__``/``.pytest_cache``/``*.pyc``/``*.egg-info``/``build`` in an
extracted archive). This test mirrors and extends that guard.
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


def _all_zips():
    """All vibe-thinker ZIPs in dist/ (release + proof bundles)."""
    if not os.path.isdir(_DIST_DIR):
        return []
    return sorted(
        os.path.join(_DIST_DIR, n)
        for n in os.listdir(_DIST_DIR)
        if n.startswith("vibe-thinker-v") and n.endswith(".zip")
    )


def _release_zips():
    """Release ZIPs only (exclude proof-bundle ZIPs)."""
    return [p for p in _all_zips() if "-proof-bundle" not in os.path.basename(p)]


def _proof_bundles():
    """Proof-bundle ZIPs only."""
    return [p for p in _all_zips() if "-proof-bundle" in os.path.basename(p)]


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


def _has_entry(zf: zipfile.ZipFile, suffix: str) -> bool:
    """True if the ZIP contains any entry ending with ``suffix``."""
    return any(n.endswith(suffix) for n in zf.namelist())


def _has_dir(zf: zipfile.ZipFile, dirname: str) -> bool:
    """True if the ZIP contains any entry under ``dirname/``."""
    return any(
        f"/{dirname}/" in n + "/" or n.startswith(dirname + "/")
        for n in zf.namelist()
    )


# ---------------------------------------------------------------------------
# Junk checks — apply to BOTH release ZIPs and proof bundles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_has_no_junk(zip_path):
    """No ZIP (release or proof) may contain junk files."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        junk = [n for n in z.namelist() if _is_junk(n)]
    assert not junk, (
        f"{os.path.basename(zip_path)} contains junk entries that the "
        f"clean builder should have excluded: {junk}. Rebuild with "
        f"./scripts/release_zip.sh (build_clean_zip.py --release)."
    )


@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_scripts_are_executable(zip_path):
    """Every shipped .sh must have the +x bit set in external_attr.

    Mirrors the check in scripts/build_clean_zip.py and test_zip_release.sh
    so extraction tools that honor external_attr produce runnable scripts.
    """
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
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


# ---------------------------------------------------------------------------
# Content checks — both ZIPs must contain essential source files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_contains_pyproject(zip_path):
    """Both release and proof ZIPs must contain pyproject.toml."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert _has_entry(z, "pyproject.toml"), (
            f"{os.path.basename(zip_path)} is missing pyproject.toml"
        )


@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_contains_demo(zip_path):
    """Both release and proof ZIPs must contain demo_verified_swarm.py."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert _has_entry(z, "demo_verified_swarm.py"), (
            f"{os.path.basename(zip_path)} is missing demo_verified_swarm.py"
        )


@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_contains_readme(zip_path):
    """Both release and proof ZIPs must contain README.md."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert _has_entry(z, "README.md"), (
            f"{os.path.basename(zip_path)} is missing README.md"
        )


@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_contains_agents_md(zip_path):
    """Both release and proof ZIPs must contain AGENTS.md."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert _has_entry(z, "AGENTS.md"), (
            f"{os.path.basename(zip_path)} is missing AGENTS.md"
        )


@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_contains_changelog(zip_path):
    """Both release and proof ZIPs must contain CHANGELOG.md."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert _has_entry(z, "CHANGELOG.md"), (
            f"{os.path.basename(zip_path)} is missing CHANGELOG.md"
        )


# ---------------------------------------------------------------------------
# Mode-specific checks — release excludes gate_results, proof includes it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zip_path", _release_zips())
def test_release_zip_excludes_gate_results(zip_path):
    """The release ZIP must NOT contain gate_results/ (proof logs are
    not source)."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no release ZIP at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert not _has_dir(z, "gate_results"), (
            f"{os.path.basename(zip_path)} contains gate_results/ — "
            f"the release ZIP must exclude proof logs. Rebuild with "
            f"./scripts/release_zip.sh (build_clean_zip.py --release)."
        )


@pytest.mark.parametrize("zip_path", _proof_bundles())
def test_proof_bundle_contains_gate_results(zip_path):
    """The proof bundle ZIP MUST contain gate_results/."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no proof bundle at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        assert _has_dir(z, "gate_results"), (
            f"{os.path.basename(zip_path)} is missing gate_results/ — "
            f"the proof bundle must include proof logs. Rebuild with "
            f"./scripts/release_zip.sh --proof-bundle."
        )


# ---------------------------------------------------------------------------
# Version consistency — the ZIP filename version must match pyproject.toml
# ---------------------------------------------------------------------------

def _get_pyproject_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(os.path.join(_PROJECT_ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


@pytest.mark.parametrize("zip_path", _all_zips())
def test_zip_version_matches_pyproject(zip_path):
    """The version embedded in the ZIP filename must match the version
    declared in pyproject.toml so the archive name is never stale."""
    if not os.path.exists(zip_path):
        pytest.skip(f"no ZIP at {zip_path}")
    basename = os.path.basename(zip_path)
    # Extract version from filename: vibe-thinker-v0.4.6a9.zip or
    # vibe-thinker-v0.4.6a9-proof-bundle.zip
    if basename.startswith("vibe-thinker-v") and basename.endswith(".zip"):
        middle = basename[len("vibe-thinker-v"):-len(".zip")]
        version = middle.replace("-proof-bundle", "")
    else:
        pytest.skip(f"unrecognized ZIP name format: {basename}")
    expected = _get_pyproject_version()
    assert version == expected, (
        f"{basename} embeds version {version} but pyproject.toml "
        f"declares {expected}. Rebuild to sync the filename."
    )
