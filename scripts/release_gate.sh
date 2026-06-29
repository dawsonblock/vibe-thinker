#!/usr/bin/env bash
# Release gate — must pass before any release.
#
# Usage:
#   ./scripts/release_gate.sh              # run all phases
#   ./scripts/release_gate.sh build        # phase 1: build wheel/sdist
#   ./scripts/release_gate.sh install-smoke # phase 2: clean wheel install + CLI
#   ./scripts/release_gate.sh core          # phase 3: dev/test core gate
#   ./scripts/release_gate.sh all           # run all phases (default)
#
# Uses fully isolated venvs for every phase so the gate never depends on
# ambient/system/user Python state. pip output is quieted to make
# failures easier to isolate (use RELEASE_GATE_VERBOSE=1 for full pip
# logs).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHASE="${1:-all}"
VERBOSE="${RELEASE_GATE_VERBOSE:-0}"
PIP_FLAGS="--quiet"
if [ "$VERBOSE" = "1" ]; then
    PIP_FLAGS=""
fi

# ---- shared helpers ----
_make_venv() {
    local vdir
    vdir="$(mktemp -d)"
    python3 -m venv "$vdir/venv"
    echo "$vdir/venv"
}

_clean_artifacts() {
    rm -rf build dist *.egg-info vibe_thinker.egg-info .pytest_cache
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
    find . -name "*.pyc" -delete
}

# ---- phase 1: build ----
phase_build() {
    echo "[release-gate] Cleaning generated artifacts"
    _clean_artifacts
    echo "[release-gate] Compile check"
    # Exclude rust/ vendor dir (contains third-party Python scripts with
    # SyntaxWarnings that are not our code).
    python3 -m compileall -q -x "rust/" .
    echo "[release-gate] Build wheel/sdist (isolated build venv)"
    local venv
    venv="$(_make_venv)"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    python -m pip install --upgrade pip setuptools wheel build $PIP_FLAGS
    python -m build
    deactivate
    rm -rf "$(dirname "$venv")"
    echo "[release-gate] phase build: OK"
}

# ---- phase 2: clean wheel install + CLI smoke ----
phase_install_smoke() {
    if [ ! -d dist ] || ! ls dist/*.whl >/dev/null 2>&1; then
        echo "[release-gate] No wheel in dist/ — run 'build' phase first."
        exit 1
    fi
    echo "[release-gate] Clean installed wheel smoke"
    local venv
    venv="$(_make_venv)"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    python -m pip install --upgrade pip $PIP_FLAGS
    python -m pip install dist/*.whl $PIP_FLAGS
    vibe-thinker --help > /dev/null
    echo "[release-gate]   --help: OK"
    vibe-thinker --version
    echo "[release-gate]   --version: OK"
    vibe-thinker doctor > /dev/null
    echo "[release-gate]   doctor: OK"
    vibe-thinker smoke
    deactivate
    rm -rf "$(dirname "$venv")"
    echo "[release-gate] phase install-smoke: OK"
}

# ---- phase 3: dev/test core gate ----
phase_core() {
    echo "[release-gate] Isolated dev/test core gate"
    local venv
    venv="$(_make_venv)"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    python -m pip install --upgrade pip setuptools wheel build $PIP_FLAGS
    python -m pip install -e ".[dev,test]" $PIP_FLAGS
    # Use 'bash' prefix so this works even if the ZIP extraction lost the
    # executable bit (some extraction tools / GitHub ZIP downloads do this).
    bash scripts/test_core.sh
    deactivate
    rm -rf "$(dirname "$venv")"
    echo "[release-gate] phase core: OK"
}

# ---- dispatch ----
case "$PHASE" in
    build)
        phase_build
        ;;
    install-smoke)
        phase_install_smoke
        ;;
    core)
        phase_core
        ;;
    all)
        phase_build
        phase_install_smoke
        phase_core
        echo ""
        echo "[release-gate] PASS"
        ;;
    *)
        echo "Usage: $0 {build|install-smoke|core|all}"
        exit 1
        ;;
esac
