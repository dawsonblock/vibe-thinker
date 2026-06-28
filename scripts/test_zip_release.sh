#!/usr/bin/env bash
# ZIP release self-test — verifies a release ZIP is clean and shippable.
# Usage: ./scripts/test_zip_release.sh path/to/release.zip
#
# Pip and build output is quieted so failures are easy to isolate. Set
# ZIP_TEST_VERBOSE=1 for full pip/build logs.
set -euo pipefail

ZIP_PATH="${1:?usage: scripts/test_zip_release.sh path/to/release.zip}"
VERBOSE="${ZIP_TEST_VERBOSE:-0}"
PIP_FLAGS="--quiet"
BUILD_FLAGS=""
if [ "$VERBOSE" = "1" ]; then
    PIP_FLAGS=""
    BUILD_FLAGS=""
fi

WORKDIR="$(mktemp -d)"

echo "Testing ZIP in $WORKDIR"
unzip -q "$ZIP_PATH" -d "$WORKDIR"
REPO_DIR="$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
cd "$REPO_DIR"

# --- Script permissions ---
echo "=== Checking script permissions ==="
test -x scripts/test_core.sh
test -x scripts/release_gate.sh
test -x scripts/mac_setup.sh
test -x scripts/mac_smoke.sh
echo "All scripts executable: OK"

# --- Hygiene: no generated junk ---
echo "=== Checking for generated junk ==="
if find . \( -name "__pycache__" -o -name ".pytest_cache" -o -name "*.pyc" \
    -o -name "*.egg-info" -o -name "build" \) | grep .; then
  echo "Release ZIP contains generated junk"
  exit 1
fi
echo "No generated junk: OK"

# --- Install + test ---
echo "=== Creating venv and installing (quiet) ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel build $PIP_FLAGS
python -m pip install -e ".[dev,test]" $PIP_FLAGS

echo "=== Compile check ==="
python -m compileall -q .

echo "=== Build wheel (quiet) ==="
python -m build $BUILD_FLAGS > /dev/null

echo "=== CLI help ==="
vibe-thinker --help > /dev/null && echo "  --help: OK"

echo "=== Version ==="
vibe-thinker --version

echo "=== Doctor ==="
vibe-thinker doctor > /dev/null && echo "  doctor: OK"

echo "=== Smoke ==="
vibe-thinker smoke

echo "=== Core tests ==="
python -m pytest --strict-markers -q \
  -m "not logic and not embeddings and not federation and not web and not sandbox and not nli and not integration"

deactivate
rm -rf "$WORKDIR"
echo ""
echo "ZIP release test PASSED."
