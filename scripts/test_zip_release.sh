#!/usr/bin/env bash
# ZIP release self-test — verifies a release ZIP is clean and shippable.
# Usage: ./scripts/test_zip_release.sh path/to/release.zip
set -euo pipefail

ZIP_PATH="${1:?usage: scripts/test_zip_release.sh path/to/release.zip}"
WORKDIR="$(mktemp -d)"

echo "Testing ZIP in $WORKDIR"
unzip -q "$ZIP_PATH" -d "$WORKDIR"
REPO_DIR="$(find "$WORKDIR" -maxdepth 1 -type d | tail -n 1)"
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
echo "=== Creating venv and installing ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e ".[dev,test]"

echo "=== Compile check ==="
python -m compileall -q .

echo "=== Build wheel ==="
python -m build

echo "=== CLI help ==="
vibe-thinker --help

echo "=== Doctor ==="
vibe-thinker doctor

echo "=== Smoke ==="
vibe-thinker smoke

echo "=== Core tests ==="
./scripts/test_core.sh

deactivate
rm -rf "$WORKDIR"
echo ""
echo "ZIP release test PASSED."
