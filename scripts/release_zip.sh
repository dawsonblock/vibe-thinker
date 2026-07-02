#!/usr/bin/env bash
# release_zip.sh — the ONE correct way to build a shippable source ZIP.
#
# Chains the two release-quality commands together so a release archive is
# never produced except through the clean builder with the self-contained
# gate, and is never shipped without passing the ZIP self-test:
#
#   1. python scripts/build_clean_zip.py --release
#      (temp venv + install .[dev,test] + core pytest gate + clean ZIP)
#   2. ./scripts/test_zip_release.sh dist/vibe-thinker-v<version>.zip
#      (extract, check +x bits, reject __pycache__/.pyc junk, fresh venv
#       install, doctor, smoke, test_core.sh)
#
# This is the command the build audit asked for: "Build the ZIP only
# through scripts/build_clean_zip.py --release." Running this
# wrapper is the only supported path; invoking build_clean_zip.py
# directly with --no-tests or without --release produces an
# archive that has NOT been proven to work from a clean install.
#
# Usage:
#   ./scripts/release_zip.sh                # build release ZIP + self-test
#   ./scripts/release_zip.sh --proof-bundle # build proof bundle ZIP (no self-test)
#   ./scripts/release_zip.sh --skip-test    # build only (NOT for release)
#   RELEASE_ZIP_VERBOSE=1 ./scripts/release_zip.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SKIP_TEST=false
BUILD_MODE="--release"
if [[ "${1:-}" == "--skip-test" ]]; then
    SKIP_TEST=true
elif [[ "${1:-}" == "--proof-bundle" ]]; then
    BUILD_MODE="--proof-bundle"
    SKIP_TEST=true  # proof bundle is not a clean-install target; no self-test
fi

echo "[release-zip] Building ZIP (mode: $BUILD_MODE) ..."
# --release / --proof-bundle: temp venv, install .[dev,test], run core
# pytest gate, then build the ZIP. Fails the whole wrapper if the gate
# fails — so a broken tree can never produce a release archive.
python3 scripts/build_clean_zip.py "$BUILD_MODE"

# Resolve the versioned ZIP name (matches pyproject.toml version).
VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
if [ "$BUILD_MODE" = "--proof-bundle" ]; then
    ZIP_PATH="dist/vibe-thinker-v${VERSION}-proof-bundle.zip"
else
    ZIP_PATH="dist/vibe-thinker-v${VERSION}.zip"
fi

if [ ! -f "$ZIP_PATH" ]; then
    echo "[release-zip] ERROR: expected $ZIP_PATH not found" >&2
    exit 1
fi

if $SKIP_TEST; then
    echo "[release-zip] --skip-test or --proof-bundle; NOT running ZIP self-test."
    echo "[release-zip] Archive left at $ZIP_PATH"
    exit 0
fi

echo "[release-zip] Running ZIP release self-test on $ZIP_PATH ..."
bash scripts/test_zip_release.sh "$ZIP_PATH"

echo ""
echo "[release-zip] PASS — release archive ready: $ZIP_PATH"
