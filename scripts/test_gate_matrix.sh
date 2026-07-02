#!/usr/bin/env bash
# test_gate_matrix.sh — run the full release gate matrix and capture a report.
#
# This is the single command the build audit asked for under fix #5:
# "Run and capture the full gate matrix on the real target Mac
# environment." It runs every release-gate profile listed in AGENTS.md,
# records PASS/FAIL/SKIP for each, prints a summary table, and writes a
# machine-readable log to dist/gate_matrix_<timestamp>.log.
#
# Gates that require unavailable prerequisites (Docker, Redis, Rust,
# optional Python extras) are reported as SKIP with the missing prereq
# named — they do NOT fail the matrix. A FAIL (gate ran and returned
# nonzero) does fail the matrix.
#
# Usage:
#   ./scripts/test_gate_matrix.sh                # run all gates
#   ./scripts/test_gate_matrix.sh --keep-going   # run all, report all, don't abort early
#   GATE_MATRIX_LOG=path.log ./scripts/test_gate_matrix.sh
set -uo pipefail
# NOTE: do NOT set -e here — we want to capture each gate's exit code and
# continue, not abort on the first failure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

KEEP_GOING=false
if [[ "${1:-}" == "--keep-going" ]]; then
    KEEP_GOING=true
fi

TS="$(date +%Y%m%d-%H%M%S)"
LOG="${GATE_MATRIX_LOG:-dist/gate_matrix_${TS}.log}"
mkdir -p "$(dirname "$LOG")"

# gate_name | run_command | prerequisite_check_command
# A gate is SKIP if its prereq check exits nonzero (missing tool/deps).
GATES=(
    "core|bash scripts/test_core.sh|python3 -c 'import pytest, aiohttp' 2>/dev/null"
    "local|bash scripts/test_local.sh|python3 -c 'import pytest, aiohttp' 2>/dev/null"
    "release|bash scripts/release_gate.sh|python3 -c 'import build' 2>/dev/null"
    "zip-release|bash scripts/release_zip.sh|python3 -c 'import pytest, aiohttp' 2>/dev/null"
    "compose|bash scripts/test_compose.sh|command -v docker >/dev/null 2>&1"
    "docker|bash scripts/test_docker.sh|command -v docker >/dev/null 2>&1"
    "embeddings|bash scripts/test_embeddings.sh|python3 -c 'import sentence_transformers' 2>/dev/null"
    "federation|bash scripts/test_federation.sh|python3 -c 'import fastapi, httpx' 2>/dev/null"
    "ruvllm|bash scripts/check_ruvllm.sh|command -v cargo >/dev/null 2>&1"
)

declare -a NAMES RESULTS REASONS
IDX=0
FAILS=0
SKIPS=0
PASSES=0

run_gate() {
    local name="$1" cmd="$2" prereq="$3"
    NAMES[$IDX]="$name"
    echo "=== gate: $name ==="
    if ! eval "$prereq" >/dev/null 2>&1; then
        RESULTS[$IDX]="SKIP"
        REASONS[$IDX]="missing prerequisite"
        SKIPS=$((SKIPS+1))
        echo "  SKIP (missing prerequisite)"
        IDX=$((IDX+1))
        return 0
    fi
    local gate_log
    gate_log="$(mktemp)"
    set +e
    eval "$cmd" >"$gate_log" 2>&1
    local rc=$?
    set -e  # harmless; -e is not globally set anyway
    tail -n 20 "$gate_log"
    cp "$gate_log" "${LOG%.log}_${name}.log"
    rm -f "$gate_log"
    if [ "$rc" -eq 0 ]; then
        RESULTS[$IDX]="PASS"
        REASONS[$IDX]=""
        PASSES=$((PASSES+1))
        echo "  PASS"
    else
        RESULTS[$IDX]="FAIL"
        REASONS[$IDX]="exit $rc (see ${LOG%.log}_${name}.log)"
        FAILS=$((FAILS+1))
        echo "  FAIL (exit $rc)"
        if ! $KEEP_GOING; then
            IDX=$((IDX+1))
            return 1
        fi
    fi
    IDX=$((IDX+1))
    return 0
}

{
    echo "vibe-thinker gate matrix — $TS"
    echo "host: $(uname -srm)"
    echo "python: $(python3 --version 2>&1)"
    echo
} | tee "$LOG"

for g in "${GATES[@]}"; do
    IFS='|' read -r name cmd prereq <<<"$g"
    if ! run_gate "$name" "$cmd" "$prereq"; then
        # abort-early path: a gate failed and --keep-going not set
        break
    fi
done

echo ""
echo "================ GATE MATRIX SUMMARY ================"
{
    echo "================ GATE MATRIX SUMMARY ================"
    printf "%-14s %-6s %s\n" "GATE" "RESULT" "DETAIL"
    for i in $(seq 0 $((IDX-1))); do
        printf "%-14s %-6s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}" "${REASONS[$i]}"
    done
    echo "----------------------------------------------------"
    printf "pass=%d fail=%d skip=%d (total gates=%d)\n" "$PASSES" "$FAILS" "$SKIPS" "$IDX"
} | tee -a "$LOG"

echo ""
echo "Full log: $LOG"
echo "Per-gate logs: ${LOG%.log}_<gate>.log"

if [ "$FAILS" -gt 0 ]; then
    exit 1
fi
exit 0
