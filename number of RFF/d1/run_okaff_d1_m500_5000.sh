#!/usr/bin/env bash
# Run the d=1 OKAFF mean-shift experiment: ARL -> EDD -> figure.
# Save this file beside the three corresponding Python scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-python3}"
N_RUNS="${N_RUNS:-200}"

case "${1:-}" in
    "")        FULL_GRID=false ;;
    --full-grid) FULL_GRID=true ;;
    -h|--help)
        printf 'Usage: bash %s [--full-grid]\n' "${0##*/}"
        printf 'Default: three thresholds (4.5, 7, 9); --full-grid: all 16 thresholds.\n'
        printf 'Optional environment variables: N_RUNS=200, PYTHON_BIN=python3.\n'
        exit 0
        ;;
    *)
        printf 'Unknown argument: %s\n' "$1" >&2
        exit 2
        ;;
esac
if (( $# > 1 )); then
    printf 'Expected at most one option: --full-grid\n' >&2
    exit 2
fi
if [[ ! "$N_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'N_RUNS must be a positive integer (got: %s)\n' "$N_RUNS" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

for script in \
    OKAFF_ARL_d1_m500_5000.py \
    OKAFF_EDD_d1_m500_5000.py \
    compare_okaff_mean_minus3_d1.py; do
    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        printf 'Missing Python script: %s\n' "$SCRIPT_DIR/$script" >&2
        exit 1
    fi
done

# Both experiments must use exactly the same features, run count, and grid.
COMMON_ARGS=(--features 500 1000 5000 --n-runs "$N_RUNS")
if "$FULL_GRID"; then
    COMMON_ARGS+=(--full-grid)
    printf 'Using all 16 thresholds; m=500, 1000, 5000; %s runs.\n' "$N_RUNS"
else
    printf 'Using thresholds 4.5, 7, 9; m=500, 1000, 5000; %s runs.\n' "$N_RUNS"
fi

cd "$SCRIPT_DIR"

printf '\n[1/3] Calculate ARL...\n'
"$PYTHON_BIN" OKAFF_ARL_d1_m500_5000.py "${COMMON_ARGS[@]}"

printf '\n[2/3] Calculate EDD and merge matching ARL results...\n'
"$PYTHON_BIN" OKAFF_EDD_d1_m500_5000.py "${COMMON_ARGS[@]}" \
    --arl-results-dir "$RESULTS_DIR"

printf '\n[3/3] Plot the ARL–EDD curves...\n'
"$PYTHON_BIN" compare_okaff_mean_minus3_d1.py \
    --n-runs "$N_RUNS" --results-dir "$RESULTS_DIR"

printf '\nFinished. Figure: %s\n' "$SCRIPT_DIR/okaff_mean03_edd-d1.pdf"
