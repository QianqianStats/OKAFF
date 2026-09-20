#!/usr/bin/env bash
# Run OKAFF ARL, EDD, then plot the ARL–EDD curves for m=500, 1000, 5000.
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-/Users/qqjiang/Desktop/OKAFF-code/OKAFF/number of RFF/d20}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
N_RUNS="${N_RUNS:-200}"
RESULTS_DIR="$SCRIPT_DIR/results"

# Fail before starting an experiment if a script is missing.
for script in \
    OKAFF_ARL_d20_m500_5000.py \
    OKAFF-edd-d20_m500_5000.py \
    compare_okaff_cov0p3_edd_d20.py; do
    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        printf 'Missing Python script: %s\n' "$SCRIPT_DIR/$script" >&2
        exit 1
    fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

printf '\n[1/3] Running ARL (m=500, 1000, 5000)...\n'
"$PYTHON_BIN" OKAFF_ARL_d20_m500_5000.py \
    --features 500 1000 5000 --n-runs "$N_RUNS" 

printf '\n[2/3] Running EDD and merging matching ARL results...\n'
"$PYTHON_BIN" OKAFF-edd-d20_m500_5000.py \
    --features 500 1000 5000 --n-runs "$N_RUNS"  \
    --arl-results-dir "$RESULTS_DIR"

# printf '\n[1/3] Running ARL (m=500, 1000, 5000)...\n'
# "$PYTHON_BIN" OKAFF_ARL_d20_m500_5000.py \
#     --features 500 1000 5000 --n-runs "$N_RUNS" --full-grid

# printf '\n[2/3] Running EDD and merging matching ARL results...\n'
# "$PYTHON_BIN" OKAFF-edd-d20_m500_5000.py \
#     --features 500 1000 5000 --n-runs "$N_RUNS" --full-grid \
#     --arl-results-dir "$RESULTS_DIR"

printf '\n[3/3] Plotting ARL–EDD curves...\n'
"$PYTHON_BIN" compare_okaff_cov0p3_edd_d20.py \
    --n-runs "$N_RUNS" --results-dir "$RESULTS_DIR"

printf '\nAll three steps completed. Figure: %s\n' "$SCRIPT_DIR/okaff_cov03_edd-d20.pdf"
