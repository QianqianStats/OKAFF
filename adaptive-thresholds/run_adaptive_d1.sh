#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$RESULTS_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

N_RUNS=500
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || "$1" != "--n-runs" ]]; then
        echo "Usage: bash $0 [--n-runs POSITIVE_INTEGER]" >&2
        exit 2
    fi
    N_RUNS="$2"
fi
if [[ ! "$N_RUNS" =~ ^[0-9]+$ || ! "$N_RUNS" =~ [1-9] ]]; then
    echo "--n-runs must be a positive integer" >&2
    exit 2
fi

run_stage() {
    local stage="$1"
    local folder="$RESULTS_DIR/$stage/d1"
    local scripts=("$folder"/*.py)
    if [[ ! -f "${scripts[0]}" ]]; then
        echo "No Python scripts found in $folder" >&2
        exit 1
    fi
    for script in "${scripts[@]}"; do
        echo "Running $script"
        (cd "$folder" && "$PYTHON_BIN" -u "$script" --n-runs "$N_RUNS")
    done
}

run_stage ARL
run_stage EDD

echo "Comparing d1 ARL and EDD results"
cd "$RESULTS_DIR"
"$PYTHON_BIN" -u "$RESULTS_DIR/compare_arl_edd.py" --dimension 1 --results-dir "$RESULTS_DIR" --n-runs "$N_RUNS"
