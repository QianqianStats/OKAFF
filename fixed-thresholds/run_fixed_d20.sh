#!/usr/bin/env bash
set -euo pipefail

FIXED_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$FIXED_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

N_RUNS=200
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

run_experiments() {
    local label="$1"
    local folder="$2"
    local scripts=("$folder"/*.py)
    if [[ ! -f "${scripts[0]}" ]]; then
        echo "No Python scripts found in $folder" >&2
        exit 1
    fi
    echo "Starting $label with $N_RUNS runs"
    for script in "${scripts[@]}"; do
        echo "Running $script"
        (cd "$folder" && "$PYTHON_BIN" -u "$script" --n-runs "$N_RUNS")
    done
}

run_plots() {
    local folder="$1"
    local scripts=("$folder"/*.py)
    if [[ ! -f "${scripts[0]}" ]]; then
        echo "No Python scripts found in $folder" >&2
        exit 1
    fi
    echo "Starting d20 ARL-vs-EDD plots"
    for script in "${scripts[@]}"; do
        echo "Running $script"
        (cd "$folder" && "$PYTHON_BIN" -u "$script" --n-runs "$N_RUNS")
    done
}

run_experiments "d20 ARL experiments" "$FIXED_DIR/ARL/d20"
run_experiments "d20 EDD experiments" "$FIXED_DIR/EDD/d20"
run_plots "$FIXED_DIR/ARL-vs-EDD-plot/d20"
