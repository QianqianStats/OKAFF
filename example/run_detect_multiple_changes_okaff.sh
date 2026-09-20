#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

usage() {
    cat <<EOF
Usage: bash ${0##*/} DATA_CSV [detector options]

Detector options:
  --quantile Q       Adaptive-threshold quantile (default: 0.92)
  --n-rff M          Number of random Fourier features (default: 500)
  --seed SEED        Random seed (default: 2026)
  --output-csv PATH  Override the default result path

Environment:
  PYTHON_BIN          Python executable (default: REPOSITORY/.venv/bin/python)
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    [[ $# -eq 0 ]] && exit 2 || exit 0
fi

DATA_CSV="$1"
shift

[[ -f "$DATA_CSV" ]] || { echo "Input CSV not found: $DATA_CSV" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }

DATA_NAME="$(basename "$DATA_CSV")"
DATA_STEM="${DATA_NAME%.*}"
QUANTILE="0.92"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    argument="${arguments[index]}"
    if [[ "$argument" == "--quantile" ]]; then
        ((index + 1 < ${#arguments[@]})) || { echo "--quantile requires a value" >&2; exit 2; }
        QUANTILE="${arguments[index + 1]}"
    elif [[ "$argument" == --quantile=* ]]; then
        QUANTILE="${argument#*=}"
    fi
done
QUANTILE_TAG="$("$PYTHON_BIN" -c 'import sys; print(f"{float(sys.argv[1]):g}".replace("-", "m").replace(".", "p"))' "$QUANTILE")"
DEFAULT_OUTPUT="$SCRIPT_DIR/results/${DATA_STEM}-OKAFF-multiple-changes-q${QUANTILE_TAG}.csv"

has_output=false
for argument in "$@"; do
    if [[ "$argument" == "--output-csv" || "$argument" == --output-csv=* ]]; then
        has_output=true
        break
    fi
done

command=(
    "$PYTHON_BIN" -u
    "$SCRIPT_DIR/detect_multiple_changes_okaff.py"
    "$DATA_CSV"
)
if [[ "$has_output" == false ]]; then
    command+=(--output-csv "$DEFAULT_OUTPUT")
fi
command+=("$@")

"${command[@]}"
