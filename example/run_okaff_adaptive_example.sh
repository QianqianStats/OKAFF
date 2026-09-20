#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
N_RUNS=500

if [[ "${1:-}" == "--n-runs" ]]; then
    [[ $# -eq 2 ]] || { echo "Usage: bash ${0##*/} [--n-runs N]" >&2; exit 2; }
    N_RUNS="$2"
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash ${0##*/} [--n-runs N]" >&2
    exit 2
fi

[[ "$N_RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "--n-runs must be a positive integer" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }

ARL_DIR="$SCRIPT_DIR/results/OKAFF-adaptive-thresholds-ARL-d20-nrun${N_RUNS}"
EDD_DIR="$SCRIPT_DIR/results/OKAFF-adaptive-thresholds-EDD-d20-nrun${N_RUNS}"
ARL_CSV="$ARL_DIR/OKAFF-adaptive-thresholds-ARL-d20.csv"
EDD_CSV="$EDD_DIR/OKAFF-adaptive-thresholds-EDD-d20.csv"
OUTPUT_CSV="$SCRIPT_DIR/results/OKAFF-adaptive-thresholds-ARL-EDD-d20-nrun${N_RUNS}.csv"

printf '\n[1/3] Running OKAFF adaptive-threshold ARL with %s runs\n' "$N_RUNS"
"$PYTHON_BIN" -u "$SCRIPT_DIR/okaff_adaptive_threshold_arl.py" --n-runs "$N_RUNS"

printf '\n[2/3] Running OKAFF adaptive-threshold EDD with %s runs\n' "$N_RUNS"
"$PYTHON_BIN" -u "$SCRIPT_DIR/okaff_adaptive_threshold_edd.py" --n-runs "$N_RUNS"

printf '\n[3/3] Combining ARL and EDD results\n'
"$PYTHON_BIN" - "$ARL_CSV" "$EDD_CSV" "$OUTPUT_CSV" <<'PY'
import sys
from pathlib import Path

import pandas as pd

arl_path, edd_path, output_path = map(Path, sys.argv[1:])
arl = pd.read_csv(arl_path)
edd = pd.read_csv(edd_path)

arl_required = {"q", "ARL", "standard_error", "censored_rate"}
edd_required = {"scenario", "q", "EDD_successful_only", "SE_successful_only", "failure_rate"}
for path, frame, required in (
    (arl_path, arl, arl_required),
    (edd_path, edd, edd_required),
):
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

arl = arl[["q", "ARL", "standard_error", "censored_rate"]].rename(
    columns={
        "standard_error": "ARL_se",
        "censored_rate": "ARL_censor_rate",
    }
)
edd = edd[["scenario", "q", "EDD_successful_only", "SE_successful_only", "failure_rate"]].rename(
    columns={
        "scenario": "postchange_distribution",
        "EDD_successful_only": "EDD",
        "SE_successful_only": "EDD_se",
        "failure_rate": "EDD_censor_rate",
    }
)

combined = edd.merge(arl, on="q", how="left", validate="many_to_one")
if combined[["ARL", "ARL_se", "ARL_censor_rate"]].isna().any().any():
    raise ValueError("An EDD quantile has no matching ARL result")
combined.insert(0, "prechange_distribution", "N(0, I_20)")
combined = combined[
    [
        "prechange_distribution",
        "postchange_distribution",
        "q",
        "ARL",
        "ARL_se",
        "ARL_censor_rate",
        "EDD",
        "EDD_se",
        "EDD_censor_rate",
    ]
]

output_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output_path, index=False)
print(combined.to_string(index=False))
print(f"\nSaved combined CSV: {output_path}")
PY
