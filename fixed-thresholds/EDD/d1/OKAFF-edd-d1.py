import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = CODE_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from shared_edd_data import PairedPostChangeData, edd_scenario_names
import pandas as pd

from kerneldetector import OKAFF
from fixedthresholds import default_band_lambda
from okaffexperiment import (
    run_one_q_prepost_and_edd, comparison_rows_for_q,
)


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

RUN_METHODS = ("OKAFF",)

band_grid_col = "chart_L"

# ARL from 100-1200
L_values = np.array([4.5, 7, 9], dtype=float)

# full ARL from 70-12000
# L_values = np.array([4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=float)

out_dir = SCRIPT_DIR / "results"
out_dir.mkdir(parents=True, exist_ok=True)

d = 1
m = 500
ref_size = 250

Sigma_d = np.eye(d)

burn_in = 250
pre_change_length = 100

change_point = pre_change_length
n_q = 1000
n_runs = parse_n_runs()

qs_seed_base = 10_000
run_seed_base = 12345

band_lambda = 0.999

algo_kwargs = dict(
    lambda0=1 - 1e-3,
    lambda1=1 - 1e-3,
    band_lambda=band_lambda,
    eta=1e-3,
    clip=(1e-3, 1 - 1e-3),
    store_lambdas=False,
    thresholding_method="fixed",
    fixed_threshold=np.inf,
    store_values=False,
)

alphas_common = np.asarray(L_values, dtype=float)
if alphas_common.size == 0:
    raise ValueError("L_values must contain at least one chart_L value.")

method_configs = {
    "OKAFF": {
        "grid_values": alphas_common,
        "dimension_d": int(d),
        "bandwidth_method": "gauss_est_gamma_resampled_pairwise_median_same_run_theory_band",
        "detector_bandwidth": "gauss_est_gamma_resampled_pairwise_median_fresh_per_run",
        "band_source": "same_run_theory_recomputed_from_detector_sigmasq",
        "band_lambda": float(default_band_lambda(algo_kwargs)),
    }
}

print(f"Using {len(alphas_common)} {band_grid_col} values without loading ARL-summary CSVs:", alphas_common)

qs = edd_scenario_names(d)

rows_all = []

for idx, name in enumerate(qs):
    print(f"\n===== Running post-change case: {name} =====")

    result = run_one_q_prepost_and_edd(
         OKAFF,
         q_obj=PairedPostChangeData(d=d, n=n_q, scenario_name=name, base_seed=qs_seed_base),
         q_name=name,
         method_configs=method_configs,
         selected_methods=RUN_METHODS,
         alphas_common=alphas_common,
         d=d,
         burn_in=burn_in,
         pre_change_length=pre_change_length,
         n_q=n_q,
         n_runs=n_runs,
         m=m,
         ref_size=ref_size,
         algo_kwargs=algo_kwargs,
         Sigma_d=Sigma_d,
         seed=run_seed_base + 1000 * idx,
    )

    rows_q = comparison_rows_for_q(
        method_configs,
        result,
        q_name=name,
        burn_in=burn_in,
        pre_change_length=pre_change_length,
        change_point=change_point,
        grid_col=band_grid_col,
    )
    rows_all.extend(rows_q)

    edd_hat, _, _, edd_se, _, edd_cens = result["edd"]["OKAFF"]
    print(
        f"  Done {name}. {band_grid_col}={alphas_common[0]:.4g}: "
        f"OKAFF={edd_hat[0]:.2f} (SE {edd_se[0]:.2f}, cens {edd_cens[0]:.3f})"
    )

cmp_df = pd.DataFrame(rows_all)

common_cols = [
    column for column in (
        "post_change_name", "q", "change_magnitude", "chart_L", "dimension_d",
        "num_features_m", "burn_in", "prechange", "change_point", "n_runs_edd",
    )
    if column in cmp_df.columns
]
metric_names = (
    "arl_hat", "arl_var", "arl_std", "arl_se_mean", "arl_censored_rate",
    "edd_hat", "edd_var", "edd_std", "edd_se_mean", "edd_censored_rate",
)
rename_map = {f"{metric}_OKAFF": metric for metric in metric_names}

def format_results_for_csv(dataframe, *, combined=False):
    columns = [
        column for column in common_cols
        if not combined or column != "prechange"
    ]
    columns += [column for column in rename_map if column in dataframe.columns]
    clean = dataframe[columns].rename(columns=rename_map).copy()
    clean.insert(0, "method", "OKAFF")
    return clean

out_path = out_dir / f"OKAFF-EDD-d{d}-nrun{n_runs}.csv"
format_results_for_csv(cmp_df).to_csv(out_path, index=False)
print("Saved OKAFF EDD CSV:", out_path)

arl_csv_path = SCRIPT_DIR.parents[1] / "ARL" / f"d{d}" / "results" / f"OKAFF-ARL-d{d}-nrun{n_runs}.csv"

if not arl_csv_path.exists():
    print(
        "Optional ARL CSV not found; EDD outputs were saved successfully and "
        "the ARL+EDD merge is being skipped:",
        arl_csv_path,
    )
    sys.exit(0)

print("Using ARL CSV:", arl_csv_path)

arl_df = pd.read_csv(arl_csv_path)

if "chart_L" not in arl_df.columns:
    raise ValueError("ARL CSV must contain column: chart_L")

required_arl_cols = [
    "arl_hat",
    "arl_var",
    "arl_std",
    "arl_se_mean",
    "arl_censored_rate",
]

missing = [c for c in required_arl_cols if c not in arl_df.columns]
if missing:
    raise ValueError("ARL CSV missing columns:\n" + "\n".join(missing))

arl_metrics = arl_df[["chart_L"] + required_arl_cols].copy()

arl_metrics = arl_metrics.rename(
    columns={column: f"{column}_OKAFF" for column in required_arl_cols}
)

df_combined = cmp_df.copy()

for c in arl_metrics.columns:
    if c != "chart_L" and c in df_combined.columns:
        df_combined = df_combined.drop(columns=[c])

df_combined = df_combined.merge(
    arl_metrics,
    on="chart_L",
    how="left",
    validate="many_to_one",
)

unmatched = df_combined[df_combined["arl_hat_OKAFF"].isna()]["chart_L"].unique()
if len(unmatched) > 0:
    print("Warning: these chart_L values did not match ARL CSV:", unmatched)

out_path_combined = out_dir / f"OKAFF-ARL-EDD-d{d}-nrun{n_runs}.csv"
format_results_for_csv(df_combined, combined=True).to_csv(out_path_combined, index=False)
print("Saved combined ARL + EDD CSV:", out_path_combined)
