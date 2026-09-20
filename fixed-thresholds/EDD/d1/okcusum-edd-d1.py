import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared_edd_data import edd_scenario_names, make_edd_gaussian_data, make_edd_postchange_data


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CODE_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kerneldetector import OKCUSUM, estimate_gaussian_gamma


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs


d = 1
n_runs = parse_n_runs()
ref_n_total = 250
pre_change_length = 100
change_point = pre_change_length
n_q = 1000

run_seed_base = 12345
qs_seed_base = 10_000

# arl from 100-1300
thresholds = np.asarray([3.3, 4.5, 5.3], dtype=float)
# full ARL from 100-20000
# thresholds = np.asarray([3.3, 4.0, 5.5, 6.0, 6.2, 6.5, 7.0, 7.3, 7.4], dtype=float)
max_threshold = thresholds[-1]

B_min = 2
B_max = 50
N_blocks = 5
alpha_level = 0.05

if ref_n_total < N_blocks * B_max:
    raise ValueError(
        f"Need ref_n_total >= N_blocks * B_max; got {ref_n_total} < {N_blocks * B_max}."
    )

out_dir = SCRIPT_DIR / "results"
out_dir.mkdir(parents=True, exist_ok=True)
arl_csv_path = SCRIPT_DIR.parents[1] / "ARL" / f"d{d}" / "results" / f"okcusum-ARL-d{d}-nrun{n_runs}.csv"
out_csv = out_dir / f"okcusum-EDD-d{d}-nrun{n_runs}.csv"

if not arl_csv_path.exists():
    raise FileNotFoundError(f"ARL CSV not found: {arl_csv_path}")

arl_df = pd.read_csv(arl_csv_path)
if "ARL_hat" in arl_df.columns and "arl_hat" not in arl_df.columns:
    arl_df = arl_df.rename(columns={"ARL_hat": "arl_hat"})
required_arl_columns = {"threshold", "arl_hat"}
if not required_arl_columns.issubset(arl_df.columns):
    raise ValueError(f"ARL CSV must contain {sorted(required_arl_columns)}")

arl_df = arl_df[["threshold", "arl_hat"]].copy()
arl_df["threshold"] = pd.to_numeric(arl_df["threshold"], errors="raise").round(12)
arl_df["arl_hat"] = pd.to_numeric(arl_df["arl_hat"], errors="raise")
if arl_df["threshold"].duplicated().any():
    raise ValueError("ARL CSV contains duplicate thresholds")


def run_scenario(name):
    pre_maxima = []
    delay_rows = []
    censor_rows = []
    n_survivors = 0

    progress = tqdm(total=n_runs, desc=f"{name} survivors")

    while n_survivors < n_runs:
        attempt_index = len(pre_maxima)
        ref, monitored_pre_data, _ = make_edd_gaussian_data(
            d=d,
            ref_size=ref_n_total,
            pre_change_length=pre_change_length,
            scenario_name=name,
            attempt_index=attempt_index,
            base_seed=run_seed_base,
        )

        detector = OKCUSUM(
            reference_sample=ref,
            B_max=B_max,
            B_min=B_min,
            N=N_blocks,
            gamma=estimate_gaussian_gamma(ref),
            alpha=alpha_level,
        )

        pre_max = -np.inf
        for elem in monitored_pre_data:
            detector.insert(np.asarray(elem))
            stat = detector.statistic()
            if stat > pre_max:
                pre_max = stat

        x_post = make_edd_postchange_data(
            d=d,
            n=n_q,
            scenario_name=name,
            attempt_index=attempt_index,
            base_seed=qs_seed_base,
        )

        first_crossing = np.zeros(len(thresholds), dtype=int)
        for delay, elem in enumerate(x_post, start=1):
            detector.insert(np.asarray(elem))
            stat = detector.statistic()
            newly_crossed = (first_crossing == 0) & (stat > thresholds)
            first_crossing[newly_crossed] = delay
            if stat > max_threshold:
                break

        censored = first_crossing == 0
        delays = np.where(censored, n_q, first_crossing)

        pre_maxima.append(pre_max)
        delay_rows.append(delays)
        censor_rows.append(censored)

        if pre_max <= thresholds[0]:
            n_survivors += 1
            progress.update(1)

    progress.close()

    pre_maxima = np.asarray(pre_maxima, dtype=float)
    delay_rows = np.asarray(delay_rows, dtype=float)
    censor_rows = np.asarray(censor_rows, dtype=bool)
    rows = []

    for j, threshold in enumerate(thresholds):
        eligible = np.flatnonzero(pre_maxima <= threshold)[:n_runs]
        if eligible.size < n_runs:
            raise RuntimeError(f"Only {eligible.size} survivors for threshold {threshold}")

        delays = delay_rows[eligible, j]
        censored = censor_rows[eligible, j]
        edd_std = float(delays.std(ddof=1))

        rows.append(
            {
                "method": "OKCUSUM",
                "post_change_name": name,
                "dimension_d": d,
                "num_features_m": np.nan,
                "threshold": float(threshold),
                "edd_hat": float(delays.mean()),
                "edd_std": edd_std,
                "edd_se_mean": float(edd_std / np.sqrt(n_runs)),
                "edd_censored_rate": float(censored.mean()),
                "n_runs_edd": n_runs,
                "burn_in": 0,
                "change_point": change_point,
            }
        )

    return rows


rows = []
for name in edd_scenario_names(d):
    rows.extend(run_scenario(name))

result = pd.DataFrame(rows)
result["threshold"] = result["threshold"].round(12)
result = result.merge(arl_df, on="threshold", how="left", validate="many_to_one")

if result["arl_hat"].isna().any():
    missing = sorted(result.loc[result["arl_hat"].isna(), "threshold"].unique())
    raise ValueError(f"Missing ARL values for thresholds: {missing}")

columns = [
    "method",
    "post_change_name",
    "arl_hat",
    "dimension_d",
    "num_features_m",
    "threshold",
    "edd_hat",
    "edd_std",
    "edd_se_mean",
    "edd_censored_rate",
    "n_runs_edd",
    "burn_in",
    "change_point",
]

result.to_csv(out_csv, columns=columns, index=False)
print("Saved:", out_csv)
