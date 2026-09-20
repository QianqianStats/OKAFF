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

from kerneldetector import ScanBStatistic, estimate_gaussian_gamma


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

d = 1
qs_seed_base = 10_000
run_seed_base = 12345
n_runs = parse_n_runs()
ref_n_total = 250
warmup = 0
pre_change_length = 100
change_point = pre_change_length
n_q = 1000
arl_low = 100
arl_high = 16000
B0 = 50
N_blocks = 5

thr_path = SCRIPT_DIR.parents[1] / "ARL" / f"d{d}" / "results" / f"Scanb-ARL-d{d}-nrun{n_runs}.csv"
out_dir = SCRIPT_DIR / "results"
out_dir.mkdir(parents=True, exist_ok=True)

if not thr_path.exists():
    raise FileNotFoundError(f"Threshold CSV not found: {thr_path}")

thr_df = pd.read_csv(thr_path)
thr_df.columns = thr_df.columns.str.strip()
if "tau" not in thr_df.columns and "threshold" in thr_df.columns:
    thr_df = thr_df.rename(columns={"threshold": "tau"})
if not {"ARL_hat", "tau"}.issubset(thr_df.columns):
    raise ValueError(f"Threshold CSV must contain 'ARL_hat' and 'tau' (or 'threshold'). Got: {thr_df.columns.tolist()}")

thr_df["ARL_hat"] = pd.to_numeric(thr_df["ARL_hat"], errors="coerce")
thr_df["tau"] = pd.to_numeric(thr_df["tau"], errors="coerce")
all_thresholds = thr_df.dropna(subset=["ARL_hat", "tau"])
thr_df = (
    all_thresholds.query("@arl_low < ARL_hat < @arl_high")
    .sort_values("ARL_hat")
    .reset_index(drop=True)
)
if thr_df.empty:
    print(
        f"Warning: no ARL estimates fall in ({arl_low}, {arl_high}); "
        "using all valid thresholds for this run count."
    )
    thr_df = all_thresholds.sort_values("ARL_hat").reset_index(drop=True)
if thr_df.empty:
    raise ValueError(f"No valid threshold rows in {thr_path}")

arl2thresh = dict(zip(thr_df["ARL_hat"].to_numpy(), thr_df["tau"].to_numpy()))
arl_values = np.asarray(list(arl2thresh.keys()), dtype=float)
thresholds = np.asarray(list(arl2thresh.values()), dtype=float)
max_tau = float(thresholds.max())
rows = []

for name in edd_scenario_names(d):
    delays_by_threshold = [[] for _ in thresholds]
    censored_by_threshold = [[] for _ in thresholds]
    attempt_index = 0
    progress = tqdm(total=n_runs * len(thresholds), desc=f"{name} survivors")

    while any(len(delays) < n_runs for delays in delays_by_threshold):
        ref, monitored_pre_data, _ = make_edd_gaussian_data(
            d=d,
            ref_size=ref_n_total,
            pre_change_length=pre_change_length,
            scenario_name=name,
            attempt_index=attempt_index,
            base_seed=run_seed_base,
        )

        detector = ScanBStatistic(
            reference_sample=ref,
            B0=B0,
            N=N_blocks,
            gamma=estimate_gaussian_gamma(ref),
        )

        pre_crossed = np.zeros(len(thresholds), dtype=bool)
        pre_data = np.vstack((ref[:warmup], monitored_pre_data))
        for i, elem in enumerate(pre_data):
            detector.insert(np.asarray(elem))
            stat = detector.statistic()
            if i >= warmup:
                pre_crossed |= stat > thresholds

        x_post = np.asarray(make_edd_postchange_data(
            d=d,
            n=n_q,
            scenario_name=name,
            attempt_index=attempt_index,
            base_seed=qs_seed_base,
        ))
        if x_post.ndim != 2 or x_post.shape[1] != d:
            raise ValueError(f"{name}: post-change data should have shape (T, {d}), got {x_post.shape}")

        first_crossing = np.full(len(thresholds), n_q, dtype=int)
        post_crossed = np.zeros(len(thresholds), dtype=bool)

        for t, elem in enumerate(x_post[:n_q], start=1):
            detector.insert(np.asarray(elem))
            stat = detector.statistic()
            newly_crossed = (~post_crossed) & (stat > thresholds)
            first_crossing[newly_crossed] = t
            post_crossed |= newly_crossed
            if stat > max_tau:
                break

        survived = ~pre_crossed
        for j in range(len(thresholds)):
            if survived[j] and len(delays_by_threshold[j]) < n_runs:
                delays_by_threshold[j].append(first_crossing[j])
                censored_by_threshold[j].append(0 if post_crossed[j] else 1)
                progress.update(1)

        attempt_index += 1

    progress.close()

    for arl_hat, threshold, delays, censored in zip(
        arl_values, thresholds, delays_by_threshold, censored_by_threshold
    ):
        delays = np.asarray(delays, dtype=float)
        censored = np.asarray(censored, dtype=float)
        edd_std = float(delays.std(ddof=1)) if len(delays) > 1 else 0.0
        rows.append({
            "method": "ScanB",
            "post_change_name": name,
            "arl_hat": float(arl_hat),
            "dimension_d": d,
            "num_features_m": np.nan,
            "threshold": float(threshold),
            "edd_hat": float(delays.mean()),
            "edd_std": edd_std,
            "edd_se_mean": float(edd_std / np.sqrt(len(delays))),
            "edd_censored_rate": float(censored.mean()),
            "n_runs_edd": len(delays),
            "burn_in": warmup,
            "change_point": change_point,
        })

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

out_csv = out_dir / f"Scanb-EDD-d{d}-nrun{n_runs}.csv"
pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)
print("Saved:", out_csv)
