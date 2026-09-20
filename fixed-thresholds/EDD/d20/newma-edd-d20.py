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
from shared_edd_data import edd_scenario_names, make_edd_gaussian_data, make_edd_postchange_data
import pandas as pd
from tqdm import tqdm
import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat

from kerneldetector import NEWMA, estimate_gaussian_gamma


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

out_dir = SCRIPT_DIR / "results"
out_dir.mkdir(parents=True, exist_ok=True)

B = 50
d = 20
qs_seed_base = 10_000
run_seed_base = 12345

burn_in = 250
pre_change_length = 100

change_point = pre_change_length
absolute_change_index = burn_in + pre_change_length
ref_n_total = 250
ref_n_sigma = 500
n_runs = parse_n_runs()
n_q = 1000

big_Lambda, small_lambda = algos.select_optimal_parameters(B)
thres_ff = small_lambda

m = 500

def generate_offdiag_pairwise_median_frequencies(m, d, reference_sample, max_len=500):
    gamma = estimate_gaussian_gamma(reference_sample[:max_len], max_len=max_len)
    sigmasq = 1 / (2 * gamma)
    return feat.generate_frequencies(m, d, sigmasq=sigmasq, choice_sigma="fixed")


qs = edd_scenario_names(d)

threshold_filename = f"NEWMA-ARL-d20-nrun{n_runs}.csv"
thr_path = SCRIPT_DIR.parents[1] / "ARL" / f"d{d}" / "results" / threshold_filename
if not thr_path.is_file():
    raise FileNotFoundError(
        f"NEWMA threshold CSV not found beside this script: {thr_path}"
    )

loarl = 70
uparl = 17000

all_thresholds = pd.read_csv(thr_path).dropna(subset=["ARL_hat", "tau"])
thr_df = (
    all_thresholds.query("@loarl < ARL_hat < @uparl")
    .sort_values("ARL_hat")
    .reset_index(drop=True)
)
if thr_df.empty:
    print(
        f"Warning: no ARL estimates fall in ({loarl}, {uparl}); "
        "using all valid thresholds for this run count."
    )
    thr_df = all_thresholds.sort_values("ARL_hat").reset_index(drop=True)
if thr_df.empty:
    raise ValueError(f"No valid threshold rows in {thr_path}")
arl2thresh = dict(zip(thr_df["ARL_hat"].to_numpy(), thr_df["tau"].to_numpy()))

def edd_summary(arl2thresh, statistics_pre, statistics_post, burn_in, max_post_len, target_n_runs):
    """Conditional capped EDD, independently for each fixed threshold."""
    out = {}
    for arl_key, thresh in arl2thresh.items():
        survived = np.asarray([
            not np.any(np.asarray(pre, dtype=float)[burn_in:] > thresh)
            for pre in statistics_pre
        ], dtype=bool)
        eligible_indices = np.flatnonzero(survived)[:target_n_runs]
        if eligible_indices.size < target_n_runs:
            raise RuntimeError(
                f"Only {eligible_indices.size} survivors for threshold {thresh}; "
                f"expected {target_n_runs}."
            )
        eligible_posts = [statistics_post[i] for i in eligible_indices]

        delays, cens_flags = [], []
        for post in eligible_posts:
            post = np.asarray(post, dtype=float)
            hits = np.flatnonzero(post > thresh)
            if hits.size:
                delays.append(float(hits[0] + 1))
                cens_flags.append(0.0)
            else:
                delays.append(max_post_len)
                cens_flags.append(1.0)

        delays = np.asarray(delays, dtype=float)
        cens_flags = np.asarray(cens_flags, dtype=float)
        n_eligible = len(delays)

        edd_hat = float(delays.mean()) if n_eligible else np.nan
        edd_var = float(delays.var(ddof=1)) if n_eligible > 1 else (0.0 if n_eligible == 1 else np.nan)
        edd_std = float(np.sqrt(edd_var))
        edd_se = float(edd_std / np.sqrt(n_eligible)) if n_eligible else np.nan
        edd_cens = float(cens_flags.mean()) if n_eligible else np.nan

        out[arl_key] = {
            "edd_hat": edd_hat,
            "edd_var": edd_var,
            "edd_std": edd_std,
            "edd_se_mean": edd_se,
            "edd_censor_rate": edd_cens,
            "n_runs": int(n_eligible),
        }

    return out

thresholds = np.asarray(list(arl2thresh.values()), dtype=float)

rows = []

for name in qs:
    pre_stats_runs = []
    post_stats_runs = []

    accepted_by_threshold = np.zeros(len(thresholds), dtype=int)
    progress = tqdm(total=n_runs * len(thresholds), desc=f"{name} survivors")

    while np.any(accepted_by_threshold < n_runs):

        ref, monitored_pre_data, initial_x = make_edd_gaussian_data(
            d=d, ref_size=ref_n_total, pre_change_length=pre_change_length,
            scenario_name=name, attempt_index=len(pre_stats_runs), base_seed=run_seed_base,
        )

        W, _ = generate_offdiag_pairwise_median_frequencies(m, d, ref, max_len=ref_n_sigma)

        def feat_func(x, W=W):
            return feat.fourier_feat(x, W)

        detector = NEWMA(
            initial_x,
            forget_factor=big_Lambda,
            forget_factor2=small_lambda,
            feat_func=feat_func,
            adapt_forget_factor=thres_ff,
        )

        if burn_in > ref.shape[0]:
            raise ValueError(
                f"Need at least {burn_in} reference observations for burn-in, "
                f"got {ref.shape[0]}"
            )
        pre_data = np.vstack((ref[:burn_in], monitored_pre_data))
        detector.apply_to_data(pre_data)

        x_post = make_edd_postchange_data(d=d, n=n_q, scenario_name=name, attempt_index=len(pre_stats_runs), base_seed=qs_seed_base)
        detector.apply_to_data(x_post)

        stat_series = [s[0] for s in detector.stat_stored]

        pre = stat_series[:absolute_change_index]
        post = stat_series[absolute_change_index:absolute_change_index + len(x_post)]

        pre_stats_runs.append(pre)
        post_stats_runs.append(post)

        monitored_pre = np.asarray(pre[burn_in:], dtype=float)
        survived = ~np.any(monitored_pre[:, None] > thresholds[None, :], axis=0)
        newly_accepted = survived & (accepted_by_threshold < n_runs)
        accepted_by_threshold[newly_accepted] += 1
        progress.update(int(newly_accepted.sum()))

    progress.close()

    arl2edd = edd_summary(
        arl2thresh=arl2thresh,
        statistics_pre=pre_stats_runs,
        statistics_post=post_stats_runs,
        burn_in=burn_in,
        max_post_len=n_q,
        target_n_runs=n_runs,
    )

    for arl_key in sorted(arl2edd.keys()):
        r = arl2edd[arl_key]
        rows.append({
            "method": "NEWMA",
            "post_change_name": name,
            "arl_hat": float(arl_key),
            "dimension_d": d,
            "num_features_m": m,
            "threshold": float(arl2thresh[arl_key]),
            "edd_hat": float(r["edd_hat"]),
            "edd_var": float(r["edd_var"]),
            "edd_std": float(r["edd_std"]),
            "edd_se_mean": float(r["edd_se_mean"]),
            "edd_censored_rate": float(r["edd_censor_rate"]),
            "n_runs_edd": int(r["n_runs"]),
            "burn_in": burn_in,
            "change_point": change_point,
        })


df_long = pd.DataFrame(rows).sort_values(
    ["post_change_name", "arl_hat"]
).reset_index(drop=True)

censor_summary = (
    df_long.groupby("post_change_name", as_index=False)["edd_censored_rate"]
    .agg(mean="mean", max="max", min="min")
    .sort_values("post_change_name")
    .reset_index(drop=True)
    .round(4)
)

csv_path = out_dir / f"NEWMA-EDD-d{d}-nrun{n_runs}.csv"
df_long.to_csv(csv_path, index=False)
print("Saved CSV:", csv_path)
print("\nCensor-rate summary (all scenarios):")
print(censor_summary.to_string(index=False))
print("\nEDD table:")
print(df_long.head(20))
