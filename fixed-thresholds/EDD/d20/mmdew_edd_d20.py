import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
import numpy as np
from shared_edd_data import edd_scenario_names, make_edd_gaussian_data, make_edd_postchange_data
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = CODE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kerneldetector import MMDEW, estimate_gaussian_gamma


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

def main():

    d = 20
    qs_seed_base = 10_000
    run_seed_base = 12345

    n_runs = parse_n_runs()
    ref_n_total = 250
    burn_in = 250
    pre_change_length = 100

    change_point = pre_change_length
    total_pre_change = burn_in + pre_change_length
    n_q = 1000
    arl_low = 100
    arl_high = 16000

    thr_path = SCRIPT_DIR.parents[1] / "ARL" / f"d{d}" / "results" / f"MMDEW-ARL-d{d}-nrun{n_runs}.csv"

    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)


    qs = edd_scenario_names(d)

    if not Path(thr_path).exists():
        raise FileNotFoundError(
            f"Threshold CSV not found: {thr_path}\n"
            "Put the threshold CSV in the same folder as this .py file."
        )

    thr_df = pd.read_csv(thr_path)
    thr_df.columns = thr_df.columns.str.strip()

    if "ARL_hat" not in thr_df.columns or "tau" not in thr_df.columns:
        raise ValueError(f"Threshold CSV must contain columns 'ARL_hat' and 'tau'. Got: {thr_df.columns.tolist()}")

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
    max_tau = float(thr_df["tau"].max())

    def edd_summary(
        arl2thresh,
        statistics_pre,
        statistics_post,
        *,
        data_name,
        algorithm="MMDEW",
        max_post_len=1000,
        burn_in=64,
        target_n_runs=200,
    ):
        rows = []

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
            stats_post = [
                np.asarray(statistics_post[i], dtype=float)[:max_post_len]
                for i in eligible_indices
            ]
            delays = []
            cens_flags = []

            for s in stats_post:
                crossed = np.flatnonzero(s > thresh)
                if len(crossed) == 0:
                    delays.append(max_post_len)
                    cens_flags.append(1)
                else:
                    delays.append(int(crossed[0]) + 1)
                    cens_flags.append(0)

            delays = np.asarray(delays, dtype=float)
            cens_flags = np.asarray(cens_flags, dtype=int)

            edd_hat = float(delays.mean())
            edd_var = float(delays.var(ddof=1)) if len(delays) > 1 else 0.0
            edd_std = float(np.sqrt(edd_var))
            edd_se = float(edd_std / np.sqrt(len(delays))) if len(delays) > 0 else np.nan
            edd_censor_rate = float(cens_flags.mean())

            rows.append({
                "method": algorithm,
                "post_change_name": data_name,
                "arl_hat": float(arl_key),
                "dimension_d": d,
                "threshold": float(thresh),
                "edd_hat": edd_hat,
                "edd_var": edd_var,
                "edd_std": edd_std,
                "edd_se_mean": edd_se,
                "edd_censored_rate": edd_censor_rate,
                "n_runs_edd": int(len(delays)),
                "burn_in": int(burn_in),
                "change_point": int(change_point),
            })
        return rows

    rows = []

    thresholds = np.asarray(list(arl2thresh.values()), dtype=float)

    for name in qs:
        pre_stats_runs = []
        post_stats_runs = []

        accepted_by_threshold = np.zeros(len(thresholds), dtype=int)
        progress = tqdm(total=n_runs * len(thresholds), desc=f"{name} survivors")

        while np.any(accepted_by_threshold < n_runs):
            ref, monitored_pre_data, _ = make_edd_gaussian_data(
                d=d, ref_size=ref_n_total, pre_change_length=pre_change_length,
                scenario_name=name, attempt_index=len(pre_stats_runs), base_seed=run_seed_base,
            )
            gamma = estimate_gaussian_gamma(ref[:500])
            detector = MMDEW(gamma=gamma)

            pre_data = np.vstack((ref[:burn_in], monitored_pre_data))
            for elem in pre_data:
                detector.insert(elem.reshape(1, -1))

            x_post = make_edd_postchange_data(d=d, n=n_q, scenario_name=name, attempt_index=len(pre_stats_runs), base_seed=qs_seed_base)
            x_post = np.asarray(x_post)
            if x_post.ndim != 2 or x_post.shape[1] != d:
                raise ValueError(f"{name}: shared post-change data should have shape (T,{d}), got {x_post.shape}")

            for elem in x_post[:n_q]:
                detector.insert(elem.reshape(1, -1))
                if detector.stats[-1] > max_tau:
                    break

            stat_series = list(detector.stats)

            pre = stat_series[:total_pre_change]
            post = stat_series[total_pre_change:total_pre_change + n_q]

            if len(post) < n_q:
                post = list(post) + [np.nan] * (n_q - len(post))
            else:
                post = list(post[:n_q])

            pre_stats_runs.append(pre)
            post_stats_runs.append(post)

            monitored_pre = np.asarray(pre[burn_in:], dtype=float)
            survived = ~np.any(monitored_pre[:, None] > thresholds[None, :], axis=0)
            newly_accepted = survived & (accepted_by_threshold < n_runs)
            accepted_by_threshold[newly_accepted] += 1
            progress.update(int(newly_accepted.sum()))

        progress.close()

        rows.extend(
            edd_summary(
                arl2thresh=arl2thresh,
                statistics_pre=pre_stats_runs,
                statistics_post=post_stats_runs,
                data_name=name,
                algorithm="MMDEW",
                max_post_len=n_q,
                burn_in=burn_in,
                target_n_runs=n_runs,
            )
        )

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"MMDEW-EDD-d{d}-nrun{n_runs}.csv"
    df.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)
    print("\nCensor-rate summary (all scenarios):")
    print(df.groupby("post_change_name")["edd_censored_rate"].agg(["mean", "max", "min"]).round(4))
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
