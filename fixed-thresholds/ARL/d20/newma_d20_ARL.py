import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


import numpy as np
import pandas as pd
from tqdm import tqdm

import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat
from kerneldetector import NEWMA, estimate_gaussian_gamma
from shared_gaussian_streams import (
    make_gaussian_initialization_rng,
    make_gaussian_rngs,
)


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

B = 50
d = 20
ref_size = 250
burn_in = 250

n_runs = parse_n_runs()
max_stats = 20000

# ARL from 70-1000
tau_min, tau_max = 0.0838, 0.087
n_tau = 3
# FULL ARL from 70-15000
# tau_min, tau_max = 0.0835, 0.0907
# n_tau = 8

out_dir = Path(__file__).resolve().parent / "results"
out_dir.mkdir(parents=True, exist_ok=True)

big_Lambda, small_lambda = algos.select_optimal_parameters(B)
thres_ff = small_lambda

m = 500

print("big_Lambda =", big_Lambda, "small_lambda =", small_lambda, "m =", m)

def offdiag_pairwise_median(reference_sample, max_len=500):
    gamma = estimate_gaussian_gamma(reference_sample[:max_len], max_len=max_len)
    return 1 / (2 * gamma)

def estimate_rl_moments_h0_newma(
    tau,
    *,
    d,
    ref_size,
    burn_in,
    n_runs,
    max_stats,
    m,
    big_Lambda,
    small_lambda,
    thres_ff,
    seed=12345,
):
    run_lengths = np.empty(n_runs, dtype=int)
    censored = 0

    for r in tqdm(range(n_runs), desc=f"RL moments @ tau={tau:.6f}", leave=False):
        reference_rng, monitoring_rng = make_gaussian_rngs(
            base_seed=seed, d=d, run_index=r
        )
        initialization_rng = make_gaussian_initialization_rng(
            base_seed=seed, d=d, run_index=r
        )
        ref = reference_rng.normal(size=(ref_size, d))
        initial_x = initialization_rng.normal(size=d)
        sigmasq = offdiag_pairwise_median(ref)
        W, _ = feat.generate_frequencies(m, d, sigmasq=sigmasq, choice_sigma="fixed")
        feat_func_local = (lambda x, W=W: feat.fourier_feat(x, W))

        detector = NEWMA(
            initial_x,
            forget_factor=big_Lambda,
            forget_factor2=small_lambda,
            feat_func=feat_func_local,
            adapt_forget_factor=thres_ff,
        )

        if burn_in > ref.shape[0]:
            raise ValueError(
                f"Need at least {burn_in} reference observations for burn-in, "
                f"got {ref.shape[0]}"
            )

        detector.apply_to_data(ref[:burn_in])

        run_lengths[r] = max_stats
        for t in range(max_stats):
            x = monitoring_rng.normal(size=d)
            stat = detector.update_stat(x)

            if stat is None:
                stat = detector.stat_stored[-1]
            stat_value = float(np.asarray(stat).reshape(-1)[0])

            if stat_value > tau:
                run_lengths[r] = t + 1
                break
        else:
            censored += 1

    arl_hat = run_lengths.mean()
    var_hat = run_lengths.var(ddof=1) if n_runs > 1 else 0.0
    std_hat = np.sqrt(var_hat)
    se_mean = std_hat / np.sqrt(n_runs) if n_runs > 0 else np.nan
    censored_rate = censored / n_runs if n_runs > 0 else np.nan

    return arl_hat, var_hat, std_hat, se_mean, censored_rate

taus = np.linspace(tau_min, tau_max, n_tau)

rows = []
out_csv = out_dir / f"NEWMA-ARL-d{d}-nrun{n_runs}.csv"

for k, tau in enumerate(taus, start=1):

    arl_hat, var_hat, std_hat, se_mean, cens = estimate_rl_moments_h0_newma(
        tau,
        d=d,
        ref_size=ref_size,
        burn_in=burn_in,
        n_runs=n_runs,
        max_stats=max_stats,
        m=m,
        big_Lambda=big_Lambda,
        small_lambda=small_lambda,
        thres_ff=thres_ff,
        seed=12345,
    )

    print(f"\n===== ({k}/{len(taus)}) tau = {tau:.6f} =====")
    print(f"ARL_hat       = {arl_hat:.3f}")
    print(f"RL_var_hat    = {var_hat:.3f}")
    print(f"RL_std_hat    = {std_hat:.3f}")
    print(f"ARL_se_hat    = {se_mean:.3f}")
    print(f"censored_rate = {cens:.3f}")
    if cens > 0.02:
        print("WARNING: censoring > 0.02, consider increasing max_stats.")

    rows.append({
        "tau": float(tau),
        "ARL_hat": float(arl_hat),
        "RL_var_hat": float(var_hat),
        "RL_std_hat": float(std_hat),
        "ARL_se_hat": float(se_mean),
        "censored_rate": float(cens),
    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Saved CSV:", out_csv)

df_tau = pd.DataFrame(rows)
max_cens = df_tau["censored_rate"].max()
if max_cens > 0.02:
    print(f"WARNING: censoring up to {max_cens:.3f}. Consider increasing max_stats.")
