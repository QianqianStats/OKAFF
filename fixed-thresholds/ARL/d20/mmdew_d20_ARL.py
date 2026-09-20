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

from kerneldetector import MMDEW, estimate_gaussian_gamma
from shared_gaussian_streams import make_gaussian_rngs


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

d = 20
ref_size = 250

burn_in_stats = 250
n_runs = parse_n_runs()
max_stats = 20000

# ARL from 100-1100
tau_min, tau_max = 0.94, 1.01
n_tau = 3
# FULL ARL from 100-16000
# tau_min, tau_max = 0.94, 1.07
# n_tau = 14

out_dir = Path(__file__).resolve().parent / "results"
out_dir.mkdir(parents=True, exist_ok=True)

def estimate_gamma_streaming_rffmmd(reference_sample):
    return estimate_gaussian_gamma(reference_sample[:500])

def estimate_rl_moments_h0_mmdew(
    tau,
    *,
    d,
    ref_size,
    burn_in_stats,
    n_runs,
    max_stats,
    seed=12345,
):
    run_lengths = np.empty(n_runs, dtype=int)
    censored = 0

    for r in tqdm(range(n_runs), desc=f"RL moments @ tau={tau:.5f}", leave=False):
        reference_rng, monitoring_rng = make_gaussian_rngs(
            base_seed=seed, d=d, run_index=r
        )
        ref = reference_rng.normal(size=(ref_size, d))
        gamma = estimate_gamma_streaming_rffmmd(ref)
        det = MMDEW(gamma=gamma)

        if burn_in_stats > ref.shape[0]:
            raise ValueError(
                f"Need at least {burn_in_stats} reference observations for "
                f"burn-in, got {ref.shape[0]}"
            )

        for x in ref[:burn_in_stats]:
            det.insert(x)

        hit = False

        for post_burn in range(1, max_stats + 1):
            x = monitoring_rng.normal(size=d)
            det.insert(x)
            s = det.stats[-1]
            if s > tau:
                run_lengths[r] = post_burn
                hit = True
                break

        if not hit:
            censored += 1
            run_lengths[r] = max_stats

    arl_hat = run_lengths.mean()
    var_hat = run_lengths.var(ddof=1) if n_runs > 1 else 0.0
    std_hat = np.sqrt(var_hat)
    se_mean = std_hat / np.sqrt(n_runs) if n_runs > 0 else np.nan
    censored_rate = censored / n_runs if n_runs > 0 else np.nan

    return arl_hat, var_hat, std_hat, se_mean, censored_rate

taus = np.linspace(tau_min, tau_max, n_tau)

rows = []
out_csv = out_dir / f"MMDEW-ARL-d{d}-nrun{n_runs}.csv"

for k, tau in enumerate(taus, start=1):

    arl_hat, var_hat, std_hat, se_mean, cens = estimate_rl_moments_h0_mmdew(
        tau,
        d=d,
        ref_size=ref_size,
        burn_in_stats=burn_in_stats,
        n_runs=n_runs,
        max_stats=max_stats,
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
        "ref_size": int(ref_size),
    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Saved CSV:", out_csv)

df_tau = pd.DataFrame(rows)
max_cens = df_tau["censored_rate"].max()
if max_cens > 0.02:
    print(f"WARNING: censoring up to {max_cens:.3f}. Consider increasing max_stats.")
