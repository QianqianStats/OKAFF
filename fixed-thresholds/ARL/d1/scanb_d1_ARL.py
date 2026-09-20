import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
import math
from tqdm import tqdm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


from kerneldetector import ScanBStatistic, estimate_gaussian_gamma
from shared_gaussian_streams import make_gaussian_rngs


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs

out_dir = Path(__file__).resolve().parent / "results"
out_dir.mkdir(parents=True, exist_ok=True)

def estimate_rl_moments_h0_scanb_tau(
    tau: float,
    *,
    d: int,
    ref_size: int,
    burn_in_stats: int,
    n_runs: int,
    max_stats: int,
    B0: int,
    N: int,
    seed: int = 12345,
):
    run_lengths = np.full(n_runs, max_stats, dtype=int)
    censored = np.ones(n_runs, dtype=bool)

    for r in tqdm(range(n_runs), desc=f"ARL runs @ tau={tau:.4f}"):
        reference_rng, monitoring_rng = make_gaussian_rngs(
            base_seed=seed, d=d, run_index=r
        )
        ref = reference_rng.normal(size=(ref_size, d))

        gamma = estimate_gaussian_gamma(ref[:500])

        det = ScanBStatistic(
            reference_sample=ref,
            B0=B0,
            N=N,
            gamma=gamma,
        )

        post_burn = 0

        for t in range(burn_in_stats + max_stats):
            x = monitoring_rng.normal(size=d)
            det.insert(x)

            if t + 1 <= burn_in_stats:
                continue

            post_burn += 1
            s = det.statistic()

            if s > tau:
                run_lengths[r] = post_burn
                censored[r] = False
                break

    arl_hat = float(run_lengths.mean())
    var_hat = float(run_lengths.var(ddof=1)) if n_runs > 1 else 0.0
    std_hat = math.sqrt(var_hat)
    se_hat = std_hat / math.sqrt(n_runs) if n_runs > 0 else float("nan")
    censored_rate = float(censored.mean()) if n_runs > 0 else float("nan")

    return {
        "ARL_hat": arl_hat,
        "RL_var_hat": var_hat,
        "RL_std_hat": std_hat,
        "ARL_se_hat": se_hat,
        "censored_rate": censored_rate,
    }

if __name__ == "__main__":

    # ARL from 100-1300
    thresholds = [1.1, 2.8, 3.2]
    # full ARL from 100-14000
    # thresholds = [1.1, 1.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    d = 1
    ref_size = 250
    burn_in_stats = 0
    n_runs = parse_n_runs()
    max_stats = 10000
    B0 = 50
    N = 5

    rows = []
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    for tau in thresholds:
        metrics = estimate_rl_moments_h0_scanb_tau(
            tau,
            d=d,
            ref_size=ref_size,
            burn_in_stats=burn_in_stats,
            n_runs=n_runs,
            max_stats=max_stats,
            B0=B0,
            N=N,
            seed=12345,
        )

        row = {
            "threshold": float(tau),
            **metrics,
            "n_runs": int(n_runs),
            "d": int(d),
            "ref_size": int(ref_size),
            "B0": int(B0),
            "N": int(N),
        }
        rows.append(row)

        print(
            f"done threshold={tau:.4f}, "
            f"ARL_hat={metrics['ARL_hat']:.3f}, "
            f"ARL_se_hat={metrics['ARL_se_hat']:.3f}, "
            f"censored_rate={metrics['censored_rate']:.3f}"
        )

    df = pd.DataFrame(rows)

    print("\nFinal summary:")
    print(df[["threshold", "ARL_hat", "ARL_se_hat", "censored_rate"]])

    out_csv = out_dir / f"Scanb-ARL-d{d}-nrun{n_runs}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    max_cens = df["censored_rate"].max()
    if max_cens > 0.02:
        print(f"WARNING: censoring up to {max_cens:.3f}. Consider increasing max_stats.")
