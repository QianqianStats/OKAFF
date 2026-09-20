import argparse
import csv
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import onlinecp.utils.feature_functions as feat
from fixedthresholds import (
    default_band_lambda,
    gaussian_kernel_theory_terms,
    theoretical_rejection_band,
)
from kerneldetector import OKAFF, estimate_gaussian_gamma
from shared_gaussian_streams import make_gaussian_rngs


def parse_n_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args.n_runs


OUT_DIR = Path(__file__).resolve().parent / "results"

D = 1
MEAN = np.zeros(D)
COVARIANCE = np.eye(D)
CHOLESKY = np.linalg.cholesky(COVARIANCE)
NUM_FEATURES = 500
REFERENCE_SIZE = 250
BANDWIDTH_REFERENCE_SIZE = 250

BURN_IN = 250
MAX_STATS = 10000
N_RUNS = parse_n_runs()
# ARL from 100-1200
L_VALUES = np.array(
    [4.5, 7, 9],
    dtype=float,
)
# full ARL from 70-12000
# L_VALUES = np.array(
#     [4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
#     dtype=float,
# )

BAND_LAMBDA = 0.999
ALGO_KWARGS = {
    "lambda0": 0.999,
    "lambda1": 0.999,
    "band_lambda": BAND_LAMBDA,
    "eta": 1e-3,
    "clip": (1e-3, 0.999),
    "store_lambdas": False,
    "thresholding_method": "fixed",
    "fixed_threshold": np.inf,
    "store_values": False,
}

ARL_SEED = 12345


def prepare_arl_run(*, run_index, base_seed):
    rng_ref, rng_null = make_gaussian_rngs(
        base_seed=base_seed, d=D, run_index=run_index
    )
    reference = (
        MEAN
        + rng_ref.standard_normal(size=(REFERENCE_SIZE, D)) @ CHOLESKY.T
    )

    gamma = estimate_gaussian_gamma(
        reference[:BANDWIDTH_REFERENCE_SIZE],
        max_len=BANDWIDTH_REFERENCE_SIZE,
    )
    sigmasq = 1.0 / (2.0 * gamma)
    frequencies, sigmasq = feat.generate_frequencies(
        NUM_FEATURES,
        D,
        choice_sigma="fixed",
        sigmasq=sigmasq,
    )

    kwargs = dict(ALGO_KWARGS)
    kwargs.pop("band_lambda", None)
    detector = OKAFF(
        **kwargs,
        feat_func=lambda x, W=frequencies: feat.fourier_feat(x, W),
        dist_func=lambda value: float(np.vdot(value, value).real),
    )

    theta_k, eta_k, zeta_k = gaussian_kernel_theory_terms(COVARIANCE, sigmasq)
    return detector, reference, rng_null, (theta_k, eta_k, zeta_k)


def run_one_arl(*, chart_l, run_index):
    detector, reference, rng_null, theory_terms = prepare_arl_run(
        run_index=run_index,
        base_seed=ARL_SEED,
    )
    if BURN_IN > len(reference):
        raise ValueError("BURN_IN cannot exceed REFERENCE_SIZE")

    for observation in reference[:BURN_IN]:
        detector.update_stat(observation)

    band = theoretical_rejection_band(
        L=float(chart_l),
        lambda_value=default_band_lambda(ALGO_KWARGS),
        theta_K=theory_terms[0],
        eta_K=theory_terms[1],
        zeta_K=theory_terms[2],
    )

    for delay in range(1, MAX_STATS + 1):
        statistic = detector.update_stat(
            MEAN + CHOLESKY @ rng_null.standard_normal(D)
        )
        if statistic < band["lo"] or statistic > band["hi"]:
            return delay, False

    return MAX_STATS, True


def estimate_arl(chart_l):
    delays = np.empty(N_RUNS, dtype=int)
    censored = np.empty(N_RUNS, dtype=bool)

    for run_id in tqdm(range(N_RUNS), desc=f"OKAFF ARL L={chart_l:g}"):
        delay, is_censored = run_one_arl(
            chart_l=chart_l,
            run_index=run_id,
        )
        delays[run_id] = delay
        censored[run_id] = is_censored

    variance = float(delays.var(ddof=1)) if N_RUNS > 1 else 0.0
    std = float(np.sqrt(variance))
    return {
        "chart_L": float(chart_l),
        "arl_hat": float(delays.mean()),
        "arl_var": variance,
        "arl_std": std,
        "arl_se_mean": std / np.sqrt(N_RUNS),
        "arl_censored_rate": float(censored.mean()),
    }


def save_summary(rows):
    output = OUT_DIR / f"OKAFF-ARL-d{D}-nrun{N_RUNS}.csv"
    fieldnames = [
        "method",
        "dimension_d",
        "num_features_m",
        "ref_size",
        "burn_in",
        "n_runs",
        "band_lambda",
        "chart_L",
        "arl_hat",
        "arl_var",
        "arl_std",
        "arl_se_mean",
        "arl_censored_rate",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": "OKAFF",
                "dimension_d": D,
                "num_features_m": NUM_FEATURES,
                "ref_size": REFERENCE_SIZE,
                "burn_in": BURN_IN,
                "n_runs": N_RUNS,
                "band_lambda": BAND_LAMBDA,
                "chart_L": row["chart_L"],
                "arl_hat": row["arl_hat"],
                "arl_var": row["arl_var"],
                "arl_std": row["arl_std"],
                "arl_se_mean": row["arl_se_mean"],
                "arl_censored_rate": row["arl_censored_rate"],
            })
    print(f"Saved ARL summary: {output}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for chart_l in L_VALUES:
        row = estimate_arl(float(chart_l))
        rows.append(row)
        print(
            f"L={chart_l:g}: ARL={row['arl_hat']:.2f}, "
            f"SE={row['arl_se_mean']:.2f}, "
            f"censored={row['arl_censored_rate']:.3f}"
        )
    save_summary(rows)


if __name__ == "__main__":
    main()
