"""Estimate OKAFF ARL separately for 500, 1000 and 5000 random features.

Run from your project with this file in its original d1 project directory.
The feature-specific summaries are saved in d1/results/.
"""

import argparse
import csv
import sys
from pathlib import Path

CODE_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / "src" / "kerneldetector.py").is_file()),
    None,
)
if CODE_ROOT is None:
    raise FileNotFoundError(
        "Put this script inside your OKAFF project (which must contain src/kerneldetector.py)."
    )
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "fixed-thresholds"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import numpy as np
from tqdm import tqdm
import onlinecp.utils.feature_functions as feat
from fixedthresholds import (
    default_band_lambda,
    gaussian_kernel_theory_terms,
    theoretical_rejection_band,
)
from kerneldetector import OKAFF, estimate_gaussian_gamma
from shared_gaussian_streams import make_gaussian_rngs

D = 1
NUM_FEATURES = (500, 1000, 5000)
REFERENCE_SIZE = 250
BANDWIDTH_REFERENCE_SIZE = 250
BURN_IN = 250
MAX_STATS = 20000
MEAN = np.zeros(D)
COVARIANCE = np.eye(D)
CHOLESKY = np.linalg.cholesky(COVARIANCE)
SHORT_L_VALUES = (4.5, 7.0, 9.0)
FULL_L_VALUES = (
    4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0,
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
)
BAND_LAMBDA = 0.999
ARL_SEED = 12345
OUT_DIR = Path(__file__).resolve().parent / "results"
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--features", nargs="+", type=int, choices=NUM_FEATURES,
                        default=NUM_FEATURES, help="Feature counts (default: all three)")
    parser.add_argument("--full-grid", action="store_true",
                        help="Use all 16 thresholds, rather than the original three")
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    if len(args.features) != len(set(args.features)):
        parser.error("--features must not contain duplicates")
    return args


def prepare_arl_run(*, num_features, run_index, base_seed):
    rng_ref, rng_null = make_gaussian_rngs(
        base_seed=base_seed, d=D, run_index=run_index
    )
    reference = (
        MEAN + rng_ref.standard_normal(size=(REFERENCE_SIZE, D)) @ CHOLESKY.T
    )
    gamma = estimate_gaussian_gamma(
        reference[:BANDWIDTH_REFERENCE_SIZE],
        max_len=BANDWIDTH_REFERENCE_SIZE,
    )
    sigmasq = 1.0 / (2.0 * gamma)
    frequencies, sigmasq = feat.generate_frequencies(
        num_features, D, choice_sigma="fixed", sigmasq=sigmasq
    )

    kwargs = dict(ALGO_KWARGS)
    kwargs.pop("band_lambda", None)
    detector = OKAFF(
        **kwargs,
        feat_func=lambda x, W=frequencies: feat.fourier_feat(x, W),
        dist_func=lambda value: float(np.vdot(value, value).real),
    )
    theory_terms = gaussian_kernel_theory_terms(COVARIANCE, sigmasq)
    return detector, reference, rng_null, theory_terms


def run_one_arl(*, chart_l, num_features, run_index):
    detector, reference, rng_null, theory_terms = prepare_arl_run(
        num_features=num_features, run_index=run_index, base_seed=ARL_SEED
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


def estimate_arl(*, chart_l, num_features, n_runs):
    delays = np.empty(n_runs, dtype=int)
    censored = np.empty(n_runs, dtype=bool)
    for run_id in tqdm(range(n_runs), desc=f"OKAFF ARL m={num_features} L={chart_l:g}"):
        delay, is_censored = run_one_arl(
            chart_l=chart_l, num_features=num_features, run_index=run_id
        )
        delays[run_id] = delay
        censored[run_id] = is_censored

    variance = float(delays.var(ddof=1)) if n_runs > 1 else 0.0
    std = float(np.sqrt(variance))
    return {
        "chart_L": float(chart_l),
        "arl_hat": float(delays.mean()),
        "arl_var": variance,
        "arl_std": std,
        "arl_se_mean": std / np.sqrt(n_runs),
        "arl_censored_rate": float(censored.mean()),
    }


def save_summary(*, rows, num_features, n_runs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"OKAFF-ARL-d{D}-m{num_features}-nrun{n_runs}.csv"
    fieldnames = (
        "method", "dimension_d", "num_features_m", "ref_size", "burn_in",
        "n_runs", "band_lambda", "chart_L", "arl_hat", "arl_var",
        "arl_std", "arl_se_mean", "arl_censored_rate",
    )
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "method": "OKAFF", "dimension_d": D,
                "num_features_m": num_features, "ref_size": REFERENCE_SIZE,
                "burn_in": BURN_IN, "n_runs": n_runs, "band_lambda": BAND_LAMBDA,
                **row,
            })
    print(f"Saved ARL summary: {output}")
    return output


def main():
    args = parse_args()
    l_values = FULL_L_VALUES if args.full_grid else SHORT_L_VALUES
    for num_features in args.features:
        rows = []
        for chart_l in l_values:
            row = estimate_arl(
                chart_l=chart_l, num_features=num_features, n_runs=args.n_runs
            )
            rows.append(row)
            print(
                f"m={num_features}, L={chart_l:g}: ARL={row['arl_hat']:.2f}, "
                f"SE={row['arl_se_mean']:.2f}, "
                f"censored={row['arl_censored_rate']:.3f}"
            )
        save_summary(rows=rows, num_features=num_features, n_runs=args.n_runs)


if __name__ == "__main__":
    main()
