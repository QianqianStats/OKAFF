"""Detect the first change in one numeric data stream with OKAFF.

The input CSV must contain more than 50 rows and only numeric feature columns.
Rows 1--50 form the reference/burn-in period. Monitoring begins at row 51 and
stops at the first adaptive-threshold alarm.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent
    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
    if (parent / "src" / "kerneldetector.py").is_file()
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptivethreholds import OKAFFAdaptiveThreshold, okaff_gaussian_theory_moments
from kerneldetector import OKAFF, estimate_gaussian_gamma

BURN_IN = 50
DEFAULT_QUANTILE = 0.92
DEFAULT_N_RFF = 500
DEFAULT_SEED = 2026
LAMBDA0 = 0.999
LAMBDA1 = 0.999
ETA = 1e-3
REFERENCE_THRESHOLD_RATE = 0.1
MONITORING_THRESHOLD_RATE = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path, help="CSV containing one numeric observation per row")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--n-rff", type=int, default=DEFAULT_N_RFF)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-csv", type=Path, help="Optional one-row detection summary CSV")
    args = parser.parse_args()
    if not 0.5 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.5 and 1")
    if args.n_rff <= 0:
        parser.error("--n-rff must be a positive integer")
    return args


def load_stream(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.shape[1] == 0:
        raise ValueError("Input CSV has no feature columns")
    try:
        stream = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Every input column must be numeric") from error
    if len(stream) <= BURN_IN:
        raise ValueError(
            f"Input stream must contain more than {BURN_IN} observations; got {len(stream)}"
        )
    if not np.isfinite(stream).all():
        raise ValueError("Input CSV contains missing or non-finite values")
    return stream


def make_rff(rng, *, gamma, dimension, n_rff):
    frequencies = rng.normal(
        scale=math.sqrt(2.0 * gamma),
        size=(n_rff, dimension),
    )

    def feature(observation):
        projection = np.einsum(
            "d,md->m",
            np.asarray(observation, dtype=float),
            frequencies,
            optimize=False,
        )
        return np.concatenate((np.cos(projection), np.sin(projection))) / math.sqrt(n_rff)

    return feature


def detect_first_change(stream, *, quantile, n_rff, seed):
    reference = stream[:BURN_IN]
    dimension = stream.shape[1]
    gamma = estimate_gaussian_gamma(reference)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False, ddof=1))
    theory_a, theory_sv = okaff_gaussian_theory_moments(
        gamma=gamma,
        covariance=covariance,
        lambda_value=LAMBDA0,
    )

    detector = OKAFF(
        lambda0=LAMBDA0,
        lambda1=LAMBDA1,
        eta=ETA,
        feat_func=make_rff(
            np.random.default_rng(seed),
            gamma=gamma,
            dimension=dimension,
            n_rff=n_rff,
        ),
        thresholding_method="fixed",
        fixed_threshold=np.inf,
        store_values=False,
    )
    threshold = OKAFFAdaptiveThreshold(
        alpha=REFERENCE_THRESHOLD_RATE,
        quantile=quantile,
        t0=BURN_IN + 1,
        initialization="theory_A_SV",
        theory_A=theory_a,
        theory_SV=theory_sv,
    )

    for observation in reference:
        detector.update(observation)
        threshold.update(detector.statistic)

    threshold.alpha = MONITORING_THRESHOLD_RATE
    for zero_based_index in range(BURN_IN, len(stream)):
        detector.update(stream[zero_based_index])
        statistic = float(detector.statistic)
        alarm = threshold.update(statistic)
        if alarm:
            lower, upper = threshold.interval
            return {
                "detected": True,
                "detected_change_point": zero_based_index + 1,
                "monitoring_run_length": zero_based_index + 1 - BURN_IN,
                "statistic": statistic,
                "threshold_lower": lower,
                "threshold_upper": upper,
                "q": quantile,
                "burn_in": BURN_IN,
                "n_observations": len(stream),
                "dimension": dimension,
                "n_rff": n_rff,
            }

    lower, upper = threshold.interval
    return {
        "detected": False,
        "detected_change_point": pd.NA,
        "monitoring_run_length": len(stream) - BURN_IN,
        "statistic": float(detector.statistic),
        "threshold_lower": lower,
        "threshold_upper": upper,
        "q": quantile,
        "burn_in": BURN_IN,
        "n_observations": len(stream),
        "dimension": dimension,
        "n_rff": n_rff,
    }


def main():
    args = parse_args()
    result = detect_first_change(
        load_stream(args.data_csv),
        quantile=args.quantile,
        n_rff=args.n_rff,
        seed=args.seed,
    )
    result_frame = pd.DataFrame([result])
    print(result_frame.to_string(index=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
