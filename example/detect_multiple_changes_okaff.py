"""Detect multiple changes in one numeric data stream with OKAFF.

The first 50 observations form the initial burn-in period. After each alarm,
the detector resets and uses the next 50 observations to learn the new baseline
before monitoring resumes. Consequently, two reported alarms are always more
than 50 observations apart.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detect_single_change_okaff import (
    BURN_IN,
    DEFAULT_N_RFF,
    DEFAULT_QUANTILE,
    DEFAULT_SEED,
    ETA,
    LAMBDA0,
    LAMBDA1,
    MONITORING_THRESHOLD_RATE,
    REFERENCE_THRESHOLD_RATE,
    OKAFF,
    OKAFFAdaptiveThreshold,
    estimate_gaussian_gamma,
    load_stream,
    make_rff,
    okaff_gaussian_theory_moments,
)

RESULT_COLUMNS = [
    "change_number",
    "detected_change_point",
    "monitoring_run_length",
    "burn_in_start",
    "burn_in_end",
    "statistic",
    "threshold_lower",
    "threshold_upper",
    "alarm_side",
    "q",
    "burn_in",
    "dimension",
    "n_rff",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path, help="CSV containing one numeric observation per row")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--n-rff", type=int, default=DEFAULT_N_RFF)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-csv", type=Path, help="Optional detected-change summary CSV")
    args = parser.parse_args()
    if not 0.5 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.5 and 1")
    if args.n_rff <= 0:
        parser.error("--n-rff must be a positive integer")
    return args


def initialize_monitor(reference, *, quantile, n_rff, seed):
    dimension = reference.shape[1]
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
    return detector, threshold


def detect_multiple_changes(stream, *, quantile, n_rff, seed):
    results = []
    burn_in_start = 0
    segment_number = 0
    n_observations, dimension = stream.shape

    while burn_in_start + BURN_IN < n_observations:
        segment_number += 1
        burn_in_end = burn_in_start + BURN_IN
        reference = stream[burn_in_start:burn_in_end]
        detector, threshold = initialize_monitor(
            reference,
            quantile=quantile,
            n_rff=n_rff,
            seed=np.random.SeedSequence([seed, segment_number]),
        )

        alarm_found = False
        for index in range(burn_in_end, n_observations):
            detector.update(stream[index])
            statistic = float(detector.statistic)
            if threshold.update(statistic):
                lower, upper = threshold.interval
                results.append(
                    {
                        "change_number": len(results) + 1,
                        "detected_change_point": index + 1,
                        "monitoring_run_length": index - burn_in_end + 1,
                        "burn_in_start": burn_in_start + 1,
                        "burn_in_end": burn_in_end,
                        "statistic": statistic,
                        "threshold_lower": lower,
                        "threshold_upper": upper,
                        "alarm_side": "lower" if statistic < lower else "upper",
                        "q": quantile,
                        "burn_in": BURN_IN,
                        "dimension": dimension,
                        "n_rff": n_rff,
                    }
                )
                burn_in_start = index + 1
                alarm_found = True
                break

        if not alarm_found:
            break

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def main():
    args = parse_args()
    stream = load_stream(args.data_csv)
    results = detect_multiple_changes(
        stream,
        quantile=args.quantile,
        n_rff=args.n_rff,
        seed=args.seed,
    )

    if results.empty:
        print("No change point was detected after the initial 50-observation burn-in.")
    else:
        print(results.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
