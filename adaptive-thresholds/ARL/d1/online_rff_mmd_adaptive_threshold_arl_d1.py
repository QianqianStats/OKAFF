import argparse
import csv
import math
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "kerneldetector.py").is_file())
for module_dir in (CODE_ROOT / "src", CODE_ROOT / "adaptive-thresholds" / "ARL", Path(__file__).resolve().parent):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptivethreholds import OnlineRFFMMDAdaptiveThreshold
from kerneldetector import (
    GaussianKernel,
    OnlineRFFMMD,
    estimate_gaussian_gamma,
)


QUANTILES = (0.99994,)
D = 1
N_RFF = 500
REFERENCE_SIZE = 50
THRESHOLD_RATE = 0.01
REFERENCE_THRESHOLD_RATE = 0.1
N_RUNS = 500
MAX_RUN_LENGTH = 10000
SEED = 2026

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / f"online_rff_mmd-adaptive-thresholds-ARL-d{D}-nrun{N_RUNS}"


def make_detector(rng, gamma):
    """Construct a reproducibly seeded Online RFF MMD detector."""
    detector = OnlineRFFMMD(
        kernel=GaussianKernel(gamma),
        d=D,
        num_omegas=N_RFF,
    )
    detector.omegas = rng.normal(
        scale=math.sqrt(2.0 * gamma),
        size=(N_RFF, D),
    )
    return detector


def make_detector_and_thresholds(rng, reference):
    """Construct and initialize the detector with observations 1,...,50."""
    gamma = estimate_gaussian_gamma(reference)
    detector = make_detector(rng, gamma)
    thresholds = {
        q: OnlineRFFMMDAdaptiveThreshold(
            rho=REFERENCE_THRESHOLD_RATE,
            quantile=q,
            t0=REFERENCE_SIZE + 1,
        )
        for q in QUANTILES
    }

    for observation in reference:
        detector.insert(observation)
        statistic = detector.statistic()
        for threshold in thresholds.values():
            threshold.update(statistic)

    for threshold in thresholds.values():
        threshold.rho = THRESHOLD_RATE

    return detector, thresholds


def simulate_one_run(rng, *, run_index=0):
    """Return the first-alarm run length after the reference period."""
    from shared_arl_data import make_reference_data, make_monitoring_rng
    monitoring_rng = make_monitoring_rng(seed=SEED, run_index=run_index, d=D)
    reference = make_reference_data(
        seed=SEED, run_index=run_index, d=D, reference_size=REFERENCE_SIZE,
    )
    detector, thresholds = make_detector_and_thresholds(rng, reference)

    run_lengths = {q: MAX_RUN_LENGTH for q in QUANTILES}
    censored = {q: True for q in QUANTILES}
    waiting_for_alarm = set(QUANTILES)

    for monitoring_time in range(1, MAX_RUN_LENGTH + 1):
        detector.insert(monitoring_rng.standard_normal(D))
        statistic = detector.statistic()

        for q in tuple(waiting_for_alarm):
            if thresholds[q].update(statistic):
                run_lengths[q] = monitoring_time
                censored[q] = False
                waiting_for_alarm.remove(q)

        if not waiting_for_alarm:
            break

    return run_lengths, censored


def calculate_summary(run_lengths, censored, completed_runs):
    """Calculate ARL statistics using completed simulations only."""
    summary = []
    for q in QUANTILES:
        values = run_lengths[q][:completed_runs].astype(float)
        standard_deviation = (
            values.std(ddof=1) if completed_runs > 1 else 0.0
        )
        summary.append(
            {
                "q": q,
                "ARL": values.mean(),
                "standard_error": (
                    standard_deviation / math.sqrt(completed_runs)
                ),
                "censored_rate": censored[q][:completed_runs].mean(),
                "completed_runs": completed_runs,
            }
        )
    return summary


def save_checkpoint(completed_runs, run_lengths, censored):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_DIR / f"online_rff_mmd-adaptive-thresholds-ARL-d{D}.csv"
    summary_temporary = (
        OUTPUT_DIR / f"online_rff_mmd-adaptive-thresholds-ARL-d{D}.csv.tmp"
    )


    summary = calculate_summary(run_lengths, censored, completed_runs)
    with summary_temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    summary_temporary.replace(summary_file)
    return summary_file


def estimate_arl():
    rng = np.random.default_rng(SEED)
    run_lengths = {
        q: np.full(N_RUNS, MAX_RUN_LENGTH, dtype=int)
        for q in QUANTILES
    }
    censored = {
        q: np.ones(N_RUNS, dtype=bool)
        for q in QUANTILES
    }

    for run in range(N_RUNS):
        lengths_one_run, censored_one_run = simulate_one_run(rng, run_index=run)
        for q in QUANTILES:
            run_lengths[q][run] = lengths_one_run[q]
            censored[q][run] = censored_one_run[q]

        save_checkpoint(run + 1, run_lengths, censored)
        print(f"Completed {run + 1}/{N_RUNS}", flush=True)

    return calculate_summary(run_lengths, censored, N_RUNS)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the adaptive-threshold simulation.")
    parser.add_argument("--n-runs", "--n_runs", type=int, default=N_RUNS, help="Number of simulation runs (default: %(default)s)")
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    return args


if __name__ == "__main__":
    args = parse_args()
    N_RUNS = args.n_runs
    OUTPUT_DIR = Path(__file__).resolve().parent / "results" / f"online_rff_mmd-adaptive-thresholds-ARL-d{D}-nrun{N_RUNS}"
    summary = estimate_arl()

    print("\nOnline RFF MMD adaptive-threshold ARL under N(0, I_d)")
    print("q       ARL          SE       censored")
    for row in summary:
        print(
            f"{row['q']:<5.3f}"
            f"{row['ARL']:>11.2f}"
            f"{row['standard_error']:>11.2f}"
            f"{row['censored_rate']:>12.3f}"
        )

    print(
        f"\nSaved: "
        f"{OUTPUT_DIR / f'online_rff_mmd-adaptive-thresholds-ARL-d{D}.csv'}"
    )
