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

from adaptivethreholds import NEWMAAdaptiveThreshold
from kerneldetector import NEWMA, estimate_gaussian_gamma
from onlinecp.algos import select_optimal_parameters

QUANTILES = (0.95,)
D = 1

REFERENCE_SIZE = 50

ADD_BURN_IN = 0
TOTAL_BURN_IN = REFERENCE_SIZE + ADD_BURN_IN

WINDOW_SIZE = 50

N_RFF = None
N_RUNS = 500
MAX_RUN_LENGTH = 10000
SEED = 2026

BIG_LAMBDA, SMALL_LAMBDA = select_optimal_parameters(WINDOW_SIZE)

BURNIN_THRESHOLD_RATE = 0.1

def resolve_n_rff(n_rff):
    """Return a user-selected RFF count or the NEWMA heuristic count."""
    if n_rff is None:
        return int(
            (1.0 / 4.0)
            / (SMALL_LAMBDA + BIG_LAMBDA) ** 2
        )
    if isinstance(n_rff, bool) or not isinstance(n_rff, (int, np.integer)):
        raise TypeError("N_RFF must be None or a positive integer")
    if n_rff <= 0:
        raise ValueError("N_RFF must be positive")
    return int(n_rff)

N_RFF_MODE = "automatic" if N_RFF is None else "user"
N_RFF = resolve_n_rff(N_RFF)

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / f"NEWMA-adaptive-thresholds-ARL-d{D}-nrun{N_RUNS}"

def make_rff(rng, gamma):
    """Draw a Gaussian random Fourier feature map."""
    frequencies = rng.normal(
        scale=math.sqrt(2.0 * gamma),
        size=(N_RFF, D),
    )

    def feature(x):
        projection = np.einsum(
            "d,md->m",
            np.asarray(x, dtype=float),
            frequencies,
            optimize=False,
        )
        return np.concatenate(
            (np.cos(projection), np.sin(projection))
        ) / math.sqrt(N_RFF)

    return feature

def make_detector_and_thresholds(rng, reference, additional_burn_in):
    """Initialise NEWMA using separate reference and post-reference threshold rates."""
    gamma = estimate_gaussian_gamma(reference)

    detector = NEWMA(
        init_sample=reference[0],
        forget_factor=BIG_LAMBDA,
        forget_factor2=SMALL_LAMBDA,
        feat_func=make_rff(rng, gamma),
        thresholding_method="fixed",
        fixed_threshold=np.inf,
        store_values=False,
    )

    thresholds = {
        q: NEWMAAdaptiveThreshold(
            slow_lambda=BURNIN_THRESHOLD_RATE,
            quantile=q,
            t0=TOTAL_BURN_IN + 1,
        )
        for q in QUANTILES
    }

    for threshold in thresholds.values():
        threshold.update(0.0)

    for observation in reference[1:]:
        statistic = detector.update_stat(observation)
        for threshold in thresholds.values():
            threshold.update(statistic)

    for observation in additional_burn_in:
        statistic = detector.update_stat(observation)
        for threshold in thresholds.values():
            threshold.update(statistic)

    for threshold in thresholds.values():
        threshold.rho = float(SMALL_LAMBDA)
        threshold.slow_lambda = threshold.rho

    return detector, thresholds

def simulate_one_run(rng, *, run_index=0):
    """Return the first-alarm run length after burn-in."""
    from shared_arl_data import make_reference_data, make_monitoring_rng
    monitoring_rng = make_monitoring_rng(seed=SEED, run_index=run_index, d=D)
    reference = make_reference_data(
        seed=SEED, run_index=run_index, d=D, reference_size=REFERENCE_SIZE,
    )
    additional_burn_in = rng.standard_normal((ADD_BURN_IN, D))
    detector, thresholds = make_detector_and_thresholds(
        rng,
        reference,
        additional_burn_in,
    )

    run_lengths = {q: MAX_RUN_LENGTH for q in QUANTILES}
    censored = {q: True for q in QUANTILES}
    waiting_for_alarm = set(QUANTILES)

    for monitoring_time in range(1, MAX_RUN_LENGTH + 1):
        statistic = detector.update_stat(monitoring_rng.standard_normal(D))

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
    """Atomically save the ARL summary after each completed Monte Carlo run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_DIR / f"NEWMA-adaptive-thresholds-ARL-d{D}.csv"
    summary_temporary = OUTPUT_DIR / f"NEWMA-adaptive-thresholds-ARL-d{D}.csv.tmp"

    summary = calculate_summary(run_lengths, censored, completed_runs)
    with summary_temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    summary_temporary.replace(summary_file)

def estimate_arl():
    """Run the Monte Carlo experiment with per-run checkpoints."""
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
    OUTPUT_DIR = Path(__file__).resolve().parent / "results" / f"NEWMA-adaptive-thresholds-ARL-d{D}-nrun{N_RUNS}"
    print(
        f"BIG_LAMBDA={BIG_LAMBDA:.8f}, "
        f"SMALL_LAMBDA={SMALL_LAMBDA:.8f}, "
        f"N_RFF={N_RFF} ({N_RFF_MODE}), "
        f"REFERENCE_SIZE={REFERENCE_SIZE}, "
        f"ADD_BURN_IN={ADD_BURN_IN}, "
        f"TOTAL_BURN_IN={TOTAL_BURN_IN}, "
        f"BURN_IN_THRESHOLD_RATE={BURNIN_THRESHOLD_RATE}, "
        f"MONITORING_THRESHOLD_RATE={SMALL_LAMBDA:.8f}"
    )
    summary = estimate_arl()

    print("\nNEWMA adaptive-threshold ARL under N(0, I_d)")
    print("q       ARL          SE       censored")
    for row in summary:
        print(
            f"{row['q']:<5.3f}"
            f"{row['ARL']:>11.2f}"
            f"{row['standard_error']:>11.2f}"
            f"{row['censored_rate']:>12.3f}"
        )

    print(f"\nSaved: {OUTPUT_DIR / f'NEWMA-adaptive-thresholds-ARL-d{D}.csv'}")
