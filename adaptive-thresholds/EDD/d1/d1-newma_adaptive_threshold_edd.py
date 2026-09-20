import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "src" / "kerneldetector.py").is_file())
SRC_DIR = REPO_ROOT / "src"

for module_dir in (REPO_ROOT / "adaptive-thresholds" / "EDD", SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from adaptivethreholds import NEWMAAdaptiveThreshold
from kerneldetector import NEWMA, estimate_gaussian_gamma
from onlinecp.algos import select_optimal_parameters

print("Using source directory:", SRC_DIR)
print("Script/output parent directory:", SCRIPT_DIR)

QUANTILES = (0.95,)
D = 1

REFERENCE_SIZE = 50

ADD_BURN_IN = 0
TOTAL_BURN_IN = REFERENCE_SIZE + ADD_BURN_IN

WINDOW_SIZE = 50

N_RFF = None

PRE_CHANGE_LENGTH = 100
EDD_CONDITIONING = "conditional_survival"
N_RUNS = 500
MAX_DELAY = 1000
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

    if isinstance(n_rff, bool) or not isinstance(
        n_rff, (int, np.integer)
    ):
        raise TypeError("N_RFF must be None or a positive integer")

    if n_rff <= 0:
        raise ValueError("N_RFF must be positive")

    return int(n_rff)


N_RFF_MODE = "automatic" if N_RFF is None else "user"
N_RFF = resolve_n_rff(N_RFF)

SCENARIOS = [
    *[
        {"name": f"MeanShift(mu={mu:g})", "family": "mean_shift", "parameter": float(mu)}
        for mu in (3, -4)
    ],
    *[
        {"name": f"CovDiag(var={variance:g})", "family": "variance_shift", "parameter": float(variance)}
        for variance in (0.1, 6)
    ],
    *[
        {"name": f"Laplace(scale={scale:.1f})", "family": "laplace", "parameter": float(scale)}
        for scale in (0.3, 3.0)
    ],
    *[
        {"name": f"Uniform(scale={scale:.1f})", "family": "uniform", "parameter": float(scale)}
        for scale in (0.5, 5.0)
    ],
    *[
        {"name": f"Exponential(scale={scale:.1f})", "family": "exponential", "parameter": float(scale)}
        for scale in (1.0, 8.0)
    ],
]



def filename_number(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def quantiles_tag():
    return "q" + "-".join(filename_number(q) for q in QUANTILES)


OUTPUT_DIR = SCRIPT_DIR / "results" / f"NEWMA-adaptive-thresholds-EDD-d{D}-nrun{N_RUNS}"




SUMMARY_CSV = OUTPUT_DIR / f"NEWMA-adaptive-thresholds-EDD-d{D}.csv"
print(
    f"BIG_LAMBDA={BIG_LAMBDA:.8f}, "
    f"SMALL_LAMBDA={SMALL_LAMBDA:.8f}, "
    f"N_RFF={N_RFF} ({N_RFF_MODE}), "
    f"REFERENCE_SIZE={REFERENCE_SIZE}, "
    f"ADD_BURN_IN={ADD_BURN_IN}, "
    f"TOTAL_BURN_IN={TOTAL_BURN_IN}, "
    f"BURNIN_THRESHOLD_RATE={BURNIN_THRESHOLD_RATE}, "
    f"MONITORING_THRESHOLD_RATE={SMALL_LAMBDA:.8f}, "
    f"PRE_CHANGE_LENGTH={PRE_CHANGE_LENGTH}"
)
print("Quantiles tag:", quantiles_tag())

def make_rff(rng, gamma):
    frequencies = rng.normal(
        scale=math.sqrt(2.0 * gamma), size=(N_RFF, D)
    )

    def feature(x):
        projection = np.einsum(
            "d,md->m", np.asarray(x, dtype=float), frequencies,
            optimize=False,
        )
        return np.concatenate((np.cos(projection), np.sin(projection))) / math.sqrt(N_RFF)

    return feature


def make_detector_and_thresholds(
    rng,
    reference,
    additional_burn_in,
):
    """Initialise NEWMA with 50 reference and 200 extra burn-in observations."""
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




def simulate_one_edd_run(rng, scenario, *, run_index=0):
    from edd_fixed_runs import make_reference_data, make_monitoring_data
    reference = make_reference_data(
        seed=SEED, scenario_name=scenario["name"], run_index=run_index,
        d=D, reference_size=REFERENCE_SIZE,
    )
    prechange, postchange = make_monitoring_data(
        seed=SEED, scenario=scenario, run_index=run_index,
        d=D, pre_length=PRE_CHANGE_LENGTH, post_length=MAX_DELAY,
    )
    additional_burn_in = rng.standard_normal((ADD_BURN_IN, D))

    detector, thresholds = make_detector_and_thresholds(
        rng,
        reference,
        additional_burn_in,
    )

    survived_prechange = {q: True for q in QUANTILES}
    false_alarm_times = {q: None for q in QUANTILES}

    for prechange_time, observation in enumerate(
        prechange, start=1
    ):
        statistic = detector.update_stat(observation)

        for q, threshold in thresholds.items():
            if threshold.update(statistic) and survived_prechange[q]:
                survived_prechange[q] = False
                false_alarm_times[q] = prechange_time

    delays = {q: MAX_DELAY for q in QUANTILES}
    censored = {q: True for q in QUANTILES}
    waiting = {
        q for q in QUANTILES
        if survived_prechange[q]
    }


    for delay, observation in enumerate(postchange, start=1):
        statistic = detector.update_stat(observation)

        for q in tuple(waiting):
            if thresholds[q].update(statistic):
                delays[q] = delay
                censored[q] = False
                waiting.remove(q)

        if not waiting:
            break

    return delays, censored, survived_prechange, false_alarm_times

def atomic_write_csv(frame, path):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)




def estimate_edd():
    from edd_fixed_runs import estimate_fixed_run_edd
    return estimate_fixed_run_edd(
        simulate_one_edd_run=simulate_one_edd_run,
        scenarios=SCENARIOS,
        quantiles=QUANTILES,
        n_runs=N_RUNS,
        seed=SEED,
        algorithm="NEWMA",
        max_delay=MAX_DELAY,
        atomic_write_csv=atomic_write_csv,
        runs_csv=None,
        summary_csv=SUMMARY_CSV,
    )

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
    OUTPUT_DIR = SCRIPT_DIR / "results" / f"NEWMA-adaptive-thresholds-EDD-d{D}-nrun{N_RUNS}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Output directory:", OUTPUT_DIR)
    SUMMARY_CSV = OUTPUT_DIR / f"NEWMA-adaptive-thresholds-EDD-d{D}.csv"
    _, edd_summary = estimate_edd()

    print(edd_summary.to_string(index=False))
    print("Saved EDD summary CSV:", SUMMARY_CSV)
