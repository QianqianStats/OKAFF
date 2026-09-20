import argparse
import sys
from pathlib import Path




SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "src" / "kerneldetector.py").is_file())
SRC_DIR = REPO_ROOT / "src"

if not SRC_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected source directory does not exist: {SRC_DIR}\n"
        "Check the repository layout or adjust REPO_ROOT."
    )

for module_dir in (REPO_ROOT / "adaptive-thresholds" / "EDD", SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptivethreholds import MMDEWAdaptiveThreshold
from kerneldetector import MMDEW, estimate_gaussian_gamma

print("Script/output parent directory:", SCRIPT_DIR)
print("Repository root:", REPO_ROOT)
print("Using source directory:", SRC_DIR)

QUANTILES = (0.9995,)

D = 1
REFERENCE_SIZE = 50
PRE_CHANGE_LENGTH = 100

THRESHOLD_RATE = 0.01
REFERENCE_THRESHOLD_RATE = 0.1
MIN_ELEMENTS_PER_WINDOW = 1
MAX_WINDOWS = 0
COOLDOWN = 500
MMDEW_SEED = 1234

EDD_CONDITIONING = "conditional_survival"
N_RUNS = 500
MAX_DELAY = 1000
SEED = 2026


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


OUTPUT_DIR = SCRIPT_DIR / "results" / f"MMDEW-adaptive-thresholds-EDD-d{D}-nrun{N_RUNS}"




SUMMARY_CSV = OUTPUT_DIR / f"MMDEW-adaptive-thresholds-EDD-d{D}.csv"

print("Quantiles tag:", quantiles_tag())

def make_detector_and_thresholds(reference):
    """Construct MMDEW and initialise it with the reference observations."""
    gamma = estimate_gaussian_gamma(reference)

    detector = MMDEW(
        gamma=gamma,
        min_elements_per_window=MIN_ELEMENTS_PER_WINDOW,
        max_windows=MAX_WINDOWS,
        cooldown=COOLDOWN,
        seed=MMDEW_SEED,
    )

    thresholds = {
        q: MMDEWAdaptiveThreshold(
            rho=REFERENCE_THRESHOLD_RATE,
            quantile=q,
            t0=REFERENCE_SIZE + 1,
        )
        for q in QUANTILES
    }

    for observation in reference:
        detector.insert(observation)
        statistic = detector.stats[-1]
        for threshold in thresholds.values():
            threshold.update(statistic)

    for threshold in thresholds.values():
        threshold.rho = THRESHOLD_RATE

    return detector, thresholds




def simulate_one_edd_run(_rng, scenario, *, run_index=0):
    """Simulate one attempted path and return quantile-specific survival and delay."""
    from edd_fixed_runs import make_reference_data, make_monitoring_data
    reference = make_reference_data(
        seed=SEED, scenario_name=scenario["name"], run_index=run_index,
        d=D, reference_size=REFERENCE_SIZE,
    )
    prechange, postchange = make_monitoring_data(
        seed=SEED, scenario=scenario, run_index=run_index,
        d=D, pre_length=PRE_CHANGE_LENGTH, post_length=MAX_DELAY,
    )
    detector, thresholds = make_detector_and_thresholds(reference)

    survived_prechange = {q: True for q in QUANTILES}
    false_alarm_times = {q: None for q in QUANTILES}

    for prechange_time, observation in enumerate(
        prechange, start=1
    ):
        detector.insert(observation)
        statistic = detector.stats[-1]

        for q, threshold in thresholds.items():
            if threshold.update(statistic) and survived_prechange[q]:
                survived_prechange[q] = False
                false_alarm_times[q] = prechange_time

    delays = {q: MAX_DELAY for q in QUANTILES}
    censored = {q: True for q in QUANTILES}
    waiting = {q for q in QUANTILES if survived_prechange[q]}


    for delay, observation in enumerate(postchange, start=1):
        detector.insert(observation)
        statistic = detector.stats[-1]

        for q in tuple(waiting):
            if thresholds[q].update(statistic):
                delays[q] = delay
                censored[q] = False
                waiting.remove(q)

        if not waiting:
            break

    return delays, censored, survived_prechange, false_alarm_times

def atomic_write_csv(frame, path):
    """Write a CSV through a temporary file to avoid partial checkpoints."""
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)




def estimate_edd():
    """Estimate conditional EDD for all scenarios and quantiles."""
    from edd_fixed_runs import estimate_fixed_run_edd
    return estimate_fixed_run_edd(
        simulate_one_edd_run=simulate_one_edd_run,
        scenarios=SCENARIOS,
        quantiles=QUANTILES,
        n_runs=N_RUNS,
        seed=SEED,
        algorithm="MMDEW",
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
    OUTPUT_DIR = SCRIPT_DIR / "results" / f"MMDEW-adaptive-thresholds-EDD-d{D}-nrun{N_RUNS}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Output directory:", OUTPUT_DIR)
    SUMMARY_CSV = OUTPUT_DIR / f"MMDEW-adaptive-thresholds-EDD-d{D}.csv"
    _, edd_summary = estimate_edd()

    print(edd_summary.to_string(index=False))
    print("Saved EDD summary CSV:", SUMMARY_CSV)
