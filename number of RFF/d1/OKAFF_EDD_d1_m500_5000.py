"""Estimate OKAFF EDD for m=500, 1000, 5000 and merge matching ARL results.

Place this file in your original d1 project directory. Run the ARL
script first, using the same --n-runs, --features and --full-grid options.
Use --arl-results-dir if your ARL results are not in the standard location.
"""

import argparse
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
import pandas as pd
from kerneldetector import OKAFF
from fixedthresholds import default_band_lambda
from okaffexperiment import run_one_q_prepost_and_edd, comparison_rows_for_q


class NegativeMeanPostChange:
    """Produce independent N(-3, 1) post-change batches, as in the original d=1 script."""

    def __init__(self, *, d, n, base_seed):
        self.d = d
        self.n = n
        self.base_seed = base_seed
        self.attempt_index = 0

    def draw(self):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.base_seed, self.d, self.attempt_index, 3])
        )
        self.attempt_index += 1
        return rng.normal(loc=-3.0, scale=1.0, size=(self.n, self.d))


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results"
D = 1
NUM_FEATURES = (500, 1000, 5000)
SHORT_L_VALUES = (4.5, 7.0, 9.0)
FULL_L_VALUES = (
    4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0,
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
)
REFERENCE_SIZE = 250
BURN_IN = 250
PRE_CHANGE_LENGTH = 100
N_Q = 1000
QS_SEED_BASE = 10_000
RUN_SEED_BASE = 12345
POST_CHANGE_CASE = "MeanShift(mu=-3)"
ALGO_KWARGS = {
    "lambda0": 0.999, "lambda1": 0.999, "band_lambda": 0.999,
    "eta": 1e-3, "clip": (1e-3, 0.999), "store_lambdas": False,
    "thresholding_method": "fixed", "fixed_threshold": np.inf,
    "store_values": False,
}
ARL_METRICS = (
    "arl_hat", "arl_var", "arl_std", "arl_se_mean", "arl_censored_rate"
)
EDD_METRICS = (
    "edd_hat", "edd_var", "edd_std", "edd_se_mean", "edd_censored_rate"
)
COMMON_COLUMNS = (
    "post_change_name", "q", "change_magnitude", "chart_L", "dimension_d",
    "num_features_m", "burn_in", "prechange", "change_point", "n_runs_edd",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--features", nargs="+", type=int, choices=NUM_FEATURES,
                        default=NUM_FEATURES, help="Feature counts (default: all three)")
    parser.add_argument("--full-grid", action="store_true",
                        help="Use all 16 thresholds, rather than the original three")
    parser.add_argument("--arl-results-dir", type=Path,
                        help="Directory containing feature-specific ARL CSVs")
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    if len(args.features) != len(set(args.features)):
        parser.error("--features must not contain duplicates")
    return args


def arl_results_dir(override: Path | None, *, num_features: int, n_runs: int) -> Path:
    if override is not None:
        return override.expanduser().resolve()

    # Prefer results alongside these scripts (e.g. "number of RFF/d1").
    filename = f"OKAFF-ARL-d{D}-m{num_features}-nrun{n_runs}.csv"
    local = SCRIPT_DIR / "results"
    sibling = CODE_ROOT / "ARL" / f"d{D}" / "results"
    for directory in (local, sibling):
        if (directory / filename).is_file():
            return directory
    return local


def method_configs(l_values):
    return {
        "OKAFF": {
            "grid_values": l_values,
            "dimension_d": D,
            "bandwidth_method": "gauss_est_gamma_resampled_pairwise_median_same_run_theory_band",
            "detector_bandwidth": "gauss_est_gamma_resampled_pairwise_median_fresh_per_run",
            "band_source": "same_run_theory_recomputed_from_detector_sigmasq",
            "band_lambda": float(default_band_lambda(ALGO_KWARGS)),
        }
    }


def format_results_for_csv(data: pd.DataFrame, metrics, *, include_prechange: bool):
    columns = [
        column for column in COMMON_COLUMNS
        if column in data.columns and (include_prechange or column != "prechange")
    ]
    rename = {f"{metric}_OKAFF": metric for metric in metrics}
    missing = [column for column in rename if column not in data.columns]
    if missing:
        raise ValueError(f"Results are missing expected columns: {missing}")
    clean = data[columns + list(rename)].rename(columns=rename).copy()
    clean.insert(0, "method", "OKAFF")
    return clean


def estimate_edd(*, num_features, n_runs, l_values) -> pd.DataFrame:
    print(f"\n===== OKAFF EDD: m={num_features}, case={POST_CHANGE_CASE} =====")
    configs = method_configs(l_values)
    result = run_one_q_prepost_and_edd(
        OKAFF,
        q_obj=NegativeMeanPostChange(d=D, n=N_Q, base_seed=QS_SEED_BASE),
        q_name=POST_CHANGE_CASE,
        method_configs=configs,
        selected_methods=("OKAFF",),
        alphas_common=l_values,
        d=D,
        burn_in=BURN_IN,
        pre_change_length=PRE_CHANGE_LENGTH,
        n_q=N_Q,
        n_runs=n_runs,
        m=num_features,
        ref_size=REFERENCE_SIZE,
        algo_kwargs=ALGO_KWARGS,
        Sigma_d=np.eye(D),
        seed=RUN_SEED_BASE,
    )
    rows = comparison_rows_for_q(
        configs, result, q_name=POST_CHANGE_CASE,
        burn_in=BURN_IN, pre_change_length=PRE_CHANGE_LENGTH,
        change_point=PRE_CHANGE_LENGTH, grid_col="chart_L",
    )
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(f"No EDD results returned for m={num_features}")
    if "num_features_m" in data and not data["num_features_m"].eq(num_features).all():
        raise ValueError(f"EDD results contain the wrong feature count for m={num_features}")
    data["num_features_m"] = num_features
    return data


def combine_arl_edd(edd_data: pd.DataFrame, *, num_features, n_runs, arl_dir: Path):
    arl_file = arl_dir / f"OKAFF-ARL-d{D}-m{num_features}-nrun{n_runs}.csv"
    if not arl_file.is_file():
        raise FileNotFoundError(
            f"Missing matching ARL results: {arl_file}. "
            "Run the ARL script first with the same --n-runs and --full-grid options. "
            "The separate EDD CSV was saved, but no combined CSV was produced."
        )

    arl_data = pd.read_csv(arl_file)
    required = ("chart_L", "num_features_m", "n_runs", *ARL_METRICS)
    missing = [column for column in required if column not in arl_data.columns]
    if missing:
        raise ValueError(f"{arl_file.name} lacks columns: {missing}")
    if not arl_data["num_features_m"].eq(num_features).all():
        raise ValueError(f"Feature count mismatch in {arl_file}")
    if not arl_data["n_runs"].eq(n_runs).all():
        raise ValueError(f"Run count mismatch in {arl_file}")

    arl_values = arl_data[["chart_L", *ARL_METRICS]].rename(
        columns={metric: f"{metric}_OKAFF" for metric in ARL_METRICS}
    )
    if arl_values["chart_L"].duplicated().any():
        raise ValueError(f"Repeated chart_L values in {arl_file}")
    expected_l = set(edd_data["chart_L"].astype(float))
    actual_l = set(arl_values["chart_L"].astype(float))
    if actual_l != expected_l:
        raise ValueError(
            f"Threshold mismatch for m={num_features}: "
            f"EDD chart_L={sorted(expected_l)}, ARL chart_L={sorted(actual_l)}. "
            "Run both scripts with matching --full-grid settings."
        )
    # The two inputs have already been selected for the SAME feature count.
    # Include num_features_m in the merge as an extra safeguard.
    arl_values["num_features_m"] = num_features
    arl_columns = {*ARL_METRICS, *(f"{metric}_OKAFF" for metric in ARL_METRICS)}
    old_arl_cols = [column for column in edd_data if column in arl_columns]
    combined = edd_data.drop(columns=old_arl_cols).merge(
        arl_values, on=["num_features_m", "chart_L"], how="left", validate="many_to_one"
    )
    if combined["arl_hat_OKAFF"].isna().any():
        missing_l = combined.loc[combined["arl_hat_OKAFF"].isna(), "chart_L"].tolist()
        raise ValueError(f"ARL results missing for m={num_features}, chart_L={missing_l}")
    return combined


def main():
    args = parse_args()
    l_values = np.asarray(FULL_L_VALUES if args.full_grid else SHORT_L_VALUES, dtype=float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for num_features in args.features:
        source = arl_results_dir(
            args.arl_results_dir, num_features=num_features, n_runs=args.n_runs
        )
        print(f"ARL results directory for m={num_features}: {source}")
        edd = estimate_edd(num_features=num_features, n_runs=args.n_runs, l_values=l_values)
        edd_file = OUT_DIR / f"OKAFF-EDD-d{D}-m{num_features}-nrun{args.n_runs}.csv"
        format_results_for_csv(edd, EDD_METRICS, include_prechange=True).to_csv(
            edd_file, index=False
        )
        print(f"Saved EDD: {edd_file}")

        combined = combine_arl_edd(
            edd, num_features=num_features, n_runs=args.n_runs, arl_dir=source
        )
        combined_file = OUT_DIR / f"OKAFF-ARL-EDD-d{D}-m{num_features}-nrun{args.n_runs}.csv"
        format_results_for_csv(
            combined, (*ARL_METRICS, *EDD_METRICS), include_prechange=False
        ).to_csv(combined_file, index=False)
        print(f"Saved combined ARL+EDD: {combined_file}")


if __name__ == "__main__":
    main()
