"""Plot OKAFF ARL–EDD for m=500, 1000, 5000 from the new combined CSVs.

By default, read results/ next to this file. Use --results-dir to point to
d1/results when the plotting script is stored elsewhere.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator
import numpy as np
import pandas as pd

CASE_NAME = "MeanShift(mu=-3)"
FEATURE_COUNTS = (500, 1000, 5000)
OUTPUT_PDF = "okaff_mean03_edd-d1.pdf"
STYLES = {
    500: {"marker": "o", "color": "#ff7f0e"},
    1000: {"marker": "^", "color": "#1f77b4"},
    5000: {"marker": "s", "color": "#2ca02c"},
}


def load_case(results_dir: Path, *, features: int, n_runs: int) -> pd.DataFrame:
    path = results_dir / f"OKAFF-ARL-EDD-d1-m{features}-nrun{n_runs}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the ARL script, then the EDD script, "
            "with the same --n-runs and --full-grid options."
        )
    data = pd.read_csv(path)
    required = {"post_change_name", "num_features_m", "chart_L", "arl_hat", "edd_hat"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    case = data.loc[
        (data["post_change_name"] == CASE_NAME)
        & (data["num_features_m"] == features)
    ].sort_values("arl_hat")
    if case.empty:
        raise ValueError(f"{path.name}: no data for {CASE_NAME}, m={features}")
    if case["chart_L"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate chart_L values for m={features}")
    if (case["arl_hat"] <= 0).any() or not np.isfinite(
        case[["arl_hat", "edd_hat"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{path.name}: ARL must be positive and ARL/EDD finite")
    return case


def plot_okaff(results_dir: Path, *, n_runs: int, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    edd_values = []
    expected_thresholds = None
    for features in FEATURE_COUNTS:
        case = load_case(results_dir, features=features, n_runs=n_runs)
        thresholds = set(case["chart_L"].astype(float))
        if expected_thresholds is None:
            expected_thresholds = thresholds
        elif thresholds != expected_thresholds:
            raise ValueError(
                f"Threshold grid differs for m={features}; "
                "re-run ARL and EDD with the same --full-grid setting for all features."
            )
        edd_values.extend(case["edd_hat"].tolist())
        ax.plot(
            case["arl_hat"], case["edd_hat"],
            marker=STYLES[features]["marker"],
            color=STYLES[features]["color"],
            linewidth=2, label=str(features),
        )

    ax.set_xscale("log")
    ax.set_xlim(70, 20000)
    ax.set_xticks((100, 1000, 10000))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))

    # The retained second figure: logarithmic ARL, LINEAR EDD.
    locator = MaxNLocator(nbins=7, integer=True, min_n_ticks=5)
    y_ticks = [
        float(value)
        for value in locator.tick_values(min(edd_values), max(edd_values) * 1.08)
        if value > 0
    ]
    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{value:g}" for value in y_ticks])
        y_min = min(min(edd_values), min(y_ticks))
        y_max = max(max(edd_values), max(y_ticks))
    else:
        y_min, y_max = min(edd_values), max(edd_values)
    padding = 0.05 * max(y_max - y_min, 1)
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xlabel("ARL")
    ax.set_ylabel("EDD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / OUTPUT_PDF)
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    print(f"Saved {plot_okaff(args.results_dir, n_runs=args.n_runs, output=args.output)}")


if __name__ == "__main__":
    main()
