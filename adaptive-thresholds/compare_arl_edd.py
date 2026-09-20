import argparse
import math
from pathlib import Path

import pandas as pd


METHODS = (
    ("OKAFF", "OKAFF"),
    ("NEWMA", "NEWMA"),
    ("Online RFF MMD", "online_rff_mmd"),
    ("MMDEW", "MMDEW"),
)
METRICS = (
    ("EDD", "EDD_successful_only"),
    ("Success", "successful_detection_rate"),
    ("False alarm", "false_alarm_rate"),
    ("Failure", "failure_rate"),
)


def load_results(root, filename, required, n_runs=None):
    paths = sorted(root.rglob(filename))
    if n_runs is not None:
        folder_name = f"{Path(filename).stem}-nrun{n_runs}"
        paths = [path for path in paths if path.parent.name == folder_name]
    if not paths:
        raise ValueError(f"Missing CSV: {filename}" + (f" in the nrun{n_runs} folder" if n_runs is not None else ""))
    if len(paths) > 1:
        raise ValueError(f"Multiple copies of {filename}; use --results-dir to select a results folder")
    frame = pd.read_csv(paths[0])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{paths[0]}: missing columns {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError(f"Empty CSV: {paths[0]}")
    for column in required:
        if column != "scenario":
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    if n_runs is not None:
        count_column = "completed_runs" if "-ARL-" in filename else "total_runs"
        if count_column not in frame or not (pd.to_numeric(frame[count_column], errors="raise") == n_runs).all():
            raise ValueError(f"{paths[0]}: expected {n_runs} completed runs")
    print(f"Loaded: {paths[0]}")
    return frame


def compare(root, dimension, output_dir, n_runs=None):
    arl_rows = []
    edd_frames = {}
    scenarios = None
    for method, prefix in METHODS:
        arl = load_results(root, f"{prefix}-adaptive-thresholds-ARL-d{dimension}.csv", ("q", "ARL", "standard_error"), n_runs)
        edd = load_results(root, f"{prefix}-adaptive-thresholds-EDD-d{dimension}.csv", ("scenario", "q", *(column for _, column in METRICS)), n_runs)
        if len(arl) != 1 or edd["q"].nunique() != 1:
            raise ValueError(f"{method} d{dimension}: expected one quantile per method")
        q = arl.iloc[0]["q"]
        if not (edd["q"] == q).all():
            raise ValueError(f"{method} d{dimension}: ARL and EDD quantiles differ")
        if edd["scenario"].duplicated().any():
            raise ValueError(f"{method} d{dimension}: duplicate EDD scenarios")
        current = edd["scenario"].tolist()
        if scenarios is None:
            scenarios = current
        elif set(current) != set(scenarios):
            raise ValueError(f"{method} d{dimension}: scenario set differs from OKAFF")
        arl_rows.append({"Method": method, "q": q, "ARL": arl.iloc[0]["ARL"], "standard_error": arl.iloc[0]["standard_error"]})
        edd_frames[method] = edd.set_index("scenario")
    arl_table = pd.DataFrame(arl_rows)
    rows = []
    for scenario in scenarios:
        for metric, column in METRICS:
            row = {"Scenario": scenario, "Metric": metric}
            for method, _ in METHODS:
                result = edd_frames[method].loc[scenario]
                value = result[column]
                if metric == "EDD" and result["successful_detection_rate"] == 0:
                    value = math.nan
                row[method] = value
            rows.append(row)
    edd_table = pd.DataFrame(rows)
    suffix = f"-nrun{n_runs}" if n_runs is not None else ""
    output_dir.mkdir(parents=True, exist_ok=True)
    arl_path = output_dir / f"adaptive-thresholds-ARL-comparison-d{dimension}{suffix}.csv"
    edd_path = output_dir / f"adaptive-thresholds-EDD-comparison-d{dimension}{suffix}.csv"
    arl_table.to_csv(arl_path, index=False)
    edd_table.to_csv(edd_path, index=False, na_rep="--")
    print(f"\nd={dimension} ARL:\n")
    print(arl_table.to_string(index=False, formatters={"ARL": "{:.2f}".format, "standard_error": "{:.2f}".format}))
    displayed = edd_table.copy()
    for method, _ in METHODS:
        displayed[method] = ["--" if pd.isna(value) else (f"{value:.2f}" if metric == "EDD" else f"{value:.3f}") for metric, value in zip(displayed["Metric"], displayed[method])]
    print(f"\nd={dimension} EDD comparison (successful detections only):\n")
    print(displayed.to_string(index=False))
    print(f"\nSaved: {arl_path}\nSaved: {edd_path}")


def main():
    parser = argparse.ArgumentParser(description="Save adaptive-threshold ARL and EDD comparison CSVs for four methods.")
    parser.add_argument("--dimension", type=int, choices=(1, 20), help="Compare one dimension; defaults to both")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent, help="CSV results folder, searched recursively")
    parser.add_argument("--output-dir", type=Path, help="Comparison output folder; defaults to RESULTS_DIR/comparison_results")
    parser.add_argument("--n-runs", type=int, help="Select results for this run count")
    args = parser.parse_args()
    if args.n_runs is not None and args.n_runs <= 0:
        parser.error("--n-runs must be a positive integer")
    failed = False
    for dimension in ((args.dimension,) if args.dimension else (1, 20)):
        try:
            compare(args.results_dir, dimension, args.output_dir or args.results_dir / "comparison_results", args.n_runs)
        except (ValueError, OSError) as error:
            print(f"\nd={dimension}: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
