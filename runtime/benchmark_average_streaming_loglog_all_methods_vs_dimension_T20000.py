"""Benchmark streaming runtime versus dimension and save one six-method log--log plot.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

from benchmark_average_streaming_loglog_all_methods_d20 import (
    BENCHMARKS,
    REFERENCE_SIZE,
)


OUT_DIR = Path(__file__).resolve().parent / "runtime_vs_dimension_gauss_T20000"
METHOD_ORDER = ("NEWMA", "OKAFF", "StreamingRFFMMD", "MMDEW", "ScanB", "OKCUSUM")
DISPLAY_NAMES = {"StreamingRFFMMD": "Online RFF MMD", "OKCUSUM": "OK-CUSUM"}
METHOD_STYLES = {
    "OKAFF": {"color": "orange", "linestyle": "--", "linewidth": 3.0, "markersize": 9},
    "MMDEW": {"color": "red", "linewidth": 2.0, "markersize": 7},
    "NEWMA": {"color": "green", "linewidth": 2.0, "markersize": 7},
    "StreamingRFFMMD": {"color": "purple", "linewidth": 2.0, "markersize": 7},
    "ScanB": {"color": "brown", "linewidth": 2.0, "markersize": 7},
    "OKCUSUM": {"color": "pink", "linewidth": 2.0, "markersize": 7},
}


def _benchmark_repeat(task):
    dimension, sequence_length, ref_size, seed, repeat = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, dimension, repeat]))
    np.random.seed(np.random.SeedSequence([seed, dimension, repeat, 1]).generate_state(1)[0])
    reference = rng.normal(size=(ref_size, dimension))
    stream = rng.normal(size=(sequence_length, dimension))
    order = list(BENCHMARKS)
    rng.shuffle(order)
    rows = []
    for method in order:
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            stream_time = BENCHMARKS[method](stream, reference)
        finally:
            if was_enabled:
                gc.enable()
        rows.append(
            {
                "algorithm": method,
                "dimension": dimension,
                "stream_time": stream_time,
            }
        )
    return dimension, repeat, rows


def benchmark_streaming(
    dimensions: list[int],
    *,
    sequence_length: int = 20_000,
    repeats: int = 10,
    ref_size: int = REFERENCE_SIZE,
    seed: int = 12345,
    prime_once: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run all six original benchmarks and record streaming time only."""
    if not dimensions or any(d <= 0 for d in dimensions):
        raise ValueError("dimensions must contain positive integers.")
    if sequence_length <= 0 or repeats < 1:
        raise ValueError("sequence_length and repeats must be positive.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    if ref_size < REFERENCE_SIZE:
        raise ValueError(
            f"ref_size must be at least {REFERENCE_SIZE} "
            "to match the fixed-threshold ARL settings."
        )
    missing = set(METHOD_ORDER) - BENCHMARKS.keys()
    if missing:
        raise ValueError(f"Benchmark functions missing: {sorted(missing)}")

    tasks = [
        (dimension, sequence_length, ref_size, seed, repeat)
        for dimension in dimensions
        for repeat in range(repeats)
    ]
    if n_jobs == 1:
        completed = map(_benchmark_repeat, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=n_jobs)
        completed = executor.map(_benchmark_repeat, tasks)

    measurements = []
    try:
        for dimension, repeat, rows in completed:
            print(
                f"\n=== dimension d = {dimension}, sequence length T = {sequence_length}, "
                f"repeat {repeat + 1}/{repeats} ==="
            )
            for row in rows:
                measurements.append(row)
                print(
                    f"{row['algorithm']:16s} rep={repeat} "
                    f"stream={row['stream_time']:9.5f}s"
                )
    finally:
        if n_jobs != 1:
            executor.shutdown()
    return pd.DataFrame(measurements)



def plot_average_streaming_only_runtime(
    measurements: pd.DataFrame, *, sequence_length: int = 20_000
) -> tuple[Path, Path, Path, Path]:
    """Plot mean streaming runtime of all methods on a single log--log chart."""
    summary = measurements.groupby(["algorithm", "dimension"], as_index=False).agg(
        stream_time_mean=("stream_time", "mean")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements_path = (
        OUT_DIR
        / f"runtime_vs_dimension_all_six_methods_measurements_T{sequence_length}.csv"
    )
    summary_path = (
        OUT_DIR / f"runtime_vs_dimension_all_six_methods_mean_T{sequence_length}.csv"
    )
    measurements.to_csv(measurements_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing = set(METHOD_ORDER) - set(summary["algorithm"])
    if missing:
        raise ValueError(f"Methods missing from measurements: {sorted(missing)}")
    if (summary["dimension"] <= 0).any() or (summary["stream_time_mean"] <= 0).any():
        raise ValueError("Log--log plots require positive dimensions and mean runtimes.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
        try:
            for method in METHOD_ORDER:
                subset = summary.loc[summary["algorithm"] == method].sort_values("dimension")
                ax.loglog(
                    subset["dimension"],
                    subset["stream_time_mean"],
                    marker="o",
                    label=DISPLAY_NAMES.get(method, method),
                    **METHOD_STYLES[method],
                )
            ax.set_xlabel("Dimension", fontsize=20)
            ax.set_ylabel("Average streaming runtime (seconds)", fontsize=20)

            # Match the reference notebook's ordinary-number labels and endpoint tick.
            minimum = int(summary["dimension"].min())
            maximum = int(summary["dimension"].max())
            ticks = sorted({t for t in (1, 10, 100, 1000, maximum) if minimum <= t <= maximum})
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 1 else f"{y:g}")
            )
            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.65)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
            ax.legend(ncol=2, frameon=True, fontsize=10)

            stem = OUT_DIR / f"runtime_vs_dimension_all_six_methods_loglog_T{sequence_length}"
            png_path, pdf_path = stem.with_suffix(".png"), stem.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
    return measurements_path, summary_path, png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel repetition workers (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be a positive integer")
    return args


if __name__ == "__main__":
    args = parse_args()
    sequence_length = 20_000
    measurements = benchmark_streaming(
        dimensions=[1, 2, 5, 10, 20, 50, 100, 1000, 5000],
        sequence_length=sequence_length,
        repeats=10,
        ref_size=REFERENCE_SIZE,
        seed=12345,
        n_jobs=args.n_jobs,
    )
    for path in plot_average_streaming_only_runtime(measurements, sequence_length=sequence_length):
        print(f"Saved {path}")
