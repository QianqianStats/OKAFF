"""Benchmark streaming-only runtime against sequence length (Gaussian data, d=20).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat
from kerneldetector import (
    GaussianKernel,
    MMDEW,
    NEWMA,
    OKAFF,
    OKCUSUM,
    OnlineRFFMMD,
    ScanBStatistic,
    estimate_gaussian_gamma,
)

OUT_DIR = SCRIPT_DIR / "runtime_vs_length_gauss_d20"

REFERENCE_SIZE = 250
BURN_IN = 250
BANDWIDTH_REFERENCE_SIZE = 250
BLOCK_SIZE = 50
NUM_BLOCKS = 5

def make_rff_feat_func_from_ref(ref: np.ndarray, m: int) -> Callable[[np.ndarray], np.ndarray]:
    """Match the clean AFF/NEWMA half-off-diagonal bandwidth procedure."""
    d = ref.shape[1]
    gamma = estimate_gaussian_gamma(
        ref[:BANDWIDTH_REFERENCE_SIZE], max_len=BANDWIDTH_REFERENCE_SIZE
    )
    sigmasq = 1.0 / (2.0 * gamma)
    W, _ = feat.generate_frequencies(
        m,
        d,
        sigmasq=sigmasq,
        choice_sigma="fixed",
    )
    return lambda x, W=W: feat.fourier_feat(x, W)


def bench_aff(data: np.ndarray, ref: np.ndarray) -> float:
    feat_func = make_rff_feat_func_from_ref(ref, m=500)
    det = OKAFF(
        lambda0=1 - 1e-3,
        lambda1=1 - 1e-3,
        eta=1e-3,
        clip=(1e-3, 1 - 1e-3),
        feat_func=feat_func,
        dist_func=lambda v: float(np.vdot(v, v).real),
        store_values=False,
    )
    for x in ref[:BURN_IN]:
        det.update_stat(x)

    start_stream = time.perf_counter()
    for x in data:
        _ = det.update_stat(x)
    stream_time = time.perf_counter() - start_stream
    return stream_time


def bench_mmdew(data: np.ndarray, ref: np.ndarray) -> float:
    gamma = estimate_gaussian_gamma(
        ref[:BANDWIDTH_REFERENCE_SIZE], max_len=BANDWIDTH_REFERENCE_SIZE
    )
    det = MMDEW(gamma=gamma)
    for x in ref[:BURN_IN]:
        det.insert(x)

    start_stream = time.perf_counter()
    for x in data:
        det.insert(x)
        _ = det.stats[-1]
    stream_time = time.perf_counter() - start_stream
    return stream_time


def bench_newma(data: np.ndarray, ref: np.ndarray) -> float:
    B = 50
    big_Lambda, small_lambda = algos.select_optimal_parameters(B)
    thres_ff = small_lambda
    m = 500

    feat_func = make_rff_feat_func_from_ref(ref, m=m)
    det = NEWMA(
        ref[0],
        forget_factor=big_Lambda,
        forget_factor2=small_lambda,
        feat_func=feat_func,
        adapt_forget_factor=thres_ff,
    )
    det.apply_to_data(ref[:BURN_IN])

    start_stream = time.perf_counter()
    det.apply_to_data(data)
    stream_time = time.perf_counter() - start_stream
    return stream_time


def bench_okcusum(data: np.ndarray, ref: np.ndarray) -> float:
    gamma = estimate_gaussian_gamma(
        ref[:BANDWIDTH_REFERENCE_SIZE], max_len=BANDWIDTH_REFERENCE_SIZE
    )
    det = OKCUSUM(
        reference_sample=ref,
        B_max=BLOCK_SIZE,
        N=NUM_BLOCKS,
        gamma=gamma,
        B_min=2,
        alpha=0.05,
    )

    start_stream = time.perf_counter()
    for x in data:
        det.insert(x)
        _ = det.statistic()
    stream_time = time.perf_counter() - start_stream
    return stream_time


def bench_streaming_rffmmd(data: np.ndarray, ref: np.ndarray) -> float:
    gamma = estimate_gaussian_gamma(
        ref[:BANDWIDTH_REFERENCE_SIZE], max_len=BANDWIDTH_REFERENCE_SIZE
    )
    det = OnlineRFFMMD(
        kernel=GaussianKernel(gamma),
        d=ref.shape[1],
        num_omegas=500,
    )
    for x in ref[:BURN_IN]:
        det.insert(x)
        _ = det.statistic()

    start_stream = time.perf_counter()
    for x in data:
        det.insert(x)
        _ = det.statistic()
    stream_time = time.perf_counter() - start_stream
    return stream_time


def bench_scanb(data: np.ndarray, ref: np.ndarray) -> float:
    gamma = estimate_gaussian_gamma(
        ref[:BANDWIDTH_REFERENCE_SIZE], max_len=BANDWIDTH_REFERENCE_SIZE
    )
    det = ScanBStatistic(
        reference_sample=ref,
        B0=BLOCK_SIZE,
        N=NUM_BLOCKS,
        gamma=gamma,
    )

    start_stream = time.perf_counter()
    for x in data:
        det.insert(x)
        _ = det.statistic()
    stream_time = time.perf_counter() - start_stream
    return stream_time


BENCHMARKS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "OKAFF": bench_aff,
    "MMDEW": bench_mmdew,
    "NEWMA": bench_newma,
    "OKCUSUM": bench_okcusum,
    "StreamingRFFMMD": bench_streaming_rffmmd,
    "ScanB": bench_scanb,
}


def _benchmark_repeat(task):
    length, d, ref_size, seed, repeat = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, length, repeat]))
    np.random.seed(np.random.SeedSequence([seed, d, length, repeat, 1]).generate_state(1)[0])
    reference = rng.normal(size=(ref_size, d))
    stream = rng.normal(size=(length, d))
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
                "length": length,
                "stream_time": stream_time,
            }
        )
    return length, repeat, rows


def benchmark_streaming(
    lengths: list[int],
    repeats: int = 10,
    d: int = 20,
    ref_size: int = REFERENCE_SIZE,
    seed: int = 12345,
    prime_once: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Time only the streaming calls; share samples and shuffle algorithm order."""
    minimum_reference_size = max(BURN_IN, NUM_BLOCKS * BLOCK_SIZE)
    if ref_size < minimum_reference_size:
        raise ValueError(
            f"ref_size must be at least {minimum_reference_size} "
            "to match the fixed-threshold ARL settings."
        )
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    if any(T <= 0 for T in lengths):
        raise ValueError("Every sequence length must be positive.")

    tasks = [
        (length, d, ref_size, seed, repeat)
        for length in lengths
        for repeat in range(repeats)
    ]
    if n_jobs == 1:
        completed = map(_benchmark_repeat, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=n_jobs)
        completed = executor.map(_benchmark_repeat, tasks)

    rows = []
    try:
        for length, repeat, result_rows in completed:
            print(
                f"\n=== sequence length T = {length}, "
                f"repeat {repeat + 1}/{repeats} ==="
            )
            for row in result_rows:
                rows.append(row)
                print(
                    f"{row['algorithm']:16s} rep={repeat} "
                    f"stream={row['stream_time']:9.5f}s"
                )
    finally:
        if n_jobs != 1:
            executor.shutdown()
    return pd.DataFrame(rows)


def plot_average_streaming_only_runtime(
    measurements: pd.DataFrame,
) -> list[Path]:
    """One log--log chart of mean streaming time for all six algorithms."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = (
        measurements.groupby(["algorithm", "length"], as_index=False)
        .agg(stream_time_mean=("stream_time", "mean"))
    )
    measurements_path = (
        OUT_DIR / "runtime_vs_length_all_six_methods_measurements_d20.csv"
    )
    summary_path = OUT_DIR / "runtime_vs_length_all_six_methods_mean_d20.csv"
    measurements.to_csv(measurements_path, index=False)
    summary.to_csv(summary_path, index=False)

    method_order = ["NEWMA", "OKAFF", "StreamingRFFMMD", "MMDEW", "ScanB", "OKCUSUM"]
    display_names = {
        "NEWMA": "NEWMA",
        "OKAFF": "OKAFF",
        "StreamingRFFMMD": "Online RFF MMD",
        "MMDEW": "MMDEW",
        "ScanB": "ScanB",
        "OKCUSUM": "OK-CUSUM",
    }
    style_map = {
        "OKAFF": {"color": "orange", "linestyle": "--", "linewidth": 3.0, "markersize": 9},
        "MMDEW": {"color": "red", "linewidth": 2.0, "markersize": 7},
        "NEWMA": {"color": "green", "linewidth": 2.0, "markersize": 7},
        "OKCUSUM": {"color": "pink", "linewidth": 2.0, "markersize": 7},
        "ScanB": {"color": "brown", "linewidth": 2.0, "markersize": 7},
        "StreamingRFFMMD": {"color": "purple", "linewidth": 2.0, "markersize": 7},
    }
    missing = set(method_order) - set(summary["algorithm"])
    if missing:
        raise ValueError(f"Methods missing from the benchmark: {sorted(missing)}")
    if (summary["length"] <= 0).any() or (summary["stream_time_mean"] <= 0).any():
        raise ValueError("Log--log axes require positive sequence lengths and runtimes.")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
        try:
            for method in method_order:
                group = summary.loc[summary["algorithm"] == method].sort_values("length")
                ax.loglog(
                    group["length"],
                    group["stream_time_mean"],
                    label=display_names[method],
                    marker="o",
                    **style_map[method],
                )

            ax.set_xlabel("Sequence length", fontsize=20)
            ax.set_ylabel("Average streaming runtime (seconds)", fontsize=20)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}")
            )
            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.65)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
            ax.legend(ncol=2, frameon=True, fontsize=10)

            stem = OUT_DIR / "runtime_vs_length_all_six_methods_loglog_d20"
            png_path, pdf_path = stem.with_suffix(".png"), stem.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
    return [measurements_path, summary_path, png_path, pdf_path]


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
    lengths = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    repeats = 10
    d = 20
    ref_size = REFERENCE_SIZE
    seed = 12345

    measurements = benchmark_streaming(
        lengths=lengths,
        repeats=repeats,
        d=d,
        ref_size=ref_size,
        seed=seed,
        n_jobs=args.n_jobs,
    )
    for path in plot_average_streaming_only_runtime(measurements):
        print(f"Saved {path}")
