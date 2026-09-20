"""Shared fixed-attempt EDD simulation and reporting utilities."""

import hashlib
import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _data_rng(*, seed, scenario_name, run_index, d, stream):
    """Select a data stream without consuming any detector randomness."""
    scenario_key = np.frombuffer(
        hashlib.sha256(scenario_name.encode("utf-8")).digest(), dtype="<u4"
    ).tolist()
    return np.random.default_rng(
        np.random.SeedSequence([seed, d, run_index, stream, *scenario_key])
    )


def make_reference_data(*, seed, scenario_name, run_index, d, reference_size):
    """Pair reference observations by scenario/run, independently of detector RNGs."""
    rng = _data_rng(
        seed=seed, scenario_name=scenario_name, run_index=run_index, d=d, stream=0
    )
    return rng.standard_normal((reference_size, d))


def make_monitoring_data(*, seed, scenario, run_index, d, pre_length, post_length):
    """Return paired pre/post-change observations for one candidate run."""
    key = dict(seed=seed, scenario_name=scenario["name"], run_index=run_index, d=d)
    pre = _data_rng(**key, stream=1).standard_normal((pre_length, d))
    post = draw_postchange(_data_rng(**key, stream=2), scenario, post_length, d=d)
    return pre, post


def draw_postchange(rng, scenario, size, *, d):
    """Use one distribution implementation for all four detectors."""
    family, parameter = scenario["family"], scenario["parameter"]
    shape = (size, d)
    if family == "normal":
        return rng.normal(size=shape)
    if family == "mean_shift":
        location = np.full(d, parameter)
        if scenario.get("scope") == "first5":
            location[5:] = 0.0
        return rng.normal(loc=location, scale=1.0, size=shape)
    if family == "variance_shift":
        variances = np.full(d, parameter)
        if scenario.get("scope") == "first5":
            variances[5:] = 1.0
        return rng.normal(scale=np.sqrt(variances), size=shape)
    if family == "mixture":
        choose_standard = (rng.uniform(size=size) <= parameter).reshape(-1, 1)
        standard = rng.normal(size=shape)
        wide = rng.normal(scale=2.0, size=shape)
        return np.where(choose_standard, standard, wide)
    if family == "laplace":
        return rng.laplace(scale=parameter, size=shape)
    if family == "uniform":
        return rng.uniform(-1.0, 1.0, size=shape) * parameter
    if family == "exponential":
        return rng.exponential(scale=parameter, size=shape)
    raise ValueError(f"Unknown post-change family: {family}")


def summarize_fixed_runs(run_frame):
    """Report outcome counts and EDD for successful detections."""
    rows = []
    for (algorithm, scenario, q), group in run_frame.groupby(
        ["algorithm", "scenario", "q"], sort=False
    ):
        false_alarm = ~group["survived_prechange"]
        successful = group["survived_prechange"] & ~group["censored"]
        failure = group["survived_prechange"] & group["censored"]
        eligible = successful | failure

        successful_delays = group.loc[successful, "delay"].to_numpy(dtype=float)
        successful_sd = (
            successful_delays.std(ddof=1)
            if len(successful_delays) > 1
            else 0.0
        )

        rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "distribution_family": group["distribution_family"].iloc[0],
                "distribution_parameter": group[
                    "distribution_parameter"
                ].iloc[0],
                "q": q,
                "total_runs": len(group),
                "successful_detections": int(successful.sum()),
                "false_alarms": int(false_alarm.sum()),
                "failures": int(failure.sum()),
                "EDD_successful_only": (
                    successful_delays.mean()
                    if len(successful_delays)
                    else np.nan
                ),
                "SE_successful_only": (
                    successful_sd / math.sqrt(len(successful_delays))
                    if len(successful_delays)
                    else np.nan
                ),
                "eligible_postchange_runs": int(eligible.sum()),
                "successful_detection_rate": successful.mean(),
                "false_alarm_rate": false_alarm.mean(),
                "failure_rate": failure.mean(),
                "max_delay": int(group["max_delay"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def estimate_fixed_run_edd(
    *,
    simulate_one_edd_run,
    scenarios,
    quantiles,
    n_runs,
    seed,
    algorithm,
    max_delay,
    atomic_write_csv,
    runs_csv,
    summary_csv,
    run_metadata=None,
):
    """Run exactly n_runs paths per scenario and classify every outcome."""
    scenario_seeds = np.random.SeedSequence(seed).spawn(len(scenarios))
    rows = []

    for scenario, scenario_seed in zip(scenarios, scenario_seeds):
        scenario_name = scenario["name"]
        rng = np.random.default_rng(scenario_seed)
        progress = tqdm(
            total=n_runs * len(quantiles),
            desc=scenario_name,
        )

        for run in range(1, n_runs + 1):
            delays, censored, survived, false_alarm_times = (
                simulate_one_edd_run(rng, scenario, run_index=run - 1)
            )
            for q in quantiles:
                false_alarm = not survived[q]
                successful = survived[q] and not censored[q]
                failure = survived[q] and censored[q]
                row = {
                    "algorithm": algorithm,
                    "scenario": scenario_name,
                    "run": run,
                    "q": q,
                    "delay": delays[q] if survived[q] else np.nan,
                    "censored": bool(censored[q]) if survived[q] else False,
                    "survived_prechange": bool(survived[q]),
                    "successful_detection": bool(successful),
                    "false_alarm": bool(false_alarm),
                    "false_alarm_time": false_alarm_times[q],
                    "failure": bool(failure),
                    "distribution_family": scenario["family"],
                    "distribution_parameter": scenario["parameter"],
                    "max_delay": max_delay,
                }
                if run_metadata is not None:
                    row.update(run_metadata())
                rows.append(row)
                progress.update(1)

        progress.close()
        run_frame = pd.DataFrame(rows)
        summary_frame = summarize_fixed_runs(run_frame)
        atomic_write_csv(run_frame, runs_csv)
        atomic_write_csv(summary_frame, summary_csv)
        print(f"Saved checkpoint after {scenario_name}")

    return run_frame, summary_frame
