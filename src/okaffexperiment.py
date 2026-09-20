"""Reusable simulation helpers for the fixed-threshold OKAFF experiment."""

import numpy as np
from tqdm import tqdm

import onlinecp.utils.feature_functions as feat
from fixedthresholds import (
    default_band_lambda,
    gaussian_kernel_theory_terms,
    theoretical_rejection_band,
)
from kerneldetector import estimate_gaussian_gamma


VALID_METHODS = ("OKAFF",)


def _normalize_run_methods(run_methods):
    if isinstance(run_methods, str):
        run_methods = (run_methods,)
    requested = tuple(str(method).strip() for method in run_methods)
    if not requested:
        raise ValueError("selected_methods must contain at least one method.")
    canonical = {method.lower(): method for method in VALID_METHODS}
    unknown = [method for method in requested if method.lower() not in canonical]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. Valid options: {VALID_METHODS}."
        )
    selected = tuple(canonical[method.lower()] for method in requested)
    if len(set(selected)) != len(selected):
        raise ValueError(f"selected_methods contains duplicates: {selected}.")
    return selected


def make_rff_feat_func(*, rng, d, m, ref_size=1000):
    """Build a fresh half-off-diagonal-median RFF map."""
    reference = rng.normal(size=(ref_size, d))
    gamma = estimate_gaussian_gamma(reference[:500], max_len=500)
    sigmasq = 1.0 / (2.0 * gamma)
    frequencies, sigmasq = feat.generate_frequencies(
        m, d, choice_sigma="fixed", sigmasq=sigmasq
    )
    feature = lambda x, W=frequencies: feat.fourier_feat(x, W)
    return feature, frequencies, float(sigmasq), reference


def _summarize_delays(delays, censored_counts, n_runs):
    mean = delays.mean(axis=0).astype(float)
    variance = np.array(
        [
            delays[:, index].var(ddof=1) if n_runs > 1 else 0.0
            for index in range(delays.shape[1])
        ],
        dtype=float,
    )
    standard_deviation = np.sqrt(variance)
    standard_error = standard_deviation / np.sqrt(n_runs)
    censored_rate = censored_counts / float(n_runs)
    return (
        mean,
        variance,
        standard_deviation,
        standard_error,
        delays,
        censored_rate,
    )


def run_one_q_prepost_and_edd(
    AlgoClass,
    *,
    q_obj,
    q_name,
    method_configs,
    selected_methods,
    alphas_common,
    d,
    burn_in,
    pre_change_length,
    n_q,
    n_runs,
    algo_kwargs,
    m,
    ref_size,
    Sigma_d=None,
    seed=12345,
):
    """Run conditional EDD for one post-change distribution."""
    selected_methods = _normalize_run_methods(selected_methods)
    rng_master = np.random.default_rng(seed)
    total_pre_change = burn_in + pre_change_length
    if Sigma_d is None:
        Sigma_d = np.eye(d)
    Sigma_d = np.asarray(Sigma_d, dtype=float)
    band_lambda_value = default_band_lambda(algo_kwargs)

    stats_runs = {
        method: {"pre": [], "post": [], "pre_lambda": [], "post_lambda": []}
        for method in selected_methods
    }
    sigmasq_used_runs = {method: [] for method in selected_methods}
    delays = {
        method: np.empty((n_runs, len(alphas_common)), dtype=int)
        for method in selected_methods
    }
    failures = {
        method: np.zeros(len(alphas_common), dtype=int)
        for method in selected_methods
    }
    accepted = {
        method: np.zeros(len(alphas_common), dtype=int)
        for method in selected_methods
    }
    attempts = {
        method: np.zeros(len(alphas_common), dtype=int)
        for method in selected_methods
    }
    band_info_runs = {
        method: {
            key: np.empty((n_runs, len(alphas_common)), dtype=float)
            for key in (
                "lo",
                "hi",
                "center",
                "SV",
                "sigmasq_for_threshold",
                "theta_K",
                "eta_K",
                "zeta_K",
            )
        }
        for method in selected_methods
    }

    progress = tqdm(
        total=n_runs * len(alphas_common) * len(selected_methods),
        desc=f"{q_name}",
        leave=True,
    )
    while any(
        np.any(accepted[method] < n_runs) for method in selected_methods
    ):
        local_kwargs = dict(algo_kwargs)
        local_kwargs.pop("band_lambda", None)
        local_kwargs["store_lambdas"] = True

        detectors = {}
        reference_samples = {}
        run_bands = {}
        for method in selected_methods:
            rng_ref = np.random.default_rng(
                rng_master.integers(0, 2**63 - 1)
            )
            feature, _, sigmasq_used, reference = make_rff_feat_func(
                rng=rng_ref, d=d, m=m, ref_size=ref_size
            )
            sigmasq_used = float(sigmasq_used)
            reference_samples[method] = reference
            sigmasq_used_runs[method].append(sigmasq_used)

            theta_K, eta_K, zeta_K = gaussian_kernel_theory_terms(
                Sigma_d, sigmasq_used
            )
            run_bands[method] = {}
            for L_value in alphas_common:
                band = theoretical_rejection_band(
                    L=float(L_value),
                    lambda_value=band_lambda_value,
                    theta_K=theta_K,
                    eta_K=eta_K,
                    zeta_K=zeta_K,
                )
                run_bands[method][float(L_value)] = {
                    **band,
                    "sigmasq": sigmasq_used,
                    "theta_K": theta_K,
                    "eta_K": eta_K,
                    "zeta_K": zeta_K,
                }

            detectors[method] = AlgoClass(
                **local_kwargs,
                feat_func=feature,
                dist_func=lambda value: float(np.vdot(value, value).real),
            )

        rng_pre = np.random.default_rng(
            rng_master.integers(0, 2**63 - 1)
        )
        monitored_pre_data = rng_pre.normal(size=(pre_change_length, d))
        pre_stats = {
            method: np.empty(total_pre_change, dtype=float)
            for method in selected_methods
        }
        for index in range(total_pre_change):
            for method in selected_methods:
                observation = (
                    reference_samples[method][index]
                    if index < burn_in
                    else monitored_pre_data[index - burn_in]
                )
                pre_stats[method][index] = detectors[method].update_stat(
                    observation
                )

        survived = {}
        for method in selected_methods:
            survived[method] = np.zeros(len(alphas_common), dtype=bool)
            monitored_stats = pre_stats[method][burn_in:]
            for grid_index, L_value in enumerate(alphas_common):
                if accepted[method][grid_index] >= n_runs:
                    continue
                attempts[method][grid_index] += 1
                band = run_bands[method][float(L_value)]
                survived[method][grid_index] = not np.any(
                    (monitored_stats < band["lo"])
                    | (monitored_stats > band["hi"])
                )

        # Advance once per candidate, including pre-change rejections, so
        # post-change data stays paired with the reference/pre-change attempt.
        post_data = q_obj.draw()
        if post_data.shape != (n_q, d):
            raise ValueError(
                f"{q_name}.draw() returned {post_data.shape}, "
                f"expected {(n_q, d)}"
            )
        if not any(
            np.any(survived[method]) for method in selected_methods
        ):
            continue

        post_stats = {
            method: np.empty(n_q, dtype=float)
            for method in selected_methods
        }
        for index in range(n_q):
            for method in selected_methods:
                post_stats[method][index] = detectors[method].update_stat(
                    post_data[index]
                )

        for method in selected_methods:
            all_lambdas = np.asarray(
                detectors[method].lambdas_stored, dtype=float
            )
            required_length = total_pre_change + n_q
            if all_lambdas.shape[0] >= required_length:
                all_lambdas = all_lambdas[-required_length:]
            if all_lambdas.shape[0] != required_length:
                raise RuntimeError(
                    f"Unexpected {method} lambda length "
                    f"{all_lambdas.shape[0]}, expected {required_length}"
                )

            if (
                survived[method][0]
                and len(stats_runs[method]["pre"]) < n_runs
            ):
                stats_runs[method]["pre"].append(pre_stats[method])
                stats_runs[method]["post"].append(post_stats[method])
                stats_runs[method]["pre_lambda"].append(
                    all_lambdas[:total_pre_change]
                )
                stats_runs[method]["post_lambda"].append(
                    all_lambdas[total_pre_change:required_length]
                )

            for grid_index, L_value in enumerate(alphas_common):
                if (
                    not survived[method][grid_index]
                    or accepted[method][grid_index] >= n_runs
                ):
                    continue
                run_index = accepted[method][grid_index]
                band = run_bands[method][float(L_value)]
                for key in (
                    "lo",
                    "hi",
                    "center",
                    "SV",
                    "theta_K",
                    "eta_K",
                    "zeta_K",
                ):
                    band_info_runs[method][key][run_index, grid_index] = (
                        band[key]
                    )
                band_info_runs[method]["sigmasq_for_threshold"][
                    run_index, grid_index
                ] = band["sigmasq"]
                hits = np.where(
                    (post_stats[method] < band["lo"])
                    | (post_stats[method] > band["hi"])
                )[0]
                if len(hits) == 0:
                    failures[method][grid_index] += 1
                    delays[method][run_index, grid_index] = n_q
                else:
                    delays[method][run_index, grid_index] = int(hits[0] + 1)
                accepted[method][grid_index] += 1
                progress.update(1)

    progress.close()
    result = {
        "selected_methods": selected_methods,
        "edd": {},
        "band_lambda": band_lambda_value,
        "edd_conditioning": "survive_last_prechange_only",
    }
    for method in selected_methods:
        for key in ("pre", "post", "pre_lambda", "post_lambda"):
            result[f"{key}_{method}"] = np.asarray(
                stats_runs[method][key], dtype=float
            )
        result[f"kernel_sigmasq_{method}"] = np.asarray(
            sigmasq_used_runs[method], dtype=float
        )
        result[f"detector_bandwidth_{method}"] = method_configs[method][
            "detector_bandwidth"
        ]
        result[f"detector_kernel_sigmasq_{method}_mean"] = float(
            np.mean(sigmasq_used_runs[method])
        )
        result[f"detector_kernel_sigmasq_{method}_std"] = (
            float(np.std(sigmasq_used_runs[method], ddof=1))
            if len(sigmasq_used_runs[method]) > 1
            else 0.0
        )
        if method_configs[method].get("kernel_sigmasq") is not None:
            result[f"detector_kernel_sigmasq_{method}"] = float(
                method_configs[method]["kernel_sigmasq"]
            )
        for key, values in band_info_runs[method].items():
            result[f"band_{key}_{method}"] = values
        result[f"band_source_{method}"] = (
            "same_run_theory_recomputed_from_detector_sigmasq"
        )
        result[f"prechange_attempts_{method}"] = attempts[method].copy()
        result[f"prechange_rejections_{method}"] = (
            attempts[method] - accepted[method]
        )
        result[f"prechange_acceptance_rate_{method}"] = (
            accepted[method] / attempts[method]
        )
        result[f"edd_failures_{method}"] = failures[method].copy()
        result["edd"][method] = _summarize_delays(
            delays[method], failures[method], n_runs
        )
    return result


def comparison_rows_for_q(
    method_configs,
    result,
    q_name,
    burn_in,
    pre_change_length,
    change_point,
    grid_col="alpha",
):
    """Convert one scenario result into long-form CSV rows."""
    selected_methods = result["selected_methods"]
    first_config = method_configs[selected_methods[0]]
    grid_values = np.asarray(first_config["grid_values"], dtype=float)
    rows = []
    for index, grid_value in enumerate(grid_values):
        row = {
            "post_change_name": q_name,
            grid_col: float(grid_value),
            "dimension_d": int(first_config["dimension_d"]),
            "burn_in": int(burn_in),
            "pre_change_length": int(pre_change_length),
            "change_point": int(change_point),
            "selected_methods": ",".join(selected_methods),
            "band_source": "same_run_theory_recomputed_from_detector_sigmasq",
            "band_lambda": float(result.get("band_lambda", np.nan)),
        }
        for method in selected_methods:
            config = method_configs[method]
            edd_hat, edd_var, edd_std, edd_se, _, edd_cens = result[
                "edd"
            ][method]
            arrays = {
                key: np.asarray(result[f"band_{key}_{method}"][:, index], dtype=float)
                for key in (
                    "lo",
                    "hi",
                    "center",
                    "SV",
                    "sigmasq_for_threshold",
                    "theta_K",
                    "eta_K",
                    "zeta_K",
                )
            }
            lo, hi = arrays["lo"], arrays["hi"]
            sigma = arrays["sigmasq_for_threshold"]
            row.update(
                {
                    f"arl_hat_{method}": np.nan,
                    f"arl_se_mean_{method}": np.nan,
                    f"region_lo_{method}": float(np.mean(lo)),
                    f"region_hi_{method}": float(np.mean(hi)),
                    f"region_lo_std_{method}": float(np.std(lo, ddof=1)) if len(lo) > 1 else 0.0,
                    f"region_hi_std_{method}": float(np.std(hi, ddof=1)) if len(hi) > 1 else 0.0,
                    f"region_lo_first_run_{method}": float(lo[0]),
                    f"region_hi_first_run_{method}": float(hi[0]),
                    f"band_center_mean_{method}": float(np.mean(arrays["center"])),
                    f"band_SV_mean_{method}": float(np.mean(arrays["SV"])),
                    f"band_kernel_sigmasq_{method}": float(np.mean(sigma)),
                    f"band_kernel_sigmasq_std_{method}": float(np.std(sigma, ddof=1)) if len(sigma) > 1 else 0.0,
                    f"band_kernel_sigmasq_first_run_{method}": float(sigma[0]),
                    f"theta_K_mean_{method}": float(np.mean(arrays["theta_K"])),
                    f"eta_K_mean_{method}": float(np.mean(arrays["eta_K"])),
                    f"zeta_K_mean_{method}": float(np.mean(arrays["zeta_K"])),
                    f"bandwidth_{method}": str(config["bandwidth_method"]),
                    f"detector_bandwidth_{method}": result.get(f"detector_bandwidth_{method}", ""),
                    f"detector_kernel_sigmasq_{method}_mean": result.get(f"detector_kernel_sigmasq_{method}_mean", ""),
                    f"detector_kernel_sigmasq_{method}_std": result.get(f"detector_kernel_sigmasq_{method}_std", ""),
                    f"edd_hat_{method}": float(edd_hat[index]),
                    f"edd_var_{method}": float(edd_var[index]),
                    f"edd_std_{method}": float(edd_std[index]),
                    f"edd_se_mean_{method}": float(edd_se[index]),
                    f"edd_censored_rate_{method}": float(edd_cens[index]),
                    f"edd_failures_{method}": int(result[f"edd_failures_{method}"][index]),
                    f"edd_conditioning_{method}": result.get("edd_conditioning", ""),
                    f"prechange_attempts_{method}": int(result[f"prechange_attempts_{method}"][index]),
                    f"prechange_rejections_{method}": int(result[f"prechange_rejections_{method}"][index]),
                    f"prechange_acceptance_rate_{method}": float(result[f"prechange_acceptance_rate_{method}"][index]),
                    f"band_source_{method}": result.get(f"band_source_{method}", ""),
                }
            )
            fixed_sigma_key = f"detector_kernel_sigmasq_{method}"
            if fixed_sigma_key in result:
                row[fixed_sigma_key] = result[fixed_sigma_key]
        rows.append(row)
    return rows


__all__ = [
    "make_rff_feat_func",
    "run_one_q_prepost_and_edd",
    "comparison_rows_for_q",
]
