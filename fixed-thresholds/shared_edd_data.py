import re
import numpy as np


def _scenario_order(d: int):
    if d == 1:
        return (
            [f"MeanShift(mu={x:g})" for x in (1, 2, 3, 4)]
            + [f"CovDiag(var={x:g})" for x in (0.3, 0.5, 2, 6)]
            + [f"MixedNormal{x:g}" for x in (0.1, 0.2, 0.3, 0.4)]
            + [f"Laplace(scale={x:.1f})" for x in (0.5, 1, 3, 4)]
            + [f"Uniform(scale={x:.1f})" for x in (0.5, 1, 3, 4)]
        )
    return (
        [f"MeanShift(mu={x})" for x in (1, 2, 3, 4)]
        + [f"MeanShift(first5coord={x:.1f})" for x in (1, 2, 3, 4)]
        + [f"CovDiag(var={x:g})" for x in (0.3, 0.5, 2, 4)]
        + [f"CovDiag(first5var={x:g})" for x in (0.3, 0.5, 2, 4)]
        + [f"MixedNormal{x:g}" for x in (0.3, 0.5, 0.7, 0.9)]
        + [f"Laplace(scale={x:.1f})" for x in (0.3, 0.5, 1, 2)]
        + [f"Uniform(scale={x:.1f})" for x in (1, 2, 3, 4)]
    )


def edd_scenario_names(d: int):
    return tuple(_scenario_order(d))


def make_edd_gaussian_data(
    *, d: int, ref_size: int, pre_change_length: int,
    scenario_name: str, attempt_index: int, base_seed: int = 12345,
):
    order = _scenario_order(d)
    if scenario_name not in order:
        raise ValueError(f"Unknown d={d} EDD scenario: {scenario_name}")
    master = np.random.default_rng(base_seed + 1000 * order.index(scenario_name))
    ref_seed = pre_seed = None
    for _ in range(attempt_index + 1):
        ref_seed = int(master.integers(0, 2**63 - 1))
        pre_seed = int(master.integers(0, 2**63 - 1))
    ref_rng = np.random.default_rng(ref_seed)
    pre_rng = np.random.default_rng(pre_seed)
    init_rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, d, order.index(scenario_name), attempt_index, 2])
    )
    return (
        ref_rng.normal(size=(ref_size, d)),
        pre_rng.normal(size=(pre_change_length, d)),
        init_rng.normal(size=d),
    )


def make_edd_postchange_data(
    *, d: int, n: int, scenario_name: str, attempt_index: int,
    base_seed: int = 10000,
):
    """Return post-change data paired by scenario and candidate attempt."""
    order = _scenario_order(d)
    if scenario_name not in order:
        raise ValueError(f"Unknown d={d} EDD scenario: {scenario_name}")
    rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, d, order.index(scenario_name), attempt_index, 3])
    )
    match = re.fullmatch(r"MeanShift\(mu=([-+0-9.]+)\)", scenario_name)
    if match:
        return rng.normal(loc=float(match.group(1)), size=(n, d))
    match = re.fullmatch(r"MeanShift\(first5coord=([-+0-9.]+)\)", scenario_name)
    if match:
        mean = np.zeros(d); mean[:5] = float(match.group(1))
        return rng.normal(loc=mean, size=(n, d))
    match = re.fullmatch(r"CovDiag\(var=([-+0-9.]+)\)", scenario_name)
    if match:
        return rng.normal(size=(n, d)) * np.sqrt(float(match.group(1)))
    match = re.fullmatch(r"CovDiag\(first5var=([-+0-9.]+)\)", scenario_name)
    if match:
        scales = np.ones(d); scales[:5] = np.sqrt(float(match.group(1)))
        return rng.normal(size=(n, d)) * scales
    match = re.fullmatch(r"MixedNormal([-+0-9.]+)", scenario_name)
    if match:
        probability = float(match.group(1))
        mask = (rng.uniform(size=n) > probability).reshape(-1, 1)
        return np.where(mask, rng.normal(scale=2, size=(n, d)), rng.normal(size=(n, d)))
    match = re.fullmatch(r"Laplace\(scale=([-+0-9.]+)\)", scenario_name)
    if match:
        return rng.laplace(scale=float(match.group(1)), size=(n, d))
    match = re.fullmatch(r"Uniform\(scale=([-+0-9.]+)\)", scenario_name)
    if match:
        return rng.uniform(-1, 1, size=(n, d)) * float(match.group(1))
    raise ValueError(f"Unsupported EDD scenario: {scenario_name}")


class PairedPostChangeData:
    """Stateful adapter for helpers that call ``q_obj.draw()``."""
    def __init__(self, *, d, n, scenario_name, base_seed=10000):
        self.d, self.n, self.scenario_name = d, n, scenario_name
        self.base_seed, self.attempt_index = base_seed, 0

    def draw(self):
        data = make_edd_postchange_data(
            d=self.d, n=self.n, scenario_name=self.scenario_name,
            attempt_index=self.attempt_index, base_seed=self.base_seed,
        )
        self.attempt_index += 1
        return data
