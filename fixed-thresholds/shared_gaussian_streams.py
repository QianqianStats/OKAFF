import numpy as np


def make_gaussian_rngs(*, base_seed: int, d: int, run_index: int):
    reference_seed = np.random.SeedSequence([base_seed, d, run_index, 0])
    monitoring_seed = np.random.SeedSequence([base_seed, d, run_index, 1])
    return (
        np.random.default_rng(reference_seed),
        np.random.default_rng(monitoring_seed),
    )


def make_gaussian_initialization_rng(*, base_seed: int, d: int, run_index: int):
    initialization_seed = np.random.SeedSequence([base_seed, d, run_index, 2])
    return np.random.default_rng(initialization_seed)
