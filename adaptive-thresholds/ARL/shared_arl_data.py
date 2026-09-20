"""Reference and monitoring data paired across adaptive-threshold ARL methods."""

import numpy as np


def make_reference_data(*, seed, run_index, d, reference_size):
    """Draw one run's reference sample independently of detector randomness."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, run_index, 0]))
    return rng.standard_normal((reference_size, d))


def make_monitoring_rng(*, seed, run_index, d):
    """Start the same independent monitoring stream for each matching run."""
    return np.random.default_rng(np.random.SeedSequence([seed, d, run_index, 1]))
