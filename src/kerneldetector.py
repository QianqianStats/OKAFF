from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmdew.RFFkernel import (
    Gauss as GaussianKernel,
    OKCUSUM,
    ScanBStatistic,
    StreamingRFFMMD,
)
from mmdew.mmdew import MMDEW
from onlinecp.algos import NEWMA


class OKAFF:

    def __init__(
        self,
        *,
        lambda0: float = 1.0 - 1e-3,
        lambda1: float = 1.0 - 1e-3,
        eta: float = 1e-3,
        feat_func: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        dist_func: Callable[[np.ndarray], float] = lambda v: float(
            np.vdot(v, v).real
        ),
        clip: Tuple[float, float] = (1e-3, 1.0 - 1e-3),
        store_lambdas: bool = False,
        thresholding_method: str = "fixed",
        fixed_threshold: Optional[float] = np.inf,
        store_values: bool = True,
    ) -> None:
        if thresholding_method != "fixed":
            raise ValueError("OKAFF supports fixed thresholding only")
        self.statistic = 0.0
        self.fixed_threshold = fixed_threshold
        if not 0.0 <= clip[0] < clip[1] <= 1.0:
            raise ValueError("clip must satisfy 0 <= lower < upper <= 1")
        if eta < 0:
            raise ValueError("eta must be non-negative")

        self.feat_func = feat_func
        self.dist_func = dist_func
        self.clip = (float(clip[0]), float(clip[1]))
        self.lambda0 = float(lambda0)
        self.lambda1 = float(lambda1)
        self.lambda_t = self._project_lambda(self.lambda0)
        self.eta = float(eta)
        self.store_lambdas = bool(store_lambdas)
        self.lambdas_stored: list[float] = []

        self.m: Optional[np.ndarray] = None
        self.w = 0.0
        self.Lm: Optional[np.ndarray] = None
        self.Lw = 0.0
        self.n_samples = 0

    def update(self, new_sample: np.ndarray) -> bool:
        self.statistic = self.update_stat(new_sample)
        return self.statistic > self.fixed_threshold

    def _project_lambda(self, value: float) -> float:
        # During early construction ``clip`` is not installed yet.
        bounds = getattr(self, "clip", (1e-3, 1.0 - 1e-3))
        return float(np.clip(value, bounds[0], bounds[1]))

    def update_stat(self, new_sample: np.ndarray) -> float:
        feature = np.asarray(self.feat_func(np.asarray(new_sample)))
        if self.m is None:
            self.m = np.zeros_like(feature, dtype=np.result_type(feature, float))
            self.Lm = np.zeros_like(self.m)

        assert self.Lm is not None
        lambda_used = self.lambda_t

        if self.w > 0.0:
            mean = self.m / self.w
            mean_derivative = (self.Lm - mean * self.Lw) / self.w
        else:
            mean = np.zeros_like(self.m)
            mean_derivative = np.zeros_like(self.m)

        previous_m = self.m.copy()
        previous_w = self.w
        self.m = lambda_used * self.m + feature
        self.w = lambda_used * self.w + 1.0
        self.Lm = previous_m + lambda_used * self.Lm
        self.Lw = previous_w + lambda_used * self.Lw

        statistic = float(self.dist_func(self.m / self.w))
        self.n_samples += 1
        if self.store_lambdas:
            self.lambdas_stored.append(lambda_used)

        if self.n_samples == 1:
            self.lambda_t = self._project_lambda(self.lambda1)
        else:
            gradient = 2.0 * float(
                np.vdot(mean_derivative.ravel(), (mean - feature).ravel()).real
            )
            self.lambda_t = self._project_lambda(
                lambda_used - self.eta * gradient
            )
        return statistic

    def reset(self) -> None:
        """Clear the streaming state while preserving configuration."""
        self.statistic = 0.0
        self.lambdas_stored.clear()
        self.lambda_t = self._project_lambda(self.lambda0)
        self.m = None
        self.w = 0.0
        self.Lm = None
        self.Lw = 0.0
        self.n_samples = 0


class OnlineRFFMMD(StreamingRFFMMD):
    """Readable public name for the project's streaming RFF-MMD detector."""


class OKCUSUMDetector(OKCUSUM):
    """Readable alias for :class:`mmdew.RFFkernel.OKCUSUM`."""


def estimate_gaussian_gamma(reference_sample: np.ndarray, max_len: int = 500) -> float:
    """Estimate the Gaussian RBF ``gamma`` using the median heuristic."""
    return float(GaussianKernel.est_gamma(np.asarray(reference_sample), max_len=max_len))


__all__ = [
    "OKAFF",
    "NEWMA",
    "MMDEW",
    "OnlineRFFMMD",
    "ScanBStatistic",
    "OKCUSUM",
    "GaussianKernel",
    "estimate_gaussian_gamma",
]
