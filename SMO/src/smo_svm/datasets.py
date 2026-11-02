"""Small reproducible NumPy datasets; no sklearn dependency."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_binary_gaussians(
    n_samples: int = 300, random_state: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return shuffled, overlapping, two-dimensional Gaussian classes.

    All samples are kept. Overlap makes the hinge-loss soft margin relevant.
    This generator is a teaching example, not evidence of general performance.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 4:
        raise ValueError("n_samples must be an integer >= 4")
    rng = np.random.default_rng(random_state)
    negative_count = n_samples // 2
    positive_count = n_samples - negative_count
    covariance = np.array([[1.0, 0.35], [0.35, 0.85]])
    negative = rng.multivariate_normal([-1.15, -0.7], covariance, negative_count)
    positive = rng.multivariate_normal([1.15, 0.7], covariance, positive_count)
    X = np.vstack((negative, positive))
    y = np.concatenate((-np.ones(negative_count, dtype=int), np.ones(positive_count, dtype=int)))
    order = rng.permutation(n_samples)
    return X[order], y[order]
