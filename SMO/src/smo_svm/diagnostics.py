"""Independent mathematical diagnostics for a feasible SVM dual iterate."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def kkt_residuals(
    alpha: FloatArray,
    y: FloatArray,
    scores: FloatArray,
    C: float,
    bound_tol: float = 1e-10,
) -> FloatArray:
    """Return nonnegative margin-KKT violations, one per training sample.

    Box feasibility and ``y @ alpha == 0`` must be checked separately.
    ``bound_tol`` is an absolute alpha tolerance, capped for very small C.
    """
    epsilon = min(bound_tol, C * 1e-8)
    margin = y * scores
    at_lower = alpha <= epsilon
    at_upper = alpha >= C - epsilon
    return np.where(
        at_lower,
        np.maximum(0.0, 1.0 - margin),
        np.where(at_upper, np.maximum(0.0, margin - 1.0), np.abs(margin - 1.0)),
    )


def objectives(
    alpha: FloatArray, y: FloatArray, scores: FloatArray, w: FloatArray, C: float
) -> tuple[float, float]:
    """Return (primal, dual), with *summed*, rather than mean, hinge loss."""
    half_norm = 0.5 * float(w @ w)
    primal = half_norm + C * float(np.maximum(0.0, 1.0 - y * scores).sum())
    dual = float(alpha.sum()) - half_norm
    return primal, dual


def bias_interval(
    alpha: FloatArray,
    y: FloatArray,
    scores_without_bias: FloatArray,
    C: float,
    bound_tol: float = 1e-10,
) -> tuple[float, float]:
    """Return the lower/upper KKT bounds on b for fixed alpha.

    Free support vectors contribute to both sets. Before convergence the
    interval may be inverted; its midpoint minimizes the largest margin-KKT
    residual over b. At an optimum without free support vectors, it gives a
    valid interval instead of taking a mean of an empty set.
    """
    epsilon = min(bound_tol, C * 1e-8)
    can_increase = alpha < C - epsilon
    can_decrease = alpha > epsilon
    lower_set = ((y > 0) & can_increase) | ((y < 0) & can_decrease)
    upper_set = ((y < 0) & can_increase) | ((y > 0) & can_decrease)
    thresholds = y - scores_without_bias
    lower = float(np.max(thresholds[lower_set])) if lower_set.any() else -np.inf
    upper = float(np.min(thresholds[upper_set])) if upper_set.any() else np.inf
    return lower, upper
