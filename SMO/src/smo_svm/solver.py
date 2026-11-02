"""SMO derived from the equality-constrained two-variable dual subproblem.

Only NumPy and the Python standard library are used here. The implementation
favors readable state transitions and exact full-data diagnostics over speed.
See docs/mathematics.md for the derivation and numerical conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from time import perf_counter
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .diagnostics import bias_interval, kkt_residuals, objectives

FloatArray = NDArray[np.float64]


class ConvergenceWarning(RuntimeWarning):
    """Training stopped before the requested full-data KKT tolerance."""


@dataclass(frozen=True)
class IterationRecord:
    """One scan, or the initial state at iteration zero."""

    iteration: int
    scan: str
    pair_updates: int
    total_pair_updates: int
    dual_objective: float
    primal_objective: float
    duality_gap: float
    max_kkt_violation: float
    n_kkt_violations: int
    equality_residual: float
    box_violation: float
    elapsed_seconds: float


def pair_bounds(ai: float, aj: float, yi: float, yj: float, C: float) -> tuple[float, float]:
    """Feasible interval for the new alpha_j, preserving the pair equality."""
    if yi != yj:
        return max(0.0, aj - ai), min(C, C + aj - ai)
    return max(0.0, ai + aj - C), min(C, ai + aj)


class LinearSMO:
    """Binary C-SVM with a linear kernel and a NumPy SMO optimizer.

    Parameters
    ----------
    C : float, default=1.0
        Positive coefficient of the **sum** of hinge losses.
    tol : float, default=1e-3
        Absolute margin-KKT tolerance, checked on every training sample.
    max_iter : int, default=2000
        Maximum number of scans (full or non-bound), not pair updates.
    random_state : int or None, default=None
        Seed for scan permutations and randomized candidate tie breaking.

    Notes
    -----
    Labels must contain both -1 and +1. Input is dense, finite float64 data.
    No scaling is performed internally; scale features before fitting.
    A dense Gram matrix uses O(n_samples**2) memory. This is an educational
    solver, without shrinking, sparse input, warm starts, or kernel options.
    """

    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 2000,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self._is_fitted = False

    def _validate_parameters(self) -> None:
        for name in ("C", "tol"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite positive real number")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive real number")
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, Integral)
            or self.max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None")

    @staticmethod
    def _as_features(X: ArrayLike) -> FloatArray:
        if np.iscomplexobj(X):
            raise ValueError("X must be real-valued")
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must have shape (n_samples, n_features), with n_features > 0")
        if not np.isfinite(array).all():
            raise ValueError("X must contain only finite values")
        return array

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearSMO:
        """Fit from alpha=0 and return self; reset all state on each call.

        On unsuccessful termination, retain the feasible current model, set
        ``converged_=False`` and emit :class:`ConvergenceWarning`.
        """
        started = perf_counter()
        self._is_fitted = False
        self._validate_parameters()
        features = self._as_features(X)
        if np.iscomplexobj(y):
            raise ValueError("y must contain both -1 and +1")
        labels = np.asarray(y, dtype=np.float64)
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("y must be one-dimensional and match the number of samples")
        if not np.array_equal(np.unique(labels), [-1.0, 1.0]):
            raise ValueError("y must contain both -1 and +1, and no other labels")

        # Copy inputs so a caller cannot silently invalidate training state.
        self.X_ = features.copy()
        self.y_ = labels.copy()
        self.n_features_in_ = features.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        with np.errstate(over="ignore", invalid="ignore"):
            self._kernel = self.X_ @ self.X_.T
        if not np.isfinite(self._kernel).all():
            raise ValueError("linear kernel overflowed; rescale X")

        self.alpha_ = np.zeros(len(labels), dtype=np.float64)
        self.bias_ = 0.0
        self._errors = -self.y_.copy()
        # Bound classification is much tighter than margin convergence.
        self._bound_tol = min(1e-10, float(self.C) * 1e-8)
        self.n_iter_ = 0
        self.n_pair_updates_ = 0
        self.n_degenerate_steps_ = 0
        self.history_: list[dict[str, object]] = []
        self.converged_ = False
        self.stop_reason_ = "max_iter"
        self._refresh_state()
        self._record("initial", 0, started)

        scan_all = True
        non_bound_scans = 0
        for iteration in range(1, int(self.max_iter) + 1):
            self.n_iter_ = iteration
            mode = "full" if scan_all else "non_bound"
            indices = np.arange(len(labels)) if scan_all else np.flatnonzero(self._free_mask())
            updates = 0
            for i in self._rng.permutation(indices):
                if self._violates_kkt(int(i)) and self._examine_example(int(i)):
                    updates += 1

            # Recompute instead of letting incremental cache drift accumulate.
            # The minimax bias also handles the absence of free support vectors.
            self._refresh_state()
            self._record(mode, updates, started)
            if self.max_kkt_violation_ <= self.tol:
                self.converged_ = True
                self.stop_reason_ = "kkt_tolerance"
                break
            if scan_all and updates == 0:
                self.stop_reason_ = "numerical_stall"
                break
            if scan_all:
                scan_all = False
                non_bound_scans = 0
            else:
                non_bound_scans += 1
                # A nearly stationary free subset may still make tiny steps.
                # Bound variables must not wait indefinitely for exactly zero
                # updates: perform a full scan after at most five free scans.
                scan_all = updates == 0 or non_bound_scans >= 5

        self.support_ = np.flatnonzero(self.alpha_ > self._bound_tol)
        self.support_vectors_ = self.X_[self.support_].copy()
        self.dual_coef_ = (self.alpha_[self.support_] * self.y_[self.support_])[None, :]
        self.coef_ = self.w_[None, :].copy()
        self.intercept_ = np.array([self.bias_])
        self.classes_ = np.array([-1, 1])
        self.fit_time_ = perf_counter() - started
        self._is_fitted = True
        if not self.converged_:
            warnings.warn(
                f"SMO stopped ({self.stop_reason_}) after {self.n_iter_} scans: "
                f"max KKT violation={self.max_kkt_violation_:.3g}, tol={self.tol:.3g}",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def _free_mask(self) -> NDArray[np.bool_]:
        return (self.alpha_ > self._bound_tol) & (self.alpha_ < self.C - self._bound_tol)

    def _violates_kkt(self, i: int) -> bool:
        # y_i E_i = y_i f_i - 1. A lower-bound variable may only increase;
        # an upper-bound variable may only decrease.
        residual = self.y_[i] * self._errors[i]
        return bool(
            (residual < -self.tol and self.alpha_[i] < self.C - self._bound_tol)
            or (residual > self.tol and self.alpha_[i] > self._bound_tol)
        )

    def _candidate_indices(self, i: int) -> NDArray[np.int64]:
        """Try non-bound j by decreasing |E_i-E_j|, then all bound j.

        A random permutation before a stable sort makes ties reproducible
        without always favoring low sample indices. Every j != i remains a
        fallback, so a single unproductive heuristic pair cannot end a scan.
        """
        candidates = self._rng.permutation(np.delete(np.arange(len(self.y_)), i))
        free = self._free_mask()[candidates]
        error_difference = np.abs(self._errors[i] - self._errors[candidates])
        free_candidates = candidates[free]
        free_order = np.argsort(-error_difference[free], kind="stable")
        bound_candidates = candidates[~free]
        bound_order = np.argsort(-error_difference[~free], kind="stable")
        return np.concatenate((free_candidates[free_order], bound_candidates[bound_order]))

    def _examine_example(self, i: int) -> bool:
        for j in self._candidate_indices(i):
            if self._take_step(i, int(j)):
                return True
        return False

    def _take_step(self, i: int, j: int) -> bool:
        """Maximize the dual on one feasible pair segment, then update b/E."""
        if i == j:
            return False
        ai, aj = float(self.alpha_[i]), float(self.alpha_[j])
        yi, yj = float(self.y_[i]), float(self.y_[j])
        low, high = pair_bounds(ai, aj, yi, yj, float(self.C))
        if high <= low:
            return False

        # For a linear kernel eta = ||x_i-x_j||^2. Computing the distance
        # directly avoids cancellation in K_ii + K_jj - 2 K_ij.
        difference = self.X_[i] - self.X_[j]
        eta = float(difference @ difference)
        slope = yj * float(self._errors[i] - self._errors[j])
        # Nearly identical errors carry no reliable directional information.
        # Rejecting their rounding-noise step is essential: otherwise a free
        # pair can repeatedly "update" and prevent the bound-j fallback.
        slope_floor = 16.0 * np.finfo(float).eps * max(
            1.0, abs(float(self._errors[i])), abs(float(self._errors[j]))
        )
        if abs(slope) <= slope_floor:
            return False
        if eta > 0.0:
            # Clip before dividing: a tiny curvature must not overflow a step
            # whose constrained value is simply one of the endpoints.
            lower_delta, upper_delta = low - aj, high - aj
            if slope <= eta * lower_delta:
                delta_j = lower_delta
            elif slope >= eta * upper_delta:
                delta_j = upper_delta
            else:
                delta_j = slope / eta
        else:
            # D(aj+t)-D(aj) = slope*t - eta*t^2/2. For eta=0 the
            # objective is linear: compare endpoints, retaining no-op on a tie.
            endpoints = np.array([low - aj, high - aj])
            gains = slope * endpoints
            best = int(np.argmax(gains))
            if gains[best] <= 0.0:
                return False
            delta_j = float(endpoints[best])

        new_aj = float(np.clip(aj + delta_j, low, high))
        delta_j = new_aj - aj
        new_ai = ai - yi * yj * delta_j
        # Snap only floating-point overshoot, updating the partner at the same
        # time. Independent clipping would break y_i*ai + y_j*aj = constant.
        if new_ai < 0.0 or new_ai > self.C:
            new_ai = float(np.clip(new_ai, 0.0, self.C))
            delta_j = -yi * yj * (new_ai - ai)
            new_aj = aj + delta_j
        delta_i = new_ai - ai
        step_floor = 8.0 * np.finfo(float).eps * max(
            abs(ai), abs(aj), abs(new_ai), abs(new_aj), np.finfo(float).tiny
        )
        if abs(delta_j) <= step_floor:
            return False
        gain = slope * delta_j - 0.5 * eta * delta_j * delta_j
        if gain <= 0.0:
            return False

        old_bias = self.bias_
        kii, kij, kjj = self._kernel[i, i], self._kernel[i, j], self._kernel[j, j]
        bi = old_bias - self._errors[i] - yi * delta_i * kii - yj * delta_j * kij
        bj = old_bias - self._errors[j] - yi * delta_i * kij - yj * delta_j * kjj
        if self._bound_tol < new_ai < self.C - self._bound_tol:
            self.bias_ = float(bi)
        elif self._bound_tol < new_aj < self.C - self._bound_tol:
            self.bias_ = float(bj)
        else:
            self.bias_ = float(0.5 * (bi + bj))
        self.alpha_[i], self.alpha_[j] = new_ai, new_aj
        self._errors += (
            yi * delta_i * self._kernel[:, i]
            + yj * delta_j * self._kernel[:, j]
            + self.bias_ - old_bias
        )
        self.n_pair_updates_ += 1
        self.n_degenerate_steps_ += int(eta == 0.0)
        return True

    def _refresh_state(self) -> None:
        self.w_ = (self.alpha_ * self.y_) @ self.X_
        raw_scores = self.X_ @ self.w_
        lower, upper = bias_interval(
            self.alpha_, self.y_, raw_scores, float(self.C), self._bound_tol
        )
        self.bias_interval_ = (lower, upper)
        if np.isfinite(lower) and np.isfinite(upper):
            self.bias_ = 0.5 * lower + 0.5 * upper
        elif np.isfinite(lower):
            self.bias_ = max(self.bias_, lower)
        elif np.isfinite(upper):
            self.bias_ = min(self.bias_, upper)
        self._errors = raw_scores + self.bias_ - self.y_

    def _record(self, scan: str, updates: int, started: float) -> None:
        scores = self._errors + self.y_
        residuals = kkt_residuals(
            self.alpha_, self.y_, scores, float(self.C), self._bound_tol
        )
        primal, dual = objectives(self.alpha_, self.y_, scores, self.w_, float(self.C))
        self.kkt_residuals_ = residuals
        self.max_kkt_violation_ = float(residuals.max())
        self.n_kkt_violations_ = int(np.count_nonzero(residuals > self.tol))
        self.primal_objective_, self.dual_objective_ = primal, dual
        # Keep the raw gap: a negative value should reveal numerical error,
        # rather than being hidden by clipping.
        self.duality_gap_ = primal - dual
        self.equality_residual_ = abs(float(self.y_ @ self.alpha_))
        box_violation = max(
            0.0, float(-self.alpha_.min()), float(self.alpha_.max() - self.C)
        )
        record = IterationRecord(
            self.n_iter_, scan, updates, self.n_pair_updates_, dual, primal,
            self.duality_gap_, self.max_kkt_violation_, self.n_kkt_violations_,
            self.equality_residual_, box_violation, perf_counter() - started,
        )
        self.history_.append(asdict(record))

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Return w @ x + b for each row; positive scores favor class +1."""
        if not self._is_fitted:
            raise RuntimeError("call fit() before prediction")
        features = self._as_features(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        return features @ self.w_ + self.bias_

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Return labels in {-1, +1}; an exactly zero score maps to +1."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
