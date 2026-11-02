import subprocess
import sys

import numpy as np
import pytest

from smo_svm import ConvergenceWarning, LinearSMO
from smo_svm.datasets import make_binary_gaussians


def assert_optimal(model, X, y):
    """Independent checks: do not use the implementation's KKT helper."""
    a, C = model.alpha_, model.C
    assert a.min() >= -1e-12
    assert a.max() <= C + 1e-12
    assert abs(y @ a) < 1e-9
    margin = y * model.decision_function(X)
    lower, upper = a <= model._bound_tol, a >= C - model._bound_tol
    free = ~(lower | upper)
    assert np.all(margin[lower] >= 1 - model.tol - 1e-10)
    assert np.all(margin[upper] <= 1 + model.tol + 1e-10)
    assert np.all(np.abs(margin[free] - 1) <= model.tol + 1e-10)
    assert model.converged_
    assert model.n_kkt_violations_ == 0
    assert model.duality_gap_ >= -1e-9
    # KKT tolerance bounds the weighted complementarity gap, not vice versa.
    assert model.duality_gap_ <= len(y) * C * model.tol + 1e-8
    history = model.history_
    assert len(history) == model.n_iter_ + 1
    assert history[0]["iteration"] == 0
    assert history[-1]["iteration"] == model.n_iter_
    assert np.diff([r["dual_objective"] for r in history]).min() >= -1e-9
    assert max(r["equality_residual"] for r in history) < 1e-9
    assert max(r["box_violation"] for r in history) < 1e-12
    assert np.diff([r["elapsed_seconds"] for r in history]).min() >= 0
    assert model.fit_time_ >= history[-1]["elapsed_seconds"] > 0


def test_known_separable_solution_and_predictions():
    X, y = np.array([[-2.0], [-1], [1], [2]]), np.array([-1, -1, 1, 1])
    model = LinearSMO(C=10, tol=1e-8, random_state=0).fit(X, y)
    assert_optimal(model, X, y)
    np.testing.assert_allclose(model.w_, [1], atol=1e-8)
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_allclose(model.bias_, 0, atol=1e-8)
    assert model.predict([[0]])[0] == 1


@pytest.mark.parametrize("C", [1e-6, 0.05, 1, 10])
@pytest.mark.parametrize("seed", [0, 7, 23])
def test_overlapping_data_feasibility_kkt_and_objective_history(C, seed):
    X, y = make_binary_gaussians(90, seed)
    model = LinearSMO(C=C, tol=1e-5, max_iter=3000, random_state=seed).fit(X, y)
    assert_optimal(model, X, y)
    if C >= 0.05:
        assert np.mean(model.predict(X) == y) >= 0.82


def test_all_identical_samples_with_unequal_class_counts():
    X = np.zeros((7, 2))
    y = np.array([-1, -1, 1, 1, 1, 1, 1])
    model = LinearSMO(C=0.3, tol=1e-8, random_state=4).fit(X, y)
    assert_optimal(model, X, y)
    assert model.n_degenerate_steps_ > 0
    assert model.bias_ == pytest.approx(1)
    assert model.dual_objective_ == pytest.approx(1.2)


def test_no_free_support_vectors_has_finite_valid_bias():
    X, y = np.array([[-1.0], [1]]), np.array([-1, 1])
    model = LinearSMO(C=0.1, tol=1e-8, random_state=0).fit(X, y)
    assert_optimal(model, X, y)
    assert not model._free_mask().any()
    assert model.bias_interval_[0] <= model.bias_ <= model.bias_interval_[1]


def test_collinear_features_and_conflicting_duplicate_samples():
    X, y = make_binary_gaussians(65, 13)
    X = np.column_stack((X, 2 * X[:, 0], np.zeros(len(X))))
    X = np.vstack((X, X[0], X[0]))
    y = np.concatenate((y, [1, -1]))
    model = LinearSMO(C=0.8, tol=1e-5, max_iter=4000, random_state=12).fit(X, y)
    assert_optimal(model, X, y)


def test_reproducibility_refit_and_input_copying():
    X, y = make_binary_gaussians(50, 4)
    model = LinearSMO(tol=1e-5, random_state=9).fit(X, y)
    old_alpha, old_scores, old_updates = model.alpha_.copy(), model.decision_function(X), model.n_pair_updates_
    assert model.fit(X, y) is model
    np.testing.assert_array_equal(model.alpha_, old_alpha)
    assert model.n_pair_updates_ == old_updates
    model2 = LinearSMO(tol=1e-5, random_state=9).fit(X, y)
    np.testing.assert_array_equal(model2.alpha_, old_alpha)
    X[:] = 100
    y[:] = -1
    assert not np.all(model.X_ == 100)
    assert set(model.y_) == {-1, 1}
    np.testing.assert_array_equal(model.decision_function(model.X_), old_scores)


def test_max_iter_reports_failure_and_keeps_feasible_model():
    X, y = make_binary_gaussians(80, 0)
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        model = LinearSMO(max_iter=1, tol=1e-10, random_state=0).fit(X, y)
    assert not model.converged_
    assert model.n_iter_ == 1
    assert model.stop_reason_ == "max_iter"
    assert abs(y @ model.alpha_) < 1e-12
    assert np.isfinite(model.decision_function(X)).all()


def test_zero_update_scan_is_not_misreported_as_convergence(monkeypatch):
    X, y = make_binary_gaussians(20, 0)
    model = LinearSMO(random_state=0)
    monkeypatch.setattr(model, "_take_step", lambda i, j: False)
    with pytest.warns(ConvergenceWarning, match="numerical_stall"):
        model.fit(X, y)
    assert not model.converged_
    assert model.max_kkt_violation_ > model.tol


@pytest.mark.parametrize("parameter,value", [
    ("C", 0), ("C", -1), ("C", np.inf), ("C", True), ("C", "1"),
    ("tol", 0), ("tol", np.nan), ("max_iter", 0), ("max_iter", 1.5),
    ("max_iter", True), ("random_state", -1), ("random_state", 1.5),
])
def test_invalid_parameters(parameter, value):
    with pytest.raises(ValueError):
        LinearSMO(**{parameter: value}).fit([[-1], [1]], [-1, 1])


@pytest.mark.parametrize("X,y", [
    ([1, 2], [-1, 1]), ([[1], [np.nan]], [-1, 1]),
    ([[1], [2]], [0, 1]), ([[1], [2]], [1, 1]),
    ([[1], [2]], [-1]), ([[1], [2]], [[-1], [1]]),
    (np.empty((2, 0)), [-1, 1]), (np.empty((0, 2)), []),
    ([[1j], [1]], [-1, 1]), ([[1e200], [-1e200]], [-1, 1]),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        LinearSMO().fit(X, y)


def test_prediction_validation_and_support_vector_representation():
    model = LinearSMO()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[0]])
    model.fit([[-1], [1]], [-1, 1])
    with pytest.raises(ValueError, match="features"):
        model.predict([[1, 2]])
    assert model.predict(np.empty((0, 1))).shape == (0,)
    scores_from_support = (model.support_vectors_ @ model.X_.T).T @ model.dual_coef_.ravel() + model.intercept_[0]
    np.testing.assert_allclose(scores_from_support, model.decision_function(model.X_))


def test_solver_runs_with_sklearn_imports_blocked():
    code = '''
import builtins
original_import = builtins.__import__
def restricted_import(name, *args, **kwargs):
    if name.split(".")[0] in {"sklearn", "scipy"}:
        raise AssertionError("optimizer must not import sklearn/scipy")
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted_import
from smo_svm import LinearSMO
model = LinearSMO(random_state=0).fit([[-1.0], [1.0]], [-1, 1])
assert model.converged_
assert list(model.predict([[-2.0], [2.0]])) == [-1, 1]
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_failed_refit_cannot_expose_partially_overwritten_model():
    model = LinearSMO().fit([[-1], [1]], [-1, 1])
    with pytest.raises(ValueError):
        model.fit([[1e200, 0], [-1e200, 0]], [-1, 1])
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1, 0]])
    model.fit([[-1], [1]], [-1, 1])
    assert model.predict([[1]])[0] == 1


@pytest.mark.parametrize("seed", [1, 14, 31])
def test_high_dimensional_noisy_problem(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 7))
    truth = rng.normal(size=7)
    y = np.where(X @ truth + 0.6 * rng.normal(size=len(X)) >= 0, 1, -1)
    model = LinearSMO(C=0.6, tol=1e-5, max_iter=4000, random_state=seed).fit(X, y)
    assert_optimal(model, X, y)
    # Label noise means even the correctly optimized solution can be below 90%.
    # Check the general-dimensional optimization invariants here; prediction
    # agreement with a reference is checked separately.


def test_full_scans_are_never_starved_by_non_bound_updates():
    X, y = make_binary_gaussians(90, 7)
    model = LinearSMO(C=0.05, tol=1e-7, random_state=7).fit(X, y)
    consecutive_free = 0
    for record in model.history_[1:]:
        consecutive_free = consecutive_free + 1 if record["scan"] == "non_bound" else 0
        assert consecutive_free <= 5
