"""sklearn is used exclusively as an external verification baseline."""

import numpy as np
import pytest
from sklearn.svm import SVC

from smo_svm import LinearSMO
from smo_svm.datasets import make_binary_gaussians


@pytest.mark.parametrize("C", [0.05, 0.7, 5.0])
@pytest.mark.parametrize("seed", [0, 5, 19])
def test_against_svc_on_objectives_weights_scores_and_unseen_predictions(C, seed):
    X, y = make_binary_gaussians(100, seed)
    query, query_y = make_binary_gaussians(400, seed + 1000)
    custom = LinearSMO(C=C, tol=1e-6, max_iter=5000, random_state=seed).fit(X, y)
    reference = SVC(C=C, kernel="linear", tol=1e-8, shrinking=False).fit(X, y)
    assert custom.converged_
    reference_alpha = np.zeros(len(y))
    reference_alpha[reference.support_] = np.abs(reference.dual_coef_[0])
    reference_dual = reference_alpha.sum() - 0.5 * np.sum(reference.coef_ ** 2)
    assert custom.dual_objective_ == pytest.approx(reference_dual, abs=2e-5, rel=2e-7)
    np.testing.assert_allclose(custom.coef_, reference.coef_, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(custom.intercept_, reference.intercept_, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(custom.decision_function(query), reference.decision_function(query), atol=2e-3)
    assert np.mean(custom.predict(query) == reference.predict(query)) >= 0.995
    assert abs(np.mean(custom.predict(query) == query_y) - np.mean(reference.predict(query) == query_y)) <= 0.005


def test_nonunique_alpha_is_compared_by_function_and_objective():
    X = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    y = np.array([-1, -1, 1, 1])
    custom = LinearSMO(C=1, tol=1e-8, random_state=2).fit(X, y)
    reference = SVC(C=1, kernel="linear", tol=1e-9).fit(X, y)
    np.testing.assert_allclose(custom.decision_function(X), reference.decision_function(X), atol=1e-8)
    assert custom.dual_objective_ == pytest.approx(0.5)


@pytest.mark.parametrize("seed", [1, 14, 31])
def test_high_dimensional_scores_match_svc(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 7))
    truth = rng.normal(size=7)
    y = np.where(X @ truth + 0.6 * rng.normal(size=len(X)) >= 0, 1, -1)
    custom = LinearSMO(C=0.6, tol=1e-5, max_iter=4000, random_state=seed).fit(X, y)
    reference = SVC(C=0.6, kernel="linear", tol=1e-8, shrinking=False).fit(X, y)
    query = rng.normal(size=(300, 7))
    np.testing.assert_allclose(custom.decision_function(query), reference.decision_function(query), atol=5e-4)
    assert np.mean(custom.predict(query) == reference.predict(query)) >= 0.995
