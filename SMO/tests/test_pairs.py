"""Check two-variable algebra independently of the outer training loop."""

import numpy as np
import pytest

from smo_svm import LinearSMO
from smo_svm.solver import pair_bounds


def state(X, y, C=1.0, alpha=None):
    """Construct a controlled feasible state for pair-component tests."""
    model = LinearSMO(C=C, tol=1e-8, random_state=0)
    model.X_ = np.asarray(X, dtype=float)
    model.y_ = np.asarray(y, dtype=float)
    model.alpha_ = np.zeros(len(y)) if alpha is None else np.asarray(alpha, dtype=float)
    model._kernel = model.X_ @ model.X_.T
    model._bound_tol = min(1e-10, C * 1e-8)
    model._rng = np.random.default_rng(0)
    model.bias_ = 0.0
    model.n_pair_updates_ = 0
    model.n_degenerate_steps_ = 0
    model._refresh_state()
    return model


def dual(model):
    signed_alpha = model.alpha_ * model.y_
    return model.alpha_.sum() - 0.5 * signed_alpha @ model._kernel @ signed_alpha


@pytest.mark.parametrize("yi,yj", [(-1, -1), (-1, 1), (1, -1), (1, 1)])
def test_pair_interval_preserves_box_and_pair_equality(yi, yj):
    C = 1.7
    for ai in (0, 0.2, C):
        for aj in (0, 0.8, C):
            low, high = pair_bounds(ai, aj, yi, yj, C)
            assert low <= aj <= high
            for new_aj in np.linspace(low, high, 11):
                new_ai = ai + yi * yj * (aj - new_aj)
                assert -1e-14 <= new_ai <= C + 1e-14
                assert yi * new_ai + yj * new_aj == pytest.approx(yi * ai + yj * aj)


def test_analytic_pair_update_and_bias():
    model = state([[-1.0], [1.0]], [-1, 1], C=2.0)
    before = dual(model)
    assert model._take_step(0, 1)
    np.testing.assert_allclose(model.alpha_, [0.5, 0.5], atol=1e-14)
    assert model.bias_ == pytest.approx(0)
    assert dual(model) > before
    margins = model.y_ * (model._kernel @ (model.alpha_ * model.y_) + model.bias_)
    np.testing.assert_allclose(margins, 1, atol=1e-14)


def test_error_cache_matches_direct_scores_after_multiple_pairs():
    model = state([[-2, -1], [-0.2, 1], [1, 1], [2, -1]], [-1, -1, 1, 1])
    for i, j in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]:
        old_alpha, old_dual = model.alpha_.copy(), dual(model)
        model._take_step(i, j)
        expected_errors = model._kernel @ (model.alpha_ * model.y_) + model.bias_ - model.y_
        np.testing.assert_allclose(model._errors, expected_errors, atol=1e-13)
        assert dual(model) >= old_dual - 1e-13
        assert model.y_ @ model.alpha_ == pytest.approx(model.y_ @ old_alpha, abs=1e-14)
        assert np.all(model.alpha_ >= -1e-14)
        assert np.all(model.alpha_ <= model.C + 1e-14)


def test_zero_curvature_conflicting_duplicates_choose_endpoint():
    model = state([[2.0, 1.0], [2.0, 1.0]], [-1, 1], C=0.7)
    assert model._take_step(0, 1)
    np.testing.assert_allclose(model.alpha_, [0.7, 0.7])
    assert dual(model) == pytest.approx(1.4)
    assert model.n_degenerate_steps_ == 1
    assert not model._take_step(0, 1)


def test_zero_curvature_flat_objective_retains_current_point():
    # The selected same-label pair has a nonzero feasible interval, but no gain.
    model = state([[1.0], [1.0], [-1.0]], [1, 1, -1], alpha=[0.2, 0.3, 0.5])
    before = model.alpha_.copy()
    assert not model._take_step(0, 1)
    np.testing.assert_array_equal(model.alpha_, before)


def test_tiny_positive_curvature_uses_clipped_step_without_overflow():
    model = state([[0.0], [1e-150]], [-1, 1])
    assert model._take_step(0, 1)
    np.testing.assert_allclose(model.alpha_, [1, 1])
    assert np.isfinite(model._errors).all()


def test_candidate_selection_covers_every_other_sample_and_prioritizes_free():
    model = state([[-2], [-1], [1], [2]], [-1, -1, 1, 1], alpha=[0, 0.4, 0.4, 0])
    candidates = model._candidate_indices(0)
    assert set(candidates) == {1, 2, 3}
    assert set(candidates[:2]) == {1, 2}
    assert not model._take_step(0, 0)
