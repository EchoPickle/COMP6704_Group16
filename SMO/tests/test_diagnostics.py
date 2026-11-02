import numpy as np
import pytest

from smo_svm.diagnostics import bias_interval, kkt_residuals, objectives


def test_kkt_residuals_cover_lower_free_and_upper_cases():
    alpha = np.array([0, 0, 0.4, 0.4, 1, 1.0])
    y = np.array([1, -1, 1, -1, 1, -1.0])
    margins = np.array([1.3, 0.7, 1.2, 0.9, 0.5, 1.4])
    np.testing.assert_allclose(
        kkt_residuals(alpha, y, y * margins, C=1), [0, 0.3, 0.2, 0.1, 0, 0.4]
    )


def test_primal_dual_and_strong_duality_on_analytic_solution():
    primal, dual = objectives(
        np.array([0.5, 0.5]), np.array([-1, 1]), np.array([-1, 1]), np.array([1.0]), 1
    )
    assert primal == pytest.approx(0.5)
    assert dual == pytest.approx(0.5)


def test_bias_interval_without_free_support_vectors():
    # w=0.2 and both alpha=C=0.1: all b in [-0.8, 0.8] satisfy KKT.
    low, high = bias_interval(
        np.array([0.1, 0.1]), np.array([-1, 1]), np.array([-0.2, 0.2]), C=0.1
    )
    assert low == pytest.approx(-0.8)
    assert high == pytest.approx(0.8)


def test_bias_midpoint_minimizes_worst_kkt_residual_even_before_convergence():
    alpha, y, raw = np.array([0.5, 0.5]), np.array([-1, 1]), np.array([-0.3, 0.3])
    low, high = bias_interval(alpha, y, raw, C=1)
    assert low > high
    midpoint = (low + high) / 2
    optimal = kkt_residuals(alpha, y, raw + midpoint, 1).max()
    assert optimal == pytest.approx(0.7)
    for bias in np.linspace(-2, 2, 41):
        assert kkt_residuals(alpha, y, raw + bias, 1).max() >= optimal - 1e-14
