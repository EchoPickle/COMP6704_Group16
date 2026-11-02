"""Binary soft-margin linear SVM, optimized entirely with NumPy."""

from .solver import ConvergenceWarning, LinearSMO

__all__ = ["ConvergenceWarning", "LinearSMO"]
