"""Central finite-difference helper shared by the autograd tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6):
    """d f / d x by central differences, (f(x+h) - f(x-h)) / 2h, one entry at a time.

    The truncation error is O(h^2), so with float64 and h = 1e-6 agreement to
    ~1e-7 relative is expected for smooth functions.
    """
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad
