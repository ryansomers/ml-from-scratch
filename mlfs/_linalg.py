"""Shared numerical helpers."""

from __future__ import annotations

import numpy as np


def largest_singular_value_sq(X: np.ndarray, n_iter: int = 100, seed: int = 0) -> float:
    """Estimate sigma_max(X)^2, the top eigenvalue of X^T X, by power iteration.

    Used to pick a safe gradient-descent step size 1/L, where L bounds the
    curvature of a loss whose Hessian is (a multiple of) X^T X.
    """
    v = np.random.default_rng(seed).normal(size=X.shape[1])
    v /= np.linalg.norm(v)
    eig = 0.0
    for _ in range(n_iter):
        w = X.T @ (X @ v)
        eig = float(np.linalg.norm(w))
        if eig == 0.0:
            return 0.0
        v = w / eig
    return eig


def log_softmax(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(Z)) computed stably by subtracting the row maximum first."""
    shifted = Z - Z.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def logsumexp(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = Z.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(Z - m).sum(axis=axis))
