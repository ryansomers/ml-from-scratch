"""Principal component analysis via the singular value decomposition.

For centred data X_c = U S V^T, the rows of V^T are the principal directions
(eigenvectors of the covariance X_c^T X_c / (n - 1)) and the variance along
direction i is s_i^2 / (n - 1). Working from the SVD of X_c rather than an
eigendecomposition of the covariance avoids squaring the condition number.
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Linear dimensionality reduction onto the top principal components.

    Parameters
    ----------
    n_components : int, float or None
        int: keep that many components. float in (0, 1): keep the smallest
        number of components whose explained variance ratio sums to at least
        this value. None: keep min(n_samples, n_features).
    """

    def __init__(self, n_components: int | float | None = None) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> PCA:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)

        # The SVD is unique only up to the sign of each singular-vector pair.
        # Fix it so the largest-magnitude loading of each component is positive,
        # which makes results reproducible across runs and platforms.
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]

        variance = S**2 / (n - 1)
        ratio = variance / variance.sum()
        k = self._resolve_n_components(ratio, max_k=min(n, d))

        self.n_components_ = k
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = ratio[:k]
        return self

    def _resolve_n_components(self, ratio: np.ndarray, max_k: int) -> int:
        nc = self.n_components
        if nc is None:
            return max_k
        if isinstance(nc, float) and 0.0 < nc < 1.0:
            # First index where the cumulative ratio reaches the target.
            # The small tolerance guards against round-off at exact targets.
            return int(np.searchsorted(np.cumsum(ratio), nc - 1e-12) + 1)
        if isinstance(nc, int) and 1 <= nc <= max_k:
            return nc
        raise ValueError(f"invalid n_components={nc!r} for max {max_k}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map projected points back to the original space (lossy if k < d)."""
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_
