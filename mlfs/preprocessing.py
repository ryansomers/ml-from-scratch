"""Feature scaling."""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Rescale each feature to zero mean and unit variance.

    Constant features (variance 0) are left centred but unscaled, so that the
    transform never divides by zero.
    """

    def fit(self, X: np.ndarray) -> StandardScaler:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) * self.scale_ + self.mean_
