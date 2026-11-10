"""Gaussian and multinomial naive Bayes classifiers.

Naive Bayes assumes features are conditionally independent given the class, so

    log P(y=k | x) = log P(y=k) + sum_j log P(x_j | y=k) - log P(x)

and the classifier picks the k with the largest joint log-likelihood. All
computation stays in log space; ``logsumexp`` normalises to probabilities.
"""

from __future__ import annotations

import numpy as np

from mlfs._linalg import logsumexp
from mlfs.metrics import accuracy_score


class _BaseNB:
    classes_: np.ndarray

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(np.asarray(X, dtype=float))
        return jll - logsumexp(jll, axis=1)[:, None]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(np.asarray(X, dtype=float))
        return self.classes_[jll.argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


class GaussianNB(_BaseNB):
    """Each feature is modelled as N(mu_kj, sigma_kj^2) within class k.

    ``var_smoothing`` adds a fraction of the largest feature variance to every
    class variance, so constant-within-class features (variance 0, common in
    image data) do not produce infinite log-densities.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNB:
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.stack([X[y_idx == c].mean(axis=0) for c in range(k)])
        self.var_ = np.stack([X[y_idx == c].var(axis=0) for c in range(k)]) + self.epsilon_
        self.class_log_prior_ = np.log(np.bincount(y_idx, minlength=k) / len(y_idx))
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        # sum_j log N(x_j; mu_kj, var_kj) for every sample/class pair, shape (n, k).
        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_), axis=1)  # (k,)
        sq = (X[:, None, :] - self.theta_[None, :, :]) ** 2 / self.var_[None, :, :]
        return self.class_log_prior_ + log_norm - 0.5 * sq.sum(axis=2)


class MultinomialNB(_BaseNB):
    """Features are non-negative counts drawn from a per-class multinomial.

    log P(x | y=k) = sum_j x_j log theta_kj + const, with Laplace/Lidstone
    smoothing theta_kj = (N_kj + alpha) / (N_k + alpha * d).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultinomialNB:
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("MultinomialNB needs non-negative feature counts")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]
        feature_counts = Y.T @ X + self.alpha  # (k, d)
        self.feature_log_prob_ = np.log(feature_counts) - np.log(
            feature_counts.sum(axis=1, keepdims=True)
        )
        self.class_log_prior_ = np.log(Y.sum(axis=0) / len(y_idx))
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return X @ self.feature_log_prob_.T + self.class_log_prior_
