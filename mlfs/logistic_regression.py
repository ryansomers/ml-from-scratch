"""Binary logistic regression and multinomial (softmax) regression with L2.

Both minimise the mean negative log-likelihood plus an L2 penalty on the
weights (never on the intercept):

    J(w, b) = (1/n) * sum_i NLL_i + (lam / 2) * ||w||^2

This matches scikit-learn's ``LogisticRegression(C=...)`` when
``lam = 1 / (C * n)``, because sklearn minimises C * sum_i NLL_i + ||w||^2 / 2.

Gradient derivations are in docs/derivations.md.
"""

from __future__ import annotations

import numpy as np

from mlfs._linalg import largest_singular_value_sq, log_softmax
from mlfs.metrics import accuracy_score


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, split by sign so exp never overflows."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _add_bias_column(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))])


class LogisticRegression:
    """Binary logistic regression.

    Parameters
    ----------
    lam : L2 strength on the weights (see module docstring).
    solver : "gd" (batch gradient descent, step 1/L) or "newton" (Newton's
        method, equivalently iteratively reweighted least squares).
    lr : gradient-descent step size; ``None`` uses 1/L where
        L = sigma_max(X)^2 / (4n) + lam bounds the Hessian (sigmoid' <= 1/4).
    max_iter, tol : stop when the gradient (gd) or Newton step (newton) has
        norm below ``tol``.
    """

    def __init__(
        self,
        lam: float = 0.0,
        solver: str = "newton",
        lr: float | None = None,
        max_iter: int = 10_000,
        tol: float = 1e-8,
    ) -> None:
        if solver not in ("gd", "newton"):
            raise ValueError(f"solver must be 'gd' or 'newton', got {solver!r}")
        self.lam = lam
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X = np.asarray(X, dtype=float)
        self.classes_, y01 = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression is binary; use SoftmaxRegression")
        y01 = y01.astype(float)
        Xb = _add_bias_column(X)
        n, d = Xb.shape
        # Penalise every parameter except the trailing bias.
        penalty = np.full(d, self.lam)
        penalty[-1] = 0.0

        theta = np.zeros(d)
        self.loss_history_: list[float] = []
        if self.solver == "gd":
            lr = self.lr or 1.0 / (largest_singular_value_sq(Xb) / (4 * n) + self.lam)
        for _ in range(self.max_iter):
            z = Xb @ theta
            p = sigmoid(z)
            self.loss_history_.append(self._loss(z, y01, theta, penalty))
            grad = Xb.T @ (p - y01) / n + penalty * theta
            if self.solver == "gd":
                if np.linalg.norm(grad) < self.tol:
                    break
                theta -= lr * grad
            else:
                # Hessian of the mean NLL is X^T S X / n with S = diag(p(1-p)).
                s = p * (1.0 - p)
                H = (Xb.T * s) @ Xb / n + np.diag(penalty)
                step = np.linalg.solve(H, grad)
                theta -= step
                if np.linalg.norm(step) < self.tol:
                    break
        self.n_iter_ = len(self.loss_history_)
        self.coef_ = theta[:-1]
        self.intercept_ = float(theta[-1])
        return self

    @staticmethod
    def _loss(z: np.ndarray, y: np.ndarray, theta: np.ndarray, penalty: np.ndarray) -> float:
        # NLL_i = log(1 + e^{z_i}) - y_i z_i, with log(1+e^z) = logaddexp(0, z).
        nll = np.mean(np.logaddexp(0.0, z) - y * z)
        return float(nll + 0.5 * np.sum(penalty * theta**2))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


class SoftmaxRegression:
    """Multinomial logistic regression trained by batch gradient descent.

    The gradient of the mean cross-entropy w.r.t. the logits is (P - Y) / n,
    where P holds softmax probabilities and Y one-hot labels, so the weight
    gradient is X^T (P - Y) / n + lam * W.

    The default step 1/L uses L = sigma_max(X)^2 / (2n) + lam: the Hessian of
    softmax cross-entropy w.r.t. one row's logits, diag(p) - p p^T, has
    eigenvalues at most 1/2.
    """

    def __init__(
        self,
        lam: float = 0.0,
        lr: float | None = None,
        max_iter: int = 5_000,
        tol: float = 1e-6,
    ) -> None:
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> SoftmaxRegression:
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        Y = np.eye(k)[y_idx]
        Xb = _add_bias_column(X)
        lr = self.lr or 1.0 / (largest_singular_value_sq(Xb) / (2 * n) + self.lam)

        W = np.zeros((d, k))
        b = np.zeros(k)
        self.loss_history_: list[float] = []
        for _ in range(self.max_iter):
            logp = log_softmax(X @ W + b)
            loss = -np.mean(np.sum(Y * logp, axis=1)) + 0.5 * self.lam * np.sum(W**2)
            self.loss_history_.append(float(loss))
            dZ = (np.exp(logp) - Y) / n
            grad_W = X.T @ dZ + self.lam * W
            grad_b = dZ.sum(axis=0)
            if np.sqrt(np.sum(grad_W**2) + np.sum(grad_b**2)) < self.tol:
                break
            W -= lr * grad_W
            b -= lr * grad_b
        self.n_iter_ = len(self.loss_history_)
        self.coef_ = W
        self.intercept_ = b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(log_softmax(self.decision_function(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
