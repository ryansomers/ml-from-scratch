"""Least-squares linear regression and ridge regression, solved three ways.

All three solvers minimise the same objective

    J(w) = ||X w - y||^2 + alpha * ||w||^2

(``alpha = 0`` gives ordinary least squares). The intercept is never
penalised: when ``fit_intercept`` is true we centre X and y, solve for w on the
centred data, and recover the intercept as ``b = mean(y) - mean(X) @ w``.

Solvers
-------
normal  Solve the normal equations (X^T X + alpha I) w = X^T y with
        ``np.linalg.solve``. Fast, but squares the condition number of X.
qr      Householder QR of X (implemented below), then back substitution on
        R w = Q^T y. Numerically stabler because it never forms X^T X. Ridge is
        handled by stacking sqrt(alpha) * I under X and zeros under y.
gd      Batch gradient descent on J / (2n). The default step size is 1/L,
        where L is the Lipschitz constant of the gradient, so it converges
        without tuning.
"""

from __future__ import annotations

import numpy as np

from mlfs.metrics import r2_score

SOLVERS = ("normal", "qr", "gd")


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduced QR decomposition A = Q R via Householder reflections.

    For an m x n matrix with m >= n, returns Q (m x n, orthonormal columns) and
    R (n x n, upper triangular).

    Column k is zeroed below the diagonal by the reflection H = I - 2 v v^T,
    where v is the unit vector pointing from x = A[k:, k] to -sign(x_0)||x|| e_1.
    Choosing the sign opposite to x_0 avoids cancellation when x is already
    close to a multiple of e_1.
    """
    R = np.array(A, dtype=float, copy=True)
    m, n = R.shape
    if m < n:
        raise ValueError(f"householder_qr needs m >= n, got shape {R.shape}")
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0.0:
            continue  # column already zero below the diagonal
        v = x.copy()
        v[0] += np.copysign(norm_x, x[0])
        v /= np.linalg.norm(v)
        # Apply H to the trailing block of R from the left: R <- H R.
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        # Accumulate Q <- Q H, so that at the end A = Q R.
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    # Entries below the diagonal are now round-off; make R exactly triangular.
    return Q[:, :n], np.triu(R[:n, :])


def back_substitution(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve R x = b for upper-triangular, non-singular R."""
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if R[i, i] == 0.0:
            raise np.linalg.LinAlgError("singular triangular matrix")
        x[i] = (b[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x


def _largest_singular_value_sq(X: np.ndarray, n_iter: int = 100, seed: int = 0) -> float:
    """Estimate sigma_max(X)^2, the top eigenvalue of X^T X, by power iteration."""
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


class Ridge:
    """Linear regression with an L2 penalty ``alpha * ||w||^2``.

    Parameters
    ----------
    alpha : regularisation strength (0 means ordinary least squares).
    solver : one of "normal", "qr", "gd".
    fit_intercept : learn an unpenalised bias term.
    lr : gradient-descent step size; ``None`` uses 1/L (see module docstring).
    max_iter, tol : gradient-descent stopping rule (gradient norm below tol).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        solver: str = "normal",
        fit_intercept: bool = True,
        lr: float | None = None,
        max_iter: int = 10_000,
        tol: float = 1e-8,
    ) -> None:
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - x_mean, y - y_mean
        else:
            x_mean, y_mean = np.zeros(X.shape[1]), 0.0
            Xc, yc = X, y

        if self.solver == "normal":
            w = self._solve_normal(Xc, yc)
        elif self.solver == "qr":
            w = self._solve_qr(Xc, yc)
        else:
            w = self._solve_gd(Xc, yc)

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    def _solve_normal(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Solving the system is cheaper and more accurate than forming the inverse.
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        return np.linalg.solve(A, X.T @ y)

    def _solve_qr(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.alpha > 0:
            # ||Xw - y||^2 + alpha||w||^2 == ||[X; sqrt(a) I] w - [y; 0]||^2
            d = X.shape[1]
            X = np.vstack([X, np.sqrt(self.alpha) * np.eye(d)])
            y = np.concatenate([y, np.zeros(d)])
        Q, R = householder_qr(X)
        return back_substitution(R, Q.T @ y)

    def _solve_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, d = X.shape
        # Minimise J/(2n); its gradient is (X^T (Xw - y) + alpha w) / n and is
        # L-Lipschitz with L = (sigma_max^2 + alpha) / n.
        lr = self.lr
        if lr is None:
            lipschitz = (_largest_singular_value_sq(X) + self.alpha) / n
            lr = 1.0 / lipschitz
        w = np.zeros(d)
        self.loss_history_: list[float] = []
        for _ in range(self.max_iter):
            residual = X @ w - y
            self.loss_history_.append(float((residual @ residual + self.alpha * w @ w) / (2 * n)))
            grad = (X.T @ residual + self.alpha * w) / n
            if np.linalg.norm(grad) < self.tol:
                break
            w -= lr * grad
        self.n_iter_ = len(self.loss_history_)
        return w

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


class LinearRegression(Ridge):
    """Ordinary least squares: ``Ridge`` with ``alpha = 0``."""

    def __init__(
        self,
        solver: str = "normal",
        fit_intercept: bool = True,
        lr: float | None = None,
        max_iter: int = 10_000,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(
            alpha=0.0,
            solver=solver,
            fit_intercept=fit_intercept,
            lr=lr,
            max_iter=max_iter,
            tol=tol,
        )
