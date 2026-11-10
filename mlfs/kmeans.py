"""k-means clustering (Lloyd's algorithm) with k-means++ initialisation."""

from __future__ import annotations

import numpy as np


def squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (len(X), len(C)).

    Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, which turns the work into one
    matrix product; tiny negatives from round-off are clipped to zero.
    """
    d2 = (X**2).sum(axis=1)[:, None] - 2.0 * X @ C.T + (C**2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose k initial centres by D^2 sampling (Arthur & Vassilvitskii, 2007).

    The first centre is a uniformly random point; each later centre is a point
    sampled with probability proportional to its squared distance to the
    nearest centre chosen so far. This spreads the centres out and gives an
    O(log k)-competitive expected inertia before any Lloyd iterations.
    """
    n = len(X)
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    closest = squared_distances(X, centers[:1]).ravel()
    for i in range(1, k):
        total = closest.sum()
        # All remaining points coincide with a centre: any choice is as good.
        idx = rng.integers(n) if total == 0 else rng.choice(n, p=closest / total)
        centers[i] = X[idx]
        closest = np.minimum(closest, squared_distances(X, centers[i : i + 1]).ravel())
    return centers


class KMeans:
    """Partition points into k clusters minimising within-cluster squared distance.

    Parameters
    ----------
    n_clusters : number of clusters k.
    n_init : independent restarts; the run with the lowest inertia is kept.
    max_iter, tol : stop a run when no centre moves more than ``tol``
        (squared distance) or after ``max_iter`` Lloyd iterations.
    init : "k-means++" or "random" (k distinct points chosen uniformly).
    random_state : seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-8,
        init: str = "k-means++",
        random_state: int | None = None,
    ) -> None:
        if init not in ("k-means++", "random"):
            raise ValueError(f"unknown init {init!r}")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=float)
        if len(X) < self.n_clusters:
            raise ValueError("need at least n_clusters samples")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run = self._single_run(X, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def _single_run(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        k = self.n_clusters
        if self.init == "k-means++":
            centers = kmeans_plus_plus(X, k, rng)
        else:
            centers = X[rng.choice(len(X), size=k, replace=False)].copy()

        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            # Assignment step: each point goes to its nearest centre.
            d2 = squared_distances(X, centers)
            labels = d2.argmin(axis=1)
            # Update step: each centre moves to the mean of its points.
            new_centers = centers.copy()
            counts = np.bincount(labels, minlength=k)
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] = X[labels == j].mean(axis=0)
                else:
                    # Empty cluster: re-seed it at the point currently worst
                    # served, which strictly lowers the objective.
                    worst = d2[np.arange(len(X)), labels].argmax()
                    new_centers[j] = X[worst]
            shift = ((new_centers - centers) ** 2).sum(axis=1).max()
            centers = new_centers
            if shift <= self.tol:
                break

        d2 = squared_distances(X, centers)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        return centers, labels, inertia, n_iter

    def predict(self, X: np.ndarray) -> np.ndarray:
        return squared_distances(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def score(self, X: np.ndarray) -> float:
        """Negative inertia on X (higher is better, as in scikit-learn)."""
        X = np.asarray(X, dtype=float)
        return -float(squared_distances(X, self.cluster_centers_).min(axis=1).sum())
