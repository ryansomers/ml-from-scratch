"""CART decision-tree classifier and a random forest built from it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfs.metrics import accuracy_score


def gini(counts: np.ndarray) -> np.ndarray:
    """Gini impurity 1 - sum_k p_k^2 for each row of class counts (0 if empty)."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return np.where(n[..., 0] > 0, 1.0 - np.sum(p**2, axis=-1), 0.0)


def entropy(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy -sum_k p_k log2 p_k (with 0 log 0 = 0) for each row."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -np.sum(terms, axis=-1)


CRITERIA = {"gini": gini, "entropy": entropy}


@dataclass
class Node:
    """A tree node. Leaves have ``feature is None``."""

    counts: np.ndarray  # class counts of the training samples that reached here
    feature: int | None = None
    threshold: float = 0.0  # go left when x[feature] <= threshold
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _n_features_to_try(max_features: int | float | str | None, d: int) -> int:
    if max_features is None:
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    if isinstance(max_features, float) and 0.0 < max_features <= 1.0:
        return max(1, int(max_features * d))
    if isinstance(max_features, int) and 1 <= max_features <= d:
        return max_features
    raise ValueError(f"invalid max_features={max_features!r}")


class DecisionTreeClassifier:
    """Binary-split classification tree grown greedily (CART).

    At every node we try each candidate feature and every threshold halfway
    between consecutive distinct sorted values, and keep the split that
    minimises the size-weighted impurity of the two children.

    Parameters
    ----------
    criterion : "gini" or "entropy".
    max_depth : maximum depth (root has depth 0); None means unlimited.
    min_samples_split : a node with fewer samples becomes a leaf.
    min_samples_leaf : each child of a split must keep at least this many.
    max_features : features examined per split (None = all; "sqrt", "log2",
        int or fraction). Random forests use this for decorrelation.
    random_state : seed for feature subsampling.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {list(CRITERIA)}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    # -- training ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        return self._fit_encoded(np.asarray(X, dtype=float), y_idx, len(self.classes_))

    def _fit_encoded(self, X: np.ndarray, y_idx: np.ndarray, n_classes: int):
        """Fit on labels already encoded as 0..n_classes-1 (used by the forest,
        whose bootstrap samples may be missing some classes)."""
        self.n_classes_ = n_classes
        self.n_features_in_ = X.shape[1]
        self._impurity = CRITERIA[self.criterion]
        self._rng = np.random.default_rng(self.random_state)
        self._m = _n_features_to_try(self.max_features, X.shape[1])
        self._importance = np.zeros(X.shape[1])
        self.root_ = self._grow(X, y_idx, depth=0)
        total = self._importance.sum()
        self.feature_importances_ = self._importance / total if total > 0 else self._importance
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        node = Node(counts=counts)
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or np.count_nonzero(counts) <= 1
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, child_impurity = split
        # Mean decrease in impurity, weighted by the fraction of samples here.
        parent_impurity = float(self._impurity(counts))
        self._importance[feature] += len(y) * (parent_impurity - child_impurity)

        go_left = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[go_left], y[go_left], depth + 1)
        node.right = self._grow(X[~go_left], y[~go_left], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        n, d = X.shape
        features = (
            np.arange(d) if self._m == d else self._rng.choice(d, size=self._m, replace=False)
        )
        onehot = np.eye(self.n_classes_)[y]
        total = onehot.sum(axis=0)
        n_left = np.arange(1, n)  # split after position i keeps i+1 samples left
        n_right = n - n_left
        leaf_ok = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)

        best: tuple[int, float, float] | None = None
        for f in features:
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            # Class counts on the left of every candidate split, all at once.
            left = np.cumsum(onehot[order], axis=0)[:-1]
            right = total - left
            weighted = (n_left * self._impurity(left) + n_right * self._impurity(right)) / n
            # Only split between distinct values.
            valid = leaf_ok & (xs[1:] > xs[:-1])
            if not valid.any():
                continue
            weighted = np.where(valid, weighted, np.inf)
            i = int(np.argmin(weighted))
            if best is None or weighted[i] < best[2]:
                best = (int(f), float((xs[i] + xs[i + 1]) / 2.0), float(weighted[i]))
        return best

    # -- inference --------------------------------------------------------

    def _leaf_counts(self, X: np.ndarray) -> np.ndarray:
        """Route all rows down the tree together, splitting index sets at each node."""
        out = np.empty((len(X), self.n_classes_))
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                out[idx] = node.counts
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        counts = self._leaf_counts(np.asarray(X, dtype=float))
        return counts / counts.sum(axis=1, keepdims=True)

    def _predict_encoded(self, X: np.ndarray) -> np.ndarray:
        return self._leaf_counts(X).argmax(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._predict_encoded(np.asarray(X, dtype=float))]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def depth_(self) -> int:
        def depth(node: Node) -> int:
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))

        return depth(self.root_)


class RandomForestClassifier:
    """Bagged decision trees with per-split feature subsampling.

    Each tree is trained on a bootstrap sample (n draws with replacement) and
    considers only ``max_features`` random features at every split, which
    decorrelates the trees so that averaging reduces variance. Prediction is a
    hard majority vote over the trees (ties go to the lowest class index).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.estimators_: list[DecisionTreeClassifier] = []
        for _ in range(self.n_estimators):
            rows = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(2**32)),
            )
            self.estimators_.append(tree._fit_encoded(X[rows], y_idx[rows], k))
        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0
        )
        return self

    def _votes(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        votes = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))
        for tree in self.estimators_:
            votes[rows, tree._predict_encoded(X)] += 1
        return votes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fraction of trees voting for each class."""
        return self._votes(X) / self.n_estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._votes(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
