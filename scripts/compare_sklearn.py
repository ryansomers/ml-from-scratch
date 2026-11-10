"""Train every mlfs model and its scikit-learn counterpart on identical splits.

    python scripts/compare_sklearn.py   ->  docs/comparison.md

Protocol: 75/25 train/test split (stratified for classification) with
random_state=0; features standardised with mlfs.StandardScaler fitted on the
training split only (MultinomialNB gets raw pixel counts). Hyperparameters are
matched between the two libraries where they have a counterpart. Metrics:
test accuracy (classification), test R^2 (regression), adjusted Rand index
against the true labels (k-means, fitted on all samples).

scikit-learn is used here only as the reference implementation, data loader
and metric provider; the mlfs models themselves use only NumPy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn import cluster, decomposition, linear_model, naive_bayes, tree
from sklearn import ensemble as sk_ensemble
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.model_selection import train_test_split

import mlfs.kmeans
import mlfs.linear_regression
import mlfs.logistic_regression
import mlfs.naive_bayes
import mlfs.pca
import mlfs.tree
from mlfs.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "docs" / "comparison.md"
SEED = 0
C = 1.0  # sklearn LogisticRegression default; mlfs uses lam = 1 / (C * n_train)


@dataclass
class Row:
    dataset: str
    model: str
    metric: str
    ours: float
    theirs: float


def split(X, y, stratify: bool):
    return train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y if stratify else None
    )


def regression_rows() -> list[Row]:
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = split(X, y, stratify=False)
    sk_ols = linear_model.LinearRegression().fit(X_tr, y_tr)
    rows = []
    for solver in ("normal", "qr", "gd"):
        ours = mlfs.linear_regression.LinearRegression(solver=solver, max_iter=200_000)
        ours.fit(X_tr, y_tr)
        rows.append(
            Row(
                "diabetes",
                f"LinearRegression ({solver})",
                "R²",
                r2_score(y_te, ours.predict(X_te)),
                r2_score(y_te, sk_ols.predict(X_te)),
            )
        )
    ours = mlfs.linear_regression.Ridge(alpha=1.0, solver="qr").fit(X_tr, y_tr)
    theirs = linear_model.Ridge(alpha=1.0).fit(X_tr, y_tr)
    rows.append(
        Row(
            "diabetes",
            "Ridge α=1 (qr)",
            "R²",
            r2_score(y_te, ours.predict(X_te)),
            r2_score(y_te, theirs.predict(X_te)),
        )
    )
    return rows


def classification_rows(name: str, X: np.ndarray, y: np.ndarray) -> list[Row]:
    X_tr_raw, X_te_raw, y_tr, y_te = split(X, y, stratify=True)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    lam = 1.0 / (C * len(X_tr))
    binary = len(np.unique(y)) == 2

    pairs: list[tuple[str, object, object, bool]] = []
    if binary:
        pairs.append(
            (
                f"LogisticRegression L2, C={C} (Newton)",
                mlfs.logistic_regression.LogisticRegression(lam=lam, solver="newton"),
                linear_model.LogisticRegression(C=C, max_iter=10_000),
                False,
            )
        )
    else:
        pairs.append(
            (
                f"SoftmaxRegression L2, C={C} (GD)",
                mlfs.logistic_regression.SoftmaxRegression(lam=lam, max_iter=20_000),
                linear_model.LogisticRegression(C=C, max_iter=10_000),
                False,
            )
        )
    pairs += [
        ("GaussianNB", mlfs.naive_bayes.GaussianNB(), naive_bayes.GaussianNB(), False),
        (
            "DecisionTree (gini)",
            mlfs.tree.DecisionTreeClassifier(criterion="gini"),
            tree.DecisionTreeClassifier(criterion="gini", random_state=SEED),
            False,
        ),
        (
            "DecisionTree (entropy, depth≤5)",
            mlfs.tree.DecisionTreeClassifier(criterion="entropy", max_depth=5),
            tree.DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=SEED),
            False,
        ),
        (
            "RandomForest (100 trees, √d features)",
            mlfs.tree.RandomForestClassifier(n_estimators=100, random_state=SEED),
            sk_ensemble.RandomForestClassifier(n_estimators=100, random_state=SEED),
            False,
        ),
    ]
    if name == "digits":
        pairs.append(
            (
                "MultinomialNB (raw counts)",
                mlfs.naive_bayes.MultinomialNB(),
                naive_bayes.MultinomialNB(),
                True,
            )
        )

    rows = []
    for label, ours, theirs, raw in pairs:
        tr, te = (X_tr_raw, X_te_raw) if raw else (X_tr, X_te)
        ours.fit(tr, y_tr)
        theirs.fit(tr, y_tr)
        rows.append(
            Row(
                name,
                label,
                "accuracy",
                accuracy_score(y_te, ours.predict(te)),
                accuracy_score(y_te, theirs.predict(te)),
            )
        )
    return rows


def clustering_row(name: str, X: np.ndarray, y: np.ndarray) -> Row:
    X = StandardScaler().fit_transform(X)
    k = len(np.unique(y))
    ours = mlfs.kmeans.KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    theirs = cluster.KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    return Row(
        name,
        f"KMeans k={k} (k-means++, n_init=10)",
        "ARI",
        adjusted_rand_score(y, ours.labels_),
        adjusted_rand_score(y, theirs.labels_),
    )


def pca_row(name: str, X: np.ndarray) -> Row:
    X = StandardScaler().fit_transform(X)
    ours = mlfs.pca.PCA(n_components=2).fit(X)
    theirs = decomposition.PCA(n_components=2).fit(X)
    return Row(
        name,
        "PCA: variance explained by 2 PCs",
        "EVR",
        ours.explained_variance_ratio_.sum(),
        theirs.explained_variance_ratio_.sum(),
    )


def to_markdown(rows: list[Row]) -> str:
    lines = [
        "| Dataset | Model | Metric | mlfs | scikit-learn | Δ |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.dataset} | {r.model} | {r.metric} | {r.ours:.4f} | {r.theirs:.4f} "
            f"| {r.ours - r.theirs:+.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "digits": load_digits(return_X_y=True),
    }
    rows = regression_rows()
    for name, (X, y) in datasets.items():
        rows += classification_rows(name, X, y)
    for name, (X, y) in datasets.items():
        rows.append(clustering_row(name, X, y))
    for name in ("wine", "digits"):
        rows.append(pca_row(name, datasets[name][0]))

    table = to_markdown(rows)
    OUT.write_text(
        "# mlfs vs scikit-learn\n\n"
        "Generated by `python scripts/compare_sklearn.py`. Same 75/25 split "
        f"(random_state={SEED}, stratified for classification), features standardised on "
        "the training split, matched hyperparameters. Accuracy and R² are on the test split; "
        "ARI is k-means agreement with the true labels on all samples; EVR is PCA's "
        "explained-variance ratio.\n\n" + table + "\n"
    )
    print(table)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
