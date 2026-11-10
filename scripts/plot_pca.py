"""Explained-variance plot for mlfs.PCA on the digits and wine datasets.

    python scripts/plot_pca.py   ->  docs/figures/pca_explained_variance.png

Both panels share one percentage axis: bars are each component's share of the
variance, the line is the running total. Features are standardised first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits, load_wine  # noqa: E402

from mlfs.pca import PCA  # noqa: E402
from mlfs.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "pca_explained_variance.png"


def panel(ax, name: str, X: np.ndarray, max_components: int) -> None:
    pca = PCA().fit(StandardScaler().fit_transform(X))
    ratio = 100 * pca.explained_variance_ratio_[:max_components]
    cumulative = 100 * np.cumsum(pca.explained_variance_ratio_)[:max_components]
    k = np.arange(1, len(ratio) + 1)
    ax.bar(k, ratio, width=0.8, color="#7aa6d6", label="per component")
    ax.plot(k, cumulative, color="#1f4e8c", lw=2, marker="o", ms=3, label="cumulative")
    k90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1)
    ax.axhline(90, color="0.6", lw=1, ls="--")
    ax.annotate(
        f"90% at {k90} components",
        xy=(k90, 90),
        xytext=(0.45, 0.62),
        textcoords="axes fraction",
        fontsize=9,
        color="0.25",
        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8},
    )
    ax.set(
        title=f"{name} ({X.shape[1]} features)",
        xlabel="principal component",
        ylabel="variance explained (%)",
        ylim=(0, 102),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panel(axes[0], "Digits", load_digits().data, max_components=40)
    panel(axes[1], "Wine", load_wine().data, max_components=13)
    axes[0].legend(frameon=False, loc="center right")
    fig.suptitle("mlfs PCA (via SVD): explained variance on standardised features")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    main()
