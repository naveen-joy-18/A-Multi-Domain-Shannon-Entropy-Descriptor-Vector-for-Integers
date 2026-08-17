"""
figures/fig_dimension_scree.py
================================
Figure: cumulative explained-variance curve across principal components
of the entropy fingerprint space -- the direct visual companion to the
"Entropy Dimension" quantity defined in analysis.py (Step 8.2 of the
theory). Shows, in one glance, how many entropy coordinates are
actually doing independent work versus how many are redundant.

A horizontal reference line at the chosen variance threshold (default
95%) and a vertical marker at the resulting dimension make the number
reported in-text immediately verifiable against the figure -- reviewers
appreciate not having to take a summary statistic on faith.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .style import new_figure, SINGLE_COLUMN_MM, clean_axes, OKABE_ITO
from .data_loader import load_fingerprints, entropy_columns


def make_figure(
    df=None,
    columns=None,
    variance_threshold=0.95,
    output_path="fig_entropy_dimension.svg",
):
    if df is None:
        df = load_fingerprints()
    if columns is None:
        columns = entropy_columns(df)

    F = df[columns].to_numpy(dtype=np.float64)
    F = np.nan_to_num(F, nan=0.0)

    n_components = min(F.shape[0], F.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(F)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dim = int(np.searchsorted(cumulative, variance_threshold) + 1)

    fig = new_figure(width_mm=SINGLE_COLUMN_MM, height_mm=SINGLE_COLUMN_MM * 0.75)
    ax = fig.add_subplot(111)

    components = np.arange(1, n_components + 1)
    ax.plot(components, cumulative, color=OKABE_ITO[4], marker="o", markersize=2.5)
    ax.axhline(variance_threshold, color="grey", linestyle="--", linewidth=0.6)
    ax.axvline(dim, color=OKABE_ITO[5], linestyle="--", linewidth=0.6)
    ax.annotate(
        f"dim = {dim}",
        xy=(dim, variance_threshold),
        xytext=(dim + 0.5, variance_threshold - 0.12),
        fontsize=6, color=OKABE_ITO[5],
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Entropy dimension of the fingerprint space", fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(components)
    clean_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_dimension_scree] wrote {output_path}  (entropy_dimension={dim} at {variance_threshold*100:.0f}% variance)")
    return output_path


if __name__ == "__main__":
    make_figure()
