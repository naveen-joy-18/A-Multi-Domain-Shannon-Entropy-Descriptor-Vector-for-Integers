"""
figures/fig_density_vs_n.py
=============================
Figure: entropy-space density (Step 8.4 of the theory, via k-NN in
fingerprint space -- see analysis.entropy_density) plotted against n
itself. This is the figure that shows whether "crowding" in entropy
space correlates with position along the number line, or whether dense
and sparse regions of entropy-space are scattered independently of
magnitude -- a genuinely new empirical question that classical integer
classification has no equivalent figure for.

Uses the same rasterization safeguard as the PCA figure: dense point
clouds get rasterized, axes and text stay vector.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from .style import new_figure, SINGLE_COLUMN_MM, clean_axes, OKABE_ITO
from .data_loader import load_fingerprints, entropy_columns


def _entropy_density(F: np.ndarray, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(F)
    distances, _ = nn.kneighbors(F)
    mean_dist = distances[:, 1:].mean(axis=1)
    return 1.0 / (mean_dist + 1e-12)


def make_figure(
    df=None,
    columns=None,
    k=10,
    output_path="fig_entropy_density_vs_n.svg",
    rasterize_threshold=20_000,
    max_points=100_000,
    random_state=0,
):
    if df is None:
        df = load_fingerprints()
    if columns is None:
        columns = entropy_columns(df)

    F = df[columns].to_numpy(dtype=np.float64)
    F = np.nan_to_num(F, nan=0.0)

    print(f"[fig_density_vs_n] computing k={k} nearest-neighbor density for N={len(df):,} points")
    density = _entropy_density(F, k=k)

    idx = np.arange(len(df))
    if len(df) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=max_points, replace=False)
        print(f"[fig_density_vs_n] subsampling {max_points:,} of {len(df):,} points for display")

    x = df["n"].to_numpy()[idx]
    y = density[idx]

    fig = new_figure(width_mm=SINGLE_COLUMN_MM, height_mm=SINGLE_COLUMN_MM * 0.75)
    ax = fig.add_subplot(111)

    rasterize = len(idx) > rasterize_threshold
    ax.scatter(
        x, y, s=1.2 if rasterize else 3.0, alpha=0.4 if rasterize else 0.7,
        color=OKABE_ITO[4], linewidths=0, rasterized=rasterize,
    )

    ax.set_xlabel("n")
    ax.set_ylabel(f"Entropy-space density (k={k})")
    ax.set_title("Entropy density across the integer line", fontsize=7)
    clean_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_density_vs_n] wrote {output_path}  (rasterized scatter: {rasterize})")
    return output_path


if __name__ == "__main__":
    make_figure()
