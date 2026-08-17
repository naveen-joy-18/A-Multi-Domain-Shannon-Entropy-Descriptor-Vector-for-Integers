"""
figures/fig_pca_projection.py
===============================
Figure: 2D PCA projection of the entropy fingerprint space -- the first
direct visualization of "Entropy Geometry" (Step 7 of the theory).
Points colored by a chosen reference quantity (default: H_algebraic, a
reasonable proxy for "how prime-like" an integer is) to show whether
classical number-theoretic structure organizes into visible manifolds
once integers are embedded via their entropy coordinates.

Scale handling
--------------
At N = 1,000,000 a literal vector SVG scatter of a million markers is
not a figure, it's a several-hundred-megabyte file that Illustrator
will choke on and NMI's production system will reject outright. So:

- If N > `rasterize_threshold` (default 20,000), the scatter LAYER is
  rasterized (rasterized=True) while axes, ticks, and text remain
  vector -- this is the standard, editor-accepted way to keep dense
  scatter plots inside a vector figure. Below the threshold, everything
  stays fully vector.
- Optionally, `max_points` subsamples the data for the plot only
  (never for the underlying PCA fit, which always uses the full
  dataset) -- keeps the figure legible rather than a solid color blob.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .style import new_figure, SINGLE_COLUMN_MM, clean_axes
from .data_loader import load_fingerprints, entropy_columns


def make_figure(
    df=None,
    columns=None,
    color_by=None,
    output_path="fig_entropy_pca_projection.svg",
    rasterize_threshold=20_000,
    max_points=100_000,
    random_state=0,
):
    if df is None:
        df = load_fingerprints()
    if columns is None:
        columns = entropy_columns(df)
    if color_by is None:
        color_by = "H_algebraic" if "H_algebraic" in df.columns else columns[0]

    F = df[columns].to_numpy(dtype=np.float64)
    F = np.nan_to_num(F, nan=0.0)

    print(f"[fig_pca_projection] fitting PCA on full N={len(df):,} rows, {F.shape[1]} coordinates")
    pca = PCA(n_components=2, random_state=random_state)
    proj = pca.fit_transform(F)
    var_explained = pca.explained_variance_ratio_

    plot_df_idx = np.arange(len(df))
    if len(df) > max_points:
        rng = np.random.default_rng(random_state)
        plot_df_idx = rng.choice(len(df), size=max_points, replace=False)
        print(f"[fig_pca_projection] subsampling {max_points:,} of {len(df):,} points for display")

    x, y = proj[plot_df_idx, 0], proj[plot_df_idx, 1]
    c = df[color_by].to_numpy()[plot_df_idx]

    fig = new_figure(width_mm=SINGLE_COLUMN_MM, height_mm=SINGLE_COLUMN_MM * 0.85)
    ax = fig.add_subplot(111)

    rasterize = len(plot_df_idx) > rasterize_threshold
    sc = ax.scatter(
        x, y, c=c, cmap="viridis", s=1.5 if rasterize else 4.0,
        alpha=0.5 if rasterize else 0.8,
        linewidths=0, rasterized=rasterize,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_by.replace("H_", "").replace("_", " "), fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var.)")
    ax.set_title("Entropy-space PCA projection", fontsize=7)
    clean_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_pca_projection] wrote {output_path}  (rasterized scatter: {rasterize})")
    return output_path


if __name__ == "__main__":
    make_figure()
