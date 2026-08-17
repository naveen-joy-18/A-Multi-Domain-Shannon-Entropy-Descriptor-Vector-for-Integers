"""
figures/fig_distributions.py
=============================
Figure: distribution of each entropy coordinate across the full
integer range, as violin panels. This is the "what does the fingerprint
space look like at a glance" figure -- typically Figure 1 or 2 of the
manuscript.

Design notes
------------
- Violins, not histograms: at N up to 1,000,000 a histogram per
  coordinate is fine too, but violins let all ~20 coordinates sit in a
  single compact multi-panel figure at double-column width, which is
  what NMI's figure budget actually rewards.
- No color-per-violin gradient nonsense -- one consistent hue, per
  Nature's "restraint over decoration" convention.
- Median and IQR shown as an overlaid boxplot skeleton (thin lines),
  not a distracting swarm of raw points.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .style import new_figure, DOUBLE_COLUMN_MM, OKABE_ITO, clean_axes, label_panel
from .data_loader import load_fingerprints, entropy_columns


def make_figure(df=None, columns=None, output_path="fig_entropy_distributions.svg",
                 ncols=5):
    """
    df, columns: optional pre-loaded data / column list. If not given,
    both are auto-discovered (folder search + 'H_' column detection).
    """
    if df is None:
        df = load_fingerprints()
    if columns is None:
        columns = entropy_columns(df)

    n_panels = len(columns)
    nrows = int(np.ceil(n_panels / ncols))
    panel_w, panel_h = 32, 28  # mm per panel, compact multi-panel grid
    fig = new_figure(width_mm=DOUBLE_COLUMN_MM, height_mm=panel_h * nrows * 0.85)

    for i, col in enumerate(columns):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        data = df[col].dropna().to_numpy()

        parts = ax.violinplot(
            data, showmeans=False, showmedians=True, showextrema=False, widths=0.8
        )
        for body in parts["bodies"]:
            body.set_facecolor(OKABE_ITO[4])
            body.set_edgecolor("none")
            body.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.8)

        short_label = col.replace("H_", "").replace("_", " ")
        ax.set_title(short_label, fontsize=6, pad=2)
        ax.set_xticks([])
        clean_axes(ax)
        ax.tick_params(labelsize=5)

    fig.suptitle(
        "Distribution of entropy fingerprint coordinates across all integers",
        fontsize=7, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_distributions] wrote {output_path}  ({n_panels} panels, N={len(df):,})")
    return output_path


if __name__ == "__main__":
    make_figure()
