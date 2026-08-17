"""
figures/fig_correlation.py
============================
Figure: pairwise correlation heatmap of all entropy coordinates.
Directly supports the manuscript's core claim (Step 4 of the theory):
that each transformation domain captures genuinely independent
information, rather than five redundant restatements of the same
underlying quantity. A near-diagonal, low-off-diagonal heatmap is the
empirical evidence for that claim -- so this figure is not decorative,
it's load-bearing for the argument.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .style import new_figure, SINGLE_COLUMN_MM, clean_axes, label_panel
from .data_loader import load_fingerprints, entropy_columns


def make_figure(df=None, columns=None, output_path="fig_entropy_correlation.svg"):
    if df is None:
        df = load_fingerprints()
    if columns is None:
        columns = entropy_columns(df)

    corr = df[columns].corr(method="pearson").to_numpy()
    short_labels = [c.replace("H_", "").replace("_", " ") for c in columns]

    size_mm = max(SINGLE_COLUMN_MM, 4.5 * len(columns))
    fig = new_figure(width_mm=size_mm, height_mm=size_mm)
    ax = fig.add_subplot(111)

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(short_labels, fontsize=5)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title("Cross-domain entropy coordinate correlation", fontsize=7, pad=8)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_correlation] wrote {output_path}  ({len(columns)}x{len(columns)} matrix, N={len(df):,})")
    return output_path


if __name__ == "__main__":
    make_figure()
