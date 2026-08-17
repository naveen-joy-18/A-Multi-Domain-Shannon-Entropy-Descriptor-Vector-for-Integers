"""
figures/fig_exemplar_fingerprints.py
======================================
Figure: parallel-coordinates comparison of full entropy fingerprint
vectors for a handful of mathematically meaningful exemplar integers
(a prime, a prime power, a highly composite number, a perfect number --
auto-selected by data_loader.find_exemplars so this never needs manual
editing when N changes). This is the figure that makes Step 3-4 of the
theory ("the fingerprint is a vector, not a single property") legible
to a reader in three seconds.

Each exemplar is one polyline across all H_* coordinates; visually
distinct trajectories = visually distinct mathematical "personalities",
which is the whole thesis of the paper stated as a picture.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .style import new_figure, DOUBLE_COLUMN_MM, clean_axes, OKABE_ITO
from .data_loader import load_fingerprints, entropy_columns, find_exemplars


def make_figure(df=None, columns=None, exemplars=None,
                 output_path="fig_exemplar_fingerprints.svg"):
    if df is None:
        df = load_fingerprints()
    if columns is None:
        columns = entropy_columns(df)
    if exemplars is None:
        exemplars = find_exemplars(df)

    if not exemplars:
        raise ValueError(
            "No exemplar integers found in the loaded dataset -- the N used "
            "for the pipeline run may be too small to contain the default "
            "reference integers (primes, 2^k, 6/28/496/8128, etc). Pass an "
            "explicit `exemplars` dict of {label: n} instead."
        )

    fig = new_figure(width_mm=DOUBLE_COLUMN_MM, height_mm=70)
    ax = fig.add_subplot(111)

    x = np.arange(len(columns))
    short_labels = [c.replace("H_", "").replace("_", " ") for c in columns]

    for i, (label, n_val) in enumerate(exemplars.items()):
        row = df.loc[df["n"] == n_val, columns]
        if row.empty:
            print(f"[fig_exemplar_fingerprints] WARNING: n={n_val} not found in data, skipping")
            continue
        y = row.to_numpy(dtype=np.float64).ravel()
        ax.plot(x, y, marker="o", markersize=2.5, color=OKABE_ITO[i % len(OKABE_ITO)],
                label=label, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Fingerprint comparison across exemplar integers", fontsize=7)
    ax.legend(frameon=False, loc="upper right", fontsize=5.5)
    clean_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_exemplar_fingerprints] wrote {output_path}  (exemplars: {list(exemplars.keys())})")
    return output_path


if __name__ == "__main__":
    make_figure()
