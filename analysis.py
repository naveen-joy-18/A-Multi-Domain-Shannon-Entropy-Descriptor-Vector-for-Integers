"""
analysis.py
===========
Implements Steps 6-8 of the theory (Entropy Equivalence, Entropy
Geometry, and the derived objects: distance, dimension, density) as
pure numerical routines over the consolidated fingerprint table.

Deliberately NO plotting anywhere in this file, per instruction --
everything here returns numpy arrays / pandas DataFrames that a
downstream (separate, reviewer-controlled) figure-generation script can
consume. Keeping analysis and visualization strictly decoupled is good
practice for a Nature Machine Intelligence submission anyway: reviewers
can rerun the numerics without needing your plotting stack.

Provided routines
------------------
- fingerprint_matrix(df, columns) -> np.ndarray            [Step 3]
- entropy_distance(F, i, j)       -> float                 [Step 8.1]
- pairwise_distance_matrix(F, ...) -> np.ndarray (chunked, tqdm)
- entropy_equivalence_classes(F, eps) -> list[list[int]]   [Step 6]
- entropy_dimension(F, variance_threshold) -> int          [Step 8.2, via PCA]
- entropy_density(F, k) -> np.ndarray                      [Step 8.4, kNN density proxy]
- entropy_flow(df, columns) -> pd.DataFrame                [Step 8.6, F(n) vs F(n+1)]
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def fingerprint_matrix(df: pd.DataFrame, columns: list) -> np.ndarray:
    """
    Extract the entropy-coordinate matrix F where F[i] = fingerprint
    vector of the i-th integer (row order preserved from df).
    """
    return df[columns].to_numpy(dtype=np.float64)


def entropy_distance(F: np.ndarray, i: int, j: int) -> float:
    """d_E(a, b) = || F(a) - F(b) ||_2 -- Step 8.1"""
    return float(np.linalg.norm(F[i] - F[j]))


def pairwise_distance_matrix(F: np.ndarray, block_size: int = 2000) -> np.ndarray:
    """
    Full pairwise entropy-distance matrix, computed in blocks with tqdm
    progress so it doesn't silently hang for large F. NOTE: this is
    O(M^2) memory/time -- intended for subsamples (e.g. M ~ few thousand
    integers), not the full N=1,000,000 dataset. For full-scale nearest-
    neighbor queries use entropy_density() below, which is O(M log M).
    """
    M = F.shape[0]
    D = np.zeros((M, M), dtype=np.float64)
    n_blocks = (M + block_size - 1) // block_size
    for bi in tqdm(range(n_blocks), desc="[analysis] pairwise distance blocks", unit="block"):
        r0, r1 = bi * block_size, min((bi + 1) * block_size, M)
        D[r0:r1, :] = cdist(F[r0:r1], F, metric="euclidean")
    return D


def entropy_equivalence_classes(F: np.ndarray, eps: float) -> list:
    """
    Step 6: partition integers into equivalence classes where
    ||F(a) - F(b)|| < eps. Implemented via connected components on an
    eps-ball graph (using NearestNeighbors radius query -- avoids the
    O(M^2) pairwise matrix for large M).

    Returns a list of classes, each a list of row-indices into F.
    """
    M = F.shape[0]
    nn = NearestNeighbors(radius=eps, algorithm="auto")
    nn.fit(F)

    visited = np.zeros(M, dtype=bool)
    classes = []

    for start in tqdm(range(M), desc="[analysis] equivalence classes", unit="int"):
        if visited[start]:
            continue
        # BFS over the eps-ball graph
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = nn.radius_neighbors([F[node]], return_distance=False)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        classes.append(component)

    return classes


def entropy_dimension(F: np.ndarray, variance_threshold: float = 0.95) -> dict:
    """
    Step 8.2: Entropy Dimension -- the minimum number of entropy
    coordinates needed to explain `variance_threshold` fraction of the
    variance in the fingerprint space, via PCA. This is an operational
    (not purely combinatorial) definition, chosen because it is directly
    computable and directly reportable as a single number in a results
    table.

    Returns dict with the chosen dimension, the explained-variance
    curve, and the fitted PCA components (for reproducibility).
    """
    pca = PCA(n_components=min(F.shape[0], F.shape[1]))
    pca.fit(F)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dim = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return {
        "entropy_dimension": dim,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumulative,
        "components": pca.components_,
    }


def entropy_density(F: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Step 8.4: Entropy Density -- proxy via inverse mean distance to the
    k nearest neighbors in entropy space. Higher value = denser region.
    O(M log M) via a KD-tree, so this DOES scale to the full N=1,000,000
    dataset (unlike the full pairwise distance matrix above).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")  # +1: includes self
    nn.fit(F)
    distances, _ = nn.kneighbors(F)
    mean_dist = distances[:, 1:].mean(axis=1)  # drop self-distance (0)
    density = 1.0 / (mean_dist + 1e-12)
    return density


def entropy_flow(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Step 8.6: Entropy Flow -- F(n) -> F(n+1) trajectories. Returns a
    DataFrame with per-coordinate deltas between consecutive integers,
    which is the natural object for later studying flow dynamics
    (e.g. autocorrelation, drift, recurrence) without producing plots.
    """
    df_sorted = df.sort_values("n").reset_index(drop=True)
    delta = df_sorted[columns].diff().add_suffix("_delta")
    result = pd.concat([df_sorted[["n"]], delta], axis=1)
    return result.dropna().reset_index(drop=True)
