#!/usr/bin/env python3
"""
UEFT REVIEWER-RERUN — SINGLE FILE
=================================

Runs the reviewer-required analyses for the UEFT manuscript using the
EXISTING fingerprint dataset. It does NOT regenerate the 1,000,000
fingerprints.

Analyses
--------
1. Standardized PCA on all 19 entropy coordinates.
2. k-NN proximity sensitivity for k = 5, 10, 20.
3. Domain ablation:
      - A+G+Ar
      - A+G+Ar+D
      - Positional only
      - All 19
4. Systematic arithmetic-class validation:
      prime, prime power, squarefree, semiprime, highly composite,
      perfect, abundant, deficient, Carmichael.
5. Collatz cap sensitivity:
      2,000 vs 5,000 iterations.
6. Reviewer-facing figures and reproducibility metadata.

Run
---
python UEFT_REVIEWER_RERUN.py

Optional
--------
python UEFT_REVIEWER_RERUN.py --data "path/to/ueft_fingerprints.csv"
python UEFT_REVIEWER_RERUN.py --data "path/to/folder/with/part_files"
python UEFT_REVIEWER_RERUN.py --output reviewer_rerun_2026
python UEFT_REVIEWER_RERUN.py --knn-sample 100000

For an exact full-dataset kNN calculation:
python UEFT_REVIEWER_RERUN.py --knn-sample 0

Dependencies
------------
numpy pandas scipy scikit-learn matplotlib tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise SystemExit(
        "Missing scikit-learn. Install with:\n"
        "pip install numpy pandas scipy scikit-learn matplotlib tqdm\n\n"
        f"Import error: {e}"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================================
# CONFIGURATION
# ============================================================================

K_VALUES = (5, 10, 20)
RANDOM_SEED = 42
DEFAULT_KNN_SAMPLE = 100_000

ARITHMETIC_CLASSES = [
    "prime",
    "prime_power",
    "squarefree",
    "semiprime",
    "highly_composite",
    "perfect",
    "abundant",
    "deficient",
    "carmichael",
]


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def log(msg=""):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def shannon_binary_counts(a, b):
    total = a + b
    if total <= 0:
        return 0.0
    p = np.array([a / total, b / total], dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def file_sha256(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def describe_environment(output_dir):
    env = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": getattr(__import__("matplotlib"), "__version__", "not installed"),
        "random_seed": RANDOM_SEED,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8"
    )


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def inspect_columns(path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=2).columns.tolist()
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path).columns.tolist()
    except Exception:
        return []
    return []


def is_candidate_dataframe(path):
    return path.suffix.lower() in {".csv", ".parquet"}


def canonical_feature_columns(df):
    """
    Resolve the manuscript's 19 entropy coordinates:

      H_algebraic
      H_geometric
      H_arithmetic
      H_pos_base2 ... H_pos_base16
      H_dynamical_collatz

    Several aliases are accepted because the supplied code uses slightly
    different names across versions.
    """
    cols = set(map(str, df.columns))
    lowmap = {c.lower(): c for c in cols}

    def find_exact(candidates):
        for c in candidates:
            if c in cols:
                return c
            if c.lower() in lowmap:
                return lowmap[c.lower()]
        return None

    algebraic = find_exact(["H_algebraic"])
    geometric = find_exact(["H_geometric"])
    arithmetic = find_exact(["H_arithmetic"])

    positional = []
    for b in range(2, 17):
        found = find_exact([
            f"H_pos_base{b}",
            f"H_positional_{b}",
            f"H_positional_base{b}",
            f"H_base{b}",
        ])
        if found is None:
            return []
        positional.append(found)

    dynamical = find_exact([
        "H_dynamical_collatz",
        "H_dynamical",
        "H_dynamic",
    ])

    if not all([algebraic, geometric, arithmetic, dynamical]):
        return []

    return [algebraic, geometric, arithmetic] + positional + [dynamical]


def discover_dataset(explicit):
    """
    Finds either one consolidated fingerprint file or all part_*.csv files
    in a directory. The search requires n plus the 19 UEFT entropy columns.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()

        if p.is_file():
            cols = inspect_columns(p)
            if "n" not in cols or len(canonical_feature_columns(
                    pd.DataFrame(columns=cols))) != 19:
                raise ValueError(
                    f"{p} does not look like a 19-coordinate UEFT fingerprint file."
                )
            return [p]

        if p.is_dir():
            files = sorted(
                x for x in p.rglob("*")
                if x.is_file() and is_candidate_dataframe(x)
            )

            valid = []
            for f in files:
                cols = inspect_columns(f)
                dummy = pd.DataFrame(columns=cols)
                if "n" in cols and len(canonical_feature_columns(dummy)) == 19:
                    valid.append(f)

            consolidated = [
                f for f in valid
                if not f.name.lower().startswith("part_")
            ]
            if consolidated:
                return [consolidated[0]]

            parts = [f for f in valid if f.name.lower().startswith("part_")]
            if parts:
                return sorted(parts)

        raise FileNotFoundError(
            f"No valid UEFT fingerprint dataset found at: {p}"
        )

    # Automatic search from the current directory.
    candidates = []
    for f in Path.cwd().rglob("*"):
        if not f.is_file() or not is_candidate_dataframe(f):
            continue
        if "reviewer_rerun" in str(f).lower():
            continue

        cols = inspect_columns(f)
        dummy = pd.DataFrame(columns=cols)
        if "n" in cols and len(canonical_feature_columns(dummy)) == 19:
            candidates.append(f)

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically find the UEFT fingerprint dataset.\n"
            "Use --data explicitly, for example:\n"
            'python UEFT_REVIEWER_RERUN.py --data "ueft_fingerprints.csv"'
        )

    consolidated = [
        f for f in candidates
        if not f.name.lower().startswith("part_")
    ]
    if consolidated:
        # Prefer names suggesting the UEFT fingerprint.
        consolidated.sort(
            key=lambda f: (
                "fingerprint" in f.name.lower(),
                "ueft" in str(f).lower()
            ),
            reverse=True
        )
        return [consolidated[0]]

    return sorted(candidates)


def load_dataset(paths):
    section("STEP 0 — LOAD EXISTING FINGERPRINT DATASET")

    if len(paths) == 1:
        path = paths[0]
        log(f"Dataset: {path}")

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    else:
        log(f"Part-files detected: {len(paths):,}")
        frames = []
        for p in tqdm(paths, desc="Reading part-files", unit="file"):
            frames.append(pd.read_csv(p))
        df = pd.concat(frames, ignore_index=True)

    if "n" not in df.columns:
        raise ValueError("Required column 'n' is missing.")

    feature_cols = canonical_feature_columns(df)
    if len(feature_cols) != 19:
        raise ValueError(
            "Could not identify the 19 canonical entropy coordinates.\n"
            f"H_* columns found: {[c for c in df.columns if str(c).startswith('H_')]}"
        )

    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(np.int64)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[feature_cols].isna().any().any():
        raise ValueError("NaN values found in entropy coordinates.")

    df = df.sort_values("n").reset_index(drop=True)

    if df["n"].duplicated().any():
        raise ValueError("Duplicate n values found in fingerprint dataset.")

    log(f"Rows: {len(df):,}")
    log(f"n range: {df['n'].min():,} .. {df['n'].max():,}")
    log(f"Entropy coordinates: {len(feature_cols)}")

    return df


# ============================================================================
# DOMAIN DEFINITIONS
# ============================================================================

def domain_columns(df):
    c = canonical_feature_columns(df)
    if len(c) != 19:
        raise ValueError("19 canonical entropy columns could not be resolved.")

    return {
        "algebraic": [c[0]],
        "geometric": [c[1]],
        "arithmetic": [c[2]],
        "positional": c[3:18],
        "dynamical": [c[18]],
        "all": c,
        "AGAr": [c[0], c[1], c[2]],
        "AGArD": [c[0], c[1], c[2], c[18]],
        "P": c[3:18],
    }


# ============================================================================
# STANDARDIZED PCA
# ============================================================================

def standardized_matrix(df, columns):
    X = df[columns].to_numpy(dtype=np.float64)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X)

    zero_variance = np.where(scaler.scale_ == 0)[0]
    if len(zero_variance):
        bad = [columns[i] for i in zero_variance]
        raise ValueError(f"Zero-variance features: {bad}")

    stats = pd.DataFrame({
        "feature": columns,
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "variance": scaler.var_,
    })

    return Z, scaler, stats


def run_pca(df, columns, outdir, tag):
    log(f"\nPCA: {tag} | {len(columns)} features")

    Z, scaler, standardization = standardized_matrix(df, columns)

    pca = PCA(
        n_components=len(columns),
        svd_solver="full",
        random_state=RANDOM_SEED,
    )

    scores = pca.fit_transform(Z)
    ev = pca.explained_variance_ratio_
    cumulative = np.cumsum(ev)

    dims = {}
    for threshold in (0.90, 0.95, 0.99):
        dims[threshold] = int(np.searchsorted(cumulative, threshold) + 1)

    standardization.to_csv(
        outdir / f"{tag}_standardization.csv", index=False
    )

    ev_df = pd.DataFrame({
        "PC": np.arange(1, len(columns) + 1),
        "explained_variance_ratio": ev,
        "explained_variance_percent": 100 * ev,
        "cumulative_variance_ratio": cumulative,
        "cumulative_variance_percent": 100 * cumulative,
    })
    ev_df.to_csv(
        outdir / f"{tag}_explained_variance.csv", index=False
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=columns,
        columns=[f"PC{i}" for i in range(1, len(columns) + 1)],
    )
    loadings.index.name = "feature"
    loadings.to_csv(outdir / f"{tag}_loadings.csv")

    # Save only the first 3 PC scores to keep the million-row output
    # manageable while retaining the full PCA loadings and variance table.
    projection = pd.DataFrame({
        "n": df["n"].to_numpy(),
        "PC1": scores[:, 0],
        "PC2": scores[:, 1] if scores.shape[1] >= 2 else np.nan,
        "PC3": scores[:, 2] if scores.shape[1] >= 3 else np.nan,
    })
    projection.to_csv(
        outdir / f"{tag}_projection_PC1_PC3.csv", index=False
    )

    summary = {
        "tag": tag,
        "n_observations": len(df),
        "n_features": len(columns),
        "standardization": (
            "z-score; mean and population standard deviation calculated "
            "across all observations in the selected dataset"
        ),
        "pca_solver": "sklearn PCA, svd_solver=full",
        "pc1_percent": float(100 * ev[0]),
        "pc2_percent": float(100 * ev[1]) if len(ev) > 1 else np.nan,
        "pc1_pc2_percent": float(100 * cumulative[1]) if len(ev) > 1 else np.nan,
        "pcs_for_90_percent": dims[0.90],
        "pcs_for_95_percent": dims[0.95],
        "pcs_for_99_percent": dims[0.99],
    }

    (outdir / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    log(f"  PC1              = {100 * ev[0]:.6f}%")
    log(f"  PC2              = {100 * ev[1]:.6f}%")
    log(f"  PC1 + PC2        = {100 * cumulative[1]:.6f}%")
    log(f"  PCs for 90%      = {dims[0.90]}")
    log(f"  PCs for 95%      = {dims[0.95]}")
    log(f"  PCs for 99%      = {dims[0.99]}")

    return {
        "Z": Z,
        "scaler": scaler,
        "pca": pca,
        "scores": scores,
        "ev": ev,
        "cumulative": cumulative,
        "summary": summary,
    }


# ============================================================================
# kNN SENSITIVITY
# ============================================================================

def run_knn_sensitivity(df, columns, outdir, sample_size):
    section("STEP 2 — k-NN PROXIMITY SENSITIVITY")

    Z, _, _ = standardized_matrix(df, columns)

    n_total = len(df)

    if sample_size and sample_size < n_total:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(
            rng.choice(n_total, size=sample_size, replace=False)
        )
        X = Z[idx]
        n_values = df["n"].to_numpy()[idx]
        sampling = (
            f"deterministic random sample of {sample_size:,} "
            f"from {n_total:,}"
        )
    else:
        X = Z
        n_values = df["n"].to_numpy()
        sampling = f"full dataset ({n_total:,})"

    log(f"kNN data: {sampling}")

    max_k = max(K_VALUES)

    nn = NearestNeighbors(
        n_neighbors=max_k + 1,
        algorithm="auto",
        metric="euclidean",
    )
    nn.fit(X)

    distances, _ = nn.kneighbors(X, return_distance=True)

    # First neighbour is the observation itself.
    neighbor_distances = distances[:, 1:]

    summary_rows = []
    score_arrays = {}

    for k in K_VALUES:
        mean_distance = neighbor_distances[:, :k].mean(axis=1)
        score = 1.0 / (mean_distance + 1e-12)

        score_arrays[k] = score

        pd.DataFrame({
            "n": n_values,
            "mean_distance_to_k_neighbors": mean_distance,
            "proximity_score": score,
        }).to_csv(
            outdir / f"knn_proximity_k{k}.csv",
            index=False,
        )

        summary_rows.append({
            "k": k,
            "n_observations": len(X),
            "mean_proximity": float(np.mean(score)),
            "std_proximity": float(np.std(score)),
            "median_proximity": float(np.median(score)),
            "min_proximity": float(np.min(score)),
            "max_proximity": float(np.max(score)),
            "mean_neighbor_distance": float(np.mean(mean_distance)),
        })

        log(
            f"  k={k:2d} | mean={np.mean(score):.8g} | "
            f"median={np.median(score):.8g}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        outdir / "knn_sensitivity_summary.csv",
        index=False,
    )

    corr = pd.DataFrame(
        index=[f"k{k}" for k in K_VALUES],
        columns=[f"k{k}" for k in K_VALUES],
        dtype=float,
    )

    for k1 in K_VALUES:
        for k2 in K_VALUES:
            corr.loc[f"k{k1}", f"k{k2}"] = np.corrcoef(
                score_arrays[k1],
                score_arrays[k2],
            )[0, 1]

    corr.to_csv(outdir / "knn_pairwise_correlations.csv")

    method = {
        "metric": "Euclidean distance",
        "space": "standardized selected entropy fingerprint",
        "definition": "1 / mean distance to k nearest non-self neighbours",
        "k_values": list(K_VALUES),
        "sampling": sampling,
        "random_seed": RANDOM_SEED,
        "interpretation": (
            "inverse-mean-kNN-distance proximity score; not claimed to be "
            "a formally normalized probability-density estimator"
        ),
    }

    (outdir / "knn_method.json").write_text(
        json.dumps(method, indent=2),
        encoding="utf-8",
    )


# ============================================================================
# NUMBER-THEORY INFRASTRUCTURE
# ============================================================================

def build_spf(N):
    section(f"STEP 4 — SPF SIEVE FOR N={N:,}")

    spf = np.zeros(N + 1, dtype=np.int32)
    spf[1] = 1

    for p in tqdm(
        range(2, int(math.isqrt(N)) + 1),
        desc="SPF sieve",
        unit="p",
    ):
        if spf[p] == 0:
            spf[p] = p
            start = p * p

            arr = spf[start:N + 1:p]
            mask = arr == 0
            arr[mask] = p
            spf[start:N + 1:p] = arr

    remaining = np.flatnonzero(spf[2:] == 0) + 2
    spf[remaining] = remaining.astype(np.int32)

    return spf


def compute_arithmetic_functions(N, spf):
    section("COMPUTE omega, Omega, tau, sigma")

    omega = np.zeros(N + 1, dtype=np.int16)
    bigomega = np.zeros(N + 1, dtype=np.int16)

    tau = np.ones(N + 1, dtype=np.int32)
    sigma = np.ones(N + 1, dtype=np.int64)

    tau[0] = 0
    sigma[0] = 0

    for n in tqdm(
        range(2, N + 1),
        desc="Arithmetic functions",
        unit="n",
    ):
        p = int(spf[n])

        m = n
        e = 0
        while m % p == 0:
            m //= p
            e += 1

        omega[n] = omega[m] + 1
        bigomega[n] = bigomega[m] + e

        tau[n] = tau[m] * (e + 1)

        # sigma(p^e) = 1+p+...+p^e
        pe_sum = 0
        term = 1
        for _ in range(e + 1):
            pe_sum += term
            term *= p

        sigma[n] = sigma[m] * pe_sum

    return omega, bigomega, tau, sigma


def highly_composite_mask(tau):
    N = len(tau) - 1
    mask = np.zeros(N + 1, dtype=bool)

    record = 0

    for n in range(1, N + 1):
        if tau[n] > record:
            mask[n] = True
            record = int(tau[n])

    mask[0] = False
    return mask


def carmichael_mask(N, spf, omega, bigomega):
    """
    Korselt criterion:
      composite + squarefree + (p-1)|(n-1) for every prime divisor p.
    """
    mask = np.zeros(N + 1, dtype=bool)

    for n in tqdm(
        range(2, N + 1),
        desc="Carmichael classification",
        unit="n",
    ):
        if omega[n] < 2:
            continue

        # Squarefree.
        if bigomega[n] != omega[n]:
            continue

        x = n
        valid = True

        while x > 1:
            p = int(spf[x])

            if (n - 1) % (p - 1) != 0:
                valid = False
                break

            while x % p == 0:
                x //= p

        if valid:
            mask[n] = True

    return mask


def build_arithmetic_classes(N, spf):
    omega, bigomega, tau, sigma = compute_arithmetic_functions(
        N, spf
    )

    n_arr = np.arange(N + 1, dtype=np.int64)

    masks = {
        "prime": (omega == 1) & (bigomega == 1),

        "prime_power": (omega == 1) & (bigomega >= 1),

        "squarefree": (
            (n_arr >= 2) &
            (bigomega == omega)
        ),

        "semiprime": (
            (n_arr >= 4) &
            (bigomega == 2)
        ),

        "highly_composite": highly_composite_mask(tau),

        "perfect": (
            (n_arr >= 2) &
            (sigma == 2 * n_arr)
        ),

        "abundant": (
            (n_arr >= 2) &
            (sigma > 2 * n_arr)
        ),

        "deficient": (
            (n_arr >= 2) &
            (sigma < 2 * n_arr)
        ),
    }

    masks["carmichael"] = carmichael_mask(
        N,
        spf,
        omega,
        bigomega,
    )

    return masks, omega, bigomega, tau, sigma


# ============================================================================
# SYSTEMATIC ARITHMETIC-CLASS VALIDATION
# ============================================================================

def run_class_validation(
    df,
    feature_cols,
    masks,
    outdir,
):
    section("STEP 5 — SYSTEMATIC ARITHMETIC-CLASS VALIDATION")

    nvals = df["n"].to_numpy(dtype=np.int64)
    Z, _, _ = standardized_matrix(df, feature_cols)

    N = int(nvals.max())

    class_counts = []
    membership = pd.DataFrame({"n": nvals})
    effects = []
    centroids = []

    for cname in ARITHMETIC_CLASSES:
        raw_mask = masks[cname]

        row_mask = raw_mask[nvals]

        membership[cname] = row_mask.astype(np.int8)

        count = int(row_mask.sum())
        prevalence = count / len(df)

        class_counts.append({
            "class": cname,
            "count": count,
            "prevalence": prevalence,
        })

        log(f"  {cname:20s}: {count:>10,}")

        if count == 0 or count == len(df):
            continue

        class_centroid = Z[row_mask].mean(axis=0)
        rest_centroid = Z[~row_mask].mean(axis=0)

        centroid_distance = float(
            np.linalg.norm(class_centroid - rest_centroid)
        )

        centroids.append({
            "class": cname,
            "count": count,
            "prevalence": prevalence,
            "centroid_distance_19D": centroid_distance,
            **{
                f"centroid_{feature}": class_centroid[j]
                for j, feature in enumerate(feature_cols)
            },
        })

        for j, feature in enumerate(feature_cols):
            x1 = Z[row_mask, j]
            x0 = Z[~row_mask, j]

            m1 = float(np.mean(x1))
            m0 = float(np.mean(x0))

            s1 = float(np.std(x1, ddof=1)) if len(x1) > 1 else np.nan
            s0 = float(np.std(x0, ddof=1)) if len(x0) > 1 else np.nan

            denom = max(len(x1) + len(x0) - 2, 1)

            pooled_var = (
                ((len(x1) - 1) * (s1 ** 2 if np.isfinite(s1) else 0.0))
                + ((len(x0) - 1) * (s0 ** 2 if np.isfinite(s0) else 0.0))
            ) / denom

            pooled_sd = math.sqrt(max(pooled_var, 1e-30))
            d = (m1 - m0) / pooled_sd

            effects.append({
                "class": cname,
                "feature": feature,
                "class_mean_standardized": m1,
                "rest_mean_standardized": m0,
                "class_sd_standardized": s1,
                "rest_sd_standardized": s0,
                "cohens_d_class_vs_rest": d,
                "abs_cohens_d": abs(d),
                "direction": "higher_in_class" if d > 0 else "lower_in_class",
                "centroid_distance_19D": centroid_distance,
            })

        # Member lists are useful for auditing and reproducing class counts.
        pd.DataFrame({
            "n": nvals[row_mask]
        }).to_csv(
            outdir / f"{cname}_members.csv",
            index=False,
        )

    pd.DataFrame(class_counts).to_csv(
        outdir / "class_counts.csv",
        index=False,
    )

    membership.to_csv(
        outdir / "arithmetic_class_membership.csv",
        index=False,
    )

    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(
        outdir / "class_feature_effects.csv",
        index=False,
    )

    centroid_df = pd.DataFrame(centroids)
    centroid_df.to_csv(
        outdir / "class_centroids_standardized_19D.csv",
        index=False,
    )

    if not effects_df.empty:
        top = (
            effects_df
            .sort_values(
                ["class", "abs_cohens_d"],
                ascending=[True, False],
            )
            .groupby("class", as_index=False)
            .head(5)
        )

        top.to_csv(
            outdir / "class_top5_effects.csv",
            index=False,
        )

    definitions = pd.DataFrame([
        {
            "class": "prime",
            "definition": "omega(n)=1 and Omega(n)=1",
        },
        {
            "class": "prime_power",
            "definition": "omega(n)=1",
        },
        {
            "class": "squarefree",
            "definition": "Omega(n)=omega(n)",
        },
        {
            "class": "semiprime",
            "definition": "Omega(n)=2",
        },
        {
            "class": "highly_composite",
            "definition": "tau(n) exceeds tau(m) for every m<n",
        },
        {
            "class": "perfect",
            "definition": "sigma(n)=2n",
        },
        {
            "class": "abundant",
            "definition": "sigma(n)>2n",
        },
        {
            "class": "deficient",
            "definition": "sigma(n)<2n",
        },
        {
            "class": "carmichael",
            "definition": (
                "Korselt criterion: composite, squarefree, and "
                "(p-1)|(n-1) for every prime p dividing n"
            ),
        },
    ])

    definitions.to_csv(
        outdir / "class_definitions.csv",
        index=False,
    )

    return membership, effects_df, centroid_df


# ============================================================================
# COLLATZ SENSITIVITY
# ============================================================================

def collatz_entropy_for_n(n, max_steps):
    """
    Reproduces the supplied dynamical.py implementation:

      even -> n/2, count even operation
      odd  -> 3n+1, count odd operation
      stop at 1 or max_steps
      entropy = Shannon entropy of [even_count, odd_count]

    n <= 1 is assigned H=0 by the supplied convention.
    """
    if n <= 1:
        return 0.0, 0, int(n), False

    m = int(n)
    even_count = 0
    odd_count = 0
    steps = 0
    max_value = int(n)
    capped = False

    while m != 1:
        if m % 2 == 0:
            m //= 2
            even_count += 1
        else:
            m = 3 * m + 1
            odd_count += 1

        max_value = max(max_value, m)

        steps += 1

        if steps >= max_steps:
            capped = True
            break

    H = shannon_binary_counts(even_count, odd_count)

    return H, steps, max_value, capped


def run_collatz_sensitivity(df, dynamical_col, outdir):
    section("STEP 6 — COLLATZ CAP SENSITIVITY")

    nvals = df["n"].to_numpy(dtype=np.int64)
    existing = df[dynamical_col].to_numpy(dtype=float)

    rows = []

    for i, n in enumerate(
        tqdm(nvals, desc="Collatz 2000 vs 5000", unit="n")
    ):
        h2000, s2000, m2000, c2000 = collatz_entropy_for_n(
            int(n), 2000
        )

        h5000, s5000, m5000, c5000 = collatz_entropy_for_n(
            int(n), 5000
        )

        rows.append({
            "n": int(n),
            "H_dynamical_existing": existing[i],
            "H_dynamical_cap2000": h2000,
            "H_dynamical_cap5000": h5000,
            "delta_5000_minus_2000": h5000 - h2000,
            "delta_existing_minus_2000": existing[i] - h2000,
            "steps_cap2000": s2000,
            "steps_cap5000": s5000,
            "max_value_cap2000": m2000,
            "max_value_cap5000": m5000,
            "capped_at_2000": c2000,
            "capped_at_5000": c5000,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        outdir / "collatz_cap_comparison.csv",
        index=False,
    )

    delta = result["delta_5000_minus_2000"].to_numpy()

    changed = np.abs(delta) > 1e-12

    summary = {
        "n_observations": len(result),
        "cap_2000_capped_count": int(
            result["capped_at_2000"].sum()
        ),
        "cap_5000_capped_count": int(
            result["capped_at_5000"].sum()
        ),
        "changed_entropy_count_abs_delta_gt_1e-12": int(
            changed.sum()
        ),
        "max_abs_entropy_difference": float(
            np.max(np.abs(delta))
        ),
        "mean_abs_entropy_difference": float(
            np.mean(np.abs(delta))
        ),
        "existing_vs_cap2000_max_abs_difference": float(
            np.max(
                np.abs(
                    result["delta_existing_minus_2000"]
                )
            )
        ),
        "implementation": (
            "Parity-bit Shannon entropy reproduced from supplied "
            "dynamical.py; n<=1 convention H=0."
        ),
    }

    (outdir / "collatz_cap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    log(
        f"  cap=2000 capped: {summary['cap_2000_capped_count']:,}"
    )
    log(
        f"  cap=5000 capped: {summary['cap_5000_capped_count']:,}"
    )
    log(
        "  changed H values: "
        f"{summary['changed_entropy_count_abs_delta_gt_1e-12']:,}"
    )
    log(
        "  max |H5000-H2000|: "
        f"{summary['max_abs_entropy_difference']:.12g}"
    )


# ============================================================================
# DOMAIN ABLATION
# ============================================================================

def run_domain_ablation(df, domains, outdir):
    section("STEP 3 — DOMAIN ABLATION")

    configs = {
        "AGAr": domains["AGAr"],
        "AGArD": domains["AGArD"],
        "P_only": domains["P"],
        "ALL_19": domains["all"],
    }

    rows = []

    for tag, cols in configs.items():
        result = run_pca(
            df,
            cols,
            outdir,
            f"ablation_{tag}",
        )

        rows.append({
            "configuration": tag,
            "n_features": len(cols),
            "features": ";".join(cols),
            "PC1_percent": result["summary"]["pc1_percent"],
            "PC2_percent": result["summary"]["pc2_percent"],
            "PC1_PC2_percent": result["summary"]["pc1_pc2_percent"],
            "PCs_for_90_percent": result["summary"]["pcs_for_90_percent"],
            "PCs_for_95_percent": result["summary"]["pcs_for_95_percent"],
            "PCs_for_99_percent": result["summary"]["pcs_for_99_percent"],
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        outdir / "domain_ablation_summary.csv",
        index=False,
    )

    return summary


# ============================================================================
# DATASET CONSISTENCY
# ============================================================================

def run_consistency_checks(df, feature_cols, outdir):
    section("DATASET CONSISTENCY CHECKS")

    nvals = df["n"].to_numpy()

    checks = [
        {
            "check": "row_count",
            "value": len(df),
            "expected": 1_000_000,
            "status": "PASS" if len(df) == 1_000_000 else "CHECK",
        },
        {
            "check": "minimum_n",
            "value": int(nvals.min()),
            "expected": 1,
            "status": "PASS" if int(nvals.min()) == 1 else "CHECK",
        },
        {
            "check": "maximum_n",
            "value": int(nvals.max()),
            "expected": 1_000_000,
            "status": "PASS" if int(nvals.max()) == 1_000_000 else "CHECK",
        },
        {
            "check": "unique_n",
            "value": int(df["n"].nunique()),
            "expected": len(df),
            "status": (
                "PASS"
                if df["n"].nunique() == len(df)
                else "FAIL"
            ),
        },
        {
            "check": "entropy_coordinate_count",
            "value": len(feature_cols),
            "expected": 19,
            "status": (
                "PASS"
                if len(feature_cols) == 19
                else "FAIL"
            ),
        },
    ]

    row1 = df[df["n"] == 1]

    if not row1.empty:
        algebraic = safe_float(row1.iloc[0][feature_cols[0]])
        dynamical = safe_float(row1.iloc[0][feature_cols[-1]])

        checks.append({
            "check": "n1_algebraic_entropy",
            "value": algebraic,
            "expected": 0.0,
            "status": (
                "PASS"
                if abs(algebraic) < 1e-12
                else "CHECK"
            ),
        })

        checks.append({
            "check": "n1_dynamical_entropy",
            "value": dynamical,
            "expected": 0.0,
            "status": (
                "PASS"
                if abs(dynamical) < 1e-12
                else "CHECK"
            ),
        })

    pd.DataFrame(checks).to_csv(
        outdir / "dataset_consistency_checks.csv",
        index=False,
    )

    log(pd.DataFrame(checks).to_string(index=False))


# ============================================================================
# FIGURES
# ============================================================================

def make_figures(
    pca_result,
    knn_dir,
    class_effects,
    outdir,
):
    if plt is None:
        log("matplotlib unavailable; figures skipped.")
        return

    section("STEP 7 — FIGURES")

    # Figure 1: standardized PCA variance.
    ev = pca_result["ev"]
    cumulative = pca_result["cumulative"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    x = np.arange(1, len(ev) + 1)

    ax.plot(
        x,
        100 * ev,
        marker="o",
        linewidth=1.5,
        label="Individual variance",
    )

    ax.plot(
        x,
        100 * cumulative,
        marker="s",
        linewidth=1.5,
        label="Cumulative variance",
    )

    ax.axhline(
        95,
        linestyle="--",
        linewidth=1.0,
        label="95%",
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Standardized PCA of the 19-coordinate UEFT fingerprint")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        outdir / "figure_standardized_pca_variance.png",
        dpi=300,
    )
    plt.close(fig)

    # Figure 2: kNN correlation matrix.
    corr_path = knn_dir / "knn_pairwise_correlations.csv"

    if corr_path.exists():
        corr = pd.read_csv(
            corr_path,
            index_col=0,
        )

        fig, ax = plt.subplots(figsize=(5.0, 4.0))

        im = ax.imshow(
            corr.values,
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns)

        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        ax.set_title("Correlation of k-NN proximity scores")

        fig.colorbar(
            im,
            ax=ax,
            label="Pearson correlation",
        )

        fig.tight_layout()
        fig.savefig(
            outdir / "figure_knn_sensitivity.png",
            dpi=300,
        )
        plt.close(fig)

    # Figure 3: strongest arithmetic-class effects.
    if class_effects is not None and not class_effects.empty:
        top = (
            class_effects
            .sort_values(
                ["class", "abs_cohens_d"],
                ascending=[True, False],
            )
            .groupby("class", as_index=False)
            .head(3)
            .sort_values("cohens_d_class_vs_rest")
        )

        fig_height = max(4.5, 0.45 * len(top))

        fig, ax = plt.subplots(
            figsize=(8.0, fig_height)
        )

        labels = [
            f"{r['class']} — {r['feature']}"
            for _, r in top.iterrows()
        ]

        values = top["cohens_d_class_vs_rest"].to_numpy()

        ax.barh(
            np.arange(len(values)),
            values,
        )

        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, fontsize=8)

        ax.axvline(
            0,
            linewidth=0.8,
        )

        ax.set_xlabel(
            "Cohen's d (class vs non-class)"
        )

        ax.set_title(
            "Largest standardized-coordinate differences by arithmetic class"
        )

        fig.tight_layout()
        fig.savefig(
            outdir / "figure_arithmetic_class_effects.png",
            dpi=300,
        )
        plt.close(fig)


# ============================================================================
# README / MASTER SUMMARY
# ============================================================================

def write_readme(
    output_dir,
    dataset_paths,
    main_pca_summary,
    knn_sample,
    ablation_summary,
):
    lines = [
        "# UEFT Reviewer Rerun Results",
        "",
        "Generated by `UEFT_REVIEWER_RERUN.py`.",
        "",
        "## Important",
        "",
        "The original UEFT fingerprint coordinates were reused. "
        "The 1,000,000 fingerprints were not regenerated.",
        "",
        "PCA was performed after explicit z-score standardization.",
        "",
        "The kNN quantity is reported as an inverse-mean-kNN-distance "
        "proximity score rather than assumed to be a formally normalized "
        "density estimator.",
        "",
        "Collatz sensitivity reproduces the supplied dynamical.py parity "
        "entropy implementation.",
        "",
        "## Input",
        "",
    ]

    for p in dataset_paths:
        lines.append(f"- `{p}`")

    lines += [
        "",
        "## Main standardized PCA",
        "",
        f"- PC1: {main_pca_summary['pc1_percent']:.6f}%",
        f"- PC2: {main_pca_summary['pc2_percent']:.6f}%",
        f"- PC1+PC2: {main_pca_summary['pc1_pc2_percent']:.6f}%",
        f"- PCs for 90%: {main_pca_summary['pcs_for_90_percent']}",
        f"- PCs for 95%: {main_pca_summary['pcs_for_95_percent']}",
        f"- PCs for 99%: {main_pca_summary['pcs_for_99_percent']}",
        "",
        "## kNN",
        "",
        f"- k values: {K_VALUES}",
        f"- kNN sample: {knn_sample:,} (0 means full dataset)",
        "",
        "## Domain ablation",
        "",
        ablation_summary.to_string(index=False),
        "",
        "## Output folders",
        "",
        "- `01_PCA_STANDARDIZED/`",
        "- `02_KNN_SENSITIVITY/`",
        "- `03_DOMAIN_ABLATION/`",
        "- `04_SYSTEMATIC_VALIDATION/`",
        "- `05_COLLATZ_SENSITIVITY/`",
        "- `06_FIGURES/`",
        "- `07_METADATA/`",
        "",
    ]

    (output_dir / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-file UEFT reviewer rerun."
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Existing consolidated fingerprint CSV/Parquet or a directory "
            "containing the part_*.csv fingerprint files."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reviewer_rerun_2026",
        help="Output directory.",
    )

    parser.add_argument(
        "--knn-sample",
        type=int,
        default=DEFAULT_KNN_SAMPLE,
        help=(
            "kNN sample size. Default 100000. "
            "Use 0 for the full dataset."
        ),
    )

    args = parser.parse_args()

    start_time = time.time()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_dir = output_dir / "01_PCA_STANDARDIZED"
    knn_dir = output_dir / "02_KNN_SENSITIVITY"
    ablation_dir = output_dir / "03_DOMAIN_ABLATION"
    validation_dir = output_dir / "04_SYSTEMATIC_VALIDATION"
    collatz_dir = output_dir / "05_COLLATZ_SENSITIVITY"
    figures_dir = output_dir / "06_FIGURES"
    metadata_dir = output_dir / "07_METADATA"

    for d in [
        pca_dir,
        knn_dir,
        ablation_dir,
        validation_dir,
        collatz_dir,
        figures_dir,
        metadata_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    describe_environment(metadata_dir)

    section("UEFT REVIEWER RERUN — START")
    log("Existing fingerprint data will be reused.")
    log("The original million-integer fingerprint will NOT be regenerated.")

    dataset_paths = discover_dataset(args.data)

    log("\nInput dataset:")
    for p in dataset_paths:
        log(f"  {p}")

    df = load_dataset(dataset_paths)

    feature_cols = canonical_feature_columns(df)
    domains = domain_columns(df)

    run_consistency_checks(
        df,
        feature_cols,
        metadata_dir,
    )

    # -----------------------------------------------------------------
    # 1. Main standardized PCA
    # -----------------------------------------------------------------
    section("STEP 1 — MAIN STANDARDIZED PCA")

    main_pca = run_pca(
        df,
        feature_cols,
        pca_dir,
        "main_19D_standardized",
    )

    # -----------------------------------------------------------------
    # 2. kNN sensitivity
    # -----------------------------------------------------------------
    run_knn_sensitivity(
        df,
        feature_cols,
        knn_dir,
        args.knn_sample,
    )

    # -----------------------------------------------------------------
    # 3. Domain ablation
    # -----------------------------------------------------------------
    ablation_summary = run_domain_ablation(
        df,
        domains,
        ablation_dir,
    )

    # -----------------------------------------------------------------
    # 4. Arithmetic classes
    # -----------------------------------------------------------------
    N = int(df["n"].max())

    spf = build_spf(N)

    masks, omega, bigomega, tau, sigma = build_arithmetic_classes(
        N,
        spf,
    )

    membership, class_effects, class_centroids = run_class_validation(
        df,
        feature_cols,
        masks,
        validation_dir,
    )

    # Save arithmetic function diagnostics for auditability.
    # This is intentionally integer-based and compact.
    pd.DataFrame({
        "n": np.arange(1, N + 1, dtype=np.int64),
        "omega": omega[1:],
        "Omega": bigomega[1:],
        "tau": tau[1:],
        "sigma": sigma[1:],
    }).to_csv(
        validation_dir / "arithmetic_function_diagnostics.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 5. Collatz sensitivity
    # -----------------------------------------------------------------
    run_collatz_sensitivity(
        df,
        domains["dynamical"][0],
        collatz_dir,
    )

    # -----------------------------------------------------------------
    # 6. Figures
    # -----------------------------------------------------------------
    make_figures(
        main_pca,
        knn_dir,
        class_effects,
        figures_dir,
    )

    # -----------------------------------------------------------------
    # 7. Metadata
    # -----------------------------------------------------------------
    dataset_metadata = []

    for p in dataset_paths:
        item = {
            "path": str(p),
            "exists": p.exists(),
        }

        if p.is_file():
            item["size_bytes"] = p.stat().st_size
            item["sha256"] = file_sha256(p)

        dataset_metadata.append(item)

    (metadata_dir / "input_dataset.json").write_text(
        json.dumps(dataset_metadata, indent=2),
        encoding="utf-8",
    )

    (metadata_dir / "run_arguments.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    # Master summary.
    master = {
        "dataset_rows": int(len(df)),
        "n_min": int(df["n"].min()),
        "n_max": int(df["n"].max()),
        "entropy_coordinates": feature_cols,
        "main_pca": main_pca["summary"],
        "knn_k_values": list(K_VALUES),
        "knn_sample": int(args.knn_sample),
        "collatz_caps": [2000, 5000],
        "runtime_seconds": float(time.time() - start_time),
    }

    (output_dir / "MASTER_SUMMARY.json").write_text(
        json.dumps(master, indent=2),
        encoding="utf-8",
    )

    write_readme(
        output_dir,
        dataset_paths,
        main_pca["summary"],
        args.knn_sample,
        ablation_summary,
    )

    section("UEFT REVIEWER RERUN — COMPLETE")

    elapsed = (time.time() - start_time) / 60.0

    log(f"Runtime: {elapsed:.2f} minutes")
    log(f"Results: {output_dir}")

    log("\nImportant result folders:")
    log("  01_PCA_STANDARDIZED/")
    log("  02_KNN_SENSITIVITY/")
    log("  03_DOMAIN_ABLATION/")
    log("  04_SYSTEMATIC_VALIDATION/")
    log("  05_COLLATZ_SENSITIVITY/")
    log("  06_FIGURES/")
    log("  07_METADATA/")
    log("  MASTER_SUMMARY.json")

    log("\nDo not delete individual CSV/JSON files before sending the results back.")


if __name__ == "__main__":
    main()
