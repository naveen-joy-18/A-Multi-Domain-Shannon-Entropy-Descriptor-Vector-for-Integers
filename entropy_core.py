"""
entropy_core.py
===============
Step 1 of UEFT, made literal: there is exactly ONE entropy definition in
this codebase. Every domain module builds a probability distribution from
its own canonical transformation, then calls shannon_entropy() on it.

No mixing of incompatible entropy definitions -- no Renyi, no Tsallis,
no differential entropy sneaking in through the back door. If a reviewer
greps this repository for "entropy", they find one function.
"""

from __future__ import annotations
import numpy as np

_LOG2 = np.log(2.0)


def shannon_entropy(counts_or_probs) -> float:
    """
    H(X) = -sum_i p_i * log2(p_i)

    Accepts either raw non-negative counts/weights or a pre-normalized
    probability vector -- it normalizes internally, so callers never need
    to worry about the distinction. Zero-probability entries are dropped
    (0 log 0 := 0 by convention).

    Degenerate cases (empty input, all-zero input, single outcome) return
    0.0, which is the correct Shannon entropy of a distribution with no
    uncertainty.
    """
    arr = np.asarray(list(counts_or_probs), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    total = arr.sum()
    if total <= 0:
        return 0.0
    probs = arr[arr > 0] / total
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs) / _LOG2))
