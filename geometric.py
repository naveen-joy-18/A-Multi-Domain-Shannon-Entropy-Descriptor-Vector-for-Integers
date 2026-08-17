"""
geometric.py
============
Domain 2 of 5: GEOMETRIC transformations (integers represented as graphs).

Canonical transformation chosen: the divisor lattice of n, viewed as a
graded poset (this IS the Hasse diagram of n's divisors under
divisibility). We do NOT instantiate a graph object per integer -- at
N=1,000,000 that would mean building and analyzing up to ~240 million
graph nodes across the dataset, which is not a research pipeline, it's
a denial-of-service attack on your own laptop.

Instead we compute the lattice's RANK-GENERATING FUNCTION directly,
which is exactly equivalent to (and fully determines) the level
structure of the Hasse diagram -- it's the canonical, reviewer-safe
shortcut.

Mathematics:
If n = p1^e1 * p2^e2 * ... * pk^ek, every divisor d of n corresponds to
an exponent vector (a1,...,ak) with 0 <= ai <= ei. Define the "rank" of
a divisor as r(d) = a1 + a2 + ... + ak (its depth/level in the Hasse
diagram, i.e. graph distance from 1).

The number of divisors at rank j is exactly the coefficient of x^j in
the polynomial product:

    prod_{i=1}^{k} (1 + x + x^2 + ... + x^{ei})

This product has degree sum(ei) <= log2(n) <= 20 for n <= 1,000,000, so
computing it via polynomial convolution is essentially free.

H_geometric(n) = Shannon entropy of the rank-count distribution
{coeff_0, coeff_1, ..., coeff_r}.

This is invariant, canonical (no arbitrary graph-construction choice
was made -- the divisor lattice IS the unique structure), and directly
answers "how is n's divisor lattice shaped" without ever drawing it.
"""

from __future__ import annotations
import numpy as np
from .entropy_core import shannon_entropy


def _rank_polynomial(exponents: list) -> np.ndarray:
    """
    Compute prod_i (1 + x + ... + x^ei) via repeated convolution.
    Returns the coefficient array (rank-count distribution).
    """
    poly = np.array([1.0])
    for e in exponents:
        factor = np.ones(e + 1)
        poly = np.convolve(poly, factor)
    return poly


def geometric_entropy(fac: dict) -> float:
    """
    fac: {prime: exponent} dict from sieve.factorize().
    n=1 (empty fac) -> single trivial divisor {1} at rank 0 -> H=0.
    """
    if not fac:
        return 0.0
    exponents = list(fac.values())
    rank_counts = _rank_polynomial(exponents)
    return shannon_entropy(rank_counts)
