"""
algebraic.py
============
Domain 1 of 5: ALGEBRAIC transformations (multiplicative structure).

Canonical transformation chosen: prime factorization.
n = p1^e1 * p2^e2 * ... * pk^ek

Induced distribution: treat the exponent multiset {e1, e2, ..., ek} as
unnormalized weights, normalize to probabilities p_i = e_i / sum(e_j).

H_algebraic(n) = Shannon entropy of this exponent distribution.

Why this and not divisor count directly: divisor count d(n) is already a
single scalar (loses all structure). The exponent distribution instead
tells you HOW the multiplicative mass is spread across distinct primes --
e.g. 2^6 (one prime, all the mass in one place -> H=0) is structurally
very different from 2*3*5*7 (four primes, evenly spread -> H=log2(4)=2),
even though both might be "highly composite" under classical metrics.

n=1 and n=prime both yield H_algebraic = 0 (no distributional freedom),
which is mathematically correct: there is nothing to distribute.
"""

from __future__ import annotations
from .entropy_core import shannon_entropy


def algebraic_entropy(fac: dict) -> float:
    """
    fac: {prime: exponent} dict, as produced by sieve.factorize().
    """
    if not fac:
        return 0.0
    exponents = list(fac.values())
    return shannon_entropy(exponents)
