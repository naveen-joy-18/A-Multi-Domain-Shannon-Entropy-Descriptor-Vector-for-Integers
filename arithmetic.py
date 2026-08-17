"""
arithmetic.py
=============
Domain 3 of 5: ARITHMETIC transformations (additive structure).

Your original draft proposed raw integer PARTITIONS for this domain.
That is a trap, not a transformation: p(n), the partition function,
grows sub-exponentially but is already astronomically large by n ~ a
few thousand (p(1000) has 31 digits), and building the actual
partition-frequency distribution per integer up to n=1,000,000 is
simply not computable in any reasonable pipeline. Flagging this now
rather than after a burned week of compute is the whole point of doing
the plan before the heist.

Canonical, tractable replacement: the SIGMA-WEIGHTED DIVISOR
DISTRIBUTION. This stays firmly inside "additive structure" (sigma(n)
is THE canonical additive/arithmetic function on divisors) while being
computable in the time it takes to enumerate n's divisors.

Definition:
    sigma(n) = sum of all divisors of n  (the arithmetic function)
    For each divisor d | n, define p_d = d / sigma(n)

This is automatically a valid probability distribution (the p_d sum to
exactly 1 by definition of sigma), and it measures how additive "mass"
is distributed across n's divisors -- a perfect square with many
mid-sized divisors looks very different from a prime power where all
the mass concentrates in {1, n}.

H_arithmetic(n) = Shannon entropy of {p_d : d | n}
"""

from __future__ import annotations
from .entropy_core import shannon_entropy
from .sieve import divisors_from_factorization


def arithmetic_entropy(fac: dict) -> float:
    """
    fac: {prime: exponent} dict from sieve.factorize().
    n=1 -> divisors=[1], single-point distribution -> H=0.
    """
    divisors = divisors_from_factorization(fac) if fac else [1]
    return shannon_entropy(divisors)
