"""
sieve.py
========
All the shared number-theoretic infrastructure lives here, computed ONCE
and reused by every domain module. This is the load-bearing wall of the
whole operation -- get this wrong and every entropy coordinate downstream
is garbage. So we test it in isolation, and we vectorize it with numpy
wherever the sieve logic allows.

Provides
--------
- smallest_prime_factor_sieve(N)  -> np.ndarray of size N+1 (SPF[n])
- sigma_sieve(N)                  -> np.ndarray of size N+1 (sum of divisors)
- factorize(n, spf)               -> dict{prime: exponent}
- divisors_from_factorization(fac)-> list[int] (all divisors of n)
"""

from __future__ import annotations
import numpy as np
from tqdm import tqdm


def smallest_prime_factor_sieve(N: int, show_progress: bool = True) -> np.ndarray:
    """
    Build the Smallest Prime Factor (SPF) sieve for all integers 1..N.

    SPF[n] = smallest prime dividing n  (SPF[1] = 1, SPF[0] undefined/0)

    This is THE fast path to factorizing any integer <= N in O(log n) time,
    which is what makes the algebraic/geometric/arithmetic domains tractable
    at N = 1,000,000.

    Complexity: O(N log log N) time, O(N) memory.
    """
    spf = np.zeros(N + 1, dtype=np.int32)
    spf[1] = 1

    # Outer loop only needs to go to sqrt(N); inner marking is vectorized
    # with numpy slicing instead of a pure-python inner loop.
    limit = int(N ** 0.5) + 1
    primes_iter = range(2, limit + 1)
    if show_progress:
        primes_iter = tqdm(primes_iter, desc="[sieve] building SPF (outer)", unit="candidate")

    for i in primes_iter:
        if spf[i] == 0:  # i is prime
            # Mark first occurrence of factor i for all unmarked multiples of i
            start = i * i
            idx = np.arange(start, N + 1, i)
            unmarked = spf[idx] == 0
            spf[idx[unmarked]] = i

    # Any number > sqrt(N) still unmarked (spf == 0) is itself prime
    remaining = np.where(spf[2:] == 0)[0] + 2
    if show_progress:
        print(f"[sieve] finalizing {len(remaining):,} primes > sqrt(N)")
    spf[remaining] = remaining

    return spf


def sigma_sieve(N: int, show_progress: bool = True) -> np.ndarray:
    """
    Build sigma(n) = sum of all divisors of n, for n = 1..N, via a
    divisor-sieve (harmonic sieve): for each d in 1..N, add d to sigma[k*d]
    for every multiple k*d <= N.

    Complexity: O(N log N) time, O(N) memory.
    This is the classic "sieve of divisors" -- much faster than factorizing
    every n individually and summing divisors.
    """
    sigma = np.zeros(N + 1, dtype=np.int64)
    d_iter = range(1, N + 1)
    if show_progress:
        d_iter = tqdm(d_iter, desc="[sieve] building sigma(n)", unit="divisor")
    for d in d_iter:
        sigma[d::d] += d
    return sigma


def factorize(n: int, spf: np.ndarray) -> dict:
    """
    Factorize a single integer n using the precomputed SPF sieve.
    Returns dict {prime: exponent}. O(log n) time. n=1 -> {}.
    """
    if n <= 1:
        return {}
    fac = {}
    while n > 1:
        p = int(spf[n])
        fac[p] = fac.get(p, 0) + 1
        n //= p
    return fac


def divisors_from_factorization(fac: dict) -> list:
    """
    Enumerate all divisors of n from its prime factorization dict.
    For n <= 1,000,000 the divisor count is bounded by 240, so this is cheap.
    """
    divisors = [1]
    for prime, exp in fac.items():
        new_divisors = []
        prime_power = 1
        for _ in range(exp + 1):
            for d in divisors:
                new_divisors.append(d * prime_power)
            prime_power *= prime
        divisors = new_divisors
    return divisors
