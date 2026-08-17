"""
dynamical.py
============
Domain 5 of 5: DYNAMICAL transformations (integer evolution).

Canonical transformation chosen: the Collatz map.
    n -> n/2         if n even
    n -> 3n + 1       if n odd
iterated until reaching 1 (or a safety cap of max_steps, to guard
against the (unproven-to-not-exist) pathological trajectory).

Induced distribution: the PARITY-BIT SEQUENCE of the trajectory --
at each step, record 0 (even step taken) or 1 (odd step taken).
This is a two-symbol distribution, so H_dynamical(n) in [0, 1] bit,
and it measures how "balanced" n's Collatz descent is between halving
and (3x+1)-ing. Numbers whose descent is almost all halving (e.g. pure
powers of 2) collapse to H=0; numbers with a long, evenly-mixed descent
approach H=1.

We also record trajectory length and max value reached as auxiliary
(non-entropy) diagnostic columns -- useful for sanity-checking the
pipeline and for later correlating entropy coordinates with classical
Collatz statistics, but they are NOT part of the Shannon-entropy
fingerprint itself (Step 1 of the theory: one entropy, everywhere).
"""

from __future__ import annotations
from .entropy_core import shannon_entropy

DEFAULT_MAX_STEPS = 2000  # generous; longest known trajectory under 1e6 is ~525 steps


def collatz_trajectory_stats(n: int, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """
    Returns dict with:
        H_dynamical_collatz : Shannon entropy of the parity-bit sequence
        collatz_steps       : trajectory length (diagnostic only)
        collatz_max_value   : peak value reached (diagnostic only)
        collatz_capped      : True if max_steps was hit without reaching 1
    """
    if n <= 1:
        return {
            "H_dynamical_collatz": 0.0,
            "collatz_steps": 0,
            "collatz_max_value": n,
            "collatz_capped": False,
        }

    m = n
    even_count = 0
    odd_count = 0
    steps = 0
    max_val = n
    capped = False

    while m != 1:
        if m % 2 == 0:
            m //= 2
            even_count += 1
        else:
            m = 3 * m + 1
            odd_count += 1
        max_val = max(max_val, m)
        steps += 1
        if steps >= max_steps:
            capped = True
            break

    entropy = shannon_entropy([even_count, odd_count])
    return {
        "H_dynamical_collatz": entropy,
        "collatz_steps": steps,
        "collatz_max_value": max_val,
        "collatz_capped": capped,
    }
