"""
positional.py
=============
Domain 4 of 5: POSITIONAL transformations (representation dependence).

Canonical transformation chosen: base-k digit expansion, for a fixed,
pre-declared, systematic family of bases k = 2..16 (this is exactly the
"Binary, Decimal, Base-k" family from the original proposal -- Gray
code and balanced ternary are deliberately deferred; see note below).

For a fixed base k, write n in base k and build the frequency
distribution of its digits (which digit values appear, how often).

H_positional_k(n) = Shannon entropy of the base-k digit-frequency
distribution.

Because this is representation-DEPENDENT by construction (that's the
entire point of this domain -- Step 5 of the theory notes positional
entropy is not invariant under change of base), we do NOT collapse it
to a single scalar. Instead the fingerprint carries the FULL VECTOR
(H_positional_2, H_positional_3, ..., H_positional_16), and the
"canonical" / invariance-preserving move (per Step 5 of the theory) is
that this entire vector, taken together across the whole declared base
family, is what's invariant -- not any single coordinate.

Note on Gray code / balanced ternary: both are legitimate positional
transforms and are implemented here as optional extras
(gray_code_entropy, balanced_ternary_entropy) but are NOT included in
the default fingerprint, to keep the primary submission's positional
block simple, systematic, and easy for a reviewer to audit. Turn them
on via `include_extras=True` in pipeline.py once the core fingerprint
is validated.
"""

from __future__ import annotations
from .entropy_core import shannon_entropy

DEFAULT_BASES = list(range(2, 17))  # 2, 3, 4, ..., 16


def _digits_in_base(n: int, base: int) -> list:
    if n == 0:
        return [0]
    digits = []
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits


def positional_entropy_vector(n: int, bases: list = None) -> dict:
    """
    Returns {f'H_pos_base{b}': entropy} for each base in `bases`.
    """
    bases = bases if bases is not None else DEFAULT_BASES
    result = {}
    for b in bases:
        digits = _digits_in_base(n, b)
        counts = {}
        for d in digits:
            counts[d] = counts.get(d, 0) + 1
        result[f"H_pos_base{b}"] = shannon_entropy(list(counts.values()))
    return result


def gray_code_entropy(n: int) -> float:
    """Optional extra: entropy of the binary digit distribution of the
    reflected binary Gray code of n. G = n XOR (n >> 1)."""
    g = n ^ (n >> 1)
    digits = _digits_in_base(g, 2)
    counts = {}
    for d in digits:
        counts[d] = counts.get(d, 0) + 1
    return shannon_entropy(list(counts.values()))


def balanced_ternary_entropy(n: int) -> float:
    """Optional extra: entropy of the digit distribution {-1, 0, 1} of
    n's balanced ternary representation."""
    digits = []
    m = n
    while m != 0:
        r = m % 3
        if r == 2:
            r = -1
            m += 1
        digits.append(r)
        m //= 3
    if not digits:
        digits = [0]
    counts = {}
    for d in digits:
        counts[d] = counts.get(d, 0) + 1
    return shannon_entropy(list(counts.values()))
