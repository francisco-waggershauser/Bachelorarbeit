"""Allocation mechanisms. Plug-in signature: mechanism(V, rng) -> A with A[g] = i.

v1 implements only round_robin. Later slots (not implemented): reversed last round,
exhaustive enumeration, alternative picking sequences, uniform random as a null model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def round_robin(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Agents 0..n-1 in order, each takes their most valuable remaining good.

    rng is unused (slot argument). RR picks by argmax, so the allocation does
    not depend on theta; exp(theta*z) preserves item ranks. Ties: np.argmax
    (first index).
    """
    _ = rng  # slot argument for later randomized mechanisms
    n, m = V.shape
    A = np.empty(m, dtype=int)
    free = np.ones(m, dtype=bool)

    for turn in range(m):
        i = turn % n
        candidates = np.flatnonzero(free)
        good = candidates[np.argmax(V[i, candidates])]
        A[good] = i
        free[good] = False

    return A


def is_round_robin(mechanism: Callable) -> bool:
    """True iff mechanism.__name__ == 'round_robin'. Gates Tests C, D, and E."""
    return getattr(mechanism, "__name__", None) == "round_robin"


# Add later mechanisms to this dict. reconstruct/load_instance look up by name.
MECHANISMS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "round_robin": round_robin,
}
