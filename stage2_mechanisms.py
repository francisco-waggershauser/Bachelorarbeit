"""Stage 2: allocation mechanisms. Plug-in signature: mechanism(V, rng) -> A with A[g] = i.

round_robin is the default. Also: exhaustive EFX enumeration (n >= 4 only in production),
uniform_random_balanced null model, and picking_sequence with RR as a wrapper.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from math import comb

import numpy as np

from stage3_evaluate import alpha_matrix, alpha_star, expected_bundle_sizes, holds_efx


def round_robin(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Agents 0..n-1 in order, each takes their most valuable remaining good.

    rng is unused (slot argument). RR picks by argmax, so the allocation does
    not depend on theta; exp(theta*z) preserves item ranks. Ties: np.argmax
    (first index).
    """
    _ = rng  # slot argument for later randomized mechanisms
    return picking_sequence(V, tuple(range(V.shape[0])))


def picking_sequence(V: np.ndarray, sigma: Sequence[int]) -> np.ndarray:
    """Serial dictatorship with turn order sigma (length n), cycling until all goods taken."""
    n, m = V.shape
    if len(sigma) != n:
        raise ValueError(f"sigma must have length n={n}, got {len(sigma)}")
    A = np.empty(m, dtype=int)
    free = np.ones(m, dtype=bool)

    for turn in range(m):
        i = sigma[turn % n] #weil so immer Zyklus über die Agenten entsteht abhängig des xten Guts
        candidates = np.flatnonzero(free)
        good = candidates[np.argmax(V[i, candidates])]
        A[good] = i
        free[good] = False

    return A


def uniform_random_balanced(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform over allocations with the Round-Robin bundle-size profile.

    Draws a uniform permutation of the goods and cuts it into blocks of
    expected_bundle_sizes(n, m): the first q = m % n agents get r+1 = m//n + 1
    goods, the rest get r. Ignores V entirely.

    Why size-matched and not i.i.d. uniform per good: with i.i.d. assignment the
    bundle sizes are multinomial, bundles can even be empty, and alpha_star
    collapses mostly because of size imbalance rather than because the choice
    was uninformed. Matching the RR size profile isolates the single thing we
    want to measure: the value of *choosing by preference* instead of at random.
    """
    n, m = V.shape
    sizes = expected_bundle_sizes(n, m)
    order = rng.permutation(m)
    A = np.empty(m, dtype=int)
    start = 0
    for agent, size in enumerate(sizes):
        A[order[start : start + size]] = agent
        start += size
    return A


def is_round_robin(mechanism: Callable) -> bool:
    """True iff mechanism.__name__ == 'round_robin'. Gates Tests C, D, and E."""
    return getattr(mechanism, "__name__", None) == "round_robin"


def full_assignment_count(n: int, m: int) -> int:
    """n^m assignments of m labeled goods to n agents."""
    return n**m


def no_empty_bundle_count(n: int, m: int) -> int:
    """Assignments where every agent receives at least one good (inclusion-exclusion)."""
    if m < n:
        return 0
    total = 0
    for k in range(n + 1):
        sign = -1 if k % 2 else 1
        total += sign * comb(n, k) * (n - k) ** m
    return total


def iter_assignments_no_empty_bundle(n: int, m: int) -> Iterator[np.ndarray]:
    """Yield A[g]=agent for each assignment with no empty bundle."""
    for assignment in product(range(n), repeat=m):
        A = np.asarray(assignment, dtype=int)
        if np.bincount(A, minlength=n).min() >= 1:
            yield A


@dataclass(frozen=True)
class EnumerateResult:
    exists_efx: bool
    alpha_opt: float
    A_opt: np.ndarray | None
    n_scanned: int
    n_total: int
    elapsed_s: float


def enumerate_efx(
    V: np.ndarray,
    on_progress: Callable[[int, int, float, float, bool], None] | None = None,
    progress_interval_s: float = 2.0,
) -> EnumerateResult:
    """Brute-force search for an EFX allocation; track best alpha* if none found.

    Scans assignments with no empty bundle only. Stops at the first EFX witness.
    If on_progress is set, it is called at most about every progress_interval_s
    seconds as (n_scanned, n_total, best_star, elapsed_s, found_efx), and once
    immediately when an EFX witness is found.
    """
    n, m = V.shape
    n_total = no_empty_bundle_count(n, m)
    t0 = time.perf_counter()
    best_star = float("-inf")
    best_A: np.ndarray | None = None
    n_scanned = 0
    last_progress = t0 - progress_interval_s

    def report(found_efx: bool) -> None:
        if on_progress is None:
            return
        star = 1.0 if found_efx else best_star
        on_progress(
            n_scanned,
            n_total,
            star,
            time.perf_counter() - t0,
            found_efx,
        )

    report(False)

    for A in iter_assignments_no_empty_bundle(n, m):
        n_scanned += 1
        if holds_efx(V, A):
            report(True)
            elapsed = time.perf_counter() - t0
            return EnumerateResult(
                exists_efx=True,
                alpha_opt=1.0,
                A_opt=A.copy(),
                n_scanned=n_scanned,
                n_total=n_total,
                elapsed_s=elapsed,
            )
        star = alpha_star(alpha_matrix(V, A))
        if star > best_star:
            best_star = star
            best_A = A.copy()
        now = time.perf_counter()
        if now - last_progress >= progress_interval_s:
            last_progress = now
            report(False)

    elapsed = time.perf_counter() - t0
    if n_scanned == 0:
        return EnumerateResult(
            exists_efx=False,
            alpha_opt=float("nan"),
            A_opt=None,
            n_scanned=0,
            n_total=n_total,
            elapsed_s=elapsed,
        )
    return EnumerateResult(
        exists_efx=False,
        alpha_opt=float(best_star),
        A_opt=best_A,
        n_scanned=n_scanned,
        n_total=n_total,
        elapsed_s=elapsed,
    )


def efx_optimum(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mechanism slot: first EFX allocation found by enumeration, else argmax alpha*."""
    _ = rng
    result = enumerate_efx(V)
    if result.A_opt is None:
        raise ValueError("enumerate_efx found no valid assignment to return")
    return result.A_opt


# Add later mechanisms to this dict. reconstruct/load_instance look up by name.
MECHANISMS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "round_robin": round_robin,
    "uniform_random_balanced": uniform_random_balanced,
    "efx_optimum": efx_optimum,
}
