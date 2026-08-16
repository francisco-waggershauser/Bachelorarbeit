"""Stage 3: evaluate an allocation (V, A) for alpha-EFX and related controls."""

from __future__ import annotations

import numpy as np


def bundle_sizes(A: np.ndarray, n: int) -> np.ndarray:
    """Length-n int array: number of goods per agent."""
    return np.bincount(A, minlength=n).astype(int)


def expected_bundle_sizes(n: int, m: int) -> np.ndarray:
    """r = m // n, q = m % n (q is the remainder). First q agents get r+1, rest get r."""
    r = m // n
    q = m % n
    sizes = np.full(n, r, dtype=int)
    sizes[:q] = r + 1
    return sizes


def skipped_fraction(alpha: np.ndarray) -> float:
    """Share of NaN among the n*(n-1) off-diagonal entries."""
    n = alpha.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    return float(np.isnan(alpha[offdiag]).mean())


def expected_skipped_fraction(n: int, m: int) -> float:
    """(n - q) / n if r == 1 (i.e. n < m < 2n), else 0.

    When r == 1 the last n-q agents hold one good; each such column j is NaN
    for all n-1 off-diagonal rows, so the share is (n-q)(n-1) / n(n-1) = (n-q)/n.
    """
    r = m // n
    q = m % n
    if r == 1:
        return (n - q) / n
    return 0.0


def collision_rate(V: np.ndarray) -> float:
    """1 - (number of distinct argmax goods) / n. From V before the mechanism."""
    n = V.shape[0]
    top = np.argmax(V, axis=1)
    return 1.0 - (len(np.unique(top)) / n)


def alpha_matrix(V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Uncapped n x n matrix. Diagonal NaN.

    Conventions (even if lognormal never hits them):
      1. denom == 0 (A_j has exactly one positive-value good) -> NaN, EFX genuinely holds
      2. A_j has no good with v_i(g) > 0 -> NaN
      3. A_j empty -> NaN (cannot occur under RR when m > n)

    Formula: v_i(A_i) / (v_i(A_j) - min{v_i(g) : g in A_j, v_i(g) > 0}).
    """
    n = V.shape[0]
    bundles = [np.flatnonzero(A == j) for j in range(n)]
    alpha = np.full((n, n), np.nan)

    for i in range(n):
        own = V[i, bundles[i]].sum()
        for j in range(n):
            if i == j:
                continue
            goods_j = bundles[j]
            # Convention 3: empty A_j -> NaN
            if goods_j.size == 0:
                continue
            vals = V[i, goods_j]
            positive = vals[vals > 0.0]
            # Convention 2: no good with v_i(g) > 0 -> NaN
            if positive.size == 0:
                continue
            # Convention 1: exactly one positive good -> denom 0, EFX holds, NaN
            if positive.size == 1:
                continue
            denom = positive.sum() - positive.min()
            if denom == 0.0:
                continue
            alpha[i, j] = own / denom

    return alpha


def alpha_star(alpha: np.ndarray) -> float:
    """min(1, nanmin(alpha)). Capping happens here only, never in the stored matrix.

    Assert that the off-diagonal is not all-NaN. An all-NaN matrix cannot occur
    when m > n under lognormal (that would mean m = n). Without the assert,
    nanmin warns, returns nan, and Python min(1, nan) is 1 — silent fake EFX.
    """
    n = alpha.shape[0]
    offdiag = alpha[~np.eye(n, dtype=bool)]
    assert not np.all(np.isnan(offdiag)), (
        "alpha matrix is all-NaN off the diagonal; cannot compute alpha_star"
    )
    return float(min(1.0, np.nanmin(alpha)))


def evaluate(V: np.ndarray, A: np.ndarray) -> dict:
    """The flowchart step: alpha_matrix, alpha_star, bundle_sizes, skipped_fraction.

    collision_rate is not included; it is computed from V before the mechanism.
    """
    n = V.shape[0]
    alpha = alpha_matrix(V, A)
    sizes = bundle_sizes(A, n)
    return {
        "alpha_matrix": alpha,
        "alpha_star": alpha_star(alpha),
        "bundle_sizes": sizes,
        "skipped_fraction": skipped_fraction(alpha),
    }


def holds_efx(V: np.ndarray, A: np.ndarray, rtol: float = 1e-12) -> bool:
    """Textbook EFX, independent of the alpha formula.

    True iff for all i != j and all g in A_j with v_i(g) > 0:
        v_i(A_i) >= v_i(A_j without g).
    Empty / all-nonpositive A_j: inner loop empty, returns True (vacuous).
    rtol 1e-12 is the same arithmetic artefact as Test C, not a fairness eps.
    """
    n = V.shape[0]
    bundles = [np.flatnonzero(A == j) for j in range(n)]

    for i in range(n):
        own = V[i, bundles[i]].sum()
        for j in range(n):
            if i == j:
                continue
            goods_j = bundles[j]
            for g in goods_j:
                if V[i, g] <= 0.0:
                    continue
                without_g = V[i, goods_j].sum() - V[i, g]
                # relative tolerance against the scale of the comparison
                scale = max(abs(own), abs(without_g), 1.0)
                if own + rtol * scale < without_g:
                    return False
    return True


def assert_ef1(V: np.ndarray, A: np.ndarray, rtol: float = 1e-12) -> None:
    """Test C: v_i(A_i) >= v_i(A_j) - max{v_i(g) : g in A_j} for all i != j.

    Relative tolerance 1e-12 is an arithmetic artefact, not a fairness eps. Never stored.
    """
    n = V.shape[0]
    bundles = [np.flatnonzero(A == j) for j in range(n)]

    for i in range(n):
        own = V[i, bundles[i]].sum()
        for j in range(n):
            if i == j:
                continue
            goods_j = bundles[j]
            if goods_j.size == 0:
                continue
            vals = V[i, goods_j]
            rhs = vals.sum() - vals.max()
            scale = max(abs(own), abs(rhs), 1.0)
            assert own + rtol * scale >= rhs, (
                f"EF1 violated for i={i}, j={j}: "
                f"v_i(A_i)={own}, v_i(A_j)-max={rhs}"
            )


def assert_bundle_sizes_and_skipped(
    n: int, m: int, sizes: np.ndarray, skipped: float
) -> None:
    """Test D: sizes match r/q; skipped_fraction matches expected_skipped_fraction."""
    expected_sizes = expected_bundle_sizes(n, m)
    assert np.array_equal(sizes, expected_sizes), (
        f"bundle sizes {sizes} != expected {expected_sizes}"
    )
    expected_skipped = expected_skipped_fraction(n, m)
    assert abs(skipped - expected_skipped) < 1e-12, (
        f"skipped_fraction {skipped} != expected {expected_skipped}"
    )
