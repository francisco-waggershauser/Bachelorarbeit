"""Sanity checks for v1: Tests C, D, E (Round Robin only) and F (mechanism-agnostic).

Run with: python test_sanity.py
"""

from __future__ import annotations

import numpy as np

from evaluate import (
    assert_bundle_sizes_and_skipped,
    assert_ef1,
    alpha_matrix,
    alpha_star,
    holds_efx,
)
from mechanisms import is_round_robin, round_robin
from simulate import simulate_one


def test_C_ef1(mechanism=round_robin, R: int = 200) -> None:
    """RR-only (skip otherwise). EF1 with relative tolerance 1e-12, a few hundred instances."""
    if not is_round_robin(mechanism):
        print("test_C_ef1: skipped (not Round Robin)")
        return

    n, m, rho, theta = 3, 7, 0.5, 1.0
    master_seed = 12345
    for k in range(R):
        one = simulate_one(n, m, rho, theta, master_seed, k, mechanism)
        assert_ef1(one["V"], one["A"])
    print(f"test_C_ef1: passed ({R} instances)")


def test_D_bundle_sizes_and_skipped(mechanism=round_robin, R: int = 50) -> None:
    """RR-only (skip otherwise). r/q bundle sizes and skipped_fraction formula,
    including a cell with r == 1 (n < m < 2n) and one with r >= 2.
    """
    if not is_round_robin(mechanism):
        print("test_D_bundle_sizes_and_skipped: skipped (not Round Robin)")
        return

    master_seed = 12345
    cases = [
        (3, 4, 0.5, 1.0),  # r == 1 (n < m < 2n)
        (3, 7, 0.5, 1.0),  # r == 2
        (4, 9, 0.25, 0.5),
    ]
    for n, m, rho, theta in cases:
        for k in range(R):
            one = simulate_one(n, m, rho, theta, master_seed, k, mechanism)
            assert_bundle_sizes_and_skipped(
                n, m, one["bundle_sizes"], one["skipped_fraction"]
            )
    print(f"test_D_bundle_sizes_and_skipped: passed ({len(cases)} cells x {R})")


def test_E_theta_invariance_of_allocation(mechanism=round_robin, R: int = 100) -> None:
    """RR-only (skip otherwise). Same master_seed, same k, theta=0.1 vs theta=1.5.
    Assert np.array_equal on A (and on Z). This is the pairing claim for theta sweeps.
    """
    if not is_round_robin(mechanism):
        print("test_E_theta_invariance_of_allocation: skipped (not Round Robin)")
        return

    n, m, rho = 3, 7, 0.5
    master_seed = 12345
    theta_lo, theta_hi = 0.1, 1.5
    for k in range(R):
        lo = simulate_one(n, m, rho, theta_lo, master_seed, k, mechanism)
        hi = simulate_one(n, m, rho, theta_hi, master_seed, k, mechanism)
        assert np.array_equal(lo["Z"], hi["Z"]), f"Z differs at k={k}"
        assert np.array_equal(lo["A"], hi["A"]), f"A differs at k={k}"
    print(f"test_E_theta_invariance_of_allocation: passed ({R} instances)")


def test_F_alpha_star_matches_efx(mechanism=round_robin, R: int = 300) -> None:
    """Mechanism-agnostic. A few hundred random instances plus the hand example (100,1,1,1), n=2.
    Assert holds_efx(V, A) == (alpha_star >= 1).
    holds_efx is written from the definition, not from the alpha formula.
    """
    # Hand example: identical values (100, 1, 1, 1), n = 2.
    # RR gives A0 = {100, 1}, A1 = {1, 1}; agent 1 fails EFX after removing cheapest.
    V_hand = np.array([[100.0, 1.0, 1.0, 1.0], [100.0, 1.0, 1.0, 1.0]])
    A_hand = round_robin(V_hand, np.random.default_rng(0))
    star_hand = alpha_star(alpha_matrix(V_hand, A_hand))
    assert holds_efx(V_hand, A_hand) == (star_hand >= 1.0), (
        f"hand example mismatch: holds_efx={holds_efx(V_hand, A_hand)}, "
        f"alpha_star={star_hand}"
    )
    assert star_hand < 1.0, f"hand example should fail EFX, got alpha_star={star_hand}"

    n, m, rho, theta = 3, 7, 0.5, 1.0
    master_seed = 99
    for k in range(R):
        one = simulate_one(n, m, rho, theta, master_seed, k, mechanism)
        efx = holds_efx(one["V"], one["A"])
        star = one["alpha_star"]
        assert efx == (star >= 1.0), (
            f"mismatch at k={k}: holds_efx={efx}, alpha_star={star}"
        )
    print(f"test_F_alpha_star_matches_efx: passed (hand example + {R} instances)")


if __name__ == "__main__":
    test_C_ef1()
    test_D_bundle_sizes_and_skipped()
    test_E_theta_invariance_of_allocation()
    test_F_alpha_star_matches_efx()
    print("All sanity tests passed.")
