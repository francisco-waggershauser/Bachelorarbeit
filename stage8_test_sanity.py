"""Stage 8: sanity checks for v1: Tests C, D, E (Round Robin only), F (mechanism-agnostic),
N (uniform_random_balanced size profile), enum G/H, pairing P, certificates Q/R,
and compare_cells S.

Run with: python stage8_test_sanity.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from stage0_theory import cell_certificate
from stage3_evaluate import (
    assert_bundle_sizes_and_skipped,
    assert_ef1,
    alpha_matrix,
    alpha_star,
    expected_bundle_sizes,
    holds_efx,
)
from stage2_mechanisms import (
    enumerate_efx,
    is_round_robin,
    round_robin,
    uniform_random_balanced,
)
from stage4_simulate import cell_filename, run_cell, simulate_one
from stage5_analyse import compare_cells


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


def test_N_null_model_bundle_sizes(R: int = 200) -> None:
    """uniform_random_balanced always matches expected_bundle_sizes(n, m)."""
    master_seed = 12345
    cases = [(2, 3), (2, 5), (3, 4), (3, 7), (4, 5), (4, 9), (5, 8)]
    for n, m in cases:
        expected = expected_bundle_sizes(n, m)
        for k in range(R):
            one = simulate_one(n, m, 0.5, 1.0, master_seed, k, uniform_random_balanced)
            assert np.array_equal(one["bundle_sizes"], expected), (
                f"bundle sizes mismatch at n={n}, m={m}, k={k}: "
                f"got {one['bundle_sizes']}, expected {expected}"
            )
    print(f"test_N_null_model_bundle_sizes: passed ({len(cases)} pairs x {R})")


def test_G_enumeration_hand_example() -> None:
    """RR fails EFX on the canonical hand example; enumeration finds a witness."""
    V_hand = np.array([[100.0, 1.0, 1.0, 1.0], [100.0, 1.0, 1.0, 1.0]])
    A_hand = round_robin(V_hand, np.random.default_rng(0))
    star_hand = alpha_star(alpha_matrix(V_hand, A_hand))
    assert star_hand < 1.0, f"hand example should fail RR EFX, got alpha_star={star_hand}"

    result = enumerate_efx(V_hand)
    assert result.exists_efx, "enumeration should find an EFX allocation"
    assert result.alpha_opt == 1.0
    assert result.A_opt is not None
    assert holds_efx(V_hand, result.A_opt)
    print("test_G_enumeration_hand_example: passed")


def test_H_enumeration_n3_always_exists(R: int = 10) -> None:
    """On n=3 additive instances, brute force must always find an EFX allocation."""
    n, m, rho, theta = 3, 7, 0.5, 1.0
    master_seed = 12345
    for k in range(R):
        one = simulate_one(n, m, rho, theta, master_seed, k, round_robin)
        result = enumerate_efx(one["V"])
        assert result.exists_efx, f"no EFX found at k={k} (n=3 should always have one)"
        assert result.alpha_opt == 1.0
    print(f"test_H_enumeration_n3_always_exists: passed ({R} instances)")


def test_P_theta_pairing_rr_and_null(R: int = 50) -> None:
    """Same (master_seed, k, n, m, rho), two thetas → same A for RR and for null."""
    n, m, rho = 3, 7, 0.5
    master_seed = 12345
    theta_lo, theta_hi = 0.1, 1.5
    for k in range(R):
        for mech in (round_robin, uniform_random_balanced):
            lo = simulate_one(n, m, rho, theta_lo, master_seed, k, mech)
            hi = simulate_one(n, m, rho, theta_hi, master_seed, k, mech)
            assert np.array_equal(lo["A"], hi["A"]), (
                f"A differs for {mech.__name__} at k={k}"
            )
    print(f"test_P_theta_pairing_rr_and_null: passed ({R} instances x 2 mechanisms)")


def test_Q_rho1_identical_valuations(R: int = 30) -> None:
    """rho = 1 ⇒ all rows of V are numerically equal (S4 sampler property)."""
    n, m, rho, theta = 4, 9, 1.0, 1.0
    master_seed = 12345
    for k in range(R):
        one = simulate_one(n, m, rho, theta, master_seed, k, round_robin)
        V = one["V"]
        assert np.allclose(V, V[0]), f"V rows differ at k={k} under rho=1"
    print(f"test_Q_rho1_identical_valuations: passed ({R} instances)")


def test_R_cell_certificate_logic() -> None:
    """Known (n, m, rho) → expected certificate key."""
    cases = [
        (2, 5, 0.0, "S1"),
        (4, 7, 0.0, "S3"),
        (4, 9, 1.0, "S4"),
        (4, 9, 0.5, None),
    ]
    for n, m, rho, expected in cases:
        got = cell_certificate(n, m, rho)
        assert got == expected, f"({n},{m},{rho}): got {got!r}, expected {expected!r}"
    print("test_R_cell_certificate_logic: passed")


def test_S_compare_cells_rejects_mismatches() -> None:
    """compare_cells raises AssertionError on mismatched seed, R, or rho."""
    with tempfile.TemporaryDirectory() as tmp:
        run_cell(
            2, 3, 0.5, 1.0, R=20, master_seed=12345, mechanism=round_robin, out_dir=tmp
        )
        path_rr = os.path.join(
            tmp, cell_filename(2, 3, 0.5, 1.0, "round_robin", 12345)
        )
        run_cell(
            2,
            3,
            0.5,
            1.0,
            R=20,
            master_seed=12345,
            mechanism=uniform_random_balanced,
            out_dir=tmp,
            store_alpha_matrix=False,
        )
        path_null = os.path.join(
            tmp, cell_filename(2, 3, 0.5, 1.0, "uniform_random_balanced", 12345)
        )
        compare_cells(path_rr, path_null)

        run_cell(
            2,
            3,
            0.0,
            1.0,
            R=20,
            master_seed=12345,
            mechanism=uniform_random_balanced,
            out_dir=tmp,
            store_alpha_matrix=False,
        )
        path_bad_rho = os.path.join(
            tmp, cell_filename(2, 3, 0.0, 1.0, "uniform_random_balanced", 12345)
        )
        try:
            compare_cells(path_rr, path_bad_rho)
            raise AssertionError("expected AssertionError for mismatched rho")
        except AssertionError as exc:
            assert "rho" in str(exc), exc

        run_cell(
            2,
            3,
            0.5,
            1.0,
            R=20,
            master_seed=99999,
            mechanism=uniform_random_balanced,
            out_dir=tmp,
            store_alpha_matrix=False,
        )
        path_bad_seed = os.path.join(
            tmp, cell_filename(2, 3, 0.5, 1.0, "uniform_random_balanced", 99999)
        )
        try:
            compare_cells(path_rr, path_bad_seed)
            raise AssertionError("expected AssertionError for mismatched master_seed")
        except AssertionError as exc:
            assert "master_seed" in str(exc), exc

        run_cell(
            2,
            3,
            0.5,
            1.0,
            R=10,
            master_seed=12345,
            mechanism=uniform_random_balanced,
            out_dir=tmp,
            store_alpha_matrix=False,
        )
        path_bad_R = os.path.join(
            tmp, cell_filename(2, 3, 0.5, 1.0, "uniform_random_balanced", 12345)
        )
        try:
            compare_cells(path_rr, path_bad_R)
            raise AssertionError("expected AssertionError for mismatched R")
        except AssertionError as exc:
            assert "R" in str(exc), exc

    print("test_S_compare_cells_rejects_mismatches: passed")


if __name__ == "__main__":
    test_C_ef1()
    test_D_bundle_sizes_and_skipped()
    test_E_theta_invariance_of_allocation()
    test_F_alpha_star_matches_efx()
    test_N_null_model_bundle_sizes()
    test_G_enumeration_hand_example()
    test_H_enumeration_n3_always_exists()
    test_P_theta_pairing_rr_and_null()
    test_Q_rho1_identical_valuations()
    test_R_cell_certificate_logic()
    test_S_compare_cells_rejects_mismatches()
    print("All sanity tests passed.")
