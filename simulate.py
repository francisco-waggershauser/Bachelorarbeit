"""Stage 2/4: run Monte Carlo cells, save .npz files, reconstruct instances."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from evaluate import collision_rate, evaluate
from mechanisms import round_robin
from sampling import build_values, build_Z, draw_base, validate_parameters


def instance_rng(master_seed: int, k: int, stream: int = 0) -> np.random.Generator:
    """The one place a per-instance generator is built.

    spawn_key=(k, stream): stream 0 = sampling, stream 1 = mechanism.
    Used by the simulation loop and by reconstruct().
    """
    return np.random.default_rng(
        np.random.SeedSequence(master_seed, spawn_key=(k, stream))
    )


def simulate_one(
    n: int,
    m: int,
    rho: float,
    theta: float,
    master_seed: int,
    k: int,
    mechanism: Callable[[np.ndarray, np.random.Generator], np.ndarray],
) -> dict:
    """One instance. Calls validate_parameters first (so run_cell and reconstruct cannot skip it).

    Then builds both streams from (master_seed, k), then
    draw_base(sample_rng) -> build_Z -> build_values -> collision_rate(V)
    -> mechanism(V, mech_rng) -> evaluate(V, A).

    Comment: draw_base depends only on (n, m, rng), build_Z only extra on rho.
    So Z (and under RR, A) is the same for fixed (master_seed, k) at every theta.
    Dropping a mechanism(Z) shortcut costs CPU, not pairing across a theta sweep.

    Returns V, Z, A, plus the evaluate dict and collision_rate.
    Internal helper; run_cell and reconstruct both call this.
    """
    validate_parameters(n, m, rho, theta)
    sample_rng = instance_rng(master_seed, k, stream=0)
    mech_rng = instance_rng(master_seed, k, stream=1)

    W, eta = draw_base(n, m, sample_rng)
    Z = build_Z(W, eta, rho)
    V = build_values(Z, theta)
    rate = collision_rate(V)
    A = mechanism(V, mech_rng)
    result = evaluate(V, A)
    result["V"] = V
    result["Z"] = Z
    result["A"] = A
    result["collision_rate"] = rate
    return result


def run_cell(
    n: int,
    m: int,
    rho: float,
    theta: float,
    R: int = 2000,
    master_seed: int = 12345,
    mechanism: Callable[[np.ndarray, np.random.Generator], np.ndarray] = round_robin,
    out_dir: str | None = None,
) -> dict:
    """R instances. No per-instance printing. No sanity asserts in this loop
    (those live in test_sanity.py). Returns a dict of arrays (see storage table).
    If out_dir is set, also writes one .npz. R is an argument, never a constant
    inside this function. mechanism= is the plug-in slot; default round_robin.
    """
    alpha_matrices = np.empty((R, n, n), dtype=np.float64)
    alpha_stars = np.empty(R, dtype=np.float64)
    sizes = np.empty((R, n), dtype=int)
    skipped = np.empty(R, dtype=np.float64)
    collisions = np.empty(R, dtype=np.float64)
    indices = np.arange(R, dtype=int)

    for k in range(R):
        one = simulate_one(n, m, rho, theta, master_seed, k, mechanism)
        alpha_matrices[k] = one["alpha_matrix"]
        alpha_stars[k] = one["alpha_star"]
        sizes[k] = one["bundle_sizes"]
        skipped[k] = one["skipped_fraction"]
        collisions[k] = one["collision_rate"]

    record = {
        "master_seed": master_seed,
        "instance_index": indices,
        "n": n,
        "m": m,
        "rho": rho,
        "theta": theta,
        "mechanism": mechanism.__name__,
        "alpha_matrix": alpha_matrices,
        "alpha_star": alpha_stars,
        "bundle_sizes": sizes,
        "skipped_fraction": skipped,
        "collision_rate": collisions,
    }

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(
            out_dir,
            cell_filename(n, m, rho, theta, mechanism.__name__, master_seed),
        )
        save_cell(path, record)

    return record


def reconstruct(
    master_seed: int,
    k: int,
    n: int,
    m: int,
    rho: float,
    theta: float,
    mechanism: Callable[[np.ndarray, np.random.Generator], np.ndarray] = round_robin,
) -> dict:
    """Rebuild one instance from two integers plus cell parameters. Never reads a file.

    Calls simulate_one. Returns V, Z, A, alpha_matrix, alpha_star, bundle_sizes, collision_rate.
    """
    return simulate_one(n, m, rho, theta, master_seed, k, mechanism)


def cell_filename(
    n: int,
    m: int,
    rho: float,
    theta: float,
    mechanism_name: str,
    master_seed: int,
) -> str:
    """The only place a cell name is built.

    n{n}_m{m}_rho{rho:.2f}_theta{theta:.2f}_{mechanism_name}_seed{master_seed}.npz
    Plot and analyse must call this, never an ad-hoc f-string.
    """
    return (
        f"n{n}_m{m}_rho{rho:.2f}_theta{theta:.2f}_"
        f"{mechanism_name}_seed{master_seed}.npz"
    )


def save_cell(path: str, record: dict) -> None:
    """np.savez_compressed. One binary file per cell."""
    np.savez_compressed(path, **record)


def load_cell(path: str) -> dict:
    """Read a cell .npz. Call .item() on every 0-d array so mechanism is a Python str,
    and n, m, rho, theta, master_seed are Python scalars.
    """
    with np.load(path, allow_pickle=False) as data:
        record = {}
        for key in data.files:
            value = data[key]
            if value.ndim == 0:
                record[key] = value.item()
            else:
                record[key] = value
    return record


def cell_path(
    results_dir: str,
    n: int,
    m: int,
    rho: float,
    theta: float,
    mechanism_name: str,
    master_seed: int,
) -> str:
    """Full path to a stored cell, built only via cell_filename."""
    return os.path.join(
        results_dir,
        cell_filename(n, m, rho, theta, mechanism_name, master_seed),
    )
