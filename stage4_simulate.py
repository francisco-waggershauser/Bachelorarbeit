"""Stage 4: run Monte Carlo cells, save .npz files, reconstruct instances."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stage0_theory import is_ordered_instance
from stage3_evaluate import collision_rate, evaluate
from stage2_mechanisms import enumerate_efx, round_robin
from stage1_sampling import build_values, build_Z, draw_base, validate_parameters


def instance_rng(master_seed: int, k: int, stream: int = 0) -> np.random.Generator:
    """The one place a per-instance generator is built.

    spawn_key=(k, stream): stream 0 = sampling, stream 1 = mechanism.
    Used by the simulation loop and by reconstruct().
    """
    return np.random.default_rng(
        np.random.SeedSequence(master_seed, spawn_key=(k, stream))
    )


def draw_instance(
    n: int,
    m: int,
    rho: float,
    theta: float,
    master_seed: int,
    k: int,
) -> dict:
    """Draw V (and latent Z) for one instance. No mechanism applied."""
    validate_parameters(n, m, rho, theta)
    sample_rng = instance_rng(master_seed, k, stream=0)
    W, eta = draw_base(n, m, sample_rng)
    Z = build_Z(W, eta, rho)
    V = build_values(Z, theta)
    return {
        "V": V,
        "Z": Z,
        "W": W,
        "eta": eta,
        "collision_rate": collision_rate(V),
    }


def _format_count(n: int) -> str:
    """Compact count for the one-line enum progress, e.g. 410k, 1.2M."""
    if n >= 1_000_000:
        value = n / 1_000_000
        text = f"{value:.1f}M"
        return text.replace(".0M", "M")
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)


def _write_enum_progress(msg: str, *, finish: bool = False) -> None:
    """Overwrite a single terminal line. finish=True then emits a newline."""
    width = getattr(_write_enum_progress, "_width", 0)
    width = max(width, len(msg))
    _write_enum_progress._width = 0 if finish else width
    sys.stdout.write("\r" + msg + " " * (width - len(msg)))
    if finish:
        sys.stdout.write("\n")
    sys.stdout.flush()


def compare_one_instance(
    n: int,
    m: int,
    rho: float,
    theta: float,
    master_seed: int,
    k: int,
    on_enum_progress: Callable[[int, int, float, float, bool], None] | None = None,
) -> dict:
    """Round Robin and optional enumeration on the same V."""
    inst = draw_instance(n, m, rho, theta, master_seed, k)
    V = inst["V"]
    mech_rng = instance_rng(master_seed, k, stream=1)
    A_rr = round_robin(V, mech_rng)
    rr_eval = evaluate(V, A_rr)
    alpha_rr = float(rr_eval["alpha_star"])
    rr_efx = alpha_rr >= 1.0

    if rr_efx:
        return {
            **inst,
            "alpha_rr": alpha_rr,
            "alpha_opt": 1.0,
            "exists_efx": True,
            "enum_skipped": True,
            "enum_elapsed_s": 0.0,
            "enum_n_scanned": 0,
        }

    enum = enumerate_efx(V, on_progress=on_enum_progress)
    return {
        **inst,
        "alpha_rr": alpha_rr,
        "alpha_opt": enum.alpha_opt,
        "exists_efx": enum.exists_efx,
        "enum_skipped": False,
        "enum_elapsed_s": enum.elapsed_s,
        "enum_n_scanned": enum.n_scanned,
        "enum_n_total": enum.n_total,
    }


def run_compare_cell(
    n: int,
    m: int,
    rho: float,
    theta: float,
    R: int,
    master_seed: int = 12345,
) -> dict:
    """R paired RR + enumeration instances. For n >= 4 only (caller enforces)."""
    alpha_rr = np.empty(R, dtype=np.float64)
    alpha_opt = np.empty(R, dtype=np.float64)
    exists_efx = np.empty(R, dtype=bool)
    enum_skipped = np.empty(R, dtype=bool)
    enum_elapsed = np.empty(R, dtype=np.float64)
    enum_n_scanned = np.empty(R, dtype=np.int64)
    cell_started = time.perf_counter()

    def _progress_line(
        k: int,
        *,
        skipped: bool = False,
        n_scanned: int = 0,
        n_total: int = 0,
        elapsed_s: float = 0.0,
        found_efx: bool = False,
        best_star: float = float("-inf"),
    ) -> str:
        prefix = f"enum  {k + 1}/{R}"
        if skipped:
            return f"{prefix}  skipped (RR already EFX)"
        if n_total > 0:
            pct = 100.0 * n_scanned / n_total
            scanned = f"{_format_count(n_scanned)}/{_format_count(n_total)} ({pct:.0f}%)"
        else:
            scanned = "0/0"
        if found_efx:
            status = "found EFX"
        elif best_star > float("-inf") and best_star == best_star:
            status = f"best α*={best_star:.2f}"
        else:
            status = "searching"
        return f"{prefix}  inst {elapsed_s:.1f}s  scanned {scanned}  {status}"

    for k in range(R):
        def on_progress(
            n_scanned: int,
            n_total: int,
            best_star: float,
            elapsed_s: float,
            found_efx: bool,
            k: int = k,
        ) -> None:
            _write_enum_progress(
                _progress_line(
                    k,
                    n_scanned=n_scanned,
                    n_total=n_total,
                    elapsed_s=elapsed_s,
                    found_efx=found_efx,
                    best_star=best_star,
                )
            )

        one = compare_one_instance(
            n, m, rho, theta, master_seed, k, on_enum_progress=on_progress
        )
        alpha_rr[k] = one["alpha_rr"]
        alpha_opt[k] = one["alpha_opt"]
        exists_efx[k] = one["exists_efx"]
        enum_skipped[k] = one["enum_skipped"]
        enum_elapsed[k] = one["enum_elapsed_s"]
        enum_n_scanned[k] = one["enum_n_scanned"]
        if one["enum_skipped"]:
            _write_enum_progress(_progress_line(k, skipped=True))
        else:
            _write_enum_progress(
                _progress_line(
                    k,
                    n_scanned=int(one["enum_n_scanned"]),
                    n_total=int(one.get("enum_n_total", 0)),
                    elapsed_s=float(one["enum_elapsed_s"]),
                    found_efx=bool(one["exists_efx"]),
                    best_star=float(one["alpha_opt"]),
                )
            )

    cell_elapsed = time.perf_counter() - cell_started
    _write_enum_progress(f"enum  {R}/{R}  done  {cell_elapsed:.1f}s", finish=True)

    return {
        "n": n,
        "m": m,
        "rho": rho,
        "theta": theta,
        "R": R,
        "master_seed": master_seed,
        "alpha_rr": alpha_rr,
        "alpha_opt": alpha_opt,
        "exists_efx": exists_efx,
        "enum_skipped": enum_skipped,
        "enum_elapsed_s": enum_elapsed,
        "enum_n_scanned": enum_n_scanned,
        "cell_elapsed_s": cell_elapsed,
    }


def simulate_one( #führt einen einzelnen Simulationsdurchlauf (Instanz k) von Anfang bis Ende aus
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
    sample_rng = instance_rng(master_seed, k, stream=0) #für das Ziehen der zufälligen Präferenzen
    mech_rng = instance_rng(master_seed, k, stream=1) #SPÄTER für eventuelle Zufallsentscheidungen der Mechanismen

    W, eta = draw_base(n, m, sample_rng) #jetzt W und eta
    Z = build_Z(W, eta, rho) #dann Z
    V = build_values(Z, theta) #jetzt zu V mit lognormal(theta*Z)
    rate = collision_rate(V)
    A = mechanism(V, mech_rng)
    result = evaluate(V, A)
    result["V"] = V
    result["Z"] = Z
    result["A"] = A
    result["collision_rate"] = rate
    return result


def run_cell( #führt ein komplettes Simulationsexperiment für eine feste Kombination von Parametern (n,m,ρ,θ) über R unabhängige Durchläufe 
#(Standard: R=2000) aus und speichert die aggregierten Ergebnisse.
    n: int,
    m: int,
    rho: float,
    theta: float,
    R: int = 2000,
    master_seed: int = 12345,
    mechanism: Callable[[np.ndarray, np.random.Generator], np.ndarray] = round_robin,
    out_dir: str | None = None,
    store_alpha_matrix: bool = True,
) -> dict:
    """R instances. No per-instance printing. No sanity asserts in this loop
    (those live in stage8_test_sanity.py). Returns a dict of arrays (see storage table).
    If out_dir is set, also writes one .npz. R is an argument, never a constant
    inside this function. mechanism= is the plug-in slot; default round_robin.
    Set store_alpha_matrix=False for null-model cells that only need alpha_star.
    """
    alpha_matrices = (
        np.empty((R, n, n), dtype=np.float64) if store_alpha_matrix else None
    )
    alpha_stars = np.empty(R, dtype=np.float64)
    sizes = np.empty((R, n), dtype=int)
    skipped = np.empty(R, dtype=np.float64)
    collisions = np.empty(R, dtype=np.float64)
    ordered = np.empty(R, dtype=bool)
    indices = np.arange(R, dtype=int)

    for k in range(R):
        one = simulate_one(n, m, rho, theta, master_seed, k, mechanism)
        if alpha_matrices is not None:
            alpha_matrices[k] = one["alpha_matrix"]
        alpha_stars[k] = one["alpha_star"]
        sizes[k] = one["bundle_sizes"]
        skipped[k] = one["skipped_fraction"]
        collisions[k] = one["collision_rate"]
        ordered[k] = is_ordered_instance(one["V"])

    record = { #speicher alles in einem Dictionary
        "master_seed": master_seed,
        "instance_index": indices,
        "n": n,
        "m": m,
        "rho": rho,
        "theta": theta,
        "mechanism": mechanism.__name__,
        "alpha_star": alpha_stars,
        "bundle_sizes": sizes,
        "skipped_fraction": skipped,
        "collision_rate": collisions,
        "ordered": ordered,
    }
    if alpha_matrices is not None:
        record["alpha_matrix"] = alpha_matrices

    if out_dir is not None: #Speicheplatz Zeugs
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

    alpha_matrix may be absent (null-model cells with store_alpha_matrix=False).
    ordered may be absent on older cells; summarise_cell rebuilds it via draw_instance.
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
