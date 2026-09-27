#!/usr/bin/env python3
"""Stage 9: run one mechanism for fixed (n, m): simulate, save a run folder, write overview PDF.

How to use
----------
1. Edit the settings below (at least N, M, R, MECHANISM).
2. In the terminal, from this folder:  ./main.py
3. When it finishes, the overview PDF is written (no plot window). Cells and a
   readable summary.csv are in results/n{N}_m{M}_R{R}_seed{SEED}/; the PDF is in
   the matching figures/ folder, named with THETA_FIXED and RHO_FIXED.
   Four pages for this fixed (n, m): survival by ρ, survival by θ, q05 vs ρ,
   collision at this m. Open the PDF from disk.

Plug in MECHANISM = round_robin or uniform_random_balanced — same PDF layout.
Same n, m, R and seed reuse that run folder. A different R is a new folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_venv() -> None:
    """Re-exec with .venv/bin/python when present (no manual venv path needed)."""
    root = Path(__file__).resolve().parent
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap_venv()

from stage5_analyse import describe_cell, describe_compare_row, write_compare_summary, write_run_summary
from stage2_mechanisms import round_robin, uniform_random_balanced
from stage7_plot import plot_run_overview
from stage4_simulate import cell_filename, compare_one_instance, load_cell, run_cell, run_compare_cell

# --- edit these, then run: ./main.py ---

N = 4  # number of agents (>= 2)
M = 13  # number of goods (must be > N)
R = 1000  # small test R; raise to 3000 / 20000 for real runs
MASTER_SEED = 12345  # same seed + same k rebuilds the same instance

# Mechanism plug-in: round_robin or uniform_random_balanced (same overview PDF).
MECHANISM = round_robin
STORE_ALPHA_MATRIX = MECHANISM.__name__ != "uniform_random_balanced"

# Brute-force EFX enumeration paired with RR on the same V. Default off.
# Forced off when n < 4 (EFX existence is known). Only runs on the main cell
# (N, M, RHO_FIXED, THETA_FIXED), not the full grid. RR-only comparison arm.
ENUMERATE = 0
ENUM_PROBE_INSTANCES = 20  # timing probe before full compare when ENUMERATE=1

# Left survival panel: one curve per rho, all at THETA_FIXED.
# Right survival panel: one curve per theta, all at RHO_FIXED.
# Reference levels are documentation defaults, not sacrosanct.
RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
THETA_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
THETA_FIXED = 1.0
RHO_FIXED = 0.5

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _run_tag() -> str:
    return f"n{N}_m{M}_R{R}_seed{MASTER_SEED}"


def _ensure_cell(run_dir: str, n: int, m: int, rho: float, theta: float) -> str:
    """Run a cell if the .npz is missing or stored with a different R."""
    name = cell_filename(n, m, rho, theta, MECHANISM.__name__, MASTER_SEED)
    path = os.path.join(run_dir, name)
    if os.path.isfile(path):
        stored_R = len(load_cell(path)["alpha_star"])
        if stored_R == R:
            return path
        os.remove(path)
    run_cell(
        n,
        m,
        rho,
        theta,
        R=R,
        master_seed=MASTER_SEED,
        mechanism=MECHANISM,
        out_dir=run_dir,
        store_alpha_matrix=STORE_ALPHA_MATRIX,
    )
    return path


def _run_enum_probe(n: int, m: int, rho: float, theta: float, probe_R: int) -> None:
    """Print median/max enum time on a few instances before a full compare cell."""
    import numpy as np

    times: list[float] = []
    for k in range(probe_R):
        one = compare_one_instance(n, m, rho, theta, MASTER_SEED, k)
        if one["enum_skipped"]:
            times.append(0.0)
        else:
            times.append(one["enum_elapsed_s"])
    arr = np.asarray(times)
    print(
        f"  Enum probe ({probe_R} instances): "
        f"median {np.median(arr) * 1000:.0f}ms, max {arr.max() * 1000:.0f}ms"
    )


def _maybe_run_enumeration(run_dir: str) -> None:
    """Optional paired RR + enumeration on the main cell only."""
    if ENUMERATE == 0:
        return
    if MECHANISM is not round_robin:
        print("Enumeration skipped: compare arm is Round-Robin only.")
        return
    if N < 4:
        print("Enumeration skipped: EFX exists by theorem for n < 4 (additive).")
        return

    print()
    print("Enumeration (paired with Round-Robin on the same V)")
    print(f"  cell n={N}, m={M}, rho={RHO_FIXED}, theta={THETA_FIXED}, R={R}")
    _run_enum_probe(N, M, RHO_FIXED, THETA_FIXED, ENUM_PROBE_INSTANCES)

    compare_record = run_compare_cell(
        N, M, RHO_FIXED, THETA_FIXED, R=R, master_seed=MASTER_SEED
    )
    describe_compare_row(compare_record)

    compare_dir = os.path.join(run_dir, "compare")
    Path(compare_dir).mkdir(parents=True, exist_ok=True)
    compare_csv = os.path.join(
        compare_dir,
        f"compare_n{N}_m{M}_rho{RHO_FIXED:.2f}_theta{THETA_FIXED:.2f}_R{R}.csv",
    )
    write_compare_summary([compare_record], compare_csv)
    print(f"  compare CSV -> {compare_csv}")


def main() -> None:
    tag = _run_tag()
    run_dir = os.path.join(RESULTS_DIR, tag)
    fig_dir = os.path.join(FIGURES_DIR, tag)
    print(f"This run: n={N}, m={M}, R={R}, mechanism={MECHANISM.__name__}")
    print(f"  cells -> {run_dir}")
    print(f"  figures -> {fig_dir}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    cell_paths: list[str] = []

    # Fixed (N, M): all rho at THETA_FIXED, all theta at RHO_FIXED
    for rho in RHO_GRID:
        cell_paths.append(_ensure_cell(run_dir, N, M, rho, THETA_FIXED))
    for theta in THETA_GRID:
        cell_paths.append(_ensure_cell(run_dir, N, M, RHO_FIXED, theta))

    # Unique paths, stable order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for path in cell_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    summary_path = os.path.join(run_dir, "summary.csv")
    write_run_summary(unique_paths, summary_path)

    overview_pdf = os.path.join(
        fig_dir,
        f"overview_{MECHANISM.__name__}_theta{THETA_FIXED:.2f}_rho{RHO_FIXED:.2f}.pdf",
    )
    plot_run_overview(
        N,
        M,
        THETA_FIXED,
        RHO_FIXED,
        RHO_GRID,
        THETA_GRID,
        run_dir,
        overview_pdf,
        master_seed=MASTER_SEED,
        mechanism_name=MECHANISM.__name__,
        show=False,
    )

    main_cell = _ensure_cell(run_dir, N, M, RHO_FIXED, THETA_FIXED)
    describe_cell(main_cell)
    _maybe_run_enumeration(run_dir)
    print()
    print("-" * 60)
    print("Done")
    print(f"  mechanism {MECHANISM.__name__}")
    print(f"  cells     {run_dir}")
    print(f"  summary   {summary_path}")
    print(f"  overview  {overview_pdf}")
    print("            4 pages for this (n, m): survival ρ/θ, q05 vs ρ, collision")
    print("            open the PDF from disk (no plot window)")
    print("-" * 60)


if __name__ == "__main__":
    main()
