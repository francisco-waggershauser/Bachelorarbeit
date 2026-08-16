"""Example run: Round Robin cells for n=3, plots, and one describe_cell block.

Edit the variables at the top to change grids and R. R is set only here.
"""

from __future__ import annotations

import os
from pathlib import Path

from analyse import describe_cell
from mechanisms import round_robin
from plot import plot_rate_vs_m, plot_survival_panels, plot_vs_collision
from simulate import cell_filename, run_cell

# --- free parameters (edit these) ---
N = 3
M = 7
R = 2_000
MASTER_SEED = 12345

RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
THETA_GRID = [0.1, 0.5, 1.0, 1.5]
THETA_FIXED = 1.0
RHO_FIXED = 0.5
M_GRID = list(range(N + 1, 2 * N + 2))

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _ensure_cell(n: int, m: int, rho: float, theta: float) -> str:
    """Run a cell if the .npz is missing; return its path."""
    name = cell_filename(n, m, rho, theta, round_robin.__name__, MASTER_SEED)
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.isfile(path):
        run_cell(
            n,
            m,
            rho,
            theta,
            R=R,
            master_seed=MASTER_SEED,
            mechanism=round_robin,
            out_dir=RESULTS_DIR,
        )
    return path


def main() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    # Survival panels: all rho at THETA_FIXED, all theta at RHO_FIXED, for (N, M)
    for rho in RHO_GRID:
        _ensure_cell(N, M, rho, THETA_FIXED)
    for theta in THETA_GRID:
        _ensure_cell(N, M, RHO_FIXED, theta)

    # Rate / collision plots: all m in M_GRID × all rho at THETA_FIXED
    for m in M_GRID:
        for rho in RHO_GRID:
            _ensure_cell(N, m, rho, THETA_FIXED)

    plot_survival_panels(
        N,
        M,
        THETA_FIXED,
        RHO_FIXED,
        RHO_GRID,
        THETA_GRID,
        RESULTS_DIR,
        os.path.join(FIGURES_DIR, "survival_panels.png"),
        master_seed=MASTER_SEED,
    )
    plot_rate_vs_m(
        N,
        THETA_FIXED,
        RHO_GRID,
        M_GRID,
        RESULTS_DIR,
        os.path.join(FIGURES_DIR, "rate_vs_m.png"),
        y="efx_rate",
        master_seed=MASTER_SEED,
    )
    plot_rate_vs_m(
        N,
        THETA_FIXED,
        RHO_GRID,
        M_GRID,
        RESULTS_DIR,
        os.path.join(FIGURES_DIR, "q05_vs_m.png"),
        y="conditional_q05",
        master_seed=MASTER_SEED,
    )
    plot_vs_collision(
        N,
        THETA_FIXED,
        RHO_GRID,
        M_GRID,
        RESULTS_DIR,
        os.path.join(FIGURES_DIR, "vs_collision.png"),
        master_seed=MASTER_SEED,
    )

    main_cell = _ensure_cell(N, M, RHO_FIXED, THETA_FIXED)
    describe_cell(main_cell)


if __name__ == "__main__":
    main()
