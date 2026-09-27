#!/usr/bin/env python3
"""Stage 9: aggregated landscape over the thesis (n, m, ρ) grid.

Fills cells for n in {2..5}, m in {n+1..4n}, ρ in RHO_GRID at THETA_FIXED,
then writes one multipage PDF (m-sweep per n, then n-sweep at fixed m/n) plus CSV.

How to use
----------
1. Edit MECHANISM (round_robin or uniform_random_balanced), THETA_FIXED, R if needed.
2. ./stage9_aggregate.py
3. Open figures/aggregate_R{R}_seed{SEED}/aggregate_{mechanism}_theta….pdf

Theta series (same as RR): set THETA_FIXED to 0.1, run; then 0.25, 0.5, 0.75,
1.0, 1.5. RR and null .npz share the results folder; cell_filename includes the
mechanism name so they do not overwrite each other. Each run writes its own
aggregate_summary_{mechanism}_theta….csv and PDF.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_venv() -> None:
    root = Path(__file__).resolve().parent
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap_venv()

from stage2_mechanisms import round_robin, uniform_random_balanced
from stage4_simulate import cell_filename, load_cell, run_cell
from stage5_analyse import write_run_summary
from stage7_plot import plot_aggregate_overview

# --- edit these, then run: ./stage9_aggregate.py ---

N_GRID = [2, 3, 4, 5]
R = 2000  # raise to 2000 / 3000 for production
MASTER_SEED = 12345
# Mechanism plug-in: round_robin or uniform_random_balanced (same PDF layout).
MECHANISM = uniform_random_balanced
STORE_ALPHA_MATRIX = MECHANISM.__name__ != "uniform_random_balanced"

RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
THETA_FIXED = 1.5
M_OVER_N = 2.0  # n-sweep pages use m = round(M_OVER_N * n)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _tag() -> str:
    return f"aggregate_R{R}_seed{MASTER_SEED}"


def _m_grid(n: int) -> list[int]:
    return list(range(n + 1, 4 * n + 1))


def _ensure_cell(run_dir: str, n: int, m: int, rho: float, theta: float) -> str:
    """Reuse .npz if present with matching R. No code-version guard yet (TODO)."""
    name = cell_filename(n, m, rho, theta, MECHANISM.__name__, MASTER_SEED)
    path = os.path.join(run_dir, name)
    if os.path.isfile(path):
        if len(load_cell(path)["alpha_star"]) == R:
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


def main() -> None:
    tag = _tag()
    run_dir = os.path.join(RESULTS_DIR, tag)
    fig_dir = os.path.join(FIGURES_DIR, tag)
    n_m_pairs = [(n, m) for n in N_GRID for m in _m_grid(n)]
    n_cells = len(n_m_pairs) * len(RHO_GRID)
    print(
        f"aggregate: n={N_GRID}, m=n+1..4n, |ρ|={len(RHO_GRID)}, "
        f"R={R}, {MECHANISM.__name__}  → {n_cells} cells"
    )
    print(f"  cells -> {run_dir}")
    print(f"  figures -> {fig_dir}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for n, m in n_m_pairs:
        for rho in RHO_GRID:
            paths.append(_ensure_cell(run_dir, n, m, rho, THETA_FIXED))

    summary_path = os.path.join(
        run_dir,
        f"aggregate_summary_{MECHANISM.__name__}_theta{THETA_FIXED:.2f}.csv",
    )
    write_run_summary(paths, summary_path)

    pdf_path = os.path.join(
        fig_dir,
        f"aggregate_{MECHANISM.__name__}_theta{THETA_FIXED:.2f}_movern{M_OVER_N:g}.pdf",
    )
    plot_aggregate_overview(
        N_GRID,
        THETA_FIXED,
        RHO_GRID,
        run_dir,
        pdf_path,
        m_over_n=M_OVER_N,
        master_seed=MASTER_SEED,
        mechanism_name=MECHANISM.__name__,
    )

    n_pages = 3 * len(N_GRID) + 3
    print()
    print("-" * 60)
    print("Done (aggregate)")
    print(f"  cells    {run_dir}  ({len(paths)} files)")
    print(f"  summary  {summary_path}")
    print(f"  overview {pdf_path}")
    print(f"           {n_pages} pages: m-sweep × each n, then n-sweep (m/n={M_OVER_N:g})")
    print("-" * 60)


if __name__ == "__main__":
    main()
