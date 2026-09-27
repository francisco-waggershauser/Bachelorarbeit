#!/usr/bin/env python3
"""Stage 10: paired RR vs mechanism X comparison over the aggregate grid.

Reads existing .npz cells (no re-sim). Default X = uniform_random_balanced.
Swap MECHANISM_B for another mechanism later (same seed / R / grids).

How to use
----------
1. Ensure Stage 9 has filled RR and MECHANISM_B cells for the θ grid.
2. Edit MECHANISM_B / grids / THETA_PLOT if needed.
3. ./stage10_compare.py
4. Open comparison_{A}_vs_{B}.csv and the Compare-PDF under figures/…
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
from stage4_simulate import cell_filename
from stage5_analyse import compare_cells, write_mechanism_comparison
from stage7_plot import plot_comparison_overview

# --- edit these, then run: ./stage10_compare.py ---

N_GRID = [2, 3, 4, 5]
R = 2000
MASTER_SEED = 12345
MECHANISM_A = round_robin  # fixed reference
MECHANISM_B = uniform_random_balanced  # plug-in X

RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
THETA_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

THETA_PLOT = 1.5  # main PDF pages filter to this θ
M_OVER_N = 2.0  # n-sweep / θ-sweep pages use m = round(M_OVER_N * n)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _tag() -> str:
    return f"aggregate_R{R}_seed{MASTER_SEED}"


def _m_grid(n: int) -> list[int]:
    return list(range(n + 1, 4 * n + 1))


def main() -> None:
    tag = _tag()
    run_dir = os.path.join(RESULTS_DIR, tag)
    fig_dir = os.path.join(FIGURES_DIR, tag)
    name_a = MECHANISM_A.__name__
    name_b = MECHANISM_B.__name__
    csv_path = os.path.join(run_dir, f"comparison_{name_a}_vs_{name_b}.csv")
    pdf_path = os.path.join(
        fig_dir,
        f"comparison_{name_a}_vs_{name_b}_theta{THETA_PLOT:.2f}.pdf",
    )

    n_m_pairs = [(n, m) for n in N_GRID for m in _m_grid(n)]
    n_expected = len(n_m_pairs) * len(RHO_GRID) * len(THETA_GRID)
    print(
        f"compare: {name_a} vs {name_b}, "
        f"{n_expected} pairs, R={R}, dir={run_dir}"
    )

    rows: list[dict] = []
    for theta in THETA_GRID:
        for n, m in n_m_pairs:
            for rho in RHO_GRID:
                path_a = os.path.join(
                    run_dir,
                    cell_filename(n, m, rho, theta, name_a, MASTER_SEED),
                )
                path_b = os.path.join(
                    run_dir,
                    cell_filename(n, m, rho, theta, name_b, MASTER_SEED),
                )
                if not os.path.isfile(path_a):
                    raise FileNotFoundError(f"missing cell A: {path_a}")
                if not os.path.isfile(path_b):
                    raise FileNotFoundError(f"missing cell B: {path_b}")
                rows.append(compare_cells(path_a, path_b))

    write_mechanism_comparison(rows, csv_path)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plot_comparison_overview(
        csv_path,
        pdf_path,
        theta_fixed=THETA_PLOT,
        m_over_n=M_OVER_N,
    )

    print()
    print("-" * 60)
    print("Done (mechanism comparison)")
    print(f"  rows   {len(rows)}")
    print(f"  csv    {csv_path}")
    print(f"  overview {pdf_path}")
    print(f"           8 pages at {theta_label_safe()}")
    print("-" * 60)


def theta_label_safe() -> str:
    return f"THETA_PLOT={THETA_PLOT:.2f}, m/n={M_OVER_N:g}"


if __name__ == "__main__":
    main()
