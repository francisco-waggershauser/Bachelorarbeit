"""Three figure types that read stored cells. None re-simulates."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse import conditional_quantile, efx_rate, survival
from scales import gini_label, spearman_from_rho, spearman_label
from simulate import cell_filename, load_cell


def _load_required_cell(
    results_dir: str,
    n: int,
    m: int,
    rho: float,
    theta: float,
    mechanism_name: str,
    master_seed: int,
) -> dict:
    """Load one cell via cell_filename, or raise FileNotFoundError with cell ids."""
    name = cell_filename(n, m, rho, theta, mechanism_name, master_seed)
    path = os.path.join(results_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing cell file for n={n}, m={m}, rho={rho}, theta={theta}: {path}"
        )
    return load_cell(path)


def plot_survival_panels(
    n: int,
    m: int,
    theta_fixed: float,
    rho_fixed: float,
    rho_grid: list[float],
    theta_grid: list[float],
    results_dir: str,
    out_path: str,
    eps: float = 0.0,
    alpha_grid: np.ndarray | None = None,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> None:
    """Two panels. Left: one curve per rho at theta_fixed (legend: Spearman).
    Right: one curve per theta at rho_fixed (legend: Gini).
    Panel titles carry n, m and the fixed scale (Gini or Spearman), not raw theta/rho.
    Title: Empirical survival function, Round Robin, R = ...
    R in the title is len(alpha_star) from the loaded cell, not an argument.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 501)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colours_rho = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))
    colours_theta = plt.cm.plasma(np.linspace(0.15, 0.85, len(theta_grid)))

    R_title: int | None = None

    for colour, rho in zip(colours_rho, rho_grid):
        cell = _load_required_cell(
            results_dir, n, m, rho, theta_fixed, mechanism_name, master_seed
        )
        stars = cell["alpha_star"]
        if R_title is None:
            R_title = len(stars)
        axes[0].plot(
            alpha_grid,
            survival(stars, alpha_grid),
            color=colour,
            label=spearman_label(rho),
        )

    for colour, theta in zip(colours_theta, theta_grid):
        cell = _load_required_cell(
            results_dir, n, m, rho_fixed, theta, mechanism_name, master_seed
        )
        stars = cell["alpha_star"]
        if R_title is None:
            R_title = len(stars)
        axes[1].plot(
            alpha_grid,
            survival(stars, alpha_grid),
            color=colour,
            label=gini_label(theta),
        )

    axes[0].set_title(f"n = {n}, m = {m}, {gini_label(theta_fixed)}")
    axes[1].set_title(f"n = {n}, m = {m}, {spearman_label(rho_fixed)}")

    for ax in axes:
        ax.set_xlabel(r"$\alpha$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"share of instances with $\alpha^* \geq \alpha$")
    fig.suptitle(
        f"Empirical survival function, Round Robin, R = {R_title}"
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rate_vs_m(
    n: int,
    theta: float,
    rho_grid: list[float],
    m_grid: list[int],
    results_dir: str,
    out_path: str,
    y: str = "efx_rate",
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> None:
    """x = m (integer axis, not m/n). One line per rho, legend Spearman.
    Title carries n and Gini(theta). Mark m where q = m % n == 0.
    Second version is the same function with y='conditional_q05'.
    """
    if y not in {"efx_rate", "conditional_q05"}:
        raise ValueError(f"y must be 'efx_rate' or 'conditional_q05', got {y!r}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))

    for colour, rho in zip(colours, rho_grid):
        ys = []
        for m in m_grid:
            cell = _load_required_cell(
                results_dir, n, m, rho, theta, mechanism_name, master_seed
            )
            stars = cell["alpha_star"]
            if y == "efx_rate":
                ys.append(efx_rate(stars, eps=eps))
            else:
                ys.append(
                    conditional_quantile(
                        stars, q=0.05, min_failures=min_failures, eps=eps
                    )
                )
        ax.plot(m_grid, ys, marker="o", color=colour, label=spearman_label(rho))

    for m in m_grid:
        if m % n == 0:
            ax.axvline(m, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("m")
    ax.set_xticks(m_grid)
    if y == "efx_rate":
        ax.set_ylabel("EFX rate")
        ax.set_ylim(0.0, 1.02)
    else:
        ax.set_ylabel("conditional 5% quantile of $\\alpha^*$")
    ax.set_title(f"n = {n}, {gini_label(theta)}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vs_collision(
    n: int,
    theta: float,
    rho_grid: list[float],
    m_grid: list[int],
    results_dir: str,
    out_path: str,
    eps: float = 0.0,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> None:
    """Scatter: x = mean collision_rate of the cell, y = EFX rate.
    One point per cell. Colour by Spearman(rho), marker by m.
    Title carries n and Gini(theta).
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.cm.viridis

    spearman_vals = [spearman_from_rho(rho) for rho in rho_grid]
    norm = plt.Normalize(min(spearman_vals), max(spearman_vals)) if spearman_vals else None

    for m_index, m in enumerate(m_grid):
        marker = markers[m_index % len(markers)]
        for rho in rho_grid:
            cell = _load_required_cell(
                results_dir, n, m, rho, theta, mechanism_name, master_seed
            )
            x = float(cell["collision_rate"].mean())
            y = efx_rate(cell["alpha_star"], eps=eps)
            colour = cmap(norm(spearman_from_rho(rho))) if norm is not None else "C0"
            ax.scatter(
                x,
                y,
                color=colour,
                marker=marker,
                s=55,
                edgecolors="k",
                linewidths=0.3,
                label=f"m={m}" if rho == rho_grid[0] else None,
            )

    ax.set_xlabel("mean collision rate")
    ax.set_ylabel("EFX rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"n = {n}, {gini_label(theta)}")
    ax.legend(loc="best", title="m")
    if norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Spearman")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
