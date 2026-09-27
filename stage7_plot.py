"""Stage 7: figures that read stored cells. None re-simulate."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from stage5_analyse import (
    conditional_quantile,
    efx_rate,
    format_float,
    format_pct,
    summarise_cell,
    survival,
)
from stage6_scales import rho_label, spearman_from_rho, theta_label
from stage4_simulate import cell_filename, load_cell


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


def _finish_figure(fig, out_path: str, show: bool) -> None:
    """Save the figure. Leave it open if show is True so the caller can plt.show()."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if not show:
        plt.close(fig)
        


def _draw_survival(
    axes: np.ndarray,
    n: int,
    m: int,
    theta_fixed: float,
    rho_fixed: float,
    rho_grid: list[float],
    theta_grid: list[float],
    results_dir: str,
    eps: float,
    alpha_grid: np.ndarray,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """Draw the two survival panels onto axes[0] and axes[1]. Returns R."""
    colours_rho = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))
    colours_theta = plt.cm.plasma(np.linspace(0.15, 0.85, len(theta_grid)))
    R_title = 0

    for colour, rho in zip(colours_rho, rho_grid):
        cell = _load_required_cell(
            results_dir, n, m, rho, theta_fixed, mechanism_name, master_seed
        )
        stars = cell["alpha_star"]
        R_title = len(stars)
        axes[0].plot(
            alpha_grid,
            survival(stars, alpha_grid),
            color=colour,
            label=rho_label(rho),
        )

    for colour, theta in zip(colours_theta, theta_grid):
        cell = _load_required_cell(
            results_dir, n, m, rho_fixed, theta, mechanism_name, master_seed
        )
        stars = cell["alpha_star"]
        axes[1].plot(
            alpha_grid,
            survival(stars, alpha_grid),
            color=colour,
            label=theta_label(theta),
        )

    axes[0].set_title(f"n = {n}, m = {m}, {theta_label(theta_fixed)}")
    axes[1].set_title(f"n = {n}, m = {m}, {rho_label(rho_fixed)}")
    for ax in axes:
        ax.set_xlabel(r"$\alpha$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"share of instances with $\alpha^* \geq \alpha$")
    return R_title


def _draw_rate_vs_m(
    ax,
    n: int,
    theta: float,
    rho_grid: list[float],
    m_grid: list[int],
    results_dir: str,
    y: str,
    eps: float,
    min_failures: int,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """Draw rate or quantile vs m onto ax. Returns R."""
    if y not in {"efx_rate", "conditional_q05"}:
        raise ValueError(f"y must be 'efx_rate' or 'conditional_q05', got {y!r}")
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))
    R_title = 0
    for colour, rho in zip(colours, rho_grid):
        ys = []
        for m in m_grid:
            cell = _load_required_cell(
                results_dir, n, m, rho, theta, mechanism_name, master_seed
            )
            stars = cell["alpha_star"]
            R_title = len(stars)
            if y == "efx_rate":
                ys.append(efx_rate(stars, eps=eps))
            else:
                ys.append(
                    conditional_quantile(
                        stars, q=0.05, min_failures=min_failures, eps=eps
                    )
                )
        ax.plot(m_grid, ys, marker="o", color=colour, label=rho_label(rho))

    for m in m_grid:
        if m % n == 0:
            ax.axvline(m, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("m")
    ax.set_xticks(m_grid)
    if y == "efx_rate":
        ax.set_ylabel("EFX rate")
        ax.set_ylim(0.0, 1.02)
        kind = "EFX rate vs m"
    else:
        ax.set_ylabel("conditional 5% quantile of $\\alpha^*$")
        kind = "conditional 5% quantile vs m"
    ax.set_title(f"{kind}\nn = {n}, R = {R_title}, {theta_label(theta)}")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)
    return R_title


def _rho_grid_for_collision_scatter(rho_grid: list[float]) -> list[float]:
    """Omit ρ = 1: identical prefs make collision deterministic (1 - 1/n)."""
    out = [rho for rho in rho_grid if rho < 1.0 - 1e-12]
    return out if out else list(rho_grid)


def _collision_scatter_norm(plot_rhos: list[float]):
    """Normalize colour by Spearman of the plotted ρ grid (dynamic min/max)."""
    if not plot_rhos:
        return None
    spearman_vals = [spearman_from_rho(rho) for rho in plot_rhos]
    return plt.Normalize(min(spearman_vals), max(spearman_vals))


def _add_collision_spearman_cbar(fig, ax, cmap, plot_rhos: list[float]) -> None:
    """Colorbar ticks exactly at plotted ρ (labels show ρ; colour = Spearman)."""
    if not plot_rhos:
        return
    spearman_vals = [spearman_from_rho(r) for r in plot_rhos]
    norm = plt.Normalize(min(spearman_vals), max(spearman_vals))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks(spearman_vals)
    cbar.set_ticklabels([f"ρ={r:.2f}" for r in plot_rhos])
    cbar.set_label("ρ (colour = Spearman)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _draw_vs_collision(
    ax,
    fig,
    n: int,
    theta: float,
    rho_grid: list[float],
    m_grid: list[int],
    results_dir: str,
    eps: float,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """Draw the collision scatter onto ax (one marker family per m). Returns R.

    Cell means only (not instance-level). Points at ρ = 1 are omitted (degenerate:
    collision = 1 - 1/n). See docs/FINDINGS_aggregate_R2000.md.
    """
    plot_rhos = _rho_grid_for_collision_scatter(rho_grid)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    cmap = plt.cm.viridis
    norm = _collision_scatter_norm(plot_rhos)
    R_title = 0

    for m_index, m in enumerate(m_grid):
        marker = markers[m_index % len(markers)]
        for rho in plot_rhos:
            cell = _load_required_cell(
                results_dir, n, m, rho, theta, mechanism_name, master_seed
            )
            R_title = len(cell["alpha_star"])
            x = float(cell["collision_rate"].mean())
            y_val = efx_rate(cell["alpha_star"], eps=eps)
            colour = cmap(norm(spearman_from_rho(rho))) if norm is not None else "C0"
            ax.scatter(
                x,
                y_val,
                color=colour,
                marker=marker,
                s=40,
                edgecolors="k",
                linewidths=0.3,
                label=f"m={m}" if rho == plot_rhos[0] else None,
            )

    ax.set_xlabel("mean collision rate (cell)")
    ax.set_ylabel("EFX rate (cell)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"EFX rate vs collision (cell means; ρ=1 omitted)\n"
        f"n = {n}, R = {R_title}, {theta_label(theta)}"
    )
    ax.legend(loc="best", fontsize=6, title="m", title_fontsize=7)
    if norm is not None:
        _add_collision_spearman_cbar(fig, ax, cmap, plot_rhos)
    ax.grid(alpha=0.3)
    return R_title


def _draw_q05_vs_rho(
    ax,
    n: int,
    m: int,
    theta: float,
    rho_grid: list[float],
    results_dir: str,
    eps: float,
    min_failures: int,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """Conditional 5% quantile of α* vs ρ at fixed m. Returns R."""
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))
    ys: list[float] = []
    R_title = 0
    for rho in rho_grid:
        cell = _load_required_cell(
            results_dir, n, m, rho, theta, mechanism_name, master_seed
        )
        R_title = len(cell["alpha_star"])
        ys.append(
            conditional_quantile(
                cell["alpha_star"], q=0.05, min_failures=min_failures, eps=eps
            )
        )
    ax.plot(rho_grid, ys, marker="o", color="0.4", linewidth=1.5, zorder=2)
    for colour, rho, y in zip(colours, rho_grid, ys):
        if y == y:  # not NaN
            ax.scatter(
                [rho], [y], color=colour, s=55, zorder=3, edgecolors="k", linewidths=0.3
            )
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"conditional 5% quantile of $\alpha^*$")
    ax.set_xticks(list(rho_grid))
    ax.set_title(
        f"Conditional 5% quantile vs ρ\nn = {n}, m = {m}, R = {R_title}, {theta_label(theta)}"
    )
    ax.grid(alpha=0.3)
    return R_title


def _draw_collision_at_m(
    ax,
    fig,
    n: int,
    m: int,
    theta: float,
    rho_grid: list[float],
    results_dir: str,
    eps: float,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """EFX rate vs mean collision at one fixed m (one point per ρ). Returns R.

    Cell means; ρ = 1 omitted (collision = 1 - 1/n). See docs/FINDINGS_aggregate_R2000.md.
    """
    plot_rhos = _rho_grid_for_collision_scatter(rho_grid)
    cmap = plt.cm.viridis
    norm = _collision_scatter_norm(plot_rhos)
    R_title = 0
    for rho in plot_rhos:
        cell = _load_required_cell(
            results_dir, n, m, rho, theta, mechanism_name, master_seed
        )
        R_title = len(cell["alpha_star"])
        x = float(cell["collision_rate"].mean())
        y_val = efx_rate(cell["alpha_star"], eps=eps)
        colour = cmap(norm(spearman_from_rho(rho))) if norm is not None else "C0"
        ax.scatter(
            x,
            y_val,
            color=colour,
            marker="o",
            s=55,
            edgecolors="k",
            linewidths=0.4,
            label=rho_label(rho),
        )
    ax.set_xlabel("mean collision rate (cell)")
    ax.set_ylabel("EFX rate (cell)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"EFX rate vs collision (cell means; ρ=1 omitted)\n"
        f"n = {n}, m = {m}, R = {R_title}, {theta_label(theta)}"
    )
    ax.legend(loc="best", fontsize=8)
    if norm is not None:
        _add_collision_spearman_cbar(fig, ax, cmap, plot_rhos)
    ax.grid(alpha=0.3)
    return R_title


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
    show: bool = False,
) -> None:
    """Two panels. Left: one curve per rho at theta_fixed. Right: one curve per theta at rho_fixed."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 501)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    R_title = _draw_survival(
        axes,
        n,
        m,
        theta_fixed,
        rho_fixed,
        rho_grid,
        theta_grid,
        results_dir,
        eps,
        alpha_grid,
        master_seed,
        mechanism_name,
    )
    fig.suptitle(
        f"Empirical survival function, Round Robin, n = {n}, m = {m}, R = {R_title}"
    )
    fig.tight_layout()
    _finish_figure(fig, out_path, show)


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
    show: bool = False,
) -> None:
    """x = m (integer axis, not m/n). One line per rho. Mark m where q = m % n == 0."""
    if y not in {"efx_rate", "conditional_q05"}:
        raise ValueError(f"y must be 'efx_rate' or 'conditional_q05', got {y!r}")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _draw_rate_vs_m(
        ax,
        n,
        theta,
        rho_grid,
        m_grid,
        results_dir,
        y,
        eps,
        min_failures,
        master_seed,
        mechanism_name,
    )
    fig.tight_layout()
    _finish_figure(fig, out_path, show)


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
    show: bool = False,
) -> None:
    """Scatter: x = mean collision_rate of the cell, y = EFX rate."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _draw_vs_collision(
        ax,
        fig,
        n,
        theta,
        rho_grid,
        m_grid,
        results_dir,
        eps,
        master_seed,
        mechanism_name,
    )
    fig.tight_layout()
    _finish_figure(fig, out_path, show)


def _cell_path(
    results_dir: str,
    n: int,
    m: int,
    rho: float,
    theta: float,
    mechanism_name: str,
    master_seed: int,
) -> str:
    """Full path to a required cell file, or raise FileNotFoundError."""
    name = cell_filename(n, m, rho, theta, mechanism_name, master_seed)
    path = os.path.join(results_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing cell file for n={n}, m={m}, rho={rho}, theta={theta}: {path}"
        )
    return path


def _short_rho(rho: float) -> str:
    return f"ρ={rho:.2f}"


def _short_theta(theta: float) -> str:
    return f"θ={theta:.2f}"


def _place_table(
    ax,
    col_labels: list[str],
    cell_text: list[list[str]],
    title: str,
    *,
    fontsize: float = 7.5,
) -> None:
    """Draw a compact table that stays inside the axes."""
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=6)
    if not cell_text:
        ax.text(0.5, 0.5, "(no rows)", ha="center", va="center", transform=ax.transAxes)
        return
    n_cols = len(col_labels)
    # Short row-key ("m") vs longer keys (ρ=/θ=); rest shared equally
    first_w = 0.10 if col_labels[0] in {"m", "n"} else 0.20
    col_widths = [first_w] + [((1.0 - first_w) / max(n_cols - 1, 1))] * (n_cols - 1)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.PAD = 0.03
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=fontsize)
            cell.set_facecolor("#e8e8e8")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        if col == 0 and row > 0:
            cell.set_text_props(fontweight="bold")
        if col > 0 and row > 0:
            cell.get_text().set_family("monospace")
            cell.get_text().set_ha("right")


def _page_figure() -> tuple[plt.Figure, object, object]:
    """Same layout on every page: usable plot on top, matrix table below."""
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.28)
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[1, 0])
    return fig, ax_plot, ax_table


def _run_banner(
    n: int,
    m: int,
    R: int,
    mechanism_name: str,
    theta_fixed: float,
    rho_fixed: float,
) -> str:
    return (
        f"{mechanism_name}  |  n={n}, m={m}, R={R}  |  "
        f"{theta_label(theta_fixed)}  |  {rho_label(rho_fixed)}"
    )


def _draw_survival_by_rho(
    ax,
    n: int,
    m: int,
    theta_fixed: float,
    rho_grid: list[float],
    results_dir: str,
    alpha_grid: np.ndarray,
    master_seed: int,
    mechanism_name: str,
) -> int:
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))
    R_title = 0
    for colour, rho in zip(colours, rho_grid):
        cell = _load_required_cell(
            results_dir, n, m, rho, theta_fixed, mechanism_name, master_seed
        )
        stars = cell["alpha_star"]
        R_title = len(stars)
        ax.plot(
            alpha_grid,
            survival(stars, alpha_grid),
            color=colour,
            label=rho_label(rho),
        )
    ax.set_title(f"Survival vs α  |  {theta_label(theta_fixed)}")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"share with $\alpha^* \geq \alpha$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return R_title


def _draw_survival_by_theta(
    ax,
    n: int,
    m: int,
    rho_fixed: float,
    theta_grid: list[float],
    results_dir: str,
    alpha_grid: np.ndarray,
    master_seed: int,
    mechanism_name: str,
) -> int:
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(theta_grid)))
    R_title = 0
    for colour, theta in zip(colours, theta_grid):
        cell = _load_required_cell(
            results_dir, n, m, rho_fixed, theta, mechanism_name, master_seed
        )
        stars = cell["alpha_star"]
        R_title = len(stars)
        ax.plot(
            alpha_grid,
            survival(stars, alpha_grid),
            color=colour,
            label=theta_label(theta),
        )
    ax.set_title(f"Survival vs α  |  {rho_label(rho_fixed)}")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"share with $\alpha^* \geq \alpha$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return R_title


def _survival_table_rows(
    *,
    by: str,
    values: list[float],
    n: int,
    m: int,
    fixed: float,
    results_dir: str,
    alpha_ticks: np.ndarray,
    master_seed: int,
    mechanism_name: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for v in values:
        if by == "rho":
            cell = _load_required_cell(
                results_dir, n, m, v, fixed, mechanism_name, master_seed
            )
            label = _short_rho(v)
        else:
            cell = _load_required_cell(
                results_dir, n, m, fixed, v, mechanism_name, master_seed
            )
            label = _short_theta(v)
        s_hat = survival(cell["alpha_star"], alpha_ticks)
        rows.append([label] + [format_pct(float(x), digits=1) for x in s_hat])
    return rows


def plot_run_overview(
    n: int,
    m: int,
    theta_fixed: float,
    rho_fixed: float,
    rho_grid: list[float],
    theta_grid: list[float],
    results_dir: str,
    out_path: str,
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
    show: bool = False,
) -> None:
    """Four PDF pages for fixed (n, m): survival ρ/θ, q05 vs ρ, collision at this m."""
    alpha_grid = np.linspace(0.0, 1.0, 501)
    alpha_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    alpha_headers = ["curve"] + [f"α={a:.1f}" for a in alpha_ticks]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    figures: list[plt.Figure] = []

    # --- Page 1: survival by rho ---
    fig1, ax1, ax1t = _page_figure()
    R_title = _draw_survival_by_rho(
        ax1,
        n,
        m,
        theta_fixed,
        rho_grid,
        results_dir,
        alpha_grid,
        master_seed,
        mechanism_name,
    )
    fig1.suptitle(_run_banner(n, m, R_title, mechanism_name, theta_fixed, rho_fixed), fontsize=10)
    _place_table(
        ax1t,
        alpha_headers,
        _survival_table_rows(
            by="rho",
            values=rho_grid,
            n=n,
            m=m,
            fixed=theta_fixed,
            results_dir=results_dir,
            alpha_ticks=alpha_ticks,
            master_seed=master_seed,
            mechanism_name=mechanism_name,
        ),
        f"Survival Ŝ(α) by ρ  at  {theta_label(theta_fixed)}  (Spearman in legend)",
    )
    figures.append(fig1)

    # --- Page 2: survival by theta ---
    fig2, ax2, ax2t = _page_figure()
    _draw_survival_by_theta(
        ax2,
        n,
        m,
        rho_fixed,
        theta_grid,
        results_dir,
        alpha_grid,
        master_seed,
        mechanism_name,
    )
    fig2.suptitle(_run_banner(n, m, R_title, mechanism_name, theta_fixed, rho_fixed), fontsize=10)
    _place_table(
        ax2t,
        alpha_headers,
        _survival_table_rows(
            by="theta",
            values=theta_grid,
            n=n,
            m=m,
            fixed=rho_fixed,
            results_dir=results_dir,
            alpha_ticks=alpha_ticks,
            master_seed=master_seed,
            mechanism_name=mechanism_name,
        ),
        f"Survival Ŝ(α) by θ  at  {rho_label(rho_fixed)}  (Gini in legend)",
    )
    figures.append(fig2)

    # --- Page 3: conditional q05 vs ρ at this m ---
    fig3, ax3, ax3t = _page_figure()
    _draw_q05_vs_rho(
        ax3,
        n,
        m,
        theta_fixed,
        rho_grid,
        results_dir,
        eps,
        min_failures,
        master_seed,
        mechanism_name,
    )
    fig3.suptitle(_run_banner(n, m, R_title, mechanism_name, theta_fixed, rho_fixed), fontsize=10)
    q05_rows: list[list[str]] = []
    for rho in rho_grid:
        path = _cell_path(
            results_dir, n, m, rho, theta_fixed, mechanism_name, master_seed
        )
        s = summarise_cell(path, eps=eps, min_failures=min_failures)
        q05_rows.append(
            [_short_rho(rho), format_float(s["conditional_q05"], digits=3)]
        )
    _place_table(
        ax3t,
        ["ρ", "q05(α*)"],
        q05_rows,
        f"Conditional 5% quantile of α*  |  m={m}, {theta_label(theta_fixed)}  "
        f"(— if fewer than {min_failures} failures)",
        fontsize=8.5,
    )
    figures.append(fig3)

    # --- Page 4: collision at this m only ---
    fig4, ax4, ax4t = _page_figure()
    _draw_collision_at_m(
        ax4,
        fig4,
        n,
        m,
        theta_fixed,
        rho_grid,
        results_dir,
        eps,
        master_seed,
        mechanism_name,
    )
    fig4.suptitle(_run_banner(n, m, R_title, mechanism_name, theta_fixed, rho_fixed), fontsize=10)
    coll_rows: list[list[str]] = []
    for rho in rho_grid:
        path = _cell_path(
            results_dir, n, m, rho, theta_fixed, mechanism_name, master_seed
        )
        s = summarise_cell(path, eps=eps, min_failures=min_failures)
        coll_rows.append(
            [
                _short_rho(rho),
                format_float(s["collision_rate_mean"], digits=3),
                format_pct(s["efx_rate"], digits=1),
            ]
        )
    _place_table(
        ax4t,
        ["ρ", "mean collision", "EFX rate"],
        coll_rows,
        f"Collision vs EFX  |  m={m}, {theta_label(theta_fixed)}",
        fontsize=8.5,
    )
    figures.append(fig4)

    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _mxrho_matrix_rows(
    m_grid: list[int],
    rho_grid: list[float],
    results_dir: str,
    n: int,
    theta: float,
    mechanism_name: str,
    master_seed: int,
    *,
    field: str,
    eps: float,
    min_failures: int,
) -> list[list[str]]:
    """Rows for an m × ρ table; field in efx_rate | conditional_q05 | collision_rate_mean."""
    rows: list[list[str]] = []
    for mm in m_grid:
        row = [str(mm)]
        for rho in rho_grid:
            path = _cell_path(
                results_dir, n, mm, rho, theta, mechanism_name, master_seed
            )
            s = summarise_cell(path, eps=eps, min_failures=min_failures)
            if field == "efx_rate":
                row.append(format_pct(s["efx_rate"], digits=1))
            elif field == "conditional_q05":
                row.append(format_float(s["conditional_q05"], digits=3))
            else:
                row.append(format_float(s["collision_rate_mean"], digits=3))
        rows.append(row)
    return rows


def _save_figures_pdf(figures: list[plt.Figure], out_path: str) -> None:
    """Write figures to a multipage PDF and close them."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _m_sweep_figures(
    n: int,
    theta_fixed: float,
    rho_grid: list[float],
    m_grid: list[int],
    results_dir: str,
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> list[plt.Figure]:
    """Three pages for fixed n, m-sweep: EFX vs m, q05 vs m, collision scatter."""
    figures: list[plt.Figure] = []
    rho_cols = ["m"] + [_short_rho(rho) for rho in rho_grid]
    mat_fs = 8.0 if len(m_grid) <= 14 else 7.0

    fig1, ax1, ax1t = _page_figure()
    R_title = _draw_rate_vs_m(
        ax1,
        n,
        theta_fixed,
        rho_grid,
        m_grid,
        results_dir,
        "efx_rate",
        eps,
        min_failures,
        master_seed,
        mechanism_name,
    )
    fig1.suptitle(
        f"{mechanism_name}  |  m-sweep  |  n={n}, R={R_title}  |  {theta_label(theta_fixed)}",
        fontsize=10,
    )
    _place_table(
        ax1t,
        rho_cols,
        _mxrho_matrix_rows(
            m_grid,
            rho_grid,
            results_dir,
            n,
            theta_fixed,
            mechanism_name,
            master_seed,
            field="efx_rate",
            eps=eps,
            min_failures=min_failures,
        ),
        f"EFX rate (%) by m × ρ  |  {theta_label(theta_fixed)}",
        fontsize=mat_fs,
    )
    figures.append(fig1)

    fig2, ax2, ax2t = _page_figure()
    _draw_rate_vs_m(
        ax2,
        n,
        theta_fixed,
        rho_grid,
        m_grid,
        results_dir,
        "conditional_q05",
        eps,
        min_failures,
        master_seed,
        mechanism_name,
    )
    fig2.suptitle(
        f"{mechanism_name}  |  m-sweep  |  n={n}, R={R_title}  |  {theta_label(theta_fixed)}",
        fontsize=10,
    )
    _place_table(
        ax2t,
        rho_cols,
        _mxrho_matrix_rows(
            m_grid,
            rho_grid,
            results_dir,
            n,
            theta_fixed,
            mechanism_name,
            master_seed,
            field="conditional_q05",
            eps=eps,
            min_failures=min_failures,
        ),
        f"Conditional q05(α*) by m × ρ  |  {theta_label(theta_fixed)}  "
        f"(— if fewer than {min_failures} failures)",
        fontsize=mat_fs,
    )
    figures.append(fig2)

    fig3, ax3, ax3t = _page_figure()
    _draw_vs_collision(
        ax3,
        fig3,
        n,
        theta_fixed,
        rho_grid,
        m_grid,
        results_dir,
        eps,
        master_seed,
        mechanism_name,
    )
    fig3.suptitle(
        f"{mechanism_name}  |  m-sweep  |  n={n}, R={R_title}  |  {theta_label(theta_fixed)}",
        fontsize=10,
    )
    _place_table(
        ax3t,
        rho_cols,
        _mxrho_matrix_rows(
            m_grid,
            rho_grid,
            results_dir,
            n,
            theta_fixed,
            mechanism_name,
            master_seed,
            field="collision_rate_mean",
            eps=eps,
            min_failures=min_failures,
        ),
        f"Mean collision rate by m × ρ  |  {theta_label(theta_fixed)}  "
        f"(cell means; H1/H4 lens — see docs/FINDINGS_aggregate_R2000.md)",
        fontsize=mat_fs,
    )
    figures.append(fig3)
    return figures


def plot_m_sweep_overview(
    n: int,
    theta_fixed: float,
    rho_grid: list[float],
    m_grid: list[int],
    results_dir: str,
    out_path: str,
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> None:
    """Three PDF pages for fixed n, m-sweep: EFX vs m, q05 vs m, collision scatter."""
    figures = _m_sweep_figures(
        n,
        theta_fixed,
        rho_grid,
        m_grid,
        results_dir,
        eps=eps,
        min_failures=min_failures,
        master_seed=master_seed,
        mechanism_name=mechanism_name,
    )
    _save_figures_pdf(figures, out_path)


def _draw_rate_vs_n(
    ax,
    n_grid: list[int],
    m_of_n: dict[int, int],
    theta: float,
    rho_grid: list[float],
    results_dir: str,
    y: str,
    eps: float,
    min_failures: int,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """EFX rate or q05 vs n at fixed m/n design (m = m_of_n[n]). Returns R."""
    if y not in {"efx_rate", "conditional_q05"}:
        raise ValueError(f"y must be 'efx_rate' or 'conditional_q05', got {y!r}")
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_grid)))
    R_title = 0
    for colour, rho in zip(colours, rho_grid):
        ys = []
        for n in n_grid:
            m = m_of_n[n]
            cell = _load_required_cell(
                results_dir, n, m, rho, theta, mechanism_name, master_seed
            )
            stars = cell["alpha_star"]
            R_title = len(stars)
            if y == "efx_rate":
                ys.append(efx_rate(stars, eps=eps))
            else:
                ys.append(
                    conditional_quantile(
                        stars, q=0.05, min_failures=min_failures, eps=eps
                    )
                )
        ax.plot(n_grid, ys, marker="o", color=colour, label=rho_label(rho))

    ax.set_xlabel("n")
    ax.set_xticks(list(n_grid))
    if y == "efx_rate":
        ax.set_ylabel("EFX rate")
        ax.set_ylim(0.0, 1.02)
        kind = "EFX rate vs n"
    else:
        ax.set_ylabel(r"conditional 5% quantile of $\alpha^*$")
        kind = "conditional 5% quantile vs n"
    ratios = sorted({m_of_n[n] / n for n in n_grid})
    ratio_txt = ", ".join(f"{r:.2f}" for r in ratios)
    ax.set_title(
        f"{kind} (m/n ≈ {ratio_txt})\nR = {R_title}, {theta_label(theta)}"
    )
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)
    return R_title


def _draw_collision_vs_n(
    ax,
    fig,
    n_grid: list[int],
    m_of_n: dict[int, int],
    theta: float,
    rho_grid: list[float],
    results_dir: str,
    eps: float,
    master_seed: int,
    mechanism_name: str,
) -> int:
    """EFX vs collision scatter; marker = n, colour = Spearman(ρ). Returns R.

    Cell means; ρ = 1 omitted (collision = 1 - 1/n). See docs/FINDINGS_aggregate_R2000.md.
    """
    plot_rhos = _rho_grid_for_collision_scatter(rho_grid)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    cmap = plt.cm.viridis
    norm = _collision_scatter_norm(plot_rhos)
    R_title = 0
    for n_index, n in enumerate(n_grid):
        marker = markers[n_index % len(markers)]
        m = m_of_n[n]
        for rho in plot_rhos:
            cell = _load_required_cell(
                results_dir, n, m, rho, theta, mechanism_name, master_seed
            )
            R_title = len(cell["alpha_star"])
            x = float(cell["collision_rate"].mean())
            y_val = efx_rate(cell["alpha_star"], eps=eps)
            colour = cmap(norm(spearman_from_rho(rho))) if norm is not None else "C0"
            ax.scatter(
                x,
                y_val,
                color=colour,
                marker=marker,
                s=45,
                edgecolors="k",
                linewidths=0.3,
                label=f"n={n} (m={m})" if rho == plot_rhos[0] else None,
            )
    ax.set_xlabel("mean collision rate (cell)")
    ax.set_ylabel("EFX rate (cell)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"EFX rate vs collision (cell means; ρ=1 omitted)\n"
        f"R = {R_title}, {theta_label(theta)}"
    )
    ax.legend(loc="best", fontsize=7)
    if norm is not None:
        _add_collision_spearman_cbar(fig, ax, cmap, plot_rhos)
    ax.grid(alpha=0.3)
    return R_title


def _n_sweep_figures(
    n_grid: list[int],
    m_over_n: float,
    theta_fixed: float,
    rho_grid: list[float],
    results_dir: str,
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> list[plt.Figure]:
    """Three pages: EFX/q05 vs n and collision at fixed m/n."""
    m_of_n = {n: int(round(m_over_n * n)) for n in n_grid}
    for n, m in m_of_n.items():
        if m <= n:
            raise ValueError(f"m={m} must be > n={n} for m_over_n={m_over_n}")

    figures: list[plt.Figure] = []
    rho_cols = ["n", "m"] + [_short_rho(rho) for rho in rho_grid]

    def _n_rows(field: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for n in n_grid:
            m = m_of_n[n]
            row = [str(n), str(m)]
            for rho in rho_grid:
                path = _cell_path(
                    results_dir, n, m, rho, theta_fixed, mechanism_name, master_seed
                )
                s = summarise_cell(path, eps=eps, min_failures=min_failures)
                if field == "efx_rate":
                    row.append(format_pct(s["efx_rate"], digits=1))
                elif field == "conditional_q05":
                    row.append(format_float(s["conditional_q05"], digits=3))
                else:
                    row.append(format_float(s["collision_rate_mean"], digits=3))
            rows.append(row)
        return rows

    fig1, ax1, ax1t = _page_figure()
    R_title = _draw_rate_vs_n(
        ax1,
        n_grid,
        m_of_n,
        theta_fixed,
        rho_grid,
        results_dir,
        "efx_rate",
        eps,
        min_failures,
        master_seed,
        mechanism_name,
    )
    fig1.suptitle(
        f"{mechanism_name}  |  n-sweep (m/n={m_over_n:g})  |  R={R_title}  |  "
        f"{theta_label(theta_fixed)}",
        fontsize=10,
    )
    _place_table(
        ax1t,
        rho_cols,
        _n_rows("efx_rate"),
        f"EFX rate (%) by n × ρ  |  m/n={m_over_n:g}, {theta_label(theta_fixed)}",
        fontsize=8.0,
    )
    figures.append(fig1)

    fig2, ax2, ax2t = _page_figure()
    _draw_rate_vs_n(
        ax2,
        n_grid,
        m_of_n,
        theta_fixed,
        rho_grid,
        results_dir,
        "conditional_q05",
        eps,
        min_failures,
        master_seed,
        mechanism_name,
    )
    fig2.suptitle(
        f"{mechanism_name}  |  n-sweep (m/n={m_over_n:g})  |  R={R_title}  |  "
        f"{theta_label(theta_fixed)}",
        fontsize=10,
    )
    _place_table(
        ax2t,
        rho_cols,
        _n_rows("conditional_q05"),
        f"Conditional q05(α*) by n × ρ  |  m/n={m_over_n:g}  "
        f"(— if fewer than {min_failures} failures)",
        fontsize=8.0,
    )
    figures.append(fig2)

    fig3, ax3, ax3t = _page_figure()
    _draw_collision_vs_n(
        ax3,
        fig3,
        n_grid,
        m_of_n,
        theta_fixed,
        rho_grid,
        results_dir,
        eps,
        master_seed,
        mechanism_name,
    )
    fig3.suptitle(
        f"{mechanism_name}  |  n-sweep (m/n={m_over_n:g})  |  R={R_title}  |  "
        f"{theta_label(theta_fixed)}",
        fontsize=10,
    )
    _place_table(
        ax3t,
        rho_cols,
        _n_rows("collision_rate_mean"),
        f"Mean collision by n × ρ  |  m/n={m_over_n:g}  "
        f"(cell means; H2/H4 — see docs/FINDINGS_aggregate_R2000.md)",
        fontsize=8.0,
    )
    figures.append(fig3)
    return figures


def plot_n_sweep_overview(
    n_grid: list[int],
    m_over_n: float,
    theta_fixed: float,
    rho_grid: list[float],
    results_dir: str,
    out_path: str,
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> None:
    """Three PDF pages: EFX/q05 vs n and collision at fixed m/n (m = round(m_over_n * n))."""
    figures = _n_sweep_figures(
        n_grid,
        m_over_n,
        theta_fixed,
        rho_grid,
        results_dir,
        eps=eps,
        min_failures=min_failures,
        master_seed=master_seed,
        mechanism_name=mechanism_name,
    )
    _save_figures_pdf(figures, out_path)


def plot_aggregate_overview(
    n_grid: list[int],
    theta_fixed: float,
    rho_grid: list[float],
    results_dir: str,
    out_path: str,
    m_over_n: float = 2.0,
    eps: float = 0.0,
    min_failures: int = 200,
    master_seed: int = 12345,
    mechanism_name: str = "round_robin",
) -> None:
    """One PDF: for each n, 3 m-sweep pages, then 3 n-sweep pages at fixed m/n.

    Expects cells for all (n, m) with m in n+1..4n and all rho at theta_fixed,
    plus the n-sweep pairs m = round(m_over_n * n) (already included when m/n=2).
    """
    figures: list[plt.Figure] = []
    for n in n_grid:
        m_grid = list(range(n + 1, 4 * n + 1))
        figures.extend(
            _m_sweep_figures(
                n,
                theta_fixed,
                rho_grid,
                m_grid,
                results_dir,
                eps=eps,
                min_failures=min_failures,
                master_seed=master_seed,
                mechanism_name=mechanism_name,
            )
        )
    figures.extend(
        _n_sweep_figures(
            n_grid,
            m_over_n,
            theta_fixed,
            rho_grid,
            results_dir,
            eps=eps,
            min_failures=min_failures,
            master_seed=master_seed,
            mechanism_name=mechanism_name,
        )
    )
    _save_figures_pdf(figures, out_path)


def _cmp_rho_colours(rhos: list[float]) -> dict[float, tuple]:
    """Map each ρ to a viridis colour via Spearman (same idea as collision scatters)."""
    if not rhos:
        return {}
    spears = [spearman_from_rho(r) for r in rhos]
    norm = plt.Normalize(min(spears), max(spears))
    cmap = plt.cm.viridis
    return {r: cmap(norm(spearman_from_rho(r))) for r in rhos}


def _cmp_filter_theta(df, theta_fixed: float):
    return df[np.isclose(df["theta"].astype(float), float(theta_fixed))]


def _cmp_n_panels(fig: plt.Figure, n_grid: list[int]) -> dict[int, object]:
    """2×2 axes keyed by n (assumes len(n_grid) <= 4)."""
    axes = fig.subplots(2, 2, sharex=False, sharey=True)
    flat = list(np.ravel(axes))
    mapping: dict[int, object] = {}
    for i, n in enumerate(n_grid):
        mapping[n] = flat[i]
    for j in range(len(n_grid), len(flat)):
        flat[j].set_axis_off()
    return mapping


def plot_comparison_overview(
    csv_path: str,
    out_path: str,
    theta_fixed: float = 1.0,
    m_over_n: float = 2.0,
) -> None:
    """Multipage PDF from a Stage-10 comparison CSV (no re-sim).

    Pages: EFX overlay, Δ EFX, pr_a_only, pr_b_only (m-sweep at theta_fixed);
    n-sweep and θ-sweep at m/n; ρ=1 anchor; paired 2×2 tables at m/n for each n.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty comparison CSV: {csv_path}")

    name_a = str(df["mechanism_a"].iloc[0])
    name_b = str(df["mechanism_b"].iloc[0])
    R = int(df["R"].iloc[0])
    n_grid = sorted(int(x) for x in df["n"].unique())
    rho_grid = sorted(float(x) for x in df["rho"].unique())
    colours = _cmp_rho_colours(rho_grid)
    pair_label = f"{name_a} vs {name_b}"
    figures: list[plt.Figure] = []

    slice_th = _cmp_filter_theta(df, theta_fixed)

    # --- Page 1: EFX overlay ---
    fig = plt.figure(figsize=(11, 8))
    axes_map = _cmp_n_panels(fig, n_grid)
    for n, ax in axes_map.items():
        sub = slice_th[slice_th["n"] == n]
        for rho in rho_grid:
            g = sub[np.isclose(sub["rho"].astype(float), rho)].sort_values("m")
            if g.empty:
                continue
            c = colours[rho]
            ax.plot(
                g["m"],
                g["efx_rate_a"],
                color=c,
                linestyle="-",
                linewidth=1.5,
                marker="o",
                markersize=3,
                label=_short_rho(rho),
            )
            ax.plot(
                g["m"],
                g["efx_rate_b"],
                color=c,
                linestyle="--",
                linewidth=1.5,
                marker="x",
                markersize=3,
            )
        ax.set_title(f"n = {n}", fontsize=9)
        ax.set_xlabel("m")
        ax.set_ylabel("EFX rate")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=5)
    fig.suptitle(
        f"{pair_label}  |  EFX overlay vs m  |  R={R}  |  {theta_label(theta_fixed)}\n"
        f"solid = {name_a}, dashed = {name_b}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figures.append(fig)

    # --- Pages 2–4: single series vs m ---
    for field, title, ylabel in (
        ("efx_rate_diff", "EFX rate diff (A − B) vs m", "efx_rate_a − efx_rate_b"),
        ("pr_a_only", "pr_a_only vs m (A EFX, B not)", "pr_a_only"),
        ("pr_b_only", "pr_b_only vs m (B EFX, A not)", "pr_b_only"),
    ):
        fig = plt.figure(figsize=(11, 8))
        axes_map = _cmp_n_panels(fig, n_grid)
        y_min, y_max = ( -1.02, 1.02) if field == "efx_rate_diff" else (0.0, 1.02)
        for n, ax in axes_map.items():
            sub = slice_th[slice_th["n"] == n]
            for rho in rho_grid:
                g = sub[np.isclose(sub["rho"].astype(float), rho)].sort_values("m")
                if g.empty:
                    continue
                ax.plot(
                    g["m"],
                    g[field],
                    color=colours[rho],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=_short_rho(rho),
                )
            ax.set_title(f"n = {n}", fontsize=9)
            ax.set_xlabel("m")
            ax.set_ylabel(ylabel)
            ax.set_ylim(y_min, y_max)
            if field == "efx_rate_diff":
                ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=6)
        fig.suptitle(
            f"{pair_label}  |  {title}  |  R={R}  |  {theta_label(theta_fixed)}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        figures.append(fig)

    # --- Page 5: n-sweep at m/n ---
    fig = plt.figure(figsize=(11, 5))
    ax_d, ax_p = fig.subplots(1, 2)
    for rho in rho_grid:
        xs, ys_d, ys_p = [], [], []
        for n in n_grid:
            m = int(round(m_over_n * n))
            row = slice_th[
                (slice_th["n"] == n)
                & (slice_th["m"] == m)
                & np.isclose(slice_th["rho"].astype(float), rho)
            ]
            if row.empty:
                continue
            xs.append(n)
            ys_d.append(float(row["efx_rate_diff"].iloc[0]))
            ys_p.append(float(row["pr_a_only"].iloc[0]))
        if not xs:
            continue
        c = colours[rho]
        ax_d.plot(xs, ys_d, color=c, marker="o", label=_short_rho(rho))
        ax_p.plot(xs, ys_p, color=c, marker="o", label=_short_rho(rho))
    ax_d.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
    ax_d.set_xlabel("n")
    ax_d.set_ylabel("efx_rate_diff")
    ax_d.set_title(f"Δ EFX vs n  |  m/n={m_over_n:g}")
    ax_d.set_xticks(n_grid)
    ax_d.set_ylim(-1.02, 1.02)
    ax_d.grid(alpha=0.3)
    ax_d.legend(fontsize=7)
    ax_p.set_xlabel("n")
    ax_p.set_ylabel("pr_a_only")
    ax_p.set_title(f"pr_a_only vs n  |  m/n={m_over_n:g}")
    ax_p.set_xticks(n_grid)
    ax_p.set_ylim(0.0, 1.02)
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=7)
    fig.suptitle(
        f"{pair_label}  |  n-sweep  |  R={R}  |  {theta_label(theta_fixed)}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    figures.append(fig)

    # --- Page 6: θ-sweep at m/n ---
    fig = plt.figure(figsize=(11, 8))
    axes_map = _cmp_n_panels(fig, n_grid)
    for n, ax in axes_map.items():
        m = int(round(m_over_n * n))
        sub = df[(df["n"] == n) & (df["m"] == m)]
        for rho in rho_grid:
            g = sub[np.isclose(sub["rho"].astype(float), rho)].sort_values("theta")
            if g.empty:
                continue
            ax.plot(
                g["theta"],
                g["efx_rate_diff"],
                color=colours[rho],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=_short_rho(rho),
            )
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
        ax.set_title(f"n = {n}, m = {m}", fontsize=9)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("efx_rate_diff")
        ax.set_ylim(-1.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=6)
    fig.suptitle(
        f"{pair_label}  |  θ-sweep of Δ EFX  |  m/n={m_over_n:g}  |  R={R}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figures.append(fig)

    # --- Page 7: ρ=1 anchor ---
    fig = plt.figure(figsize=(11, 8))
    axes_map = _cmp_n_panels(fig, n_grid)
    slice_r1 = slice_th[np.isclose(slice_th["rho"].astype(float), 1.0)]
    for n, ax in axes_map.items():
        g = slice_r1[slice_r1["n"] == n].sort_values("m")
        if g.empty:
            continue
        ax.plot(
            g["m"],
            g["efx_rate_a"],
            color="C0",
            marker="o",
            linestyle="-",
            label=name_a,
        )
        ax.plot(
            g["m"],
            g["efx_rate_b"],
            color="C1",
            marker="x",
            linestyle="--",
            label=name_b,
        )
        ax.set_title(f"n = {n}", fontsize=9)
        ax.set_xlabel("m")
        ax.set_ylabel("EFX rate")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        f"{pair_label}  |  ρ = 1 anchor (S4: EFX exists; mechanism gap)  |  "
        f"R={R}  |  {theta_label(theta_fixed)}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figures.append(fig)

    # --- Page 8: paired tables at m/n for each n ---
    fig = plt.figure(figsize=(11, 9))
    axes_map = _cmp_n_panels(fig, n_grid)
    col_labels = [
        "ρ",
        "EFX A",
        "EFX B",
        "Δ",
        "both",
        "a_only",
        "b_only",
        "neither",
        "cert",
    ]
    for n, ax in axes_map.items():
        m = int(round(m_over_n * n))
        tab = slice_th[
            (slice_th["n"] == n) & (slice_th["m"] == m)
        ].sort_values("rho")
        rows: list[list[str]] = []
        for _, r in tab.iterrows():
            cert = r["certificate"]
            cert_s = "—" if (cert != cert or cert is None or str(cert) == "nan") else str(cert)
            rows.append(
                [
                    _short_rho(float(r["rho"])),
                    format_pct(float(r["efx_rate_a"])),
                    format_pct(float(r["efx_rate_b"])),
                    f"{float(r['efx_rate_diff']):+.3f}",
                    format_pct(float(r["pr_both"])),
                    format_pct(float(r["pr_a_only"])),
                    format_pct(float(r["pr_b_only"])),
                    format_pct(float(r["pr_neither"])),
                    cert_s,
                ]
            )
        _place_table(
            ax,
            col_labels,
            rows,
            f"n={n}, m={m} (m/n={m_over_n:g})",
            fontsize=6.5,
        )
    fig.suptitle(
        f"{pair_label}  |  paired 2×2 tables at m/n={m_over_n:g}  |  "
        f"R={R}  |  {theta_label(theta_fixed)}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig)

    _save_figures_pdf(figures, out_path)
