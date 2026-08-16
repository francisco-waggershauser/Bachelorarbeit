"""Stage 4: summarise stored cells. Fairness eps lives here, not in the simulation.

At R = 2000 the EFX-rate standard error is at most about 0.011, so the rate
converges quickly. Conditional statistics are the constraint: with
min_failures = 200, a cell goes grey once the EFX rate exceeds
1 - min_failures/R (0.90 at R = 2000, 0.99 at R = 20000). That greying in
high-rate cells is expected, not a bug.
"""

from __future__ import annotations

import numpy as np

from mechanisms import MECHANISMS
from scales import gini_label, spearman_label
from simulate import load_cell, reconstruct


def survival(alpha_star: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """S_hat(alpha) = (1/R) * #{k : alpha_star_k >= alpha}. Empirical, hence the hat."""
    return (alpha_star[None, :] >= grid[:, None]).mean(axis=1)


def is_failure(alpha_star: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """True where alpha_star < 1 - eps. The one failure definition; used by rate and conditionals."""
    return alpha_star < (1.0 - eps)


def efx_rate(alpha_star: np.ndarray, eps: float = 0.0) -> float:
    """1 - is_failure(alpha_star, eps).mean(). Same threshold as the conditionals."""
    return float(1.0 - is_failure(alpha_star, eps).mean())


def conditional_quantile(
    alpha_star: np.ndarray,
    q: float = 0.05,
    min_failures: int = 200,
    eps: float = 0.0,
) -> float:
    """q-quantile over failures (alpha_star < 1 - eps). NaN if fewer than min_failures failures."""
    failures = alpha_star[is_failure(alpha_star, eps)]
    if failures.size < min_failures:
        return float("nan")
    return float(np.quantile(failures, q))


def conditional_mean(
    alpha_star: np.ndarray,
    min_failures: int = 200,
    eps: float = 0.0,
) -> float:
    """E[alpha_star | failure]. NaN if fewer than min_failures failures."""
    failures = alpha_star[is_failure(alpha_star, eps)]
    if failures.size < min_failures:
        return float("nan")
    return float(failures.mean())


def raw_mean_diagnostic(alpha_star: np.ndarray) -> float:
    """E[alpha_star]. Diagnostic only; do not put this in a results table.

    Warning: E[alpha_star] = S(1) + (1 - S(1)) * E[alpha_star | failure], so the
    raw mean mixes the height of the point mass at 1 with the severity of the
    left tail. "80% perfect, 20% catastrophic" and "100% narrowly off" look alike.
    """
    return float(alpha_star.mean())


def summarise_cell(
    path: str, eps: float = 0.0, min_failures: int = 200
) -> dict:
    """Numbers 1-3 plus raw_mean_diagnostic, R, n_failures, mean/std collision_rate.

    In-memory dict. No CSV.
    """
    cell = load_cell(path)
    stars = cell["alpha_star"]
    failures = is_failure(stars, eps)
    return {
        "n": cell["n"],
        "m": cell["m"],
        "rho": cell["rho"],
        "theta": cell["theta"],
        "mechanism": cell["mechanism"],
        "master_seed": cell["master_seed"],
        "R": int(len(stars)),
        "efx_rate": efx_rate(stars, eps=eps),
        "conditional_q05": conditional_quantile(
            stars, q=0.05, min_failures=min_failures, eps=eps
        ),
        "conditional_mean": conditional_mean(
            stars, min_failures=min_failures, eps=eps
        ),
        "raw_mean_diagnostic": raw_mean_diagnostic(stars),
        "n_failures": int(failures.sum()),
        "collision_rate_mean": float(cell["collision_rate"].mean()),
        "collision_rate_std": float(cell["collision_rate"].std(ddof=1))
        if len(stars) > 1
        else float("nan"),
        "bundle_sizes": cell["bundle_sizes"][0],
        "skipped_fraction": float(cell["skipped_fraction"][0]),
    }


def describe_cell(path: str, eps: float = 0.0, min_failures: int = 200) -> None:
    """Print one compact block for one stored cell."""
    s = summarise_cell(path, eps=eps, min_failures=min_failures)
    sizes = np.asarray(s["bundle_sizes"])
    print(
        f"n={s['n']}, m={s['m']}, rho={s['rho']}, theta={s['theta']}, "
        f"{gini_label(s['theta'])}, {spearman_label(s['rho'])}, "
        f"R={s['R']}, mechanism={s['mechanism']}, master_seed={s['master_seed']}"
    )
    print(f"  bundle_sizes={sizes.tolist()}, skipped_fraction={s['skipped_fraction']:.4f}")
    print(
        f"  EFX rate={s['efx_rate']:.4f}, "
        f"conditional 5% quantile={s['conditional_q05']:.4f}, "
        f"conditional mean={s['conditional_mean']:.4f}, "
        f"raw_mean_diagnostic={s['raw_mean_diagnostic']:.4f}"
    )
    print(f"  n_failures={s['n_failures']}")
    print(
        f"  collision_rate mean={s['collision_rate_mean']:.4f}, "
        f"std={s['collision_rate_std']:.4f}"
    )


def load_instance(path: str, k: int) -> dict:
    """Stored fields for instance k, plus reconstructed V, Z, A via reconstruct()."""
    cell = load_cell(path)
    mechanism = MECHANISMS[cell["mechanism"]]
    rebuilt = reconstruct(
        cell["master_seed"],
        k,
        cell["n"],
        cell["m"],
        cell["rho"],
        cell["theta"],
        mechanism=mechanism,
    )
    return {
        "master_seed": cell["master_seed"],
        "instance_index": k,
        "n": cell["n"],
        "m": cell["m"],
        "rho": cell["rho"],
        "theta": cell["theta"],
        "mechanism": cell["mechanism"],
        "alpha_matrix": cell["alpha_matrix"][k],
        "alpha_star": float(cell["alpha_star"][k]),
        "bundle_sizes": cell["bundle_sizes"][k],
        "skipped_fraction": float(cell["skipped_fraction"][k]),
        "collision_rate": float(cell["collision_rate"][k]),
        "V": rebuilt["V"],
        "Z": rebuilt["Z"],
        "A": rebuilt["A"],
    }
