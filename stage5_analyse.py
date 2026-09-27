"""Stage 5: summarise stored cells. Fairness eps lives here, not in the simulation.

At R = 2000 the EFX-rate standard error is at most about 0.011, so the rate
converges quickly. Conditional statistics are the constraint: with
min_failures = 200, a cell goes grey once the EFX rate exceeds
1 - min_failures/R (0.90 at R = 2000, 0.99 at R = 20000). That greying in
high-rate cells is expected, not a bug.
"""

from __future__ import annotations

import numpy as np

from stage0_theory import cell_certificate, is_ordered_instance
from stage2_mechanisms import MECHANISMS, full_assignment_count, no_empty_bundle_count
from stage6_scales import gini_from_theta, rho_label, spearman_from_rho, theta_label
from stage4_simulate import draw_instance, load_cell, reconstruct


def survival(alpha_star: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """S_hat(alpha) = (1/R) * #{k : alpha_star_k >= alpha}. Empirical, hence the hat."""
    return (alpha_star[None, :] >= grid[:, None]).mean(axis=1)


def is_failure(alpha_star: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """True where alpha_star < 1 - eps. The one failure definition; used by rate and conditionals."""
    return alpha_star < (1.0 - eps)


def efx_rate(alpha_star: np.ndarray, eps: float = 0.0) -> float:
    """1 - is_failure(alpha_star, eps).mean(). Same threshold as the conditionals."""
    return float(1.0 - is_failure(alpha_star, eps).mean())


def format_pct(x: float, digits: int = 2) -> str:
    """Format a rate in [0, 1] as a percentage string, e.g. 0.8734 -> '87.34%'."""
    if x != x:  # NaN
        return "—"
    return f"{100.0 * x:.{digits}f}%"


def format_float(x: float, digits: int = 4) -> str:
    """Format a float for tables; NaN becomes an em dash."""
    if x != x:  # NaN
        return "—"
    return f"{x:.{digits}f}"


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


def resolve_ordered(cell: dict) -> np.ndarray:
    """Bool array length R: ordered instances. Uses stored field or rebuilds V only."""
    R = int(len(cell["alpha_star"]))
    if "ordered" in cell:
        return np.asarray(cell["ordered"], dtype=bool)
    out = np.empty(R, dtype=bool)
    n = int(cell["n"])
    m = int(cell["m"])
    rho = float(cell["rho"])
    theta = float(cell["theta"])
    master_seed = int(cell["master_seed"])
    for k in range(R):
        inst = draw_instance(n, m, rho, theta, master_seed, k)
        out[k] = is_ordered_instance(inst["V"])
    return out


def summarise_cell(
    path: str, eps: float = 0.0, min_failures: int = 200
) -> dict:
    """Numbers 1-3 plus raw_mean_diagnostic, R, n_failures, mean/std collision_rate.

    In-memory dict. Also used to build the per-run summary.csv.
    For round_robin, bundle sizes and skipped_fraction are asserted constant
    across instances (RR size profile is deterministic); summary still reports
    instance 0 for those two fields.

    Certificate fields:
    - certificate: cell_certificate(n, m, rho)
    - pr_exists_lower_bound: 1.0 if cell certificate; else for RR
      mean(ordered OR rr_is_efx) (boolean OR, constructive existence lower bound);
      for non-RR mean(ordered) (or 1.0 if certified).
    - pr_rr_efx_given_exists: efx_rate if certified and mechanism is round_robin,
      else NaN. Never condition on the ordered subset (that would bias rho < 1).
    Note: efx_rate is always a lower bound on Pr[RR EFX | exists] because
    Pr[exists] <= 1, even without a certificate.
    """
    cell = load_cell(path)
    stars = cell["alpha_star"]
    failures = is_failure(stars, eps)
    sizes = np.asarray(cell["bundle_sizes"])
    skipped = np.asarray(cell["skipped_fraction"], dtype=np.float64)
    if cell["mechanism"] == "round_robin":
        if not np.all(sizes == sizes[0]):
            raise AssertionError(
                f"round_robin bundle_sizes vary across instances in {path}"
            )
        if not np.allclose(skipped, skipped[0], rtol=0.0, atol=0.0):
            raise AssertionError(
                f"round_robin skipped_fraction varies across instances in {path}"
            )

    n = int(cell["n"])
    m = int(cell["m"])
    rho = float(cell["rho"])
    cert = cell_certificate(n, m, rho)
    rate = efx_rate(stars, eps=eps)
    is_rr = cell["mechanism"] == "round_robin"

    if cert is not None:
        pr_exists_lb = 1.0
        pr_rr_given = rate if is_rr else float("nan")
    else:
        ordered = resolve_ordered(cell)
        if is_rr:
            rr_efx = ~is_failure(stars, eps)
            pr_exists_lb = float(np.mean(ordered | rr_efx))
        else:
            pr_exists_lb = float(ordered.mean())
        pr_rr_given = float("nan")

    return {
        "n": n,
        "m": m,
        "rho": rho,
        "theta": cell["theta"],
        "gini": gini_from_theta(float(cell["theta"])),
        "spearman": spearman_from_rho(float(cell["rho"])),
        "mechanism": cell["mechanism"],
        "master_seed": cell["master_seed"],
        "R": int(len(stars)),
        "efx_rate": rate,
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
        "bundle_sizes": sizes[0],
        "skipped_fraction": float(skipped[0]),
        "certificate": cert,
        "pr_exists_lower_bound": pr_exists_lb,
        "pr_rr_efx_given_exists": pr_rr_given,
    }


def compare_cells(
    path_a: str,
    path_b: str,
    eps: float = 0.0,
    min_failures: int = 200,
) -> dict:
    """Paired mechanism comparison of two stored cells (same V worlds via CRN).

    Asserts identical n, m, rho, theta, master_seed, R. Convention for aggregate
    drivers: a = round_robin (reference), b = mechanism X.
    """
    a = load_cell(path_a)
    b = load_cell(path_b)
    for key in ("n", "m", "rho", "theta", "master_seed"):
        if a[key] != b[key]:
            raise AssertionError(
                f"compare_cells: mismatched {key}: a={a[key]!r}, b={b[key]!r} "
                f"({path_a} vs {path_b})"
            )
    stars_a = np.asarray(a["alpha_star"], dtype=np.float64)
    stars_b = np.asarray(b["alpha_star"], dtype=np.float64)
    if len(stars_a) != len(stars_b):
        raise AssertionError(
            f"compare_cells: mismatched R: a={len(stars_a)}, b={len(stars_b)} "
            f"({path_a} vs {path_b})"
        )
    idx_a = np.asarray(a["instance_index"])
    idx_b = np.asarray(b["instance_index"])
    if not np.array_equal(idx_a, idx_b):
        raise AssertionError(
            f"compare_cells: instance_index mismatch ({path_a} vs {path_b})"
        )

    R = int(len(stars_a))
    efx_a = ~is_failure(stars_a, eps)
    efx_b = ~is_failure(stars_b, eps)
    both = efx_a & efx_b
    a_only = efx_a & ~efx_b
    b_only = ~efx_a & efx_b
    neither = ~efx_a & ~efx_b

    n_disc = int(a_only.sum() + b_only.sum())
    if n_disc > 0:
        share_a_on_disc = float(a_only.sum() / n_disc)
        se_a_on_disc = float(
            np.sqrt(share_a_on_disc * (1.0 - share_a_on_disc) / n_disc)
        )
    else:
        share_a_on_disc = float("nan")
        se_a_on_disc = float("nan")

    delta = stars_a - stars_b
    rate_a = efx_rate(stars_a, eps=eps)
    rate_b = efx_rate(stars_b, eps=eps)
    n = int(a["n"])
    m = int(a["m"])
    rho = float(a["rho"])
    cert = cell_certificate(n, m, rho)

    return {
        "n": n,
        "m": m,
        "rho": rho,
        "theta": float(a["theta"]),
        "master_seed": int(a["master_seed"]),
        "R": R,
        "mechanism_a": a["mechanism"],
        "mechanism_b": b["mechanism"],
        "certificate": cert,
        "efx_rate_a": rate_a,
        "efx_rate_b": rate_b,
        "efx_rate_diff": rate_a - rate_b,
        "pr_both": float(both.mean()),
        "pr_a_only": float(a_only.mean()),
        "pr_b_only": float(b_only.mean()),
        "pr_neither": float(neither.mean()),
        "n_discordant": n_disc,
        "share_a_on_discordant": share_a_on_disc,
        "se_share_a_on_discordant": se_a_on_disc,
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_q05": float(np.quantile(delta, 0.05)),
        "delta_q95": float(np.quantile(delta, 0.95)),
        "share_a_better": float(np.mean(delta > 0.0)),
        "share_tie": float(np.mean(delta == 0.0)),
        "share_b_better": float(np.mean(delta < 0.0)),
        "q05_a": conditional_quantile(
            stars_a, q=0.05, min_failures=min_failures, eps=eps
        ),
        "q05_b": conditional_quantile(
            stars_b, q=0.05, min_failures=min_failures, eps=eps
        ),
    }


def write_mechanism_comparison(rows: list[dict], out_path: str) -> None:
    """Write one CSV row per compared cell pair."""
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_path, index=False)


def describe_comparison(result: dict) -> None:
    """Print one readable block for a compare_cells result."""
    label_w = 28

    def row(label: str, value: str) -> None:
        print(f"  {label:<{label_w}} {value}")

    print(
        f"Compare {result['mechanism_a']} vs {result['mechanism_b']}  "
        f"n={result['n']} m={result['m']} rho={result['rho']:.2f} "
        f"theta={result['theta']:.2f} R={result['R']}"
    )
    row("certificate", str(result["certificate"]))
    row(
        "EFX a / b / Δ",
        f"{format_pct(result['efx_rate_a'])} / "
        f"{format_pct(result['efx_rate_b'])} / "
        f"{result['efx_rate_diff']:+.4f}",
    )
    row(
        "both / a_only / b_only / neither",
        f"{format_pct(result['pr_both'])} / "
        f"{format_pct(result['pr_a_only'])} / "
        f"{format_pct(result['pr_b_only'])} / "
        f"{format_pct(result['pr_neither'])}",
    )
    row(
        "share a on discordant (±SE)",
        f"{format_float(result['share_a_on_discordant'])} "
        f"(±{format_float(result['se_share_a_on_discordant'])}, "
        f"n={result['n_discordant']})",
    )
    row(
        "Δα* mean / med / q05 / q95",
        f"{format_float(result['delta_mean'])} / "
        f"{format_float(result['delta_median'])} / "
        f"{format_float(result['delta_q05'])} / "
        f"{format_float(result['delta_q95'])}",
    )
    row(
        "α* a> / tie / b>",
        f"{format_pct(result['share_a_better'])} / "
        f"{format_pct(result['share_tie'])} / "
        f"{format_pct(result['share_b_better'])}",
    )
    row(
        "q05 a / b",
        f"{format_float(result['q05_a'])} / {format_float(result['q05_b'])}",
    )


def format_ms(seconds: float) -> str:
    """Format seconds for terminal output."""
    if seconds != seconds:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f}ms"
    return f"{seconds:.2f}s"


def summarise_compare(record: dict) -> dict:
    """Aggregate paired RR + enumeration results for one cell."""
    alpha_rr = np.asarray(record["alpha_rr"], dtype=np.float64)
    exists = np.asarray(record["exists_efx"], dtype=bool)
    enum_skipped = np.asarray(record["enum_skipped"], dtype=bool)
    enum_elapsed = np.asarray(record["enum_elapsed_s"], dtype=np.float64)
    R = int(len(alpha_rr))

    rr_efx = alpha_rr >= 1.0
    pr_exists = float(exists.mean())
    pr_rr_efx = float(rr_efx.mean())
    pr_no_efx = float((~exists).mean())
    pr_rr_fail_exists = float((exists & ~rr_efx).mean())

    exists_mask = exists
    if exists_mask.any():
        pr_rr_given_exists = float(rr_efx[exists_mask].mean())
    else:
        pr_rr_given_exists = float("nan")

    enum_times = enum_elapsed[~enum_skipped]
    return {
        "n": int(record["n"]),
        "m": int(record["m"]),
        "rho": float(record["rho"]),
        "theta": float(record["theta"]),
        "R": R,
        "master_seed": int(record["master_seed"]),
        "full_assignments": full_assignment_count(int(record["n"]), int(record["m"])),
        "no_empty_assignments": no_empty_bundle_count(
            int(record["n"]), int(record["m"])
        ),
        "pr_exists_efx": pr_exists,
        "pr_rr_efx": pr_rr_efx,
        "pr_rr_efx_given_exists": pr_rr_given_exists,
        "pr_no_efx": pr_no_efx,
        "pr_rr_fail_but_exists": pr_rr_fail_exists,
        "enum_skipped_frac": float(enum_skipped.mean()),
        "enum_time_median_s": float(np.median(enum_elapsed)),
        "enum_time_p95_s": float(np.quantile(enum_elapsed, 0.95)),
        "enum_time_max_s": float(enum_elapsed.max()),
        "enum_time_mean_s": float(enum_elapsed.mean()),
        "enum_time_median_nonskip_s": float(np.median(enum_times))
        if enum_times.size
        else 0.0,
        "enum_time_p95_nonskip_s": float(np.quantile(enum_times, 0.95))
        if enum_times.size
        else 0.0,
        "enum_time_max_nonskip_s": float(enum_times.max()) if enum_times.size else 0.0,
        "cell_elapsed_s": float(record["cell_elapsed_s"]),
    }


def describe_compare_row(record: dict) -> None:
    """Print search space, timing, rates, and one English prose sentence."""
    s = summarise_compare(record)
    n, m = s["n"], s["m"]
    full = s["full_assignments"]
    no_empty = s["no_empty_assignments"]

    print(
        f"n={n}, m={m}, rho={s['rho']:.2f}, theta={s['theta']:.2f}, R={s['R']}"
    )
    print(
        f"  search space: {n}^{m} = {full} assignments "
        f"({no_empty} with no empty bundle)"
    )
    skipped_pct = 100.0 * s["enum_skipped_frac"]
    print(
        f"  enum time: median {format_ms(s['enum_time_median_nonskip_s'])}, "
        f"p95 {format_ms(s['enum_time_p95_nonskip_s'])}, "
        f"max {format_ms(s['enum_time_max_nonskip_s'])} "
        f"(skipped {skipped_pct:.0f}% because RR already EFX); "
        f"cell {format_ms(s['cell_elapsed_s'])}"
    )
    print(
        f"  exists={s['pr_exists_efx']:.3f}, RR_EFX={s['pr_rr_efx']:.3f}, "
        f"RR|exists={format_float(s['pr_rr_efx_given_exists'])}, "
        f"no_EFX={s['pr_no_efx']:.3f}, "
        f"RR_fail_but_exists={s['pr_rr_fail_but_exists']:.3f}"
    )

    exists_pct = format_pct(s["pr_exists_efx"])
    rr_pct = format_pct(s["pr_rr_efx"])
    no_efx_pct = format_pct(s["pr_no_efx"])
    fail_pct = format_pct(s["pr_rr_fail_but_exists"])
    med = format_ms(s["enum_time_median_nonskip_s"])

    if s["pr_exists_efx"] == 0.0:
        print(
            f"  No instance had an EFX allocation. Round-Robin was EFX in "
            f"{rr_pct} of instances. RR|exists is undefined."
        )
    else:
        print(
            f"  An EFX allocation existed in {exists_pct} of instances. "
            f"Round-Robin was EFX in {rr_pct} of instances "
            f"({format_pct(s['pr_rr_efx_given_exists'])} of those where EFX exists). "
            f"On {no_efx_pct} no EFX allocation existed; on {fail_pct} EFX existed "
            f"but Round-Robin missed it. Enumeration median {med}."
        )


def write_compare_summary(records: list[dict], out_path: str) -> None:
    """Write one CSV row per compare cell."""
    import pandas as pd

    rows = [summarise_compare(record) for record in records]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def describe_cell(path: str, eps: float = 0.0, min_failures: int = 200) -> None:
    """Print one labeled block for one stored cell (scannable terminal summary)."""
    s = summarise_cell(path, eps=eps, min_failures=min_failures)
    sizes = np.asarray(s["bundle_sizes"]).tolist()
    label_w = 22

    def row(label: str, value: str) -> None:
        print(f"  {label:<{label_w}} {value}")

    print("Main cell")
    row("n / m / R", f"{s['n']} / {s['m']} / {s['R']}")
    row("rho", rho_label(s["rho"]))
    row("theta", theta_label(s["theta"]))
    row("mechanism", str(s["mechanism"]))
    row("master_seed", str(s["master_seed"]))
    row("certificate", str(s["certificate"]))
    row("EFX rate", format_pct(s["efx_rate"]))
    row("Pr[exists] LB", format_float(s["pr_exists_lower_bound"]))
    row("RR EFX | exists", format_float(s["pr_rr_efx_given_exists"]))
    row("failures", f"{s['n_failures']} / {s['R']}")
    row("cond. 5% quantile", format_float(s["conditional_q05"]))
    row("conditional mean", format_float(s["conditional_mean"]))
    row("raw_mean_diagnostic", format_float(s["raw_mean_diagnostic"]))
    row("bundle_sizes", str(sizes))
    row("skipped_fraction", format_float(s["skipped_fraction"]))
    row(
        "mean collision",
        f"{format_float(s['collision_rate_mean'])}  "
        f"(std {format_float(s['collision_rate_std'])})",
    )


def write_run_summary(
    cell_paths: list[str], out_path: str, eps: float = 0.0, min_failures: int = 200
) -> None:
    """Write one readable CSV row per cell in this run."""
    import pandas as pd

    rows = []
    for path in cell_paths:
        s = summarise_cell(path, eps=eps, min_failures=min_failures)
        sizes = np.asarray(s["bundle_sizes"]).tolist()
        rows.append(
            {
                "n": s["n"],
                "m": s["m"],
                "rho": s["rho"],
                "theta": s["theta"],
                "gini": s["gini"],
                "spearman": s["spearman"],
                "R": s["R"],
                "mechanism": s["mechanism"],
                "master_seed": s["master_seed"],
                "certificate": s["certificate"],
                "efx_rate": s["efx_rate"],
                "pr_exists_lower_bound": s["pr_exists_lower_bound"],
                "pr_rr_efx_given_exists": s["pr_rr_efx_given_exists"],
                "conditional_q05": s["conditional_q05"],
                "conditional_mean": s["conditional_mean"],
                "raw_mean_diagnostic": s["raw_mean_diagnostic"],
                "n_failures": s["n_failures"],
                "collision_rate_mean": s["collision_rate_mean"],
                "collision_rate_std": s["collision_rate_std"],
                "skipped_fraction": s["skipped_fraction"],
                "bundle_sizes": sizes,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


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
        "alpha_matrix": (
            cell["alpha_matrix"][k]
            if "alpha_matrix" in cell
            else rebuilt["alpha_matrix"]
        ),
        "alpha_star": float(cell["alpha_star"][k]),
        "bundle_sizes": cell["bundle_sizes"][k],
        "skipped_fraction": float(cell["skipped_fraction"][k]),
        "collision_rate": float(cell["collision_rate"][k]),
        "V": rebuilt["V"],
        "Z": rebuilt["Z"],
        "A": rebuilt["A"],
    }
