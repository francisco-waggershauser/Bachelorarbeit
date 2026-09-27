"""Stage 0: literature certificates for EFX existence (not results of this thesis).

Cell certificates (S1–S4) depend only on (n, m, rho). ORDERED is an
instance-level certificate checked on V.
"""

from __future__ import annotations

import numpy as np

CERTIFICATES: dict[str, str] = {
    "S1": (
        "S1_N2: EFX exists for n = 2 (general valuations, cut-and-choose). "
        "Plaut & Roughgarden, SIDMA 2020, Thm 4.3."
    ),
    "S2": (
        "S2_N3: EFX exists for n = 3 under additive valuations. "
        "Chaudhury, Garg, Mehlhorn 2020/2024."
    ),
    "S3": (
        "S3_M_LE_N_PLUS_3: EFX exists for m <= n + 3 (general valuations). "
        "Mahara 2021, Thm 3."
    ),
    "S4": (
        "S4_IDENTICAL: EFX exists under identical valuations (leximin++). "
        "Plaut & Roughgarden, SIDMA 2020, Thm 4.2."
    ),
    "ORDERED": (
        "ORDERED: EFX exists for ordered instances (common ranking; "
        "envy-cycle elimination). Plaut & Roughgarden, SIDMA 2020."
    ),
}


def cell_certificate(n: int, m: int, rho: float) -> str | None:
    """Return a cell-level existence certificate key, or None.

    Priority (first match wins):
    1. S4 if rho == 1 — in *this* sampler, rho = 1 makes Z[i,g] = W[g] for all i,
       so all agents have identical valuations. That is a property of the DGP,
       not of the theorem itself.
    2. S1 if n == 2
    3. S2 if n == 3 (additivity holds for our additive V)
    4. S3 if m <= n + 3
    """
    if abs(float(rho) - 1.0) < 1e-12:
        return "S4"
    if n == 2:
        return "S1"
    if n == 3:
        return "S2"
    if m <= n + 3:
        return "S3"
    return None


def is_ordered_instance(V: np.ndarray) -> bool:
    """True iff every agent has the same argsort ranking of goods (ties broken by index)."""
    ranks = np.argsort(-V, axis=1, kind="mergesort")
    return bool(np.all(ranks == ranks[0]))
