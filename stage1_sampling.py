"""Stage 1: draw the latent structure and turn it into valuations."""

from __future__ import annotations

import numpy as np


def validate_parameters(n: int, m: int, rho: float, theta: float) -> None:
    """Raise ValueError if n < 2, m <= n, rho not in [0, 1], or theta <= 0."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if m <= n:
        raise ValueError(f"m must be > n, got m={m}, n={n}")
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if theta <= 0.0:
        raise ValueError(f"theta must be > 0, got {theta}")


def draw_base(
    n: int, m: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw W shape (m,) and eta shape (n, m) as independent N(0, 1). No rho, no theta."""
    W = rng.standard_normal(m)
    eta = rng.standard_normal((n, m))
    return W, eta


def build_Z(W: np.ndarray, eta: np.ndarray, rho: float) -> np.ndarray:
    """Z[i,g] = sqrt(rho)*W[g] + sqrt(1-rho)*eta[i,g]. Deterministic. Var(Z)=1, Corr=rho."""
    return np.sqrt(rho) * W + np.sqrt(1.0 - rho) * eta


def build_values(Z: np.ndarray, theta: float) -> np.ndarray:
    """V = exp(theta * Z). Deterministic. Strictly increasing in Z for theta > 0."""
    return np.exp(theta * Z)
