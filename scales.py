"""Human-facing scale conversions. Files stay in (rho, theta); only labels convert."""

from __future__ import annotations

import math


def gini_from_theta(theta: float) -> float:
    """2 * Phi(theta / sqrt(2)) - 1, with Phi the standard normal cdf.

    This is the Gini coefficient of a lognormal with sigma = theta, because
    v = exp(theta * Z) and Z ~ N(0, 1).
    """
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), so Phi(theta/sqrt(2)) = 0.5 * (1 + erf(theta/2))
    return math.erf(theta / 2.0)


def spearman_from_rho(rho: float) -> float:
    """(6 / pi) * asin(rho / 2). Equals 0 at rho=0 and 1 at rho=1."""
    return (6.0 / math.pi) * math.asin(rho / 2.0)


def gini_label(theta: float) -> str:
    """The one Gini legend/title string, e.g. 'Gini = 0.52'. Plots must use this."""
    return f"Gini = {gini_from_theta(theta):.2f}"


def spearman_label(rho: float) -> str:
    """The one Spearman legend/title string, e.g. 'Spearman = 0.48'. Plots must use this."""
    return f"Spearman = {spearman_from_rho(rho):.2f}"
