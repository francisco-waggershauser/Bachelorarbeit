"""Stage 6: human-facing scale conversions. Files stay in (rho, theta); labels show both."""

from __future__ import annotations

import math

#berechnung von Gini und Spearman aus theta und rho

def gini_from_theta(theta: float) -> float:
    """2 * Phi(theta / sqrt(2)) - 1, with Phi the standard normal cdf.

    This is the Gini coefficient of a lognormal with sigma = theta, because
    v = exp(theta * Z) and Z ~ N(0, 1).
    """
    return math.erf(theta / 2.0)


def spearman_from_rho(rho: float) -> float:
    """(6 / pi) * asin(rho / 2). Equals 0 at rho=0 and 1 at rho=1."""
    return (6.0 / math.pi) * math.asin(rho / 2.0)


def theta_label(theta: float) -> str:
    """The one theta legend/title string, e.g. 'theta = 1.00 (Gini = 0.52)'."""
    return f"theta = {theta:.2f} (Gini = {gini_from_theta(theta):.2f})"


def rho_label(rho: float) -> str:
    """The one rho legend/title string, e.g. 'rho = 0.50 (Spearman = 0.48)'."""
    return f"rho = {rho:.2f} (Spearman = {spearman_from_rho(rho):.2f})"
