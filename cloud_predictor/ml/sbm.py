"""NASA Standard Breakup Model reference formulas (Johnson et al., 2001).

Lc is characteristic length in metres, lambda_c = log10(Lc), chi = log10(A/M) with A/M in
m^2/kg, and delta-v is in m/s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .schema import BreakupEvent

CATASTROPHIC_EMR_J_PER_G = 40.0
LC_ANCHOR_M = 0.1

ALPHA = {"collision": 1.71, "explosion": 1.6}
DV_COEF = {"explosion": (0.2, 1.85), "collision": (0.9, 2.9)}
DV_SIGMA = 0.4

# Explosion scaling factor S by rocket-body type. The input carries no rocket-body type,
# so every explosion uses S = 1 until such a field exists.
EXPLOSION_S: dict[str, float] = {}


def target_mass_kg(event: BreakupEvent) -> float:
    return event.target.dry_mass_kg + event.target.propellant_mass_kg


def emr_j_per_g(impactor_mass_kg: float, v_rel_km_s: float, target_mass: float) -> float:
    energy_j = 0.5 * impactor_mass_kg * (v_rel_km_s * 1000.0) ** 2
    return energy_j / (target_mass * 1000.0)


def is_catastrophic(emr: float) -> bool:
    return emr >= CATASTROPHIC_EMR_J_PER_G


def sbm_mass_param(target_mass: float, impactor_mass_kg: float, v_rel_km_s: float) -> float:
    """M in kg if catastrophic, else impactor mass x v^2 in kg*(km/s)^2."""
    if is_catastrophic(emr_j_per_g(impactor_mass_kg, v_rel_km_s, target_mass)):
        return float(target_mass + impactor_mass_kg)
    return float(impactor_mass_kg * v_rel_km_s ** 2)


def n_cum_collision(lc, mass_param: float):
    return 0.1 * mass_param ** 0.75 * np.asarray(lc, dtype=float) ** -ALPHA["collision"]


def n_cum_explosion(lc, s: float = 1.0):
    return 6.0 * s * np.asarray(lc, dtype=float) ** -ALPHA["explosion"]


@dataclass(frozen=True)
class Derived:
    emr_j_per_g: Optional[float]
    is_catastrophic: Optional[bool]
    sbm_mass_param: Optional[float]


def derive(event: BreakupEvent) -> Derived:
    if event.event_type == "explosion":
        return Derived(None, None, None)
    m_t = target_mass_kg(event)
    imp = event.impactor
    emr = emr_j_per_g(imp.mass_kg, imp.v_rel_km_s, m_t)
    return Derived(emr, is_catastrophic(emr), sbm_mass_param(m_t, imp.mass_kg, imp.v_rel_km_s))


def n_cum_sbm(lc, event: BreakupEvent, s: float = 1.0):
    """Unmodified SBM cumulative fragment count N(>= Lc) for this event."""
    if event.event_type == "collision":
        return n_cum_collision(lc, derive(event).sbm_mass_param)
    return n_cum_explosion(lc, s)


def am_mu(lambda_c):
    """Mean of log10(A/M), SBM small-fragment curve.

    The contract applies this curve at every size, although the full SBM switches to a
    bimodal distribution above 11 cm. Labels use the same curve so shifts mean the same thing.
    """
    lc = np.asarray(lambda_c, dtype=float)
    return np.where(lc <= -1.75, -0.3, np.where(lc < -1.25, -0.3 - 1.4 * (lc + 1.75), -1.0))


def am_sigma(lambda_c):
    """Standard deviation of log10(A/M), SBM small-fragment curve."""
    lc = np.asarray(lambda_c, dtype=float)
    return np.where(lc <= -3.5, 0.2, 0.2 + 0.1333 * (lc + 3.5))


def dv_mu(chi, event_type: str):
    a, b = DV_COEF[event_type]
    return a * np.asarray(chi, dtype=float) + b
