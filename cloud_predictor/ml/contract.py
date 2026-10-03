"""How the fragmentation engine applies a BreakupParameters record to the SBM.

The engine should import these functions rather than re-deriving them, so every correction
has exactly one meaning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from . import sbm
from .schema import BAND_FIELDS, BreakupEvent, BreakupParameters

NEUTRAL: dict[str, float] = {
    "n_multiplier": 1.0,
    "slope_delta": 0.0,
    "am_mu_shift": 0.0,
    "am_sigma_scale": 1.0,
    "dv_mu_shift": 0.0,
    "dv_sigma_scale": 1.0,
}

BOUNDS: dict[str, tuple[float, float]] = {
    "n_multiplier": (0.2, 5.0),
    "slope_delta": (-0.5, 0.5),
    "am_mu_shift": (-0.5, 0.5),
    "am_sigma_scale": (0.5, 2.0),
    "dv_mu_shift": (-0.5, 0.5),
    "dv_sigma_scale": (0.5, 2.0),
}

# Multipliers and scales are sampled (and learned) in log space so they stay positive.
LOG_SPACE = frozenset({"n_multiplier", "am_sigma_scale", "dv_sigma_scale"})

# Distance between the 10th and 90th percentiles of a standard normal.
P10_P90_Z_SPAN = 2.563


@dataclass(frozen=True)
class ParameterSet:
    """One concrete draw of the six corrections, used for a whole Monte Carlo run."""
    n_multiplier: float = 1.0
    slope_delta: float = 0.0
    am_mu_shift: float = 0.0
    am_sigma_scale: float = 1.0
    dv_mu_shift: float = 0.0
    dv_sigma_scale: float = 1.0

    @classmethod
    def from_p50(cls, params: BreakupParameters) -> "ParameterSet":
        return cls(**{f.name: getattr(params, f.name).p50 for f in fields(cls)})


def corrected_count(lc, ps: ParameterSet, event: BreakupEvent, s: float = 1.0):
    """Cumulative fragment count N(>= Lc) after the n_multiplier and slope_delta corrections."""
    lc = np.asarray(lc, dtype=float)
    alpha = sbm.ALPHA[event.event_type]
    anchor = sbm.n_cum_sbm(sbm.LC_ANCHOR_M, event, s)
    above = ps.n_multiplier * sbm.n_cum_sbm(lc, event, s)
    below = ps.n_multiplier * anchor * (lc / sbm.LC_ANCHOR_M) ** -(alpha + ps.slope_delta)
    return np.where(lc >= sbm.LC_ANCHOR_M, above, below)


def am_params(lambda_c, ps: ParameterSet):
    """(mean, std) of log10(A/M) at the given log10(Lc)."""
    return sbm.am_mu(lambda_c) + ps.am_mu_shift, ps.am_sigma_scale * sbm.am_sigma(lambda_c)


def dv_params(chi, ps: ParameterSet, event_type: str):
    """(mean, std) of log10(delta-v) at the given log10(A/M)."""
    return sbm.dv_mu(chi, event_type) + ps.dv_mu_shift, sbm.DV_SIGMA * ps.dv_sigma_scale


def sample_parameter_set(params: BreakupParameters, rng: np.random.Generator) -> ParameterSet:
    """Draw one parameter set per Monte Carlo run, never per particle."""
    draws = {}
    for name in BAND_FIELDS:
        band = getattr(params, name)
        lo, hi = BOUNDS[name]
        if name in LOG_SPACE:
            center = math.log(band.p50)
            sd = (math.log(band.p90) - math.log(band.p10)) / P10_P90_Z_SPAN
            value = math.exp(center + sd * rng.standard_normal())
        else:
            sd = (band.p90 - band.p10) / P10_P90_Z_SPAN
            value = band.p50 + sd * rng.standard_normal()
        draws[name] = min(max(value, lo), hi)
    return ParameterSet(**draws)
