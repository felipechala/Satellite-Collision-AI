"""
Step 1: fragmentation engine (NASA Standard Breakup Model, small-fragment regime).

Given a breakup event, produce K weighted "representative particles" between 2 mm and
10 cm, each with a size, area-to-mass ratio, mass, and ejection velocity vector.

All SBM formulas come from ml.sbm, and the ML corrections are applied through
ml.contract, so every correction has exactly one meaning (see ml/contract.py).
Pass a ParameterSet from the estimator to apply corrections; the default is the
uncorrected SBM.

Units: metres, kilograms, m/s (convert dv to km/s before adding to an orbital state).
Run the demo from cloud_predictor/: python -m engine.fragmentation
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ml import sbm
from ml.contract import ParameterSet, am_params, corrected_count, dv_params
from ml.schema import BreakupEvent

L_MIN, L_MAX = 0.002, 0.10  # characteristic length range [m]


def area(lc):
    """Average cross-section A(Lc) [m^2] (Johnson et al., 2001)."""
    lc = np.asarray(lc, dtype=float)
    return np.where(lc < 0.00167, 0.540424 * lc**2, 0.556945 * lc**2.0047077)


def _isotropic(speed, rng):
    v = rng.normal(size=(speed.size, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * speed[:, None]


def generate_cloud(
    event: BreakupEvent,
    params: Optional[ParameterSet] = None,
    k: int = 10_000,
    n_bins: int = 50,
    s: float = 1.0,
    seed: int = 0,
) -> dict:
    """Sample ~k representative particles in n_bins log-spaced size bins.

    Each particle carries a weight = number of real fragments it represents, so
    downstream density grids should sum weights, not particles.
    """
    ps = params if params is not None else ParameterSet()
    rng = np.random.default_rng(seed)

    edges = np.logspace(np.log10(L_MIN), np.log10(L_MAX), n_bins + 1)
    n_cum = corrected_count(edges, ps, event, s)
    real_in_bin = n_cum[:-1] - n_cum[1:]  # real fragments per size bin

    # per_bin particles in each bin, log-uniform within the bin
    per_bin = k // n_bins
    lo = np.repeat(np.log10(edges[:-1]), per_bin)
    hi = np.repeat(np.log10(edges[1:]), per_bin)
    lam = rng.uniform(lo, hi)  # log10(Lc)
    lc = 10**lam
    weight = np.repeat(real_in_bin / per_bin, per_bin)

    chi = rng.normal(*am_params(lam, ps))  # log10(A/M)
    am = 10**chi
    speed = 10 ** rng.normal(*dv_params(chi, ps, event.event_type))  # m/s

    return {
        "lc": lc,                   # m
        "weight": weight,           # real fragments represented
        "am": am,                   # m^2/kg
        "mass": area(lc) / am,      # kg
        "dv": _isotropic(speed, rng),  # m/s, shape (K, 3)
    }


if __name__ == "__main__":
    from ml.schema import Impactor, Spacecraft

    event = BreakupEvent(
        event_type="collision",
        epoch="2026-10-03T00:00:00Z",
        target=Spacecraft(object_class="payload", dry_mass_kg=950.0),
        impactor=Impactor(mass_kg=50.0, v_rel_km_s=10.0),
    )
    cloud = generate_cloud(event)
    speed = np.linalg.norm(cloud["dv"], axis=1)
    expected = sbm.n_cum_sbm(L_MIN, event) - sbm.n_cum_sbm(L_MAX, event)
    print(f"catastrophic:         {sbm.derive(event).is_catastrophic}")
    print(f"particles:            {cloud['lc'].size}")
    print(f"real fragments > 2mm: {cloud['weight'].sum():,.0f}")
    print(f"expected from model:  {expected:,.0f}")
    print(f"median A/M:           {np.median(cloud['am']):.3f} m^2/kg")
    print(f"median dv:            {np.median(speed):.0f} m/s")
    print(f"fragment mass total:  {(cloud['mass'] * cloud['weight']).sum():.0f} kg")
