"""The product's modeling core: breakup event -> debris cloud density frames.

This is the function API everything else wraps (app.py serves it over HTTP,
publish_breakup.py streams its output into SpacetimeDB). It chains the NN-corrected
NASA breakup model (ml/) and the physics engine (engine/).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from engine.density_grid import density_timeline
from engine.fragmentation import generate_cloud
from engine.propagator import circular_state, propagate_cloud
from ml import sbm
from ml.contract import ParameterSet
from ml.schema import BreakupEvent

# Dense early (the point-source -> band transition is fast), sparse late.
DEFAULT_TIMES_S = [0.0, 600.0, 1800.0] + [h * 3600.0 for h in (1, 2, 3, 4, 5, 6)] \
    + [h * 3600.0 for h in range(9, 49, 3)]


def model_breakup(
    event: BreakupEvent,
    alt_km: Optional[float] = None,
    inc_deg: Optional[float] = None,
    params: Optional[ParameterSet] = None,
    times_s: Sequence[float] = tuple(DEFAULT_TIMES_S),
    k: int = 10_000,
    voxel_km: float = 20.0,
    sigma_voxels: float = 1.0,
    max_voxels: int = 1000,
    seed: int = 0,
    raan_deg: float = 0.0,
    arg_lat_deg: float = 0.0,
    r0_km: Optional[Sequence[float]] = None,
    v0_km_s: Optional[Sequence[float]] = None,
) -> dict:
    """Model the untrackable (2 mm - 10 cm) debris cloud of one breakup.

    The parent orbit is either a circular orbit (alt_km + inc_deg, optionally
    raan_deg/arg_lat_deg) or an explicit ECI state (r0_km, v0_km_s). params is a
    ml.contract.ParameterSet of NN corrections; None means the uncorrected SBM.

    Returns a dict with:
      times_s            (T,) snapshot times [s after breakup]
      frames             list of T {"xyz_km": (n, 3), "density": (n,)} sparse voxel fields
      in_orbit_fraction  (T,) weighted share of fragments still in orbit
      fragments_total    total real fragments represented (weighted)
      derived            ml.sbm.Derived (EMR, catastrophic flag, SBM mass param)
      voxel_km           voxel edge length [km]
    """
    if r0_km is not None and v0_km_s is not None:
        r0 = np.asarray(r0_km, dtype=float)
        v0 = np.asarray(v0_km_s, dtype=float)
    elif alt_km is not None and inc_deg is not None:
        r0, v0 = circular_state(alt_km, inc_deg, raan_deg, arg_lat_deg)
    else:
        raise ValueError("provide either (alt_km, inc_deg) or (r0_km, v0_km_s)")

    cloud = generate_cloud(event, params=params, k=k, seed=seed)
    out = propagate_cloud(cloud, r0, v0, times_s=np.asarray(times_s, dtype=float))
    frames = density_timeline(out["r_km"], out["alive"], out["weight"],
                              voxel_km=voxel_km, sigma_voxels=sigma_voxels,
                              max_voxels=max_voxels)
    total = float(cloud["weight"].sum())
    in_orbit = (out["alive"] * cloud["weight"]).sum(axis=1) / total
    return {
        "times_s": out["times_s"],
        "frames": frames,
        "in_orbit_fraction": in_orbit,
        "fragments_total": total,
        "derived": sbm.derive(event),
        "voxel_km": float(voxel_km),
    }
