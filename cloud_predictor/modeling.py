"""The product's modeling core: breakup event -> debris cloud density frames.

This is the function API everything else wraps (app.py serves it over HTTP,
publish_breakup.py streams its output into SpacetimeDB, worker.py runs it for queued
/v1/simulations requests). It chains the NN-corrected NASA breakup model (ml/) and the
physics engine (engine/): one run, or a Monte Carlo ensemble (engine.ensemble) whose runs
sample the model's correction bands, giving per-voxel p10/p50/p90.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from engine import ensemble
from engine.density_grid import density_timeline
from engine.fragmentation import generate_cloud
from engine.propagator import circular_state, particle_orbits, propagate_cloud
from ml import sbm
from ml.contract import ParameterSet
from ml.schema import BreakupEvent, BreakupParameters

# Dense early (the point-source -> band transition is fast), then hourly to 48 h.
DEFAULT_TIMES_S = [0.0, 600.0, 1800.0] + [h * 3600.0 for h in range(1, 49)]


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
    n_runs: int = 1,
    bands: Optional[BreakupParameters] = None,
    progress: Optional[Callable[[float], None]] = None,
    with_particles: bool = False,
) -> dict:
    """Model the untrackable (2 mm - 10 cm) debris cloud of one breakup.

    The parent orbit is either a circular orbit (alt_km + inc_deg, optionally
    raan_deg/arg_lat_deg) or an explicit ECI state (r0_km, v0_km_s).

    n_runs == 1: one run with params (a ml.contract.ParameterSet of NN corrections; None
    means the uncorrected SBM). n_runs > 1: an ensemble whose runs each draw corrections
    from bands (the estimator's BreakupParameters; None = uncorrected SBM every run).
    progress(fraction) is called as runs finish. with_particles adds "particles": each
    representative fragment's orbit (engine.propagator.particle_orbits) for one run at params,
    so a viewer can animate the cloud continuously instead of stepping between frames.

    Returns a dict with:
      times_s            (T,) snapshot times [s after breakup]
      frames             list of T {"xyz_km": (n, 3), "density": (n,) ensemble mean,
                         "p10", "p50", "p90": (n,) percentiles across runs} sparse voxel
                         fields, densest first (one run: p10 = p50 = p90 = density)
      in_orbit_fraction  (T,) weighted share of fragments still in orbit (ensemble mean)
      fragments_total    total real fragments represented (weighted; ensemble mean)
      n_runs             number of runs
      particles          (with_particles) dict of (K,) arrays from particle_orbits
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
    if n_runs > 1:
        result = _model_ensemble(event, r0, v0, times_s, bands, n_runs, k, voxel_km, sigma_voxels,
                                 max_voxels, seed, progress)
        if with_particles:
            # The median scenario (params = p50 corrections), not one random ensemble draw.
            cloud = generate_cloud(event, params=params, k=k, seed=seed)
            result["particles"] = particle_orbits(cloud, r0, v0, float(np.max(times_s)))
        return result

    cloud = generate_cloud(event, params=params, k=k, seed=seed)
    out = propagate_cloud(cloud, r0, v0, times_s=np.asarray(times_s, dtype=float))
    frames = density_timeline(out["r_km"], out["alive"], out["weight"],
                              voxel_km=voxel_km, sigma_voxels=sigma_voxels,
                              max_voxels=max_voxels)
    for f in frames:
        f["p10"] = f["p50"] = f["p90"] = f["density"]
    total = float(cloud["weight"].sum())
    in_orbit = (out["alive"] * cloud["weight"]).sum(axis=1) / total
    if progress is not None:
        progress(1.0)
    result = {
        "times_s": out["times_s"],
        "frames": frames,
        "in_orbit_fraction": in_orbit,
        "fragments_total": total,
        "derived": sbm.derive(event),
        "voxel_km": float(voxel_km),
        "n_runs": 1,
    }
    if with_particles:
        result["particles"] = particle_orbits(cloud, r0, v0, float(np.max(times_s)))
    return result


def _model_ensemble(event, r0, v0, times_s, bands, n_runs, k, voxel_km, sigma_voxels, max_voxels,
                    seed, progress) -> dict:
    res = ensemble.simulate(event, r0, v0, times_s, params=bands, n_runs=n_runs, k=k, seed=seed,
                            voxel_km=voxel_km, sigma_voxels=sigma_voxels, max_voxels=max_voxels,
                            progress=progress)
    total = float(np.mean([r["fragments_total"] for r in res["runs"]]))
    in_orbit = np.mean([r["fragments_in_orbit"] for r in res["runs"]], axis=0) / total
    frames = [{"xyz_km": f["xyz_km"], "density": f["mean"], "p10": f["p10"], "p50": f["p50"], "p90": f["p90"]}
              for f in res["frames"]]
    return {
        "times_s": res["times_s"],
        "frames": frames,
        "in_orbit_fraction": in_orbit,
        "fragments_total": total,
        "derived": sbm.derive(event),
        "voxel_km": float(voxel_km),
        "n_runs": n_runs,
    }
