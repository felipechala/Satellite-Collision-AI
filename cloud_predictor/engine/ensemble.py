"""
Step 4: Monte Carlo ensemble — ML corrections in, density with uncertainty out.

Runs the fragmentation -> propagation -> voxel pipeline n_runs times. Each run draws one
concrete ParameterSet from the estimator's p10/p50/p90 bands (ml.contract.sample_parameter_set:
one draw per run, never per particle) and its own particle seed, so the spread across runs
covers both the model's uncertainty and Monte Carlo noise.

Per snapshot, the mean density is the smoothed mean of the runs' raw grids (smoothing is
linear, so this equals the mean of the smoothed runs). The densest max_voxels voxels of that
mean are kept, and each run's smoothed density is evaluated only there (smoothed_at) to get
the per-voxel p10/p50/p90 across runs. Keeping every run's fully smoothed grid would cost
gigabytes; the raw grids are a few thousand voxels each.

Units: km, km/s, seconds in; densities in fragments/km^3 out.
Demo / CLI (from cloud_predictor/): python -m engine.ensemble --out cloud.json [--model-dir models/estimator-v1]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ml.contract import ParameterSet, sample_parameter_set
from ml.schema import BreakupEvent, BreakupParameters, params_to_dict

from .density_grid import _accumulate, gaussian_smooth, smoothed_at, voxelize
from .fragmentation import generate_cloud
from .propagator import propagate_cloud

PERCENTILES = (10, 50, 90)


def _frame(grids, n_runs, voxel_km, sigma_voxels, max_voxels):
    """Aggregate one snapshot's per-run raw grids into the densest voxels with run percentiles."""
    filled = [g for g in grids if g["indices"].shape[0]]
    if not filled:
        empty = np.empty(0)
        return {"xyz_km": np.empty((0, 3)), "mean": empty, "p10": empty, "p50": empty, "p90": empty}
    rows, total = _accumulate(np.concatenate([g["indices"] for g in filled]),
                              np.concatenate([g["density"] for g in filled]))
    mean = gaussian_smooth({"indices": rows, "density": total / n_runs, "voxel_km": voxel_km}, sigma_voxels)
    order = np.argsort(mean["density"])[::-1][:max_voxels]
    selected = mean["indices"][order]
    per_run = np.stack([smoothed_at(g, selected, sigma_voxels) for g in grids], axis=1)
    p10, p50, p90 = np.percentile(per_run, PERCENTILES, axis=1)
    return {"xyz_km": (selected + 0.5) * voxel_km, "mean": mean["density"][order], "p10": p10, "p50": p50, "p90": p90}


def simulate(event: BreakupEvent, r0_km, v0_km_s, times_s, params: Optional[BreakupParameters] = None,
             n_runs: int = 20, k: int = 10_000, seed: int = 0, voxel_km: float = 20.0,
             sigma_voxels: float = 1.0, max_voxels: int = 1000) -> dict:
    """Monte Carlo density field of a breakup cloud.

    params: the estimator's BreakupParameters; None runs the uncorrected SBM in every run
    (only Monte Carlo noise varies).

    Returns {"times_s": (T,),
             "frames": T dicts {"xyz_km": (n, 3) voxel centers [ECI km], "mean", "p10", "p50", "p90": (n,)
                                fragments/km^3}, densest mean first, n <= max_voxels,
             "runs": n_runs dicts {"params": drawn ParameterSet, "fragments_in_orbit": (T,) list},
             "params": params as a dict or None}.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    rng = np.random.default_rng(seed)
    times = np.sort(np.unique(np.asarray(times_s, dtype=float)))
    grids = [[] for _ in times]  # grids[j][i]: run i's raw grid at snapshot j
    runs = []
    for _ in range(n_runs):
        ps = sample_parameter_set(params, rng) if params is not None else ParameterSet()
        cloud = generate_cloud(event, ps, k=k, seed=int(rng.integers(2**31)))
        out = propagate_cloud(cloud, r0_km, v0_km_s, times)
        in_orbit = []
        for j in range(times.size):
            w = np.where(out["alive"][j], out["weight"], 0.0)
            grids[j].append(voxelize(out["r_km"][j], w, voxel_km))
            in_orbit.append(float(w.sum()))
        runs.append({"params": asdict(ps), "fragments_in_orbit": in_orbit})
    frames = [_frame(g, n_runs, voxel_km, sigma_voxels, max_voxels) for g in grids]
    return {"times_s": times, "frames": frames, "runs": runs,
            "params": params_to_dict(params) if params is not None else None}


def to_json(result: dict) -> dict:
    """Frontend / SpacetimeDB-shaped output: one entry per snapshot, voxels densest first."""
    frames = []
    in_orbit = np.array([r["fragments_in_orbit"] for r in result["runs"]])  # (runs, T)
    for j, (t, f) in enumerate(zip(result["times_s"], result["frames"])):
        lo, mid, hi = np.percentile(in_orbit[:, j], PERCENTILES)
        frames.append({
            "t_s": float(t),
            "fragments_in_orbit": {"mean": float(in_orbit[:, j].mean()), "p10": float(lo), "p50": float(mid),
                                   "p90": float(hi)},
            "voxels": [
                {"x": round(float(x), 1), "y": round(float(y), 1), "z": round(float(z), 1),
                 "density_mean": float(f"{m:.6g}"), "density_p10": float(f"{a:.6g}"),
                 "density_p50": float(f"{b:.6g}"), "density_p90": float(f"{c:.6g}")}
                for (x, y, z), m, a, b, c in zip(f["xyz_km"], f["mean"], f["p10"], f["p50"], f["p90"])
            ],
        })
    return {"params": result["params"], "runs": result["runs"], "frames": frames}


def demo_event() -> BreakupEvent:
    """The scenario the other engine demos use: 950 kg payload hit by 50 kg at 10 km/s."""
    from ml.schema import Impactor, Spacecraft
    return BreakupEvent(
        event_type="collision",
        epoch="2026-10-03T00:00:00Z",
        target=Spacecraft(object_class="payload", dry_mass_kg=950.0),
        impactor=Impactor(mass_kg=50.0, v_rel_km_s=10.0),
    )


def main(argv: Optional[list[str]] = None) -> None:
    from .propagator import circular_state

    p = argparse.ArgumentParser(prog="python -m engine.ensemble", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="JSON file for the frontend / SpacetimeDB")
    p.add_argument("--model-dir", help="trained estimator (models/estimator-v1); omit for the uncorrected SBM")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--particles", type=int, default=10_000)
    p.add_argument("--hours", type=float, default=48.0)
    p.add_argument("--step-hours", type=float, default=3.0)
    p.add_argument("--alt-km", type=float, default=780.0)
    p.add_argument("--inc-deg", type=float, default=86.4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    event = demo_event()
    params = None
    if a.model_dir:
        from ml.estimator import BreakupParameterEstimator  # lazy: the engine itself never needs torch
        params = BreakupParameterEstimator.load(a.model_dir).predict(event)
        b = params.n_multiplier
        print(f"n_multiplier p10/p50/p90 = {b.p10:.3g}/{b.p50:.3g}/{b.p90:.3g}; warnings: {params.warnings}")
    else:
        print("WARNING: no --model-dir, running the uncorrected SBM (no model uncertainty)")

    r0, v0 = circular_state(alt_km=a.alt_km, inc_deg=a.inc_deg)
    times = np.arange(0.0, a.hours * 3600.0 + 1.0, a.step_hours * 3600.0)
    result = simulate(event, r0, v0, times, params, n_runs=a.runs, k=a.particles, seed=a.seed)
    out = to_json(result)
    Path(a.out).write_text(json.dumps(out) + "\n", encoding="utf-8")

    for f in out["frames"]:
        v, n = f["voxels"], f["fragments_in_orbit"]
        peak = f"peak {v[0]['density_mean']:.3g} (p10 {v[0]['density_p10']:.3g}, p90 {v[0]['density_p90']:.3g})" if v else "empty"
        print(f"t={f['t_s'] / 3600.0:5.1f} h: fragments in orbit {n['p50']:,.0f} (p10 {n['p10']:,.0f} - "
              f"p90 {n['p90']:,.0f}), {len(v)} voxels, {peak} fragments/km^3")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
