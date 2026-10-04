"""Model a breakup end-to-end and stream the voxel frames into SpacetimeDB.

    python publish_breakup.py --event-json examples/event.json --event-id 1 \
        --sat-name "Demo Sat" --alt-km 780 --inc-deg 86.4 \
        [--model-dir models/estimator-v1] [--host http://127.0.0.1:3000] [--db debris-tracker]

model_breakup() is the modeling API: BreakupEvent in, voxel density frames out. The
SpacetimeDB publishing below is one consumer of it; a REST wrapper can be another.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import httpx
import numpy as np

from engine.density_grid import density_timeline
from engine.fragmentation import generate_cloud
from engine.propagator import circular_state, propagate_cloud
from ml import sbm
from ml.contract import ParameterSet
from ml.schema import BreakupEvent, event_from_dict

# Dense early (the point-source -> band transition is fast), sparse late.
DEFAULT_TIMES_S = [0.0, 600.0, 1800.0] + [h * 3600.0 for h in (1, 2, 3, 4, 5, 6)] \
    + [h * 3600.0 for h in range(9, 49, 3)]


def model_breakup(
    event: BreakupEvent,
    alt_km: float,
    inc_deg: float,
    params: Optional[ParameterSet] = None,
    times_s: Sequence[float] = tuple(DEFAULT_TIMES_S),
    k: int = 10_000,
    voxel_km: float = 20.0,
    sigma_voxels: float = 1.0,
    max_voxels: int = 1000,
    seed: int = 0,
) -> dict:
    """Breakup event -> time series of voxel density frames (the modeling API).

    Returns {"times_s", "frames": [{"xyz_km", "density"}, ...], "derived", "voxel_km"}.
    """
    cloud = generate_cloud(event, params=params, k=k, seed=seed)
    r0, v0 = circular_state(alt_km, inc_deg)
    out = propagate_cloud(cloud, r0, v0, times_s=np.asarray(times_s, dtype=float))
    frames = density_timeline(out["r_km"], out["alive"], out["weight"],
                              voxel_km=voxel_km, sigma_voxels=sigma_voxels,
                              max_voxels=max_voxels)
    return {"times_s": out["times_s"], "frames": frames,
            "derived": sbm.derive(event), "voxel_km": voxel_km}


def _estimate_params(event: BreakupEvent, model_dir: Optional[str]) -> tuple[Optional[ParameterSet], str]:
    if not model_dir:
        return None, "neutral-sbm"
    from ml import BreakupParameterEstimator  # imports torch; optional path

    estimator = BreakupParameterEstimator.load(model_dir)
    predicted = estimator.predict(event)
    return ParameterSet.from_p50(predicted), predicted.model_version


class SpacetimeClient:
    """Minimal reducer caller over SpacetimeDB's HTTP API (no SDK dependency)."""

    def __init__(self, host: str, db: str):
        self.base = f"{host.rstrip('/')}/v1/database/{db}/call"
        self.http = httpx.Client(timeout=30.0)

    def call(self, reducer: str, *args) -> None:
        r = self.http.post(f"{self.base}/{reducer}", json=list(args))
        if r.status_code >= 300:
            raise RuntimeError(f"{reducer} failed ({r.status_code}): {r.text}")


def publish(client: SpacetimeClient, event_id: int, sat_name: str, event: BreakupEvent,
            result: dict, model_version: str, speed: float, autoplay: bool = True) -> None:
    d = result["derived"]
    mass = event.target.dry_mass_kg + event.target.propellant_mass_kg
    client.call("start_breakup", event_id, sat_name, mass,
                d.emr_j_per_g or 0.0, bool(d.is_catastrophic), model_version,
                result["voxel_km"])
    for i, (t, frame) in enumerate(zip(result["times_s"], result["frames"])):
        xyz, rho = frame["xyz_km"], frame["density"]
        client.call("publish_frame", event_id, i, float(t),
                    xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist(),
                    rho.tolist())
    client.call("finalize_breakup", event_id, speed, autoplay)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event-json", required=True, help="BreakupEvent as JSON (ml.schema shape)")
    p.add_argument("--event-id", type=int, required=True)
    p.add_argument("--sat-name", required=True)
    p.add_argument("--alt-km", type=float, required=True, help="parent circular orbit altitude")
    p.add_argument("--inc-deg", type=float, required=True, help="parent orbit inclination")
    p.add_argument("--model-dir", help="trained estimator dir; omitted = uncorrected SBM")
    p.add_argument("--host", default="http://127.0.0.1:3000")
    p.add_argument("--db", default="debris-tracker")
    p.add_argument("--voxel-km", type=float, default=20.0)
    p.add_argument("--max-voxels", type=int, default=1000)
    p.add_argument("--speed", type=float, default=3600.0, help="simulated s per wall s")
    p.add_argument("--replace", action="store_true", help="delete the event first if it exists")
    a = p.parse_args(argv)

    event = event_from_dict(json.loads(open(a.event_json, encoding="utf-8").read()))
    params, model_version = _estimate_params(event, a.model_dir)
    print(f"modeling with {model_version} corrections...")
    result = model_breakup(event, a.alt_km, a.inc_deg, params,
                           voxel_km=a.voxel_km, max_voxels=a.max_voxels)

    client = SpacetimeClient(a.host, a.db)
    if a.replace:
        try:
            client.call("delete_breakup", a.event_id)
        except RuntimeError:
            pass  # didn't exist
    publish(client, a.event_id, a.sat_name, event, result, model_version, a.speed)
    n = sum(f["density"].size for f in result["frames"])
    print(f"published event {a.event_id}: {len(result['frames'])} frames, {n} voxel rows")


if __name__ == "__main__":
    main()
