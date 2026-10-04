"""Model a breakup end-to-end and stream the voxel frames into SpacetimeDB.

    python publish_breakup.py --event-json examples/event.json --event-id 1 \
        --sat-name "Demo Sat" --alt-km 780 --inc-deg 86.4 \
        [--model-dir models/estimator-v1] [--host http://127.0.0.1:3000] [--db debris-tracker]

The modeling itself lives in modeling.model_breakup (the product's function API);
this file is the SpacetimeDB consumer of it. app.py is the HTTP consumer.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import httpx
import numpy as np

from ml.contract import ParameterSet
from ml.schema import BreakupEvent, event_from_dict
from modeling import model_breakup


def _estimate_params(event: BreakupEvent, model_dir: Optional[str]) -> tuple[Optional[ParameterSet], str]:
    if not model_dir:
        return None, "neutral-sbm"
    from ml import BreakupParameterEstimator  # imports torch; optional path

    estimator = BreakupParameterEstimator.load(model_dir)
    predicted = estimator.predict(event)
    return ParameterSet.from_p50(predicted), predicted.model_version


class SpacetimeClient:
    """Minimal SpacetimeDB client over its HTTP API (no SDK dependency): reducer calls + SQL reads."""

    def __init__(self, host: str, db: str):
        self.base = f"{host.rstrip('/')}/v1/database/{db}"
        self.http = httpx.Client(timeout=30.0)

    def call(self, reducer: str, *args) -> None:
        try:
            r = self.http.post(f"{self.base}/call/{reducer}", json=list(args))
        except httpx.HTTPError as e:
            raise RuntimeError(f"{reducer}: cannot reach SpacetimeDB at {self.base}: {e}") from e
        if r.status_code >= 300:
            raise RuntimeError(f"{reducer} failed ({r.status_code}): {r.text}")

    def sql(self, query: str) -> list[dict]:
        """Rows of a single SELECT as dicts keyed by column name."""
        try:
            r = self.http.post(f"{self.base}/sql", content=query.encode("utf-8"),
                               headers={"content-type": "text/plain"})
        except httpx.HTTPError as e:
            raise RuntimeError(f"sql: cannot reach SpacetimeDB at {self.base}: {e}") from e
        if r.status_code >= 300:
            raise RuntimeError(f"sql failed ({r.status_code}): {r.text}")
        return sql_rows(r.json())


def _column_name(element: dict) -> str:
    name = element.get("name")
    return name.get("some") if isinstance(name, dict) else name


def sql_rows(statements: list[dict]) -> list[dict]:
    """Decode a /sql response (one statement: schema.elements + positional rows) into dicts."""
    if not statements:
        return []
    first = statements[0]
    names = [_column_name(e) for e in first["schema"]["elements"]]
    return [dict(zip(names, row)) for row in first["rows"]]


def publish(client: SpacetimeClient, event_id: int, sat_name: str, event: BreakupEvent,
            result: dict, model_version: str, speed: float, autoplay: bool = True) -> None:
    """Write a modeling.model_breakup result as event `event_id` (frames carry p10/p50/p90)."""
    d = result["derived"]
    mass = event.target.dry_mass_kg + event.target.propellant_mass_kg
    client.call("start_breakup", event_id, sat_name, mass,
                d.emr_j_per_g or 0.0, bool(d.is_catastrophic), model_version,
                result["voxel_km"])
    total = result["fragments_total"]
    for i, (t, frac, frame) in enumerate(zip(result["times_s"], result["in_orbit_fraction"], result["frames"])):
        xyz = frame["xyz_km"]
        client.call("publish_frame", event_id, i, float(t), float(frac) * total,
                    xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist(),
                    frame["density"].tolist(), frame["p10"].tolist(), frame["p50"].tolist(),
                    frame["p90"].tolist())
    if "particles" in result:
        publish_particles(client, event_id, result["particles"])
    client.call("finalize_breakup", event_id, speed, autoplay)


# Column order of the publish_particles reducer (and the cloud_particle table).
PARTICLE_COLUMNS = ("a_km", "e", "inc", "raan", "argp", "mean_anomaly", "raan_dot", "argp_dot",
                    "m_dot", "a_dot", "m_ddot", "t_decay_s", "lc_m", "weight")
PARTICLES_PER_CALL = 2500


def publish_particles(client, event_id: int, particles: dict) -> None:
    """Upload engine.propagator.particle_orbits arrays. Never-bound particles (t_decay 0) are
    skipped; survivors' t_decay (inf) is sent as -1, since JSON has no infinity."""
    keep = particles["t_decay_s"] > 0
    cols = {c: np.asarray(particles[c], dtype=float)[keep] for c in PARTICLE_COLUMNS}
    cols["t_decay_s"] = np.where(np.isfinite(cols["t_decay_s"]), cols["t_decay_s"], -1.0)
    for start in range(0, int(keep.sum()), PARTICLES_PER_CALL):
        client.call("publish_particles", event_id,
                    *(cols[c][start:start + PARTICLES_PER_CALL].tolist() for c in PARTICLE_COLUMNS))


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
