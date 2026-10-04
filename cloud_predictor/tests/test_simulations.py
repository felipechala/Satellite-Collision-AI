"""The /v1/simulations flow: API -> SpacetimeDB request row -> worker -> stored frames -> API reads.

FakeSpacetime is an in-memory stand-in for the debris-tracker module: it implements the
reducers this flow calls (with the module's rules) and the simple SELECT ... WHERE col = v
[AND ...] queries the API and worker issue.
"""
import copy
import itertools
import re

import pytest
from fastapi.testclient import TestClient

import worker
from app import create_app
from modeling import model_breakup
from ml.schema import event_from_dict
from publish_breakup import sql_rows

EVENT = {
    "event_type": "collision",
    "epoch": "2026-10-03T12:00:00Z",
    "target": {"object_class": "payload", "dry_mass_kg": 950.0, "propellant_mass_kg": 50.0},
    "impactor": {"mass_kg": 50.0, "v_rel_km_s": 10.0},
}
TIMES = [0.0, 3600.0, 21600.0]
BODY = {"event": EVENT, "orbit": {"alt_km": 780, "inc_deg": 86.4}, "k": 1000, "times_s": TIMES,
        "max_voxels": 150, "n_runs": 3}


class FakeSpacetime:
    def __init__(self):
        self.tables = {"simulation_request": [], "breakup_event": [], "cloud_frame": [], "debris_voxel": [],
                       "cloud_particle": []}
        self.calls = []
        self._ids = itertools.count(1)
        self._clock = itertools.count(1_700_000_000_000_000, 1_000_000)

    # --- SQL ---
    def sql(self, query):
        m = re.fullmatch(r"SELECT \* FROM (\w+)(?: WHERE (.+))?", query.strip())
        assert m, f"unsupported query: {query}"
        rows = self.tables[m.group(1)]
        for cond in (m.group(2).split(" AND ") if m.group(2) else []):
            col, raw = (s.strip() for s in cond.split("="))
            value = raw.strip("'") if raw.startswith("'") else int(raw)
            rows = [r for r in rows if r[col] == value]
        return copy.deepcopy(rows)

    # --- reducers (same rules as spacetimedb/src/lib.rs) ---
    def call(self, reducer, *args):
        self.calls.append((reducer, args))
        getattr(self, f"_{reducer}")(*args)

    def _req(self, sim_id):
        for r in self.tables["simulation_request"]:
            if r["id"] == sim_id:
                return r
        raise RuntimeError(f"unknown simulation {sim_id}")

    def _remove_event(self, event_id):
        for t in ("breakup_event", "cloud_frame", "debris_voxel", "cloud_particle"):
            self.tables[t] = [r for r in self.tables[t] if r["event_id"] != event_id]

    def _request_simulation(self, key, params_json):
        if any(r["request_key"] == key for r in self.tables["simulation_request"]):
            raise RuntimeError("duplicate key")
        self.tables["simulation_request"].append({
            "id": next(self._ids), "request_key": key, "params_json": params_json, "status": "queued",
            "progress": 0.0, "error": "",
            "created_at": [next(self._clock)]})  # Timestamp as the /sql endpoint returns it

    def _claim_simulation(self, sim_id):
        r = self._req(sim_id)
        if r["status"] != "queued":
            raise RuntimeError("not queued")
        r["status"] = "running"

    def _report_progress(self, sim_id, progress):
        self._req(sim_id)["progress"] = min(max(progress, 0.0), 1.0)

    def _complete_simulation(self, sim_id):
        r = self._req(sim_id)
        if not any(e["event_id"] == sim_id for e in self.tables["breakup_event"]):
            raise RuntimeError("no published event")
        r["status"], r["progress"] = "done", 1.0

    def _fail_simulation(self, sim_id, error):
        r = self._req(sim_id)
        self._remove_event(sim_id)
        r["status"], r["error"] = "failed", error

    def _delete_simulation(self, sim_id):
        r = self._req(sim_id)
        self._remove_event(sim_id)
        self.tables["simulation_request"].remove(r)

    def _start_breakup(self, event_id, sat_name, mass_kg, emr, catastrophic, model_version, voxel_km):
        self.tables["breakup_event"].append({
            "event_id": event_id, "sat_name": sat_name, "mass_kg": mass_kg, "emr_j_per_g": emr,
            "is_catastrophic": catastrophic, "model_version": model_version, "voxel_km": voxel_km})

    def _publish_frame(self, event_id, frame, t, in_orbit, x, y, z, d, p10, p50, p90):
        assert len({len(x), len(y), len(z), len(d), len(p10), len(p50), len(p90)}) == 1
        for i in range(len(x)):
            self.tables["debris_voxel"].append({
                "event_id": event_id, "frame": frame, "x": x[i], "y": y[i], "z": z[i], "density": d[i],
                "density_p_10": p10[i], "density_p_50": p50[i], "density_p_90": p90[i]})  # SQL column names
        self.tables["cloud_frame"].append({
            "event_id": event_id, "frame": frame, "t_sim_s": t, "n_voxels": len(x),
            "peak_density": max(d), "fragments_in_orbit": in_orbit})

    def _publish_particles(self, event_id, *cols):
        from publish_breakup import PARTICLE_COLUMNS

        assert len(cols) == len(PARTICLE_COLUMNS) and len({len(c) for c in cols}) == 1
        assert len(cols[0]) <= 5000  # the module's MAX_PARTICLES_PER_CALL
        for values in zip(*cols):
            self.tables["cloud_particle"].append({"event_id": event_id, **dict(zip(PARTICLE_COLUMNS, values))})

    def _finalize_breakup(self, event_id, speed, autoplay):
        assert autoplay is False  # the worker never starts the shared playback clock


@pytest.fixture
def fake():
    return FakeSpacetime()


@pytest.fixture
def client(fake, monkeypatch):
    monkeypatch.delenv("BREAKUP_MODEL_DIR", raising=False)
    return TestClient(create_app(spacetime=fake, start_worker=False))


def test_full_flow_post_run_read_delete(client, fake):
    r = client.post("/v1/simulations", json=BODY)
    assert r.status_code == 202
    sim = r.json()
    assert sim["status"] == "queued" and sim["n_runs"] == 3
    sim_id = sim["id"]
    assert client.get(f"/v1/simulations/{sim_id}").json()["status"] == "queued"
    assert client.get(f"/v1/simulations/{sim_id}/frame?t_s=0").status_code == 409

    assert worker.run_once(fake, None) == sim_id
    assert worker.run_once(fake, None) is None  # queue is empty now

    info = client.get(f"/v1/simulations/{sim_id}").json()
    assert info["status"] == "done" and info["progress"] == 1.0 and info["error"] is None
    assert info["request"]["n_runs"] == 3 and info["event"]["model_version"] == "neutral-sbm"
    assert [f["t_s"] for f in info["frames"]] == TIMES
    assert info["created_at"].startswith("2023-")

    # A time between snapshots returns the latest frame at or before it.
    f = client.get(f"/v1/simulations/{sim_id}/frame", params={"t_s": 5000}).json()
    assert f["t_s"] == 3600.0 and f["requested_t_s"] == 5000.0
    assert 0 < f["n_voxels"] == len(f["voxels"]) <= 150
    means = [v[3] for v in f["voxels"]]
    assert means == sorted(means, reverse=True) and f["peak_density"] == means[0]
    assert all(v[4] <= v[5] <= v[6] for v in f["voxels"])
    assert any(v[4] < v[6] for v in f["voxels"])  # 3 runs give a real spread
    assert f["fragments_in_orbit"] > 0

    rng = client.get(f"/v1/simulations/{sim_id}/frames", params={"start_s": 0, "end_s": 3600}).json()
    assert [fr["t_s"] for fr in rng["frames"]] == [0.0, 3600.0]
    assert rng["frames"][1]["voxels"] == f["voxels"]
    assert len(client.get(f"/v1/simulations/{sim_id}/frames").json()["frames"]) == 3

    assert [s["id"] for s in client.get("/v1/simulations").json()] == [sim_id]
    assert client.delete(f"/v1/simulations/{sim_id}").status_code == 204
    assert client.get(f"/v1/simulations/{sim_id}").status_code == 404
    assert not fake.tables["debris_voxel"] and not fake.tables["cloud_frame"]


def test_progress_is_reported_per_run(client, fake):
    sim_id = client.post("/v1/simulations", json=BODY).json()["id"]
    worker.run_once(fake, None)
    progress = [args[1] for name, args in fake.calls if name == "report_progress"]
    assert progress == sorted(progress) and progress[-1] == 1.0 and len(progress) == 3


def test_bad_job_fails_without_leaving_frames(client, fake):
    fake.call("request_simulation", "manual-key", "{not json")
    sim_id = fake.tables["simulation_request"][0]["id"]
    assert worker.run_once(fake, None) == sim_id
    info = client.get(f"/v1/simulations/{sim_id}").json()
    assert info["status"] == "failed" and "JSONDecodeError" in info["error"]
    assert info["request"] == "{not json"
    r = client.get(f"/v1/simulations/{sim_id}/frame")
    assert r.status_code == 409 and "failed" in r.json()["detail"]
    assert not fake.tables["breakup_event"]


def test_invalid_requests_are_rejected_before_queueing(client, fake):
    r = client.post("/v1/simulations", json={**BODY, "n_runs": 99, "orbit": {"alt_km": 5}})
    assert r.status_code == 422
    locs = {d["loc"] for d in r.json()["detail"]}
    assert {"n_runs", "orbit.alt_km", "orbit.inc_deg"} <= locs
    assert client.post("/v1/simulations", json={**BODY, "corrections": "p50"}).status_code == 503
    assert not fake.tables["simulation_request"]


def test_unknown_ids_and_bad_times(client, fake):
    assert client.get("/v1/simulations/42").status_code == 404
    assert client.get("/v1/simulations/42/frame").status_code == 404
    assert client.delete("/v1/simulations/42").status_code == 404
    assert client.get("/v1/simulations/1/frame", params={"t_s": -5}).status_code == 422
    assert client.get("/v1/simulations/1/frames", params={"start_s": 10, "end_s": 5}).status_code == 422


def test_unreachable_spacetimedb_is_502(monkeypatch):
    from publish_breakup import SpacetimeClient

    app = create_app(spacetime=SpacetimeClient("http://127.0.0.1:1", "nope"), start_worker=False)
    c = TestClient(app)
    assert c.post("/v1/simulations", json=BODY).status_code == 502
    assert c.get("/v1/simulations/1").status_code == 502


def test_prune_keeps_newest_finished(fake):
    for i in range(14):
        fake.call("request_simulation", f"k{i}", "{}")
    for r in fake.tables["simulation_request"][:12]:
        r["status"] = "done"
    deleted = worker.prune(fake, keep=10)
    assert deleted == [2, 1]
    statuses = [r["status"] for r in fake.tables["simulation_request"]]
    assert statuses.count("done") == 10 and statuses.count("queued") == 2


def test_model_breakup_ensemble_percentiles():
    event = event_from_dict(EVENT)
    res = model_breakup(event, 780.0, 86.4, n_runs=3, k=500, times_s=[0.0, 3600.0], max_voxels=100)
    assert res["n_runs"] == 3 and len(res["frames"]) == 2
    for f in res["frames"]:
        assert ((f["p10"] <= f["p50"]) & (f["p50"] <= f["p90"])).all()
    assert 0 < res["in_orbit_fraction"][-1] <= res["in_orbit_fraction"][0] <= 1.0
    single = model_breakup(event, 780.0, 86.4, k=500, times_s=[0.0])
    assert single["n_runs"] == 1 and (single["frames"][0]["p10"] == single["frames"][0]["density"]).all()


def test_sync_endpoint_returns_percentile_voxels_for_ensembles(client):
    b = client.post("/v1/debris-cloud", json=BODY).json()
    assert b["n_runs"] == 3 and len(b["frames"][1]["voxels"][0]) == 7
    assert len(client.post("/v1/debris-cloud", json={**BODY, "n_runs": 1}).json()["frames"][0]["voxels"][0]) == 4


def test_sql_rows_decodes_spacetimedb_response():
    resp = [{"schema": {"elements": [{"name": {"some": "id"}, "algebraic_type": {"U64": []}},
                                     {"name": {"some": "status"}, "algebraic_type": {"String": []}}]},
             "rows": [[1, "queued"], [2, "done"]]}]
    assert sql_rows(resp) == [{"id": 1, "status": "queued"}, {"id": 2, "status": "done"}]
    assert sql_rows([]) == []


def test_particles_are_stored_and_served(client, fake):
    sim_id = client.post("/v1/simulations", json=BODY).json()["id"]
    assert client.get(f"/v1/simulations/{sim_id}/particles").status_code == 409
    worker.run_once(fake, None)
    b = client.get(f"/v1/simulations/{sim_id}/particles").json()
    p = b["particles"]
    assert 0 < b["n_particles"] <= BODY["k"] and b["t_end_s"] == TIMES[-1]
    assert len(p["a_km"]) == len(p["t_decay_s"]) == b["n_particles"]
    assert all(t == -1 or t > 0 for t in p["t_decay_s"])  # never-bound particles are not stored
    # Bound orbits only (sub-orbital fragments are stored too; they re-enter at the first step).
    assert all(a > 0 for a in p["a_km"]) and all(0 <= e < 0.99 for e in p["e"])
    assert client.delete(f"/v1/simulations/{sim_id}").status_code == 204
    assert not fake.tables["cloud_particle"]


def test_particle_orbits_track_the_propagator():
    """The browser animates particles from particle_orbits; they must follow the full propagator."""
    import numpy as np

    from engine.fragmentation import generate_cloud
    from engine.propagator import circular_state, orbit_positions, particle_orbits, propagate_cloud

    event = event_from_dict(EVENT)
    r0, v0 = circular_state(780.0, 86.4)
    cloud = generate_cloud(event, k=2000, seed=1)
    orbits = particle_orbits(cloud, r0, v0, 24 * 3600.0)
    ref = propagate_cloud(cloud, r0, v0, [0.0, 6 * 3600.0, 24 * 3600.0])
    for j, t in enumerate(ref["times_s"]):
        p = orbit_positions(orbits, t)
        assert np.array_equal(np.isfinite(p).all(axis=1), ref["alive"][j])  # same re-entry times
        err = np.linalg.norm(p[ref["alive"][j]] - ref["r_km"][j][ref["alive"][j]], axis=1)
        assert np.median(err) < 1.0 and np.percentile(err, 90) < 50.0
    assert np.allclose(orbit_positions(orbits, 0.0)[ref["alive"][0]], r0, atol=1e-6)
