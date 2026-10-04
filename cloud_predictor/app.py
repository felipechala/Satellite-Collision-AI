"""Debris Cloud Modeling API — the product.

    BREAKUP_MODEL_DIR=models/estimator-v1 uvicorn app:app        # from cloud_predictor/

Endpoints (interactive docs at /docs):
  GET    /v1/health                       service + model status
  POST   /v1/breakup-parameters           NN correction bands for one event (p10/p50/p90)
<<<<<<< HEAD
=======
  POST   /v1/satellite-image              Grok Imagine illustration of the target satellite (XAI_TOKEN)
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
  POST   /v1/debris-cloud                 full model, computed in the request: event + orbit -> frames
  POST   /v1/debris-cloud:publish         same, then stream the frames into SpacetimeDB
  POST   /v1/simulations                  queue a simulation (stored in SpacetimeDB) -> 202 {id}
  GET    /v1/simulations                  recent simulations and their status
  GET    /v1/simulations/{id}             status, progress, event metadata, frame times
  GET    /v1/simulations/{id}/frame?t_s=  the cloud at a time (latest frame at or before t_s)
  GET    /v1/simulations/{id}/frames?start_s=&end_s=   every frame in a time range
  DELETE /v1/simulations/{id}

Request body (shared by /v1/debris-cloud and /v1/simulations; see request_schema.py):
  {
    "event":   { ...ml.schema.BreakupEvent shape... },
    "orbit":   {"alt_km": 780, "inc_deg": 86.4, "raan_deg": 0, "arg_lat_deg": 0}
               or {"r_km": [x,y,z], "v_km_s": [vx,vy,vz]},
    "corrections": "auto" | "p50" | "neutral" | {"n_multiplier": 1.4, ...},   # default auto
    "times_s": [...optional snapshot times...],
    "k": 10000, "voxel_km": 20, "sigma_voxels": 1, "max_voxels": 1000, "seed": 0,
    "n_runs": 1 for /v1/debris-cloud, 10 for /v1/simulations (Monte Carlo runs, 1-30)
  }

/v1/simulations runs in a worker (worker.py, started with the app unless SIMULATION_WORKER=0)
and stores results in SpacetimeDB (SPACETIME_HOST, SPACETIME_DB). Simulation frame voxels are
[x, y, z, mean, p10, p50, p90] (ECI km, fragments/km^3, densest first).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
<<<<<<< HEAD
=======
from starlette.concurrency import run_in_threadpool
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb

from ml.contract import ParameterSet
from ml.schema import EventValidationError, FieldError, event_from_dict, params_to_dict
from modeling import model_breakup
<<<<<<< HEAD
from request_schema import RequestError, parse_request, resolve_corrections
=======
from request_schema import RequestError, check_orbit, parse_request, resolve_corrections
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb

MODEL_DIR_ENV = "BREAKUP_MODEL_DIR"
SIMULATION_RUNS = 10
DEMO_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _error(errors: list[FieldError], status: int = 422) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"detail": [{"loc": e.loc, "msg": e.msg} for e in errors]})


def _detail(status: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": msg})


async def _json_body(request: Request) -> tuple[Any, Optional[JSONResponse]]:
    try:
        body = await request.json()
    except ValueError:
        return None, _error([FieldError("body", "invalid JSON")])
    if not isinstance(body, dict):
        return None, _error([FieldError("body", "must be an object")])
    return body, None


def _frames_json(result: dict) -> list[dict]:
    """Synchronous-endpoint frames: voxels [x, y, z, density], or [x, y, z, mean, p10, p50, p90]
    when the result is a Monte Carlo ensemble (n_runs > 1)."""
    ensemble = result.get("n_runs", 1) > 1
    frames = []
    for t, frac, frame in zip(result["times_s"], result["in_orbit_fraction"], result["frames"]):
        xyz = np.round(frame["xyz_km"], 2)
        cols = [frame["density"]] + ([frame["p10"], frame["p50"], frame["p90"]] if ensemble else [])
        rho = frame["density"]
        frames.append({
            "t_s": float(t),
            "in_orbit_fraction": round(float(frac), 4),
            "n_voxels": int(rho.size),
            "peak_density": float(f"{rho[0]:.5g}") if rho.size else 0.0,
            "voxels": [[float(x), float(y), float(z), *(float(f"{c[i]:.5g}") for c in cols)]
                       for i, (x, y, z) in enumerate(xyz)],
        })
    return frames


def _timestamp_iso(value: Any) -> Any:
    """SQL rows encode a Timestamp as a one-field product: [micros_since_unix_epoch]."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)):
        return datetime.fromtimestamp(value[0] / 1e6, tz=timezone.utc).isoformat()
    return value


# SpacetimeDB's SQL names snake_case fields with an underscore before digits:
# the Rust field density_p10 is the SQL column density_p_10.
PERCENTILE_COLUMNS = ("density_p_10", "density_p_50", "density_p_90")


def _sim_frame(frame: dict, voxels: list[dict]) -> dict:
    rows = sorted(voxels, key=lambda v: v["density"], reverse=True)
    return {
        "frame": frame["frame"],
        "t_s": frame["t_sim_s"],
        "fragments_in_orbit": frame["fragments_in_orbit"],
        "n_voxels": frame["n_voxels"],
        "peak_density": frame["peak_density"],
        "voxels": [[v["x"], v["y"], v["z"], v["density"], *(v[c] for c in PERCENTILE_COLUMNS)] for v in rows],
    }


<<<<<<< HEAD
def create_app(estimator=None, spacetime=None, start_worker: Optional[bool] = None) -> FastAPI:
    """estimator: preloaded BreakupParameterEstimator (else loaded from BREAKUP_MODEL_DIR on demand).
    spacetime: SpacetimeDB client with call()/sql() (else publish_breakup.SpacetimeClient from env).
    start_worker: run worker.py's loop in a background thread (default: SIMULATION_WORKER != "0")."""
    state = {"estimator": estimator, "spacetime": spacetime}
=======
def create_app(estimator=None, spacetime=None, start_worker: Optional[bool] = None, imager=None) -> FastAPI:
    """estimator: preloaded BreakupParameterEstimator (else loaded from BREAKUP_MODEL_DIR on demand).
    spacetime: SpacetimeDB client with call()/sql() (else publish_breakup.SpacetimeClient from env).
    start_worker: run worker.py's loop in a background thread (default: SIMULATION_WORKER != "0").
    imager: prompt -> imagine.GeneratedImage (else imagine.XaiImager from XAI_TOKEN on demand)."""
    state = {"estimator": estimator, "spacetime": spacetime, "imager": imager}
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
    if start_worker is None:
        start_worker = os.environ.get("SIMULATION_WORKER", "1") != "0"

    def get_estimator():
        if state["estimator"] is None and os.environ.get(MODEL_DIR_ENV):
            from ml import BreakupParameterEstimator  # lazy: imports torch

            state["estimator"] = BreakupParameterEstimator.load(os.environ[MODEL_DIR_ENV])
        return state["estimator"]

<<<<<<< HEAD
=======
    def get_imager():
        if state["imager"] is None:
            from imagine import imager_from_env

            state["imager"] = imager_from_env()
        return state["imager"]

>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
    def stdb():
        if state["spacetime"] is None:
            from worker import client_from_env

            state["spacetime"] = client_from_env()
        return state["spacetime"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = None
        if start_worker:
            from worker import client_from_env, start_thread

            # The worker gets its own HTTP client; the API's stays on the request threads.
            stop = start_thread(spacetime or client_from_env(), get_estimator)
        yield
        if stop is not None:
            stop.set()

    app = FastAPI(title="Debris Cloud Modeling API", version="1.1", lifespan=lifespan,
                  description="Predicts the untrackable (<10 cm) debris cloud of a satellite "
                              "breakup as a time series of 3D voxel density fields.")
    app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", DEMO_ORIGINS).split(","),
                       allow_methods=["*"], allow_headers=["*"])

    def run_model(body: dict) -> tuple[Optional[dict], Optional[dict], Optional[JSONResponse]]:
        """-> (response JSON, raw modeling result, error response)"""
        try:
            req = parse_request(body)
            corr = resolve_corrections(req.corrections, req.event, get_estimator())
        except RequestError as e:
            return None, None, _error(e.errors, e.status)
        result = model_breakup(req.event, params=corr.point, bands=corr.bands, **req.orbit, **req.options)
        result["event"] = req.event
        d = result["derived"]
        return {
            "model_version": corr.source,
            "corrections": dataclasses.asdict(corr.point or ParameterSet()),
            "parameters": corr.parameters,
            "derived": {"emr_j_per_g": d.emr_j_per_g, "is_catastrophic": d.is_catastrophic,
                        "sbm_mass_param": d.sbm_mass_param},
            "n_runs": result["n_runs"],
            "fragments_total": round(result["fragments_total"]),
            "voxel_km": result["voxel_km"],
            "frames": _frames_json(result),
        }, result, None

    @app.get("/v1/health")
    def health():
        est = state["estimator"]
        return {"status": "ok",
                "model_dir": os.environ.get(MODEL_DIR_ENV),
                "model_loaded": est is not None,
                "model_version": est.model_version if est else None}

    @app.post("/v1/breakup-parameters")
    async def breakup_parameters(request: Request):
        est = get_estimator()
        if est is None:
            return _detail(503, f"{MODEL_DIR_ENV} is not set")
        body, err = await _json_body(request)
        if err:
            return err
        try:
            return params_to_dict(est.predict(event_from_dict(body)))
        except EventValidationError as e:
            return _error(e.errors)

<<<<<<< HEAD
=======
    @app.post("/v1/satellite-image")
    async def satellite_image(request: Request):
        """Artist's impression of the event's target satellite in its orbit (not simulation output).
        Body: {"event": ..., "orbit": ...} as for /v1/debris-cloud; orbit is optional."""
        from imagine import TOKEN_ENV, altitude_of, satellite_prompt

        body, err = await _json_body(request)
        if err:
            return err
        errors: list[FieldError] = []
        event = None
        try:
            event = event_from_dict(body.get("event"))
        except EventValidationError as e:
            errors.extend(FieldError(f"event.{x.loc}", x.msg) for x in e.errors)
        orbit = check_orbit(body["orbit"], errors) if "orbit" in body else {}
        if errors:
            return _error(errors)
        imager = get_imager()
        if imager is None:
            return _detail(503, f"{TOKEN_ENV} is not set (environment or repo-root .env)")
        prompt = satellite_prompt(event, *altitude_of(orbit))
        try:
            image = await run_in_threadpool(imager, prompt)
        except RuntimeError as e:
            return _detail(502, str(e))
        return {"image": image.data_url(), "prompt": prompt, "model": image.model}

>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
    @app.post("/v1/debris-cloud")
    async def debris_cloud(request: Request):
        body, err = await _json_body(request)
        if err:
            return err
        response, _, err = run_model(body)
        return err if err else response

    @app.post("/v1/debris-cloud:publish")
    async def debris_cloud_publish(request: Request):
        import publish_breakup  # module attribute lookup, so tests can swap SpacetimeClient

        body, err = await _json_body(request)
        if err:
            return err
        errors = []
        event_id = body.get("event_id")
        if not isinstance(event_id, int) or event_id < 0:
            errors.append(FieldError("event_id", "must be a non-negative integer"))
        sat_name = body.get("sat_name")
        if not isinstance(sat_name, str) or not sat_name:
            errors.append(FieldError("sat_name", "is required"))
        if errors:
            return _error(errors)
        response, result, err = run_model(body)
        if err:
            return err

        stdb_opts = body.get("spacetimedb") or {}
        client = publish_breakup.SpacetimeClient(stdb_opts.get("host", "http://127.0.0.1:3000"),
                                                 stdb_opts.get("db", "debris-tracker"))
        try:
            if body.get("replace"):
                try:
                    client.call("delete_breakup", event_id)
                except RuntimeError:
                    pass
            publish_breakup.publish(client, event_id, sat_name, result["event"], result,
                                    response["model_version"], float(body.get("speed", 3600.0)), True)
        except (RuntimeError, OSError) as e:
            return _detail(502, f"SpacetimeDB publish failed: {e}")
        response["published"] = {"event_id": event_id, "frames": len(response["frames"])}
        return response

    # ---------- stored simulations (SpacetimeDB) ----------

    def request_row(sim_id: int) -> Optional[dict]:
        rows = stdb().sql(f"SELECT * FROM simulation_request WHERE id = {int(sim_id)}")
        return rows[0] if rows else None

    def frames_of(sim_id: int) -> list[dict]:
        return sorted(stdb().sql(f"SELECT * FROM cloud_frame WHERE event_id = {int(sim_id)}"),
                      key=lambda f: f["t_sim_s"])

    def summary(row: dict) -> dict:
        return {"id": row["id"], "status": row["status"], "progress": row["progress"],
                "error": row["error"] or None, "created_at": _timestamp_iso(row["created_at"])}

    def done_or_error(sim_id: int) -> Optional[JSONResponse]:
        row = request_row(sim_id)
        if row is None:
            return _detail(404, f"unknown simulation {sim_id}")
        if row["status"] != "done":
            return _detail(409, f"simulation {sim_id} is {row['status']}"
                                + (f": {row['error']}" if row["error"] else ""))
        return None

    @app.post("/v1/simulations", status_code=202)
    async def create_simulation(request: Request):
        body, err = await _json_body(request)
        if err:
            return err
        try:
            req = parse_request(body, default_runs=SIMULATION_RUNS)
            resolve_corrections(req.corrections, req.event, get_estimator())  # 503 now, not in the worker
        except RequestError as e:
            return _error(e.errors, e.status)
        key = uuid.uuid4().hex
        try:
            stdb().call("request_simulation", key, json.dumps(body, sort_keys=True, separators=(",", ":")))
            rows = stdb().sql(f"SELECT * FROM simulation_request WHERE request_key = '{key}'")
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")
        if not rows:
            return _detail(502, "SpacetimeDB accepted the request but the row was not found")
        sim_id = rows[0]["id"]
        return JSONResponse(status_code=202, content={
            "id": sim_id, "status": rows[0]["status"], "n_runs": req.options["n_runs"],
            "links": {"self": f"/v1/simulations/{sim_id}", "frame": f"/v1/simulations/{sim_id}/frame?t_s=0",
                      "frames": f"/v1/simulations/{sim_id}/frames"}})

    @app.get("/v1/simulations")
    def list_simulations():
        try:
            rows = stdb().sql("SELECT * FROM simulation_request")
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")
        return [summary(r) for r in sorted(rows, key=lambda r: r["id"], reverse=True)]

    @app.get("/v1/simulations/{sim_id}")
    def get_simulation(sim_id: int):
        try:
            row = request_row(sim_id)
            if row is None:
                return _detail(404, f"unknown simulation {sim_id}")
            out = summary(row)
            try:
                out["request"] = json.loads(row["params_json"])
            except ValueError:  # queued straight through the reducer with bad JSON; the worker fails it
                out["request"] = row["params_json"]
            if row["status"] == "done":
                events = stdb().sql(f"SELECT * FROM breakup_event WHERE event_id = {int(sim_id)}")
                if events:
                    ev = events[0]
                    out["event"] = {k: ev[k] for k in ("sat_name", "mass_kg", "emr_j_per_g", "is_catastrophic",
                                                       "model_version", "voxel_km")}
                out["frames"] = [{"frame": f["frame"], "t_s": f["t_sim_s"], "n_voxels": f["n_voxels"],
                                  "peak_density": f["peak_density"], "fragments_in_orbit": f["fragments_in_orbit"]}
                                 for f in frames_of(sim_id)]
            return out
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")

    @app.get("/v1/simulations/{sim_id}/frame")
    def get_frame(sim_id: int, t_s: float = 0.0):
        if not math.isfinite(t_s) or t_s < 0:
            return _error([FieldError("t_s", "must be a time >= 0 in seconds after the breakup")])
        try:
            err = done_or_error(sim_id)
            if err:
                return err
            frames = frames_of(sim_id)
            at_or_before = [f for f in frames if f["t_sim_s"] <= t_s]
            frame = at_or_before[-1] if at_or_before else frames[0]
            voxels = stdb().sql(f"SELECT * FROM debris_voxel WHERE event_id = {int(sim_id)} "
                                f"AND frame = {int(frame['frame'])}")
            return {"id": sim_id, "requested_t_s": t_s, **_sim_frame(frame, voxels)}
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")

    @app.get("/v1/simulations/{sim_id}/frames")
    def get_frames(sim_id: int, start_s: float = 0.0, end_s: Optional[float] = None):
        if end_s is not None and end_s < start_s:
            return _error([FieldError("end_s", "must be >= start_s")])
        try:
            err = done_or_error(sim_id)
            if err:
                return err
            frames = [f for f in frames_of(sim_id)
                      if f["t_sim_s"] >= start_s and (end_s is None or f["t_sim_s"] <= end_s)]
            by_frame: dict[int, list[dict]] = {f["frame"]: [] for f in frames}
            if frames:
                for v in stdb().sql(f"SELECT * FROM debris_voxel WHERE event_id = {int(sim_id)}"):
                    if v["frame"] in by_frame:
                        by_frame[v["frame"]].append(v)
            return {"id": sim_id, "start_s": start_s, "end_s": end_s,
                    "frames": [_sim_frame(f, by_frame[f["frame"]]) for f in frames]}
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")

    @app.get("/v1/simulations/{sim_id}/particles")
    def get_particles(sim_id: int):
        """Representative fragment orbits for continuous animation (see engine.propagator.particle_orbits):
        position at t from a = a_km + a_dot t, raan/argp + rate t, M = mean_anomaly + m_dot t + m_ddot t^2/2;
        t_decay_s -1 = in orbit at the horizon. Column arrays, one entry per particle."""
        from publish_breakup import PARTICLE_COLUMNS

        try:
            err = done_or_error(sim_id)
            if err:
                return err
            rows = stdb().sql(f"SELECT * FROM cloud_particle WHERE event_id = {int(sim_id)}")
            frames = frames_of(sim_id)
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")
        return {"id": sim_id, "n_particles": len(rows), "t_end_s": frames[-1]["t_sim_s"] if frames else 0.0,
                "particles": {c: [r[c] for r in rows] for c in PARTICLE_COLUMNS}}

    @app.delete("/v1/simulations/{sim_id}", status_code=204)
    def delete_simulation(sim_id: int):
        try:
            if request_row(sim_id) is None:
                return _detail(404, f"unknown simulation {sim_id}")
            stdb().call("delete_simulation", int(sim_id))
        except RuntimeError as e:
            return _detail(502, f"SpacetimeDB: {e}")
        return Response(status_code=204)

    return app


app = create_app()
