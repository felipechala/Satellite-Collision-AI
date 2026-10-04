"""Debris Cloud Modeling API — the product.

    BREAKUP_MODEL_DIR=models/estimator-v1 uvicorn app:app        # from cloud_predictor/

Endpoints (interactive docs at /docs):
  GET  /v1/health                 service + model status
  POST /v1/breakup-parameters     NN correction bands for one event (p10/p50/p90)
  POST /v1/debris-cloud           full model: event + orbit -> voxel density frames
  POST /v1/debris-cloud:publish   same, then stream the frames into SpacetimeDB

/v1/debris-cloud body:
  {
    "event":   { ...ml.schema.BreakupEvent shape... },
    "orbit":   {"alt_km": 780, "inc_deg": 86.4, "raan_deg": 0, "arg_lat_deg": 0}
               or {"r_km": [x,y,z], "v_km_s": [vx,vy,vz]},
    "corrections": "auto" | "p50" | "neutral" | {"n_multiplier": 1.4, ...},   # default auto
    "times_s": [...optional snapshot times...],
    "k": 10000, "voxel_km": 20, "sigma_voxels": 1, "max_voxels": 1000, "seed": 0
  }
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ml.contract import BOUNDS, ParameterSet
from ml.schema import EventValidationError, FieldError, event_from_dict, params_to_dict
from modeling import DEFAULT_TIMES_S, model_breakup

MODEL_DIR_ENV = "BREAKUP_MODEL_DIR"

# Request limits: keep one call bounded in time and payload.
LIMITS = {"k": (100, 100_000), "voxel_km": (1.0, 500.0), "sigma_voxels": (0.0, 5.0),
          "max_voxels": (10, 2000), "seed": (0, 2**31 - 1)}
MAX_TIMES = 100
MAX_HORIZON_S = 30 * 86400.0


def _error(errors: list[FieldError]) -> JSONResponse:
    return JSONResponse(status_code=422,
                        content={"detail": [{"loc": e.loc, "msg": e.msg} for e in errors]})


async def _json_body(request: Request) -> tuple[Any, Optional[JSONResponse]]:
    try:
        body = await request.json()
    except ValueError:
        return None, _error([FieldError("body", "invalid JSON")])
    if not isinstance(body, dict):
        return None, _error([FieldError("body", "must be an object")])
    return body, None


def _check_orbit(orbit: Any, errors: list[FieldError]) -> dict:
    if not isinstance(orbit, dict):
        errors.append(FieldError("orbit", "is required (alt_km+inc_deg or r_km+v_km_s)"))
        return {}
    if "r_km" in orbit or "v_km_s" in orbit:
        for key in ("r_km", "v_km_s"):
            v = orbit.get(key)
            if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)):
                errors.append(FieldError(f"orbit.{key}", "must be a list of 3 numbers"))
        return {"r0_km": orbit.get("r_km"), "v0_km_s": orbit.get("v_km_s")}
    out = {}
    for key, lo, hi, required in (("alt_km", 150.0, 50_000.0, True), ("inc_deg", 0.0, 180.0, True),
                                  ("raan_deg", -360.0, 360.0, False), ("arg_lat_deg", -360.0, 360.0, False)):
        if key not in orbit:
            if required:
                errors.append(FieldError(f"orbit.{key}", "is required"))
            continue
        v = orbit[key]
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            errors.append(FieldError(f"orbit.{key}", f"must be a number in [{lo:g}, {hi:g}]"))
        else:
            out[key] = float(v)
    return out


def _check_options(body: dict, errors: list[FieldError]) -> dict:
    out = {}
    for key, (lo, hi) in LIMITS.items():
        if key in body:
            v = body[key]
            if not isinstance(v, (int, float)) or not lo <= v <= hi:
                errors.append(FieldError(key, f"must be a number in [{lo}, {hi}]"))
            else:
                out[key] = int(v) if isinstance(lo, int) else float(v)
    times = body.get("times_s")
    if times is not None:
        ok = (isinstance(times, list) and 0 < len(times) <= MAX_TIMES
              and all(isinstance(t, (int, float)) and 0 <= t <= MAX_HORIZON_S for t in times))
        if ok:
            out["times_s"] = [float(t) for t in times]
        else:
            errors.append(FieldError("times_s", f"must be 1..{MAX_TIMES} times in [0, {MAX_HORIZON_S:g}] s"))
    return out


def _frames_json(result: dict) -> list[dict]:
    frames = []
    for t, frac, frame in zip(result["times_s"], result["in_orbit_fraction"], result["frames"]):
        xyz = np.round(frame["xyz_km"], 2)
        rho = frame["density"]
        frames.append({
            "t_s": float(t),
            "in_orbit_fraction": round(float(frac), 4),
            "n_voxels": int(rho.size),
            "peak_density": float(f"{rho[0]:.5g}") if rho.size else 0.0,
            "voxels": [[float(x), float(y), float(z), float(f"{d:.5g}")]
                       for (x, y, z), d in zip(xyz, rho)],
        })
    return frames


def create_app(estimator=None) -> FastAPI:
    app = FastAPI(title="Debris Cloud Modeling API", version="1.0",
                  description="Predicts the untrackable (<10 cm) debris cloud of a satellite "
                              "breakup as a time series of 3D voxel density fields.")
    state = {"estimator": estimator}

    def get_estimator():
        if state["estimator"] is None and os.environ.get(MODEL_DIR_ENV):
            from ml import BreakupParameterEstimator  # lazy: imports torch

            state["estimator"] = BreakupParameterEstimator.load(os.environ[MODEL_DIR_ENV])
        return state["estimator"]

    def resolve_corrections(body: dict, event) -> tuple[Optional[ParameterSet], Optional[dict], str,
                                                        Optional[JSONResponse]]:
        """-> (params, parameter_bands, source, error_response)"""
        choice = body.get("corrections", "auto")
        if isinstance(choice, dict):
            unknown = set(choice) - set(BOUNDS)
            if unknown or not all(isinstance(v, (int, float)) for v in choice.values()):
                return None, None, "", _error([FieldError("corrections",
                                                          f"numeric values for any of {sorted(BOUNDS)}")])
            clamped = {k: min(max(float(v), BOUNDS[k][0]), BOUNDS[k][1]) for k, v in choice.items()}
            return ParameterSet(**clamped), None, "overrides", None
        if choice not in ("auto", "p50", "neutral"):
            return None, None, "", _error([FieldError("corrections",
                                                      "must be 'auto', 'p50', 'neutral' or an object")])
        est = get_estimator() if choice in ("auto", "p50") else None
        if choice == "p50" and est is None:
            return None, None, "", JSONResponse(status_code=503, content={
                "detail": f"corrections='p50' needs a trained model; set {MODEL_DIR_ENV}"})
        if est is None:
            return None, None, "neutral-sbm", None
        predicted = est.predict(event)
        return ParameterSet.from_p50(predicted), params_to_dict(predicted), est.model_version, None

    def run_model(body: dict) -> tuple[Optional[dict], Optional[JSONResponse]]:
        errors: list[FieldError] = []
        orbit = _check_orbit(body.get("orbit"), errors)
        options = _check_options(body, errors)
        event = None
        if "event" not in body:
            errors.append(FieldError("event", "is required"))
        else:
            try:
                event = event_from_dict(body["event"])
            except EventValidationError as e:
                errors.extend(FieldError(f"event.{x.loc}", x.msg) for x in e.errors)
        if errors:
            return None, _error(errors)
        params, bands, source, err = resolve_corrections(body, event)
        if err:
            return None, err
        result = model_breakup(event, params=params, **orbit, **options)
        d = result["derived"]
        return {
            "model_version": source,
            "corrections": dataclasses.asdict(params) if params else dataclasses.asdict(ParameterSet()),
            "parameters": bands,
            "derived": {"emr_j_per_g": d.emr_j_per_g, "is_catastrophic": d.is_catastrophic,
                        "sbm_mass_param": d.sbm_mass_param},
            "fragments_total": round(result["fragments_total"]),
            "voxel_km": result["voxel_km"],
            "frames": _frames_json(result),
        }, None

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
            return JSONResponse(status_code=503,
                                content={"detail": f"{MODEL_DIR_ENV} is not set"})
        body, err = await _json_body(request)
        if err:
            return err
        try:
            return params_to_dict(est.predict(event_from_dict(body)))
        except EventValidationError as e:
            return _error(e.errors)

    @app.post("/v1/debris-cloud")
    async def debris_cloud(request: Request):
        body, err = await _json_body(request)
        if err:
            return err
        response, err = run_model(body)
        return err if err else response

    @app.post("/v1/debris-cloud:publish")
    async def debris_cloud_publish(request: Request):
        from publish_breakup import SpacetimeClient  # local import: optional feature

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
        response, err = run_model(body)
        if err:
            return err

        stdb = body.get("spacetimedb") or {}
        client = SpacetimeClient(stdb.get("host", "http://127.0.0.1:3000"),
                                 stdb.get("db", "debris-tracker"))
        try:
            if body.get("replace"):
                try:
                    client.call("delete_breakup", event_id)
                except RuntimeError:
                    pass
            d = response["derived"]
            client.call("start_breakup", event_id, sat_name,
                        float(body["event"]["target"].get("dry_mass_kg", 0.0))
                        + float(body["event"]["target"].get("propellant_mass_kg") or 0.0),
                        d["emr_j_per_g"] or 0.0, bool(d["is_catastrophic"]),
                        response["model_version"], response["voxel_km"])
            for i, frame in enumerate(response["frames"]):
                v = frame["voxels"]
                client.call("publish_frame", event_id, i, frame["t_s"],
                            [r[0] for r in v], [r[1] for r in v], [r[2] for r in v],
                            [r[3] for r in v])
            client.call("finalize_breakup", event_id,
                        float(body.get("speed", 3600.0)), True)
        except (RuntimeError, OSError) as e:
            return JSONResponse(status_code=502, content={"detail": f"SpacetimeDB publish failed: {e}"})
        response["published"] = {"event_id": event_id, "frames": len(response["frames"])}
        return response

    return app


app = create_app()
