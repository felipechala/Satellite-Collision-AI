"""Request parsing shared by the HTTP API (app.py) and the simulation worker (worker.py).

A request body has the /v1/debris-cloud shape:
  {
    "event":   { ...ml.schema.BreakupEvent shape... },
    "orbit":   {"alt_km": 780, "inc_deg": 86.4, "raan_deg": 0, "arg_lat_deg": 0}
               or {"r_km": [x,y,z], "v_km_s": [vx,vy,vz]},
    "corrections": "auto" | "p50" | "neutral" | {"n_multiplier": 1.4, ...},   # default auto
    "times_s": [...optional snapshot times...],
    "k": 10000, "voxel_km": 20, "sigma_voxels": 1, "max_voxels": 1000, "seed": 0,
    "n_runs": 10                                    # Monte Carlo runs (1 = single run)
  }
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

from ml.contract import BOUNDS, ParameterSet
from ml.schema import (
    BAND_FIELDS,
    Band,
    BreakupEvent,
    BreakupParameters,
    EventValidationError,
    FieldError,
    event_from_dict,
    params_to_dict,
)

# Request limits: keep one call bounded in time and payload.
LIMITS = {"k": (100, 100_000), "voxel_km": (1.0, 500.0), "sigma_voxels": (0.0, 5.0),
          "max_voxels": (10, 2000), "seed": (0, 2**31 - 1), "n_runs": (1, 30)}
MAX_TIMES = 100
MAX_HORIZON_S = 30 * 86400.0


class RequestError(Exception):
    """A request the API should reject: status 422 with field errors, or another status with a message."""

    def __init__(self, status: int, errors: list[FieldError]):
        self.status = status
        self.errors = errors
        super().__init__("; ".join(f"{e.loc}: {e.msg}" for e in errors))


@dataclass
class ModelRequest:
    event: BreakupEvent
    orbit: dict    # keyword arguments for modeling.model_breakup's orbit
    options: dict  # k, voxel_km, sigma_voxels, max_voxels, seed, times_s, n_runs
    corrections: Any


@dataclass
class Corrections:
    point: Optional[ParameterSet]       # single-run corrections (None = uncorrected SBM)
    bands: Optional[BreakupParameters]  # what each ensemble run draws from (None = uncorrected SBM)
    parameters: Optional[dict]          # the estimator's full prediction, for responses
    source: str                         # model version, "neutral-sbm" or "overrides"


def check_orbit(orbit: Any, errors: list[FieldError]) -> dict:
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


def check_options(body: dict, errors: list[FieldError]) -> dict:
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


def parse_request(body: Any, default_runs: int = 1) -> ModelRequest:
    """Validate a request body; raises RequestError(422) listing every problem at once."""
    if not isinstance(body, dict):
        raise RequestError(422, [FieldError("body", "must be an object")])
    errors: list[FieldError] = []
    orbit = check_orbit(body.get("orbit"), errors)
    options = check_options(body, errors)
    options.setdefault("n_runs", default_runs)
    event = None
    if "event" not in body:
        errors.append(FieldError("event", "is required"))
    else:
        try:
            event = event_from_dict(body["event"])
        except EventValidationError as e:
            errors.extend(FieldError(f"event.{x.loc}", x.msg) for x in e.errors)
    choice = body.get("corrections", "auto")
    if isinstance(choice, dict):
        if set(choice) - set(BOUNDS) or not all(isinstance(v, (int, float)) for v in choice.values()):
            errors.append(FieldError("corrections", f"numeric values for any of {sorted(BOUNDS)}"))
    elif choice not in ("auto", "p50", "neutral"):
        errors.append(FieldError("corrections", "must be 'auto', 'p50', 'neutral' or an object"))
    if errors:
        raise RequestError(422, errors)
    return ModelRequest(event, orbit, options, choice)


def fixed_bands(ps: ParameterSet, model_version: str) -> BreakupParameters:
    """Degenerate bands (p10 = p50 = p90), so every ensemble run uses exactly ps."""
    return BreakupParameters(
        **{f.name: Band(getattr(ps, f.name), getattr(ps, f.name), getattr(ps, f.name)) for f in fields(ps)},
        emr_j_per_g=None, is_catastrophic=None, sbm_mass_param=None, model_version=model_version)


def resolve_corrections(choice: Any, event: BreakupEvent, estimator) -> Corrections:
    """Turn the request's corrections choice into single-run and ensemble parameters.

    auto: the estimator's bands if a model is loaded (ensemble runs sample them), else the
    uncorrected SBM; p50: the model's p50 in every run (503 without a model); neutral: SBM;
    an object: those values (clamped to ml.contract.BOUNDS) in every run.
    """
    if isinstance(choice, dict):
        clamped = {k: min(max(float(v), BOUNDS[k][0]), BOUNDS[k][1]) for k, v in choice.items()}
        ps = ParameterSet(**clamped)
        return Corrections(ps, fixed_bands(ps, "overrides"), None, "overrides")
    est = estimator if choice in ("auto", "p50") else None
    if choice == "p50" and est is None:
        raise RequestError(503, [FieldError("corrections", "'p50' needs a trained model; set BREAKUP_MODEL_DIR")])
    if est is None:
        return Corrections(None, None, None, "neutral-sbm")
    predicted = est.predict(event)
    point = ParameterSet.from_p50(predicted)
    bands = predicted if choice == "auto" else fixed_bands(point, predicted.model_version)
    return Corrections(point, bands, params_to_dict(predicted), est.model_version)


assert set(BAND_FIELDS) == {f.name for f in fields(ParameterSet)}  # fixed_bands covers every band
