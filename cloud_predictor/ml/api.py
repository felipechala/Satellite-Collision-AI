"""HTTP wrapper: `BREAKUP_MODEL_DIR=models/estimator-v1 uvicorn ml.api:app` from cloud_predictor/."""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .estimator import BreakupParameterEstimator
from .schema import EventValidationError, FieldError, event_from_dict, params_to_dict

MODEL_DIR_ENV = "BREAKUP_MODEL_DIR"


def _error_response(errors: list[FieldError]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [{"loc": e.loc, "msg": e.msg} for e in errors]})


async def _json_body(request: Request) -> tuple[Any, Optional[JSONResponse]]:
    try:
        return await request.json(), None
    except ValueError:
        return None, _error_response([FieldError("body", "invalid JSON")])


def create_app(estimator: Optional[BreakupParameterEstimator] = None) -> FastAPI:
    app = FastAPI(title="Breakup Parameter Estimator")
    state = {"estimator": estimator}

    def get_estimator() -> Optional[BreakupParameterEstimator]:
        if state["estimator"] is None and os.environ.get(MODEL_DIR_ENV):
            state["estimator"] = BreakupParameterEstimator.load(os.environ[MODEL_DIR_ENV])
        return state["estimator"]

    def unavailable() -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": f"{MODEL_DIR_ENV} is not set"})

    @app.post("/v1/breakup-parameters")
    async def predict_one(request: Request):
        est = get_estimator()
        if est is None:
            return unavailable()
        body, err = await _json_body(request)
        if err:
            return err
        try:
            event = event_from_dict(body)
        except EventValidationError as e:
            return _error_response(e.errors)
        return params_to_dict(est.predict(event))

    @app.post("/v1/breakup-parameters:batch")
    async def predict_batch(request: Request):
        est = get_estimator()
        if est is None:
            return unavailable()
        body, err = await _json_body(request)
        if err:
            return err
        if not isinstance(body, list):
            return _error_response([FieldError("body", "must be an array of events")])
        events, errors = [], []
        for i, item in enumerate(body):
            try:
                events.append(event_from_dict(item))
            except EventValidationError as e:
                errors.extend(FieldError(f"[{i}].{x.loc}", x.msg) for x in e.errors)
        if errors:
            return _error_response(errors)
        return [params_to_dict(p) for p in est.predict_batch(events)]

    return app


app = create_app()
