"""Simulation worker: runs queued /v1/simulations requests and stores the results in SpacetimeDB.

    python -m worker [--host http://127.0.0.1:3000] [--db debris-tracker]   # from cloud_predictor/

app.py also starts it as a background thread (SIMULATION_WORKER=1, the default), so the demo
needs one Python process. Flow per job: claim_simulation -> parse + model (progress reported
per Monte Carlo run) -> publish frames as event_id = request id -> complete_simulation;
any error -> fail_simulation (which drops partial frames). SpacetimeDB is a temporary store:
after each job only the KEEP_LATEST newest finished simulations are kept.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from typing import Callable, Optional

from modeling import model_breakup
from publish_breakup import SpacetimeClient, publish
from request_schema import parse_request, resolve_corrections

log = logging.getLogger("worker")

DEFAULT_HOST = "http://127.0.0.1:3000"
DEFAULT_DB = "debris-tracker"
DEFAULT_RUNS = 10
POLL_S = 2.0
KEEP_LATEST = 10
PROGRESS_STEP = 0.05  # report progress at most every 5%
PLAYBACK_SPEED = 3600.0


def client_from_env() -> SpacetimeClient:
    return SpacetimeClient(os.environ.get("SPACETIME_HOST", DEFAULT_HOST),
                           os.environ.get("SPACETIME_DB", DEFAULT_DB))


def _progress_reporter(client, sim_id: int) -> Callable[[float], None]:
    last = [0.0]

    def report(fraction: float) -> None:
        if fraction - last[0] >= PROGRESS_STEP or fraction >= 1.0:
            client.call("report_progress", sim_id, float(fraction))
            last[0] = fraction

    return report


def run_job(client, request: dict, estimator) -> None:
    """Run one claimed simulation request; raises on any failure."""
    sim_id = int(request["id"])
    body = json.loads(request["params_json"])
    req = parse_request(body, default_runs=DEFAULT_RUNS)
    corr = resolve_corrections(req.corrections, req.event, estimator)
    result = model_breakup(req.event, params=corr.point, bands=corr.bands, with_particles=True,
                           progress=_progress_reporter(client, sim_id), **req.orbit, **req.options)
    sat_name = body.get("sat_name") if isinstance(body.get("sat_name"), str) else f"simulation {sim_id}"
    publish(client, sim_id, sat_name, req.event, result, corr.source, PLAYBACK_SPEED, autoplay=False)
    client.call("complete_simulation", sim_id)


def prune(client, keep: int = KEEP_LATEST) -> list[int]:
    """Delete finished simulations beyond the newest `keep`; returns the deleted ids."""
    finished = [r for r in client.sql("SELECT * FROM simulation_request")
                if r["status"] in ("done", "failed")]
    old = sorted((int(r["id"]) for r in finished), reverse=True)[keep:]
    for sim_id in old:
        client.call("delete_simulation", sim_id)
    return old


def run_once(client, estimator) -> Optional[int]:
    """Process the oldest queued request, if any. Returns its id."""
    queued = client.sql("SELECT * FROM simulation_request WHERE status = 'queued'")
    if not queued:
        return None
    request = min(queued, key=lambda r: int(r["id"]))
    sim_id = int(request["id"])
    client.call("claim_simulation", sim_id)
    log.info("simulation %d: running", sim_id)
    try:
        run_job(client, request, estimator)
        log.info("simulation %d: done", sim_id)
    except Exception as e:  # any failure is reported on the request row, never kills the loop
        log.exception("simulation %d failed", sim_id)
        client.call("fail_simulation", sim_id, f"{type(e).__name__}: {e}")
    prune(client)
    return sim_id


def loop(client, get_estimator: Callable[[], object], stop: Optional[threading.Event] = None,
         poll_s: float = POLL_S) -> None:
    """Poll forever (or until stop is set). SpacetimeDB being down is logged and retried."""
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            if run_once(client, get_estimator()) is not None:
                continue  # look for the next job right away
        except RuntimeError as e:
            log.warning("SpacetimeDB unavailable: %s", e)
        stop.wait(poll_s)


def start_thread(client, get_estimator: Callable[[], object]) -> threading.Event:
    """Run loop() in a daemon thread; set the returned event to stop it."""
    stop = threading.Event()
    threading.Thread(target=loop, args=(client, get_estimator, stop), name="simulation-worker",
                     daemon=True).start()
    return stop


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m worker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("SPACETIME_HOST", DEFAULT_HOST))
    p.add_argument("--db", default=os.environ.get("SPACETIME_DB", DEFAULT_DB))
    p.add_argument("--model-dir", default=os.environ.get("BREAKUP_MODEL_DIR"),
                   help="trained estimator; omitted = uncorrected SBM")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    estimator = None
    if a.model_dir:
        from ml import BreakupParameterEstimator  # imports torch; optional path

        estimator = BreakupParameterEstimator.load(a.model_dir)
    log.info("watching %s/%s (model: %s)", a.host, a.db, estimator.model_version if estimator else "none")
    loop(SpacetimeClient(a.host, a.db), lambda: estimator)


if __name__ == "__main__":
    main()
