import copy

import pytest
from fastapi.testclient import TestClient

import publish_breakup
from app import create_app
from ml import BreakupParameterEstimator
from ml.synthetic import write
from ml.train_breakup_scaler import train

AS_OF = "2026-01-01T00:00:00Z"

EVENT = {
    "event_type": "collision",
    "epoch": "2026-10-03T12:00:00Z",
    "target": {"object_class": "payload", "dry_mass_kg": 950.0, "propellant_mass_kg": 50.0,
               "structure_material": "cfrp", "mli_fraction": 0.5, "launch_year": 2018},
    "impactor": {"mass_kg": 50.0, "v_rel_km_s": 10.0},
}
FAST = {"k": 2000, "times_s": [0.0, 3600.0, 21600.0], "max_voxels": 200}


def _body(**overrides):
    body = {"event": copy.deepcopy(EVENT), "orbit": {"alt_km": 780, "inc_deg": 86.4}, **FAST}
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def estimator(tmp_path_factory):
    data = write(tmp_path_factory.mktemp("synth"), n_events=200, seed=0, as_of=AS_OF)
    out = tmp_path_factory.mktemp("models") / "api-test"
    train(str(data / "historical_breakups.csv"), str(data / "gunter_satellites.json"), str(out),
          fragments_path=str(data / "fragments.csv"), as_of=AS_OF, log=lambda _: None)
    return BreakupParameterEstimator.load(str(out))


@pytest.fixture
def neutral_client(monkeypatch):
    monkeypatch.delenv("BREAKUP_MODEL_DIR", raising=False)
    return TestClient(create_app())


@pytest.fixture(scope="module")
def model_client(estimator):
    return TestClient(create_app(estimator))


def test_health_reports_model_state(neutral_client, model_client):
    assert neutral_client.get("/v1/health").json()["model_loaded"] is False
    h = model_client.get("/v1/health").json()
    assert h["model_loaded"] is True and h["model_version"] == "api-test"


def test_debris_cloud_neutral_shape_and_physics(neutral_client):
    r = neutral_client.post("/v1/debris-cloud", json=_body())
    assert r.status_code == 200
    b = r.json()
    assert b["model_version"] == "neutral-sbm" and b["parameters"] is None
    assert b["derived"]["is_catastrophic"] is True
    assert [f["t_s"] for f in b["frames"]] == FAST["times_s"]
    for f in b["frames"]:
        assert 0 < f["n_voxels"] <= FAST["max_voxels"] == 200
        assert len(f["voxels"]) == f["n_voxels"] and len(f["voxels"][0]) == 4
        densities = [v[3] for v in f["voxels"]]
        assert densities == sorted(densities, reverse=True) and f["peak_density"] == densities[0]
    # Point source at t=0 spreads out: peak density falls, the cloud occupies more voxels.
    first, last = b["frames"][0], b["frames"][-1]
    assert last["peak_density"] < first["peak_density"]
    assert last["n_voxels"] > first["n_voxels"]
    assert last["in_orbit_fraction"] <= first["in_orbit_fraction"] <= 1.0


def test_auto_uses_model_and_returns_bands(model_client):
    b = model_client.post("/v1/debris-cloud", json=_body()).json()
    assert b["model_version"] == "api-test"
    assert b["parameters"]["n_multiplier"]["p10"] <= b["parameters"]["n_multiplier"]["p50"]
    assert b["corrections"]["n_multiplier"] == b["parameters"]["n_multiplier"]["p50"]


def test_neutral_choice_skips_model(model_client):
    b = model_client.post("/v1/debris-cloud", json=_body(corrections="neutral")).json()
    assert b["model_version"] == "neutral-sbm" and b["corrections"]["n_multiplier"] == 1.0


def test_override_corrections_scale_fragment_count(neutral_client):
    base = neutral_client.post("/v1/debris-cloud", json=_body()).json()
    doubled = neutral_client.post("/v1/debris-cloud",
                                  json=_body(corrections={"n_multiplier": 2.0})).json()
    assert doubled["model_version"] == "overrides"
    assert doubled["fragments_total"] == pytest.approx(2 * base["fragments_total"], rel=1e-6)
    clamped = neutral_client.post("/v1/debris-cloud",
                                  json=_body(corrections={"n_multiplier": 99.0})).json()
    assert clamped["corrections"]["n_multiplier"] == 5.0  # contract BOUNDS


def test_p50_without_model_is_503(neutral_client):
    assert neutral_client.post("/v1/debris-cloud", json=_body(corrections="p50")).status_code == 503


def test_explicit_state_vector_orbit(neutral_client):
    r = neutral_client.post("/v1/debris-cloud",
                            json=_body(orbit={"r_km": [7158.137, 0, 0], "v_km_s": [0, 0.4668, 7.4475]}))
    assert r.status_code == 200 and r.json()["frames"][0]["n_voxels"] > 0


def test_deterministic_for_same_seed(neutral_client):
    a = neutral_client.post("/v1/debris-cloud", json=_body(seed=7)).json()
    b = neutral_client.post("/v1/debris-cloud", json=_body(seed=7)).json()
    assert a == b


def test_validation_reports_all_errors_at_once(neutral_client):
    r = neutral_client.post("/v1/debris-cloud", json={
        "event": {"event_type": "collision", "epoch": "2026-01-01T00:00:00Z",
                  "target": {"object_class": "payload", "dry_mass_kg": 100}},
        "orbit": {"alt_km": 5},
        "k": 10**9,
        "times_s": [-1],
    })
    assert r.status_code == 422
    locs = {d["loc"] for d in r.json()["detail"]}
    assert {"event.impactor", "orbit.alt_km", "orbit.inc_deg", "k", "times_s"} <= locs


@pytest.mark.parametrize("payload", ["not json", "[1, 2]"])
def test_non_object_body_is_422(neutral_client, payload):
    r = neutral_client.post("/v1/debris-cloud", content=payload,
                            headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_bad_corrections_are_422(neutral_client):
    for c in ("magic", {"not_a_field": 1.0}, {"n_multiplier": "big"}):
        assert neutral_client.post("/v1/debris-cloud", json=_body(corrections=c)).status_code == 422


def test_breakup_parameters_endpoint(neutral_client, model_client):
    assert neutral_client.post("/v1/breakup-parameters", json=EVENT).status_code == 503
    b = model_client.post("/v1/breakup-parameters", json=EVENT).json()
    assert b["model_version"] == "api-test" and "am_mu_shift" in b
    assert model_client.post("/v1/breakup-parameters", json={"event_type": "x"}).status_code == 422


class _FakeSpacetime:
    calls: list = []

    def __init__(self, host, db):
        self.host, self.db = host, db

    def call(self, reducer, *args):
        if reducer == "delete_breakup":
            raise RuntimeError("no such event")
        _FakeSpacetime.calls.append((reducer, args))


def test_publish_streams_every_frame(neutral_client, monkeypatch):
    _FakeSpacetime.calls = []
    monkeypatch.setattr(publish_breakup, "SpacetimeClient", _FakeSpacetime)
    r = neutral_client.post("/v1/debris-cloud:publish",
                            json=_body(event_id=5, sat_name="Test Sat", replace=True))
    assert r.status_code == 200
    assert r.json()["published"] == {"event_id": 5, "frames": 3}
    names = [c[0] for c in _FakeSpacetime.calls]
    assert names == ["start_breakup"] + ["publish_frame"] * 3 + ["finalize_breakup"]
    start_args = _FakeSpacetime.calls[0][1]
    assert start_args[:3] == (5, "Test Sat", 1000.0)
    _, frame_args = _FakeSpacetime.calls[1]
    in_orbit, xs, ys, zs, rho, p10, p50, p90 = frame_args[3:]
    assert len(xs) == len(ys) == len(zs) == len(rho) == r.json()["frames"][0]["n_voxels"]
    assert p10 == p50 == p90 == rho  # a single run has no spread across runs
    assert in_orbit == pytest.approx(r.json()["fragments_total"] * r.json()["frames"][0]["in_orbit_fraction"],
                                     rel=1e-3)


def test_publish_requires_ids(neutral_client):
    r = neutral_client.post("/v1/debris-cloud:publish", json=_body())
    locs = {d["loc"] for d in r.json()["detail"]}
    assert r.status_code == 422 and {"event_id", "sat_name"} <= locs


def test_publish_reports_unreachable_spacetimedb(neutral_client):
    r = neutral_client.post("/v1/debris-cloud:publish", json=_body(
        event_id=6, sat_name="x", spacetimedb={"host": "http://127.0.0.1:1", "db": "nope"}))
    assert r.status_code == 502
