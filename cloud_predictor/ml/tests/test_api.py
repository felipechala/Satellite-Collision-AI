import pytest
from fastapi.testclient import TestClient

from ml.api import MODEL_DIR_ENV, create_app
from ml.schema import params_from_dict


@pytest.fixture(scope="module")
def client(estimator):
    return TestClient(create_app(estimator))


def test_predict_returns_parameters(client, spec_request):
    r = client.post("/v1/breakup-parameters", json=spec_request)
    assert r.status_code == 200
    params = params_from_dict(r.json())
    assert params.is_catastrophic is True
    assert params.sbm_mass_param == 755.0


def test_invalid_event_returns_field_errors(client, spec_request):
    spec_request["target"]["dry_mass_kg"] = -1
    del spec_request["impactor"]
    r = client.post("/v1/breakup-parameters", json=spec_request)
    assert r.status_code == 422
    locs = {e["loc"] for e in r.json()["detail"]}
    assert locs == {"target.dry_mass_kg", "impactor"}


def test_malformed_json_returns_422(client):
    r = client.post("/v1/breakup-parameters", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == "body"


def test_batch(client, spec_request):
    explosion = {"event_type": "explosion", "epoch": "2001-05-01T00:00:00Z",
                 "target": {"object_class": "rocket_body", "dry_mass_kg": 1400}}
    r = client.post("/v1/breakup-parameters:batch", json=[spec_request, explosion])
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2
    assert out[1]["emr_j_per_g"] is None


def test_batch_errors_are_indexed(client, spec_request):
    bad = {**spec_request, "event_type": "explosion"}
    r = client.post("/v1/breakup-parameters:batch", json=[spec_request, bad])
    assert r.status_code == 422
    assert [e["loc"] for e in r.json()["detail"]] == ["[1].impactor"]


def test_batch_requires_array(client, spec_request):
    r = client.post("/v1/breakup-parameters:batch", json=spec_request)
    assert r.status_code == 422


def test_unconfigured_app_returns_503(monkeypatch, spec_request):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    r = TestClient(create_app()).post("/v1/breakup-parameters", json=spec_request)
    assert r.status_code == 503


def test_app_loads_model_from_env(monkeypatch, model_dir, spec_request):
    monkeypatch.setenv(MODEL_DIR_ENV, str(model_dir))
    r = TestClient(create_app()).post("/v1/breakup-parameters", json=spec_request)
    assert r.status_code == 200
