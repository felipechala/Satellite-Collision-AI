import base64

import httpx
from fastapi.testclient import TestClient

import imagine
from app import create_app
from imagine import GeneratedImage, XaiImager, satellite_prompt
from ml.schema import event_from_dict

EVENT = {
    "event_type": "collision",
    "epoch": "2026-10-03T12:00:00Z",
    "target": {"object_class": "payload", "dry_mass_kg": 950.0, "propellant_mass_kg": 50.0,
               "structure_material": "cfrp", "mli_fraction": 0.5, "launch_year": 2018},
    "impactor": {"mass_kg": 50.0, "v_rel_km_s": 10.0},
}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _event(**target):
    return event_from_dict({**EVENT, "target": {**EVENT["target"], **target}})


def test_prompt_reflects_size_construction_and_orbit():
    p = satellite_prompt(_event(), 780.0, 86.4)
    assert "car-sized" in p and "1,000 kg" in p
    assert "carbon-fibre" in p and "gold multi-layer insulation" in p and "2018" in p
    assert "low Earth orbit" in p and "polar" in p and "No text" in p
    assert "CubeSat" in satellite_prompt(_event(dry_mass_kg=4.0, propellant_mass_kg=0.0), 500.0, 51.6)
    assert "polar" not in satellite_prompt(_event(), 500.0, 51.6)
    assert "distant globe" in satellite_prompt(_event(), 35_786.0, 0.0)
    assert "rocket upper stage" in satellite_prompt(_event(object_class="rocket_body"), 800.0, 98.0)


def test_altitude_from_state_vector():
    alt, inc = imagine.altitude_of({"r0_km": [7158.137, 0.0, 0.0], "v0_km_s": [0, 7.5, 0]})
    assert abs(alt - 780.0) < 1e-6 and inc is None


def test_endpoint_returns_data_url_from_imager():
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return GeneratedImage(PNG, "image/png", "fake-model")

    client = TestClient(create_app(start_worker=False, imager=fake))
    r = client.post("/v1/satellite-image", json={"event": EVENT, "orbit": {"alt_km": 780, "inc_deg": 86.4}})
    assert r.status_code == 200
    b = r.json()
    assert b["image"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert b["model"] == "fake-model" and b["prompt"] == prompts[0] and "polar" in prompts[0]


def test_endpoint_validates_and_reports_missing_key(monkeypatch):
    client = TestClient(create_app(start_worker=False, imager=lambda p: GeneratedImage(PNG, "image/png", "m")))
    r = client.post("/v1/satellite-image", json={"event": {"event_type": "collision"}, "orbit": {"alt_km": 1}})
    assert r.status_code == 422
    locs = {d["loc"] for d in r.json()["detail"]}
    assert "orbit.alt_km" in locs and any(loc.startswith("event.") for loc in locs)

    monkeypatch.setattr(imagine, "api_key", lambda: None)
    r = TestClient(create_app(start_worker=False)).post("/v1/satellite-image", json={"event": EVENT})
    assert r.status_code == 503 and "XAI_TOKEN" in r.json()["detail"]


def test_endpoint_maps_xai_failure_to_502():
    def failing(prompt):
        raise RuntimeError("xAI image API returned 401: bad key")

    r = TestClient(create_app(start_worker=False, imager=failing)).post("/v1/satellite-image", json={"event": EVENT})
    assert r.status_code == 502 and "401" in r.json()["detail"]


def test_xai_imager_request_decoding_and_cache():
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    imager = XaiImager("secret", "grok-imagine-image", http=httpx.Client(transport=httpx.MockTransport(handler)))
    first = imager("a satellite")
    again = imager("a satellite")
    assert first is again and len(calls) == 1
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert first.mime == "image/png" and first.data == PNG
