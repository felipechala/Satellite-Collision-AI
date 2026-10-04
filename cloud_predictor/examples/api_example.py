"""Minimal example: model a satellite breakup's debris cloud with the API.
Start the API first (from cloud_predictor/):
    uvicorn app:app --port 8000
Then run:
    python examples/api_example.py
"""
import httpx

API = "http://127.0.0.1:8000"

request = {
    # The breakup: a 1,000 kg carbon-fiber satellite hit by a 50 kg object at 10 km/s.
    "event": {
        "event_type": "collision",
        "epoch": "2026-10-04T12:00:00Z",
        "target": {"object_class": "payload", "dry_mass_kg": 950, "propellant_mass_kg": 50,
                   "structure_material": "cfrp", "mli_fraction": 0.5},
        "impactor": {"mass_kg": 50, "v_rel_km_s": 10},
    },
    # Where it happened: a circular orbit at 780 km, inclined 86.4 degrees.
    "orbit": {"alt_km": 780, "inc_deg": 86.4},
    # Snapshots at 0, 6 and 48 hours after the breakup.
    "times_s": [0, 6 * 3600, 48 * 3600],
}

result = httpx.post(f"{API}/v1/debris-cloud", json=request, timeout=120).json()

print(f"Model: {result['model_version']}, catastrophic: {result['derived']['is_catastrophic']}")
print(f"Fragments 2 mm - 10 cm: {result['fragments_total']:,}")
for frame in result["frames"]:
    x, y, z, density = frame["voxels"][0][:4]  # densest 20 km cube, ECI km
    print(f"t = {frame['t_s'] / 3600:4.0f} h: {frame['in_orbit_fraction']:.0%} still in orbit, "
          f"densest cube at ({x:,.0f}, {y:,.0f}, {z:,.0f}) km holds {density:.3g} fragments/km^3")
