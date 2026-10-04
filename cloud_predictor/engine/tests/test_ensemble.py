import json

import numpy as np
import pytest

from engine.density_grid import gaussian_smooth, smoothed_at, voxelize
from engine.ensemble import _frame, demo_event, main, simulate
from engine.propagator import circular_state
from ml.contract import NEUTRAL
from ml.schema import BAND_FIELDS, Band, BreakupParameters

TIMES = [0.0, 3600.0]


@pytest.fixture(scope="module")
def parent_state():
    return circular_state(alt_km=780.0, inc_deg=86.4)


def _params(n_band: Band) -> BreakupParameters:
    bands = {name: Band(NEUTRAL[name], NEUTRAL[name], NEUTRAL[name]) for name in BAND_FIELDS}
    bands["n_multiplier"] = n_band
    return BreakupParameters(**bands, emr_j_per_g=None, is_catastrophic=None, sbm_mass_param=None,
                             model_version="test")


def _random_grid(rng, n=200):
    r = rng.normal(0, 60.0, size=(n, 3)) + 7000.0
    return voxelize(r, rng.uniform(0.5, 2.0, n), voxel_km=20.0)


def test_smoothed_at_matches_full_smoothing():
    rng = np.random.default_rng(1)
    grid = _random_grid(rng)
    full = gaussian_smooth(grid, sigma_voxels=1.0)
    pick = rng.choice(full["indices"].shape[0], 300, replace=False)
    assert np.allclose(smoothed_at(grid, full["indices"][pick], 1.0), full["density"][pick])
    far = full["indices"][:5] + 1000
    assert np.all(smoothed_at(grid, far, 1.0) == 0.0)
    assert np.allclose(smoothed_at(grid, grid["indices"], 0.0), grid["density"])


def test_frame_mean_is_mean_of_runs_and_percentiles_ordered():
    rng = np.random.default_rng(2)
    grids = [_random_grid(rng) for _ in range(5)]
    f = _frame(grids, 5, 20.0, 1.0, max_voxels=400)
    sel = np.floor(f["xyz_km"] / 20.0).astype(np.int64)
    per_run = np.stack([smoothed_at(g, sel, 1.0) for g in grids], axis=1)
    assert np.allclose(per_run.mean(axis=1), f["mean"])
    assert np.all(f["p10"] <= f["p50"]) and np.all(f["p50"] <= f["p90"])
    assert np.all(np.diff(f["mean"]) <= 0)  # densest first


def test_mass_is_conserved_and_deterministic(parent_state):
    r0, v0 = parent_state
    kw = dict(n_runs=3, k=600, seed=7, max_voxels=10**7)
    a = simulate(demo_event(), r0, v0, TIMES, None, **kw)
    b = simulate(demo_event(), r0, v0, TIMES, None, **kw)
    for fa, fb in zip(a["frames"], b["frames"]):
        assert np.array_equal(fa["mean"], fb["mean"]) and np.array_equal(fa["p90"], fb["p90"])
    in_orbit = np.array([r["fragments_in_orbit"] for r in a["runs"]]).mean(axis=0)
    for j, f in enumerate(a["frames"]):
        assert f["mean"].sum() * 20.0**3 == pytest.approx(in_orbit[j], rel=1e-9)
    assert all(r["params"]["n_multiplier"] == 1.0 for r in a["runs"])


def test_wider_count_band_gives_wider_density_spread(parent_state):
    r0, v0 = parent_state
    kw = dict(n_runs=12, k=600, seed=3)
    narrow = simulate(demo_event(), r0, v0, [0.0], _params(Band(0.9, 1.0, 1.1)), **kw)["frames"][0]
    wide = simulate(demo_event(), r0, v0, [0.0], _params(Band(0.2, 1.0, 5.0)), **kw)["frames"][0]

    def rel_spread(f):
        return (f["p90"][0] - f["p10"][0]) / f["mean"][0]

    assert rel_spread(wide) > 3 * rel_spread(narrow)


def test_cli_writes_frontend_json(tmp_path):
    out = tmp_path / "cloud.json"
    main(["--out", str(out), "--runs", "2", "--particles", "400", "--hours", "2", "--step-hours", "1"])
    data = json.loads(out.read_text())
    assert data["params"] is None and len(data["runs"]) == 2
    assert [f["t_s"] for f in data["frames"]] == [0.0, 3600.0, 7200.0]
    v = data["frames"][1]["voxels"][0]
    assert set(v) == {"x", "y", "z", "density_mean", "density_p10", "density_p50", "density_p90"}
    assert v["density_p10"] <= v["density_p50"] <= v["density_p90"]
