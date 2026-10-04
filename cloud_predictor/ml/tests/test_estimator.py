import copy
import json
import math
import statistics
import time

import pytest

from ml import BreakupParameterEstimator
from ml.contract import BOUNDS
from ml.data import build_training_rows, load_events, load_specs
from ml.schema import BAND_FIELDS, Spacecraft, event_from_dict, params_to_dict


def _dump(p):
    return json.dumps(params_to_dict(p), sort_keys=True)


@pytest.fixture(scope="module")
def training_events(synth_dir):
    rows, _ = build_training_rows(load_events(synth_dir / "historical_breakups.csv"),
                                  load_specs(synth_dir / "gunter_satellites.json"))
    return [r.event for r in rows]


def _check_bands(p):
    for name in BAND_FIELDS:
        b = getattr(p, name)
        lo, hi = BOUNDS[name]
        assert lo <= b.p10 <= b.p50 <= b.p90 <= hi, (name, b)


def test_manifest_ships_signal_heads_and_falls_back_on_noise(estimator):
    heads = estimator.manifest["heads"]
    assert heads["n_multiplier"]["status"] == "shipped"
    assert heads["dv_sigma_scale"]["status"] == "fallback"


def test_spec_example(estimator, spec_request):
    p = estimator.predict(event_from_dict(spec_request))
    _check_bands(p)
    assert p.emr_j_per_g == pytest.approx(333.3333333)
    assert p.is_catastrophic is True
    assert p.sbm_mass_param == 755.0
    assert p.model_version == "estimator-test"
    assert p.imputed_fields == ["target.solar_array_area_m2", "target.bus_volume_m3", "target.bus_family"]
    assert p.warnings == ["low_support_collision"]
    # dv_sigma_scale fell back, so this event reports fallback with a neutral centre.
    assert p.fallback is True
    assert p.dv_sigma_scale.p50 == 1.0
    assert p.dv_sigma_scale.p10 < 1.0 < p.dv_sigma_scale.p90
    # Synthetic truth for this event: k = exp(0.44) ~ 1.55, am shift = 0.23.
    assert 1.1 < p.n_multiplier.p50 < 2.1
    assert 0.05 < p.am_mu_shift.p50 < 0.4


def test_every_training_event_gives_ordered_bounded_bands(estimator, training_events):
    for p in estimator.predict_batch(training_events):
        _check_bands(p)


def test_no_construction_info_falls_back_to_neutral(estimator):
    event = event_from_dict({
        "event_type": "explosion",
        "epoch": "2001-05-01T00:00:00Z",
        "target": {"object_class": "rocket_body", "dry_mass_kg": 1400, "launch_year": 1990},
    })
    p = estimator.predict(event)
    assert p.fallback is True
    assert p.n_multiplier.p50 == 1.0
    assert p.am_mu_shift.p50 == 0.0
    assert p.am_sigma_scale.p50 == 1.0
    assert p.dv_mu_shift.p50 == 0.0
    assert p.emr_j_per_g is None and p.is_catastrophic is None and p.sbm_mass_param is None
    assert "target.structure_material" in p.imputed_fields


def test_warnings(estimator, spec_request):
    spec_request["target"]["dry_mass_kg"] = 1e6
    spec_request["target"]["launch_year"] = 2099
    p = estimator.predict(event_from_dict(spec_request))
    assert p.warnings == ["ood_mass", "ood_era", "low_support_collision"]

    explosion = event_from_dict({
        "event_type": "explosion", "epoch": "2001-05-01T00:00:00Z",
        "target": {"object_class": "payload", "dry_mass_kg": 900, "structure_material": "aluminum"},
    })
    assert estimator.predict(explosion).warnings == []


def test_am_shift_is_monotone_in_mli(estimator, spec_request):
    previous = None
    for i in range(21):
        spec_request["target"]["mli_fraction"] = i / 20
        b = estimator.predict(event_from_dict(spec_request)).am_mu_shift
        if previous is not None:
            assert b.p10 >= previous.p10 and b.p50 >= previous.p50 and b.p90 >= previous.p90
        previous = b


def test_output_is_byte_identical_across_calls_and_reloads(estimator, model_dir, spec_request):
    event = event_from_dict(spec_request)
    first = _dump(estimator.predict(event))
    assert _dump(estimator.predict(event)) == first
    assert _dump(BreakupParameterEstimator.load(str(model_dir)).predict(event)) == first


def test_batch_matches_single_predictions(estimator, training_events):
    events = training_events[:20]
    batch = estimator.predict_batch(events)
    assert [_dump(p) for p in batch] == [_dump(estimator.predict(e)) for e in events]
    assert estimator.predict_batch([]) == []


def test_single_prediction_latency(estimator, spec_request):
    event = event_from_dict(spec_request)
    estimator.predict(event)
    times = []
    for _ in range(50):
        t = time.perf_counter()
        estimator.predict(event)
        times.append(time.perf_counter() - t)
    assert statistics.median(times) < 0.010


def test_invalid_event_raises_value_error(estimator, spec_request):
    event = event_from_dict(spec_request)
    bad = copy.deepcopy(event)
    bad.impactor = None
    with pytest.raises(ValueError, match="impactor"):
        estimator.predict(bad)
    with pytest.raises(ValueError, match=r"\[1\]\.impactor"):
        estimator.predict_batch([event, bad])


def test_unseen_bus_family_is_handled(estimator, spec_request):
    spec_request["target"]["bus_family"] = "never-seen-bus"
    _check_bands(estimator.predict(event_from_dict(spec_request)))


def _gated(estimator, spread=(-1.5, 1.1)):
    manifest = copy.deepcopy(estimator.manifest)
    manifest["heads"]["n_multiplier"]["collision_gate"] = {
        "n_labeled": 4, "min_rows": 30, "use_model": False, "spread": list(spread)}
    return BreakupParameterEstimator(manifest, estimator.heads, estimator.lab_rules)


def test_collision_gate_uses_sbm_band_for_collisions_only(estimator, spec_request):
    gated = _gated(estimator)
    p = gated.predict(event_from_dict(spec_request))
    assert p.n_multiplier.p50 == 1.0
    assert p.n_multiplier.p10 == pytest.approx(math.exp(-1.5), abs=1e-6)
    assert p.n_multiplier.p90 == pytest.approx(math.exp(1.1), abs=1e-6)
    assert p.fallback is True
    assert p.warnings == ["low_support_collision", "collision_count_from_sbm"]
    # Other heads and explosions are untouched by the gate.
    ungated = estimator.predict(event_from_dict(spec_request))
    assert p.am_mu_shift == ungated.am_mu_shift
    explosion = event_from_dict({
        "event_type": "explosion", "epoch": "2001-05-01T00:00:00Z", "explosion_cause": "propulsion",
        "target": {"object_class": "rocket_body", "dry_mass_kg": 1400, "bus_family": "stage-1"},
    })
    assert _dump(gated.predict(explosion)) == _dump(estimator.predict(explosion))


def test_manifest_without_gate_uses_learned_head(estimator, spec_request):
    manifest = copy.deepcopy(estimator.manifest)
    manifest["heads"]["n_multiplier"].pop("collision_gate", None)
    old = BreakupParameterEstimator(manifest, estimator.heads, estimator.lab_rules)
    assert _dump(old.predict(event_from_dict(spec_request))) == _dump(estimator.predict(event_from_dict(spec_request)))


def test_load_rejects_other_feature_sets(estimator):
    manifest = copy.deepcopy(estimator.manifest)
    manifest["feature_columns"] = manifest["feature_columns"][:-1]
    with pytest.raises(ValueError, match="feature set"):
        BreakupParameterEstimator(manifest, estimator.heads, estimator.lab_rules)


def test_spacecraft_defaults_match_spec():
    s = Spacecraft("payload", 100.0)
    assert s.propellant_mass_kg == 0.0 and s.structure_material == "unknown"
