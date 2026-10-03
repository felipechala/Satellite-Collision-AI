import json

import pytest

from ml.schema import (
    Band,
    BreakupParameters,
    EventValidationError,
    event_from_dict,
    event_to_dict,
    params_from_dict,
    params_to_dict,
)


def test_spec_request_parses_and_round_trips(spec_request):
    event = event_from_dict(spec_request)
    assert event.target.structure_material == "cfrp"
    assert event.impactor.v_rel_km_s == 10
    assert event_from_dict(json.loads(json.dumps(event_to_dict(event)))) == event


def test_minimal_explosion_needs_four_fields():
    event = event_from_dict({
        "event_type": "explosion",
        "epoch": "2001-05-01T00:00:00Z",
        "target": {"object_class": "rocket_body", "dry_mass_kg": 1400},
    })
    assert event.target.propellant_mass_kg == 0.0
    assert event.target.structure_material == "unknown"


def test_null_defaulted_fields_use_defaults(spec_request):
    spec_request["target"]["propellant_mass_kg"] = None
    spec_request["target"]["structure_material"] = None
    event = event_from_dict(spec_request)
    assert event.target.propellant_mass_kg == 0.0
    assert event.target.structure_material == "unknown"


def _set(path, value):
    def mutate(d):
        *parents, leaf = path.split(".")
        for p in parents:
            d = d[p]
        if value is _DELETE:
            del d[leaf]
        else:
            d[leaf] = value
    return mutate


_DELETE = object()


@pytest.mark.parametrize("mutate, loc", [
    (_set("target.dry_mass_kg", 0), "target.dry_mass_kg"),
    (_set("target.dry_mass_kg", -5), "target.dry_mass_kg"),
    (_set("target.propellant_mass_kg", -1), "target.propellant_mass_kg"),
    (_set("impactor", None), "impactor"),
    (_set("impactor.v_rel_km_s", 0), "impactor.v_rel_km_s"),
    (_set("impactor.v_rel_km_s", 20.01), "impactor.v_rel_km_s"),
    (_set("impactor.mass_kg", 0), "impactor.mass_kg"),
    (_set("target.mli_fraction", 1.2), "target.mli_fraction"),
    (_set("target.mli_fraction", -0.1), "target.mli_fraction"),
    (_set("event_type", "implosion"), "event_type"),
    (_set("target.object_class", "station"), "target.object_class"),
    (_set("target.structure_material", "steel"), "target.structure_material"),
    (_set("impactor.structure_material", "steel"), "impactor.structure_material"),
    (_set("explosion_cause", "aliens"), "explosion_cause"),
    (_set("epoch", "yesterday"), "epoch"),
    (_set("target.dry_mass_kg", "heavy"), "target.dry_mass_kg"),
    (_set("target.dry_mass_kg", True), "target.dry_mass_kg"),
    (_set("target.launch_year", 2019.5), "target.launch_year"),
    (_set("target.dry_mass_kg", _DELETE), "target.dry_mass_kg"),
    (_set("target.colour", "red"), "target.colour"),
])
def test_invalid_events_are_rejected(spec_request, mutate, loc):
    mutate(spec_request)
    with pytest.raises(EventValidationError) as exc:
        event_from_dict(spec_request)
    assert loc in [e.loc for e in exc.value.errors]
    assert isinstance(exc.value, ValueError)


def test_explosion_with_impactor_is_rejected(spec_request):
    spec_request["event_type"] = "explosion"
    with pytest.raises(EventValidationError) as exc:
        event_from_dict(spec_request)
    assert [e.loc for e in exc.value.errors] == ["impactor"]


def test_v_rel_upper_bound_is_inclusive(spec_request):
    spec_request["impactor"]["v_rel_km_s"] = 20
    assert event_from_dict(spec_request).impactor.v_rel_km_s == 20


def test_body_must_be_object():
    with pytest.raises(EventValidationError) as exc:
        event_from_dict([1, 2])
    assert exc.value.errors[0].loc == "body"


def test_parameters_json_round_trip_is_lossless():
    b = Band(0.1234567, 0.5, 0.9)
    params = BreakupParameters(
        n_multiplier=Band(0.8, 1.3, 2.1), slope_delta=Band(0.0, 0.0, 0.0), am_mu_shift=b,
        am_sigma_scale=Band(0.9, 1.1, 1.3), dv_mu_shift=Band(-0.1, 0.0, 0.1), dv_sigma_scale=Band(0.9, 1.0, 1.1),
        emr_j_per_g=333.3333333333333, is_catastrophic=True, sbm_mass_param=755.0, model_version="v",
        imputed_fields=["target.bus_family"], fallback=True, warnings=["low_support_collision"],
    )
    assert params_from_dict(json.loads(json.dumps(params_to_dict(params)))) == params
