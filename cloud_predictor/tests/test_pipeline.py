import math

import pandas as pd
import pytest

from ml.data import build_training_rows
from pipeline import build_events, build_specs, fetch_gunter


def _obj(satno, mass=1000.0, frags=50, cospar=None, cls="Payload", name=None):
    return {"satno": satno, "mass": mass, "cataloguedFragments": frags, "objectClass": cls,
            "cosparId": cospar or f"2000-{satno:03d}A", "name": name or f"Sat {satno}"}


def _discos(fragmentations, objects):
    return {"fragmentations": fragmentations, "objects": objects}


def _ev(eid, etype, object_ids, epoch="2005-01-01"):
    return {"id": eid, "eventType": etype, "epoch": epoch, "object_ids": object_ids}


def test_event_type_mapping_and_exclusions():
    d = _discos(
        [_ev("1", "Propulsion", ["a"]), _ev("2", "Anomalous", ["b"]), _ev("3", "Battery", ["c"]),
         _ev("4", "Brand New Type", ["d"])],
        {k: _obj(i) for i, k in enumerate("abcd", start=1)},
    )
    events, prov = build_events.build(d, {}, {})
    assert list(events["event_id"]) == ["discos-1", "discos-3"]
    assert list(events["explosion_cause"]) == ["propulsion", "battery"]
    reasons = dict(zip(prov["event_id"], prov["reason"]))
    assert reasons["discos-2"] == "not a structural breakup"
    assert reasons["discos-4"] == "unmapped DISCOS eventType"


def test_count_is_ambiguous_when_parent_breaks_up_twice():
    d = _discos([_ev("1", "Propulsion", ["a"]), _ev("2", "Propulsion", ["a"]), _ev("3", "Unknown", ["b"])],
                {"a": _obj(1, frags=40), "b": _obj(2, frags=7)})
    events, prov = build_events.build(d, {}, {})
    counts = dict(zip(events["event_id"], events["n_cataloged"]))
    assert pd.isna(counts["discos-1"])
    assert counts["discos-3"] == 7
    assert prov.set_index("event_id").loc["discos-1", "count_method"].startswith("ambiguous")


def test_collision_sums_both_clouds_and_takes_impactor_from_discos():
    d = _discos([_ev("224", "Collision", ["c", "i"])],
                {"c": _obj(22675, mass=900, frags=1715), "i": _obj(24946, mass=662, frags=660)})
    coll = {"224": {"target_satno": 22675, "impactor_mass_kg": float("nan"), "impactor_v_rel_km_s": 11.7,
                    "impactor_material": float("nan"), "source": "test"}}
    events, _ = build_events.build(d, {}, coll)
    row = events.iloc[0]
    assert row["event_type"] == "collision"
    assert row["parent_norad_id"] == 22675
    assert row["n_cataloged"] == 2375
    assert row["impactor_mass_kg"] == 662
    assert row["impactor_v_rel_km_s"] == 11.7
    assert row["impactor_material"] is None


def test_collisions_csv_turns_asat_into_collision():
    d = _discos([_ev("182", "ASAT", ["f"]), _ev("71", "ASAT", ["g"])], {"f": _obj(25730), "g": _obj(3504)})
    coll = {"182": {"target_satno": float("nan"), "impactor_mass_kg": 600.0, "impactor_v_rel_km_s": 9.0,
                    "impactor_material": float("nan"), "source": "test"}}
    events, _ = build_events.build(d, {}, coll)
    types = dict(zip(events["event_id"], events["event_type"]))
    assert types == {"discos-182": "collision", "discos-71": "explosion"}


@pytest.mark.parametrize("shape,dims,expected", [
    ("Cyl", dict(height=2.0, diameter=1.0), math.pi * 0.25 * 2.0),
    ("Cyl + 2 Pan", dict(height=2.0, diameter=1.0), math.pi * 0.25 * 2.0),
    ("Box + 1 Pan", dict(width=1.0, height=2.0, depth=3.0), 6.0),
    ("Sphere", dict(diameter=2.0), math.pi * 8 / 6),
    ("Cone", dict(height=3.0, diameter=2.0), math.pi * 3.0 / 3),
    ("Sphere + Cyl", dict(height=2.0, diameter=1.0), math.pi * 0.25 * 2.0),
    ("Irr + 6 Pan", dict(height=2.0, diameter=1.0), None),
    ("Cyl", dict(height=None, diameter=1.0), None),
    (None, {}, None),
])
def test_bus_volume_by_shape(shape, dims, expected):
    kw = dict(width=None, height=None, depth=None, diameter=None) | dims
    got = build_specs.bus_volume_m3(shape, **kw)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_name_family_and_configuration():
    assert build_specs.name_family("Delta K (Delta 5920)", "rocket_body") == "Delta K"
    assert build_specs.name_family("L-14B (YF40B) (Long March)", "rocket_body") == "L-14B"
    assert build_specs.name_family("Transtage 24 (Titan IIIC)", "rocket_body") == "Transtage"
    assert build_specs.name_family("Cosmos-1408", "payload") == "Cosmos"
    assert build_specs.bus_from_configuration("Yantar Bus, main reentry module") == "Yantar"
    assert build_specs.bus_from_configuration("US-Bus") == "US"
    assert build_specs.bus_from_configuration("LM-700A") == "LM-700A"
    assert build_specs.bus_from_configuration("Octagonal body, dust shields") is None
    assert build_specs.gunter_mass_kg("954 kg (#1C, 1D); 750 kg (#1A)") == 954
    assert build_specs.gunter_mass_kg("~ 2,000 kg") == 2000


def test_spec_priority_manual_over_discos_over_gunter():
    d = _discos([], {"a": _obj(1, mass=None) | {"shape": "Cyl", "height": 2.0, "diameter": 1.0},
                     "b": _obj(2, mass=500.0)})
    gunter = {"1": {"slug": "x", "Mass": "700 kg", "Configuration": "LM-700A"}, "2": {"slug": "y", "Mass": "900 kg"}}
    overrides = {2: {"structure_material": "cfrp", "dry_mass_kg": 480.0}}
    specs, prov = build_specs.build({1, 2}, d, gunter, {1: 1990}, overrides)
    by_id = {s["norad_id"]: s for s in specs}
    assert by_id[1]["dry_mass_kg"] == 700.0  # DISCOS mass missing -> Gunter
    assert by_id[1]["bus_family"] == "LM-700A"
    assert by_id[1]["launch_year"] == 1990
    assert by_id[2]["dry_mass_kg"] == 480.0  # manual beats DISCOS
    assert by_id[2]["structure_material"] == "cfrp"
    assert by_id[2]["bus_family"] == "y"
    assert set(prov.set_index("norad_id")["mass_source"]) == {"Gunter", "manual"}


def test_built_files_load_through_ml_data_without_skips():
    d = _discos([_ev("1", "Propulsion", ["a"]), _ev("2", "Collision", ["b", "c"])],
                {"a": _obj(1, cls="Rocket Body"), "b": _obj(2, mass=900), "c": _obj(3, mass=500)})
    coll = {"2": {"target_satno": float("nan"), "impactor_mass_kg": float("nan"), "impactor_v_rel_km_s": 10.0,
                  "impactor_material": float("nan"), "source": "test"}}
    events, _ = build_events.build(d, {}, coll)
    specs, _ = build_specs.build(set(events["parent_norad_id"].astype(int)), d, {}, {}, {})
    rows, skipped = build_training_rows(events, {s["norad_id"]: s for s in specs})
    assert skipped == []
    assert {r.event.event_type for r in rows} == {"explosion", "collision"}


def test_gunter_payload_matching():
    one_type = [("Iridium 33", "iridium"), ("Iridium 34", "iridium")]
    assert fetch_gunter.match_payload("Iridium 33", one_type) == ("iridium", "only payload type on launch")
    mixed = [("Kosmos 1408 (Tselina-D #38)", "tselina-d"), ("Radio 7", "radio-sputnik")]
    slug, how = fetch_gunter.match_payload("Cosmos-1408", mixed)
    assert slug == "tselina-d" and how.startswith("name match")
    assert fetch_gunter.match_payload("X", [])[0] is None


def test_parse_chronology_and_spacecraft():
    chron = b"""<table><tr><th>ID</th></tr>
      <tr><td>1982-092</td><td>16.09.1982</td><td><a href="../doc_sdat/tselina-d.htm">Kosmos 1408</a></td></tr>
      <tr><td>January</td></tr></table>"""
    assert fetch_gunter.parse_chronology(chron) == {"1982-092": [("Kosmos 1408", "tselina-d")]}
    page = b"""<table id="satdata"><tr><th>Configuration:</th><td>LM-700A</td></tr>
      <tr><th>Mass:</th><td>689 kg</td></tr><tr><th>Equipment:</th><td>?</td></tr></table>"""
    assert fetch_gunter.parse_spacecraft(page) == {"Configuration": "LM-700A", "Mass": "689 kg"}
