"""Build data/historical_breakups.csv from the DISCOS fragmentation events.

    python -m pipeline.build_events

Only structural breakups go into the training CSV. Every DISCOS event, kept or not, is listed with
the reason in data/reports/event_provenance.csv, together with the SATCAT cross-check count.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Optional

import pandas as pd

from . import config, fetch_discos, satcat

COLLISIONS_PATH = config.MANUAL / "collisions.csv"

EVENT_COLUMNS = [
    "event_id", "event_type", "epoch", "parent_norad_id", "n_cataloged", "explosion_cause",
    "impactor_mass_kg", "impactor_v_rel_km_s", "impactor_material",
]

PROPULSION = ("explosion", "propulsion")
BATTERY = ("explosion", "battery")
DELIBERATE = ("explosion", "deliberate")
UNKNOWN = ("explosion", "unknown")
COLLISION = ("collision", None)

# DISCOS eventType -> (event_type, explosion_cause). None marks events that are not structural
# breakups (debris shedding, re-entry break-up, coolant release, unconfirmed): they release a handful
# of pieces and would drag the SBM count correction far below what a breakup does.
# Kinetic ASAT intercepts are listed in data/manual/collisions.csv, which turns them into collisions;
# the remaining ASAT events are co-orbital interceptors that blew themselves up.
EVENT_TYPE_MAP: dict[str, Optional[tuple[str, Optional[str]]]] = {
    "Propulsion": PROPULSION,
    "Proton Ullage Motor": PROPULSION,
    "Ariane Upper Stage": PROPULSION,
    "Delta Upper Stage": PROPULSION,
    "Delta 4 Class": PROPULSION,
    "Zenit-2 Upper Stage": PROPULSION,
    "Briz-M": PROPULSION,
    "Tsyklon Upper Stage": PROPULSION,
    "Titan Transtage": PROPULSION,
    "CZ-6A Upper Stage": PROPULSION,
    "H-IIA Class": PROPULSION,
    "Scout Class": PROPULSION,
    "Cosmos-3 Class": PROPULSION,
    "Battery": BATTERY,
    "Electrical": BATTERY,
    "DMSP/NOAA Class": BATTERY,
    "Deliberate": DELIBERATE,
    "Cosmos 862 Class (Explosive Charge)": DELIBERATE,
    "Cosmos 2031 Class": DELIBERATE,
    "Payload Recovery Failure": DELIBERATE,
    "ASAT": DELIBERATE,
    "Unknown": UNKNOWN,
    "Cosmos 699 Class (EORSAT)": UNKNOWN,
    "Meteor Class": UNKNOWN,
    "Accidental": UNKNOWN,
    "Collision": COLLISION,
    "Small Impactor": COLLISION,
    "Anomalous": None,
    "Aerodynamics": None,
    "Unconfirmed": None,
    "RORSAT Reactor Core Ejection Class": None,
    "TOPAZ Leakage Class": None,
    "Transit Class": None,
    "ERS/SPOT Class": None,
    "Vostok Class": None,
    "L-14B Class": None,
}


def _num(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _str(v) -> Optional[str]:
    return v.strip() or None if isinstance(v, str) else None


def load_collisions() -> dict[str, dict]:
    df = pd.read_csv(COLLISIONS_PATH, dtype={"discos_event_id": str})
    return {r["discos_event_id"]: r for r in df.to_dict(orient="records")}


def _pick_target(objs: list[dict], target_satno: Optional[float]) -> dict:
    if target_satno is not None:
        for o in objs:
            if o.get("satno") == int(target_satno):
                return o
    return max(objs, key=lambda o: o.get("mass") or -1.0)


def build(discos: dict, debris_by_prefix: dict[str, int], collisions: dict[str, dict]
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (training events, provenance for every DISCOS event)."""
    objects = discos["objects"]
    events_per_object = Counter(oid for ev in discos["fragmentations"] for oid in ev["object_ids"])
    rows, prov = [], []
    unmapped = Counter()
    for ev in discos["fragmentations"]:
        objs = [objects[oid] | {"_id": oid} for oid in ev["object_ids"] if oid in objects]
        coll = collisions.get(str(ev["id"]))
        kind = COLLISION if coll is not None else EVENT_TYPE_MAP.get(ev["eventType"], "unmapped")
        p = {
            "event_id": f"discos-{ev['id']}",
            "epoch": ev["epoch"],
            "discos_event_type": ev["eventType"],
            "n_objects": len(objs),
            "collision_source": coll["source"] if coll is not None else None,
        }
        reason = None
        if kind == "unmapped":
            unmapped[ev["eventType"]] += 1
            reason = "unmapped DISCOS eventType"
        elif kind is None:
            reason = "not a structural breakup"
        elif not objs:
            reason = "no objects linked"

        target = _pick_target(objs, _num(coll.get("target_satno")) if coll else None) if objs else None
        if target is not None:
            p.update(parent_norad_id=target.get("satno"), parent_name=target.get("name"),
                     object_class=target.get("objectClass"))
            if reason is None and target.get("satno") is None:
                reason = "parent has no NORAD id"

        ambiguous = [o for o in objs if events_per_object[o["_id"]] > 1]
        if not objs:
            n_cat, method = None, None
        elif ambiguous:
            n_cat = None
            method = f"ambiguous: object in {max(events_per_object[o['_id']] for o in ambiguous)} DISCOS events"
        else:
            n_cat = int(sum(o.get("cataloguedFragments") or 0 for o in objs))
            method = "DISCOS cataloguedFragments" + (f" (sum of {len(objs)} objects)" if len(objs) > 1 else "")
        prefixes = {o["cosparId"][:8] for o in objs if o.get("cosparId")}
        p.update(n_cataloged=n_cat, count_method=method,
                 n_debris_satcat_prefix=sum(debris_by_prefix.get(x, 0) for x in prefixes) if prefixes else None)

        if reason is None:
            event_type, cause = kind
            row = {"event_id": p["event_id"], "event_type": event_type, "epoch": f"{ev['epoch']}T00:00:00Z",
                   "parent_norad_id": int(target["satno"]), "n_cataloged": n_cat, "explosion_cause": cause}
            if event_type == "collision":
                others = [o for o in objs if o is not target]
                mass = _num(coll.get("impactor_mass_kg")) if coll else None
                if mass is None and others:
                    mass = _num(others[0].get("mass"))
                row.update(impactor_mass_kg=mass,
                           impactor_v_rel_km_s=_num(coll.get("impactor_v_rel_km_s")) if coll else None,
                           impactor_material=_str(coll.get("impactor_material")) if coll else None)
                if others:
                    p["impactor_norad_id"] = others[0].get("satno")
            rows.append(row)
            p.update(event_type=event_type, explosion_cause=cause)
        p.update(included=reason is None, reason=reason)
        prov.append(p)

    for t, n in unmapped.items():
        print(f"  WARNING unmapped DISCOS eventType {t!r} ({n} events) - add it to EVENT_TYPE_MAP")
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for c in ("parent_norad_id", "n_cataloged"):
        events[c] = events[c].astype("Int64")
    return events, pd.DataFrame(prov)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.build_events", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    events, prov = build(fetch_discos.load(), satcat.debris_by_prefix(satcat.load()), load_collisions())
    events.to_csv(config.EVENTS_PATH, index=False)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    prov.to_csv(config.PROVENANCE_PATH, index=False)
    print(f"{len(events)} breakup events of {len(prov)} DISCOS events -> {config.EVENTS_PATH}")
    print(prov.loc[~prov["included"], "reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
