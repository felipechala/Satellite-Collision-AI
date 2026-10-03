"""Build data/gunter_satellites.json: one spec record per breakup parent in historical_breakups.csv.

    python -m pipeline.build_specs

Field priority: data/manual/spec_overrides.csv > DISCOS (mass, dimensions, object class) > Gunter
(bus, mass fallback) > name heuristics (bus_family only). Where each field came from is written to
data/reports/spec_provenance.csv. DISCOS `mass` is used as dry mass: propellant is unknown for almost
every parent, and most breakups happen long after the propellant was spent.
"""
from __future__ import annotations

import argparse
import math
import re
from typing import Optional

import pandas as pd

from . import config, fetch_discos, fetch_gunter, satcat
from ml.data import SPEC_FIELDS

OVERRIDES_PATH = config.MANUAL / "spec_overrides.csv"
SPEC_PROVENANCE_PATH = config.REPORTS / "spec_provenance.csv"

OBJECT_CLASS_MAP = {
    "Payload": "payload",
    "Payload Mission Related Object": "payload",
    "Rocket Body": "rocket_body",
    "Rocket Mission Related Object": "rocket_body",
    "Payload Fragmentation Debris": "debris",
    "Payload Debris": "debris",
    "Rocket Fragmentation Debris": "debris",
    "Rocket Debris": "debris",
    "Other Mission Related Object": "debris",
    "Other Debris": "debris",
}
NUMERIC_OVERRIDES = {"dry_mass_kg", "propellant_mass_kg", "solar_array_area_m2", "bus_volume_m3", "mli_fraction"}
_APPENDAGE = re.compile(r"^\d+\s+\w+$")  # "2 Pan", "1 Nozzle", "4 Ant"


def bus_volume_m3(shape: Optional[str], width: Optional[float], height: Optional[float],
                  depth: Optional[float], diameter: Optional[float]) -> Optional[float]:
    """Main-body volume from a DISCOS shape string; appendages (panels, nozzles, antennas) are ignored.

    Box -> w*h*d; Sphere -> pi*d^3/6; Cone -> pi*r^2*h/3; any other mix of round bodies (Cyl, Oct Cyl,
    Sphere + Cyl, Cone + Cyl, ...) -> the bounding cylinder pi*r^2*h. Irregular shapes give None.
    """
    if not shape:
        return None
    parts = [p.strip() for p in shape.split("+")]
    body = {p.split()[-1] for p in parts if p and not _APPENDAGE.match(p)}
    if not body or body & {"Irr", "Torus"}:
        return None
    if "Box" in body:
        return width * height * depth if width and height and depth else None
    if body == {"Sphere"}:
        return math.pi * diameter ** 3 / 6 if diameter else None
    if not (diameter and height):
        return None
    cylinder = math.pi * (diameter / 2) ** 2 * height
    return cylinder / 3 if body == {"Cone"} else cylinder


def name_family(name: str, object_class: str) -> str:
    """Fallback bus family from an object name: 'Delta K (Delta 5920)' -> 'Delta K', 'NOAA 11' -> 'NOAA'."""
    stem = name.split(" (")[0].strip()
    # Payload serials are often hyphenated ("Cosmos-1408"); stage names keep theirs ("L-14B", "Blok-DM-2").
    pattern = r"[\s\-]+#?[^\s\-]*\d[^\s\-]*$" if object_class == "payload" else r"\s+#?\d+$"
    return re.sub(pattern, "", stem) or stem


_NOT_A_BUS = re.compile(r"sphere|balloon|prism|octagonal|hexag|body|orbiter|stage", re.IGNORECASE)


def bus_from_configuration(text: Optional[str]) -> Optional[str]:
    """Bus name from a Gunter Configuration line: 'Yantar Bus, main reentry module' -> 'Yantar'.

    None when the line describes a shape ('Octagonal body, dust shields') rather than a bus.
    """
    if not text:
        return None
    head = re.split(r",| with ", text)[0]
    head = re.sub(r"[\s\-]*\bbus\b.*$", "", head, flags=re.IGNORECASE).strip(" ?")
    return None if not head or _NOT_A_BUS.search(head) else head


def gunter_mass_kg(text: Optional[str]) -> Optional[float]:
    """First 'NNN kg' figure on a Gunter Mass line ('~ 2000 kg', '954 kg (#1C, 1D)')."""
    if not text:
        return None
    m = re.search(r"([\d][\d,.]*)\s*kg", text)
    return float(m.group(1).replace(",", "")) if m else None


def load_overrides() -> dict[int, dict]:
    if not OVERRIDES_PATH.exists():
        return {}
    df = pd.read_csv(OVERRIDES_PATH, comment="#", dtype=str)
    out: dict[int, dict] = {}
    for r in df.to_dict(orient="records"):
        field = r["field"].strip()
        if field not in SPEC_FIELDS or field == "norad_id":
            raise ValueError(f"{OVERRIDES_PATH}: unknown field {field!r}")
        value = r["value"].strip()
        if field in NUMERIC_OVERRIDES:
            value = float(value)
        elif field == "launch_year":
            value = int(value)
        out.setdefault(int(r["norad_id"]), {})[field] = value
    return out


def build(parent_ids: set[int], discos: dict, gunter: dict[str, dict], launch_years: dict[int, int],
          overrides: dict[int, dict]) -> tuple[list[dict], pd.DataFrame]:
    """Returns (spec records, per-parent provenance including dropped parents)."""
    by_satno = {o["satno"]: o for o in discos["objects"].values() if o.get("satno") is not None}
    specs, prov = [], []
    for norad in sorted(parent_ids):
        o = by_satno.get(norad, {})
        g = gunter.get(str(norad), {})
        ov = overrides.get(norad, {})
        object_class = OBJECT_CLASS_MAP.get(o.get("objectClass"))
        p = {"norad_id": norad, "name": o.get("name"), "discos_object_class": o.get("objectClass"),
             "gunter_match": g.get("match"), "gunter_slug": g.get("slug")}

        mass, mass_src = o.get("mass"), "DISCOS"
        if mass is None and gunter_mass_kg(g.get("Mass")) is not None:
            mass, mass_src = gunter_mass_kg(g.get("Mass")), "Gunter"
        volume = bus_volume_m3(o.get("shape"), o.get("width"), o.get("height"), o.get("depth"), o.get("diameter"))
        if bus_from_configuration(g.get("Configuration")):
            family, family_src = bus_from_configuration(g["Configuration"]), "Gunter configuration"
        elif g.get("slug"):
            family, family_src = g["slug"], "Gunter page"
        elif o.get("name") and object_class:
            family, family_src = name_family(o["name"], object_class), "name"
        else:
            family, family_src = None, None

        rec = {
            "norad_id": norad,
            "object_class": object_class,
            "dry_mass_kg": round(mass, 3) if mass is not None else None,
            "bus_volume_m3": round(volume, 4) if volume is not None else None,
            "launch_year": launch_years.get(norad) or (int(o["firstEpoch"][:4]) if o.get("firstEpoch") else None),
            "bus_family": family,
        }
        for field, value in ov.items():
            rec[field] = value
        if "dry_mass_kg" in ov:
            mass_src = "manual"
        if "bus_family" in ov:
            family_src = "manual"
        rec = {k: v for k, v in rec.items() if v is not None}

        dropped = None
        if "object_class" not in rec:
            dropped = f"unmapped DISCOS objectClass {o.get('objectClass')!r}" if o else "no DISCOS object"
        elif "dry_mass_kg" not in rec or rec["dry_mass_kg"] <= 0:
            dropped = "no mass from any source"
        p.update(mass_source=mass_src if "dry_mass_kg" in rec else None,
                 volume_source="DISCOS shape" if volume is not None and "bus_volume_m3" not in ov else
                 ("manual" if "bus_volume_m3" in ov else None),
                 bus_family=rec.get("bus_family"), bus_family_source=family_src,
                 overrides=",".join(sorted(ov)) or None, dropped=dropped)
        prov.append(p)
        if dropped is None:
            specs.append(rec)
    return specs, pd.DataFrame(prov)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pipeline.build_specs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    events = pd.read_csv(config.EVENTS_PATH)
    parents = set(events["parent_norad_id"].dropna().astype(int))
    specs, prov = build(parents, fetch_discos.load(), fetch_gunter.load(), satcat.launch_years(satcat.load()),
                        load_overrides())
    config.write_json(config.SPECS_PATH, specs)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    prov.to_csv(SPEC_PROVENANCE_PATH, index=False)
    print(f"{len(specs)} spec records for {len(parents)} parents -> {config.SPECS_PATH}")
    if prov["dropped"].notna().any():
        print(prov.loc[prov["dropped"].notna(), ["norad_id", "name", "dropped"]].to_string(index=False))


if __name__ == "__main__":
    main()
