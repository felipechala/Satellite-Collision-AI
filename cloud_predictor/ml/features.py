from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from . import sbm
from .schema import EVENT_TYPES, EXPLOSION_CAUSES, MATERIALS, OBJECT_CLASSES, BreakupEvent, parse_epoch

NUMERIC_COLUMNS = [
    "log10_dry_mass",
    "log10_total_mass",
    "propellant_fraction",
    "solar_array_area_m2",
    "solar_area_per_kg",
    "bus_volume_m3",
    "bulk_density_kg_m3",
    "mli_fraction",
    "launch_year",
    "epoch_year",
    "log10_emr",
    "is_catastrophic",
    "log10_v_rel",
    "log10_mass_ratio",
]

# "unknown" enum values are treated as missing, like nulls.
FIXED_CATEGORIES: dict[str, list[str]] = {
    "object_class": list(OBJECT_CLASSES),
    "event_type": list(EVENT_TYPES),
    "explosion_cause": [c for c in EXPLOSION_CAUSES if c != "unknown"],
    "structure_material": [m for m in MATERIALS if m != "unknown"],
    "impactor_material": [m for m in MATERIALS if m != "unknown"],
}
CATEGORICAL_COLUMNS = list(FIXED_CATEGORIES) + ["bus_family"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
CATEGORICAL_INDEX = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]

NULLABLE_TARGET_FIELDS = ("solar_array_area_m2", "bus_volume_m3", "mli_fraction", "launch_year", "bus_family")
CONSTRUCTION_FIELDS = ("solar_array_area_m2", "bus_volume_m3", "mli_fraction", "bus_family")

NAN = float("nan")


def fit_categories(events: Sequence[BreakupEvent]) -> dict[str, list[str]]:
    families = sorted({e.target.bus_family for e in events if e.target.bus_family is not None})
    return {**FIXED_CATEGORIES, "bus_family": families}


def _fractional_year(epoch: str) -> float:
    dt = parse_epoch(epoch)
    start = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


def _opt(v: Optional[float]) -> float:
    return NAN if v is None else float(v)


def _code(value: Optional[str], categories: list[str]) -> float:
    if value is None:
        return NAN
    try:
        return float(categories.index(value))
    except ValueError:
        return NAN


def feature_row(event: BreakupEvent, categories: dict[str, list[str]]) -> list[float]:
    t = event.target
    total = t.dry_mass_kg + t.propellant_mass_kg
    row = {
        "log10_dry_mass": math.log10(t.dry_mass_kg),
        "log10_total_mass": math.log10(total),
        "propellant_fraction": t.propellant_mass_kg / total,
        "solar_array_area_m2": _opt(t.solar_array_area_m2),
        "solar_area_per_kg": NAN if t.solar_array_area_m2 is None else t.solar_array_area_m2 / total,
        "bus_volume_m3": _opt(t.bus_volume_m3),
        "bulk_density_kg_m3": NAN if t.bus_volume_m3 is None else total / t.bus_volume_m3,
        "mli_fraction": _opt(t.mli_fraction),
        "launch_year": _opt(t.launch_year),
        "epoch_year": _fractional_year(event.epoch),
        "log10_emr": NAN,
        "is_catastrophic": NAN,
        "log10_v_rel": NAN,
        "log10_mass_ratio": NAN,
    }
    imp = event.impactor
    if event.event_type == "collision":
        d = sbm.derive(event)
        row["log10_emr"] = math.log10(d.emr_j_per_g)
        row["is_catastrophic"] = float(d.is_catastrophic)
        row["log10_v_rel"] = math.log10(imp.v_rel_km_s)
        row["log10_mass_ratio"] = math.log10(imp.mass_kg / total)

    cause = event.explosion_cause if event.event_type == "explosion" else None
    cats = {
        "object_class": t.object_class,
        "event_type": event.event_type,
        "explosion_cause": cause,
        "structure_material": t.structure_material,
        "impactor_material": imp.structure_material if imp is not None else None,
        "bus_family": t.bus_family,
    }
    return [row[c] for c in NUMERIC_COLUMNS] + [_code(cats[c], categories[c]) for c in CATEGORICAL_COLUMNS]


def build_features(events: Sequence[BreakupEvent], categories: dict[str, list[str]]) -> np.ndarray:
    return np.array([feature_row(e, categories) for e in events], dtype=np.float64).reshape(
        len(events), len(FEATURE_COLUMNS)
    )


def imputed_fields(event: BreakupEvent) -> list[str]:
    t = event.target
    out = ["target.structure_material"] if t.structure_material == "unknown" else []
    return out + [f"target.{f}" for f in NULLABLE_TARGET_FIELDS if getattr(t, f) is None]


def has_construction_info(event: BreakupEvent) -> bool:
    t = event.target
    return t.structure_material != "unknown" or any(getattr(t, f) is not None for f in CONSTRUCTION_FIELDS)
