"""Loaders for the training inputs.

Specs JSON: a list of parent records with norad_id, object_class, dry_mass_kg and optionally
propellant_mass_kg, structure_material, solar_array_area_m2, bus_volume_m3, mli_fraction,
launch_year, bus_family.

Events CSV, one row per fragmentation event: event_id, event_type, epoch, parent_norad_id,
n_cataloged (pieces >= 10 cm ever cataloged, decayed included), and optionally explosion_cause,
propellant_mass_kg (residual at breakup; overrides the spec), impactor_mass_kg,
impactor_v_rel_km_s, impactor_material, group (overrides the CV group).

Fragments CSV, one row per tracked fragment: event_id, lc_m, and nullable am_m2_kg, dv_m_s,
tle_span_days.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .schema import BreakupEvent, FieldError, Impactor, Spacecraft, validate

EVENT_COLUMNS = ["event_id", "event_type", "epoch", "parent_norad_id", "n_cataloged"]
FRAGMENT_COLUMNS = ["event_id", "lc_m"]
OPTIONAL_FRAGMENT_COLUMNS = ["am_m2_kg", "dv_m_s", "tle_span_days"]
SPEC_FIELDS = {f.name for f in fields(Spacecraft)}


@dataclass
class TrainingRow:
    event_id: str
    event: BreakupEvent
    n_cataloged: Optional[float]
    group: str


def _opt(v: Any) -> Any:
    """CSV/JSON missing values (None, NaN, NA, blank string) -> None."""
    if isinstance(v, str):
        return v if v.strip() else None
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _num(v: Any) -> Optional[float]:
    v = _opt(v)
    return None if v is None else float(v)


def _int(v: Any) -> Optional[int]:
    v = _num(v)
    return None if v is None else int(v)


def load_specs(path: str | Path) -> dict[int, dict]:
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path} is empty")
    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of spacecraft records")
    specs = {}
    for i, rec in enumerate(records):
        unknown = set(rec) - SPEC_FIELDS
        if unknown:
            raise ValueError(f"{path}[{i}]: unknown fields {sorted(unknown)}")
        for req in ("norad_id", "object_class", "dry_mass_kg"):
            if rec.get(req) is None:
                raise ValueError(f"{path}[{i}]: {req} is required")
        specs[int(rec["norad_id"])] = rec
    return specs


def _read_csv(path: str | Path, required: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"event_id": str})
    except pd.errors.EmptyDataError:
        raise ValueError(f"{path} is empty") from None
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def load_events(path: str | Path) -> pd.DataFrame:
    return _read_csv(path, EVENT_COLUMNS)


def load_fragments(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path, FRAGMENT_COLUMNS)
    for c in OPTIONAL_FRAGMENT_COLUMNS:
        if c not in df.columns:
            df[c] = float("nan")
    return df


def _spacecraft(spec: dict, propellant_override: Optional[float]) -> Spacecraft:
    kw = {k: _opt(v) for k, v in spec.items()}
    kw["propellant_mass_kg"] = (
        propellant_override if propellant_override is not None else (kw.get("propellant_mass_kg") or 0.0)
    )
    kw["structure_material"] = kw.get("structure_material") or "unknown"
    return Spacecraft(**kw)


def build_training_rows(events: pd.DataFrame, specs: dict[int, dict]
                        ) -> tuple[list[TrainingRow], list[tuple[str, list[FieldError]]]]:
    """Join events to parent specs. Returns valid rows and (event_id, errors) for skipped ones."""
    rows: list[TrainingRow] = []
    skipped: list[tuple[str, list[FieldError]]] = []
    for rec in events.to_dict(orient="records"):
        event_id = str(rec["event_id"])
        norad = _int(rec.get("parent_norad_id"))
        if norad is None or norad not in specs:
            skipped.append((event_id, [FieldError("parent_norad_id", "no matching spec record")]))
            continue
        target = _spacecraft(specs[norad], _num(rec.get("propellant_mass_kg")))
        impactor = None
        if _opt(rec.get("impactor_mass_kg")) is not None or _opt(rec.get("impactor_v_rel_km_s")) is not None:
            impactor = Impactor(
                mass_kg=_num(rec.get("impactor_mass_kg")),
                v_rel_km_s=_num(rec.get("impactor_v_rel_km_s")),
                structure_material=_opt(rec.get("impactor_material")) or "unknown",
            )
        event = BreakupEvent(
            event_type=_opt(rec.get("event_type")),
            epoch=_opt(rec.get("epoch")),
            target=target,
            impactor=impactor,
            explosion_cause=_opt(rec.get("explosion_cause")),
        )
        errors = validate(event)
        if errors:
            skipped.append((event_id, errors))
            continue
        group = _opt(rec.get("group")) or target.bus_family or f"event:{event_id}"
        rows.append(TrainingRow(event_id, event, _num(rec.get("n_cataloged")), str(group)))
    return rows, skipped
