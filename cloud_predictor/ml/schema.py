from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Literal, Optional, get_args

Material = Literal["aluminum", "cfrp", "mixed", "unknown"]
ObjectClass = Literal["payload", "rocket_body", "debris"]
EventType = Literal["explosion", "collision"]
ExplosionCause = Literal["propulsion", "battery", "deliberate", "unknown"]

MATERIALS: tuple[str, ...] = get_args(Material)
OBJECT_CLASSES: tuple[str, ...] = get_args(ObjectClass)
EVENT_TYPES: tuple[str, ...] = get_args(EventType)
EXPLOSION_CAUSES: tuple[str, ...] = get_args(ExplosionCause)

MAX_V_REL_KM_S = 20.0


@dataclass
class Spacecraft:
    object_class: ObjectClass
    dry_mass_kg: float
    propellant_mass_kg: float = 0.0
    structure_material: Material = "unknown"
    solar_array_area_m2: Optional[float] = None
    bus_volume_m3: Optional[float] = None
    mli_fraction: Optional[float] = None
    launch_year: Optional[int] = None
    bus_family: Optional[str] = None
    norad_id: Optional[int] = None


@dataclass
class Impactor:
    mass_kg: float
    v_rel_km_s: float
    structure_material: Material = "unknown"


@dataclass
class BreakupEvent:
    event_type: EventType
    epoch: str  # ISO 8601 UTC
    target: Spacecraft
    impactor: Optional[Impactor] = None  # required iff collision
    explosion_cause: Optional[ExplosionCause] = None


@dataclass
class Band:
    p10: float
    p50: float
    p90: float


@dataclass
class BreakupParameters:
    n_multiplier: Band
    slope_delta: Band
    am_mu_shift: Band
    am_sigma_scale: Band
    dv_mu_shift: Band
    dv_sigma_scale: Band
    emr_j_per_g: Optional[float]
    is_catastrophic: Optional[bool]
    sbm_mass_param: Optional[float]
    model_version: str
    imputed_fields: list[str] = field(default_factory=list)
    fallback: bool = False
    warnings: list[str] = field(default_factory=list)


BAND_FIELDS: tuple[str, ...] = (
    "n_multiplier",
    "slope_delta",
    "am_mu_shift",
    "am_sigma_scale",
    "dv_mu_shift",
    "dv_sigma_scale",
)


@dataclass(frozen=True)
class FieldError:
    loc: str
    msg: str


class EventValidationError(ValueError):
    def __init__(self, errors: list[FieldError]):
        self.errors = errors
        super().__init__("; ".join(f"{e.loc}: {e.msg}" for e in errors))


def parse_epoch(epoch: str) -> datetime:
    dt = datetime.fromisoformat(epoch)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_number(errors: list[FieldError], loc: str, v: Any, *, positive: bool = False,
                  non_negative: bool = False) -> bool:
    if not _is_number(v) or not math.isfinite(v):
        errors.append(FieldError(loc, "must be a finite number"))
        return False
    if positive and v <= 0:
        errors.append(FieldError(loc, "must be > 0"))
        return False
    if non_negative and v < 0:
        errors.append(FieldError(loc, "must be >= 0"))
        return False
    return True


def _check_enum(errors: list[FieldError], loc: str, v: Any, allowed: tuple[str, ...]) -> None:
    if v not in allowed:
        errors.append(FieldError(loc, f"must be one of {list(allowed)}"))


def _check_int(errors: list[FieldError], loc: str, v: Any) -> None:
    if not isinstance(v, int) or isinstance(v, bool):
        errors.append(FieldError(loc, "must be an integer"))


def validate(event: BreakupEvent) -> list[FieldError]:
    errors: list[FieldError] = []

    _check_enum(errors, "event_type", event.event_type, EVENT_TYPES)
    if not isinstance(event.epoch, str):
        errors.append(FieldError("epoch", "must be an ISO 8601 string"))
    else:
        try:
            parse_epoch(event.epoch)
        except ValueError:
            errors.append(FieldError("epoch", "must be an ISO 8601 string"))
    if event.explosion_cause is not None:
        _check_enum(errors, "explosion_cause", event.explosion_cause, EXPLOSION_CAUSES)

    t = event.target
    if not isinstance(t, Spacecraft):
        errors.append(FieldError("target", "is required"))
    else:
        _check_enum(errors, "target.object_class", t.object_class, OBJECT_CLASSES)
        _check_number(errors, "target.dry_mass_kg", t.dry_mass_kg, positive=True)
        _check_number(errors, "target.propellant_mass_kg", t.propellant_mass_kg, non_negative=True)
        _check_enum(errors, "target.structure_material", t.structure_material, MATERIALS)
        for name in ("solar_array_area_m2", "bus_volume_m3"):
            v = getattr(t, name)
            if v is not None:
                _check_number(errors, f"target.{name}", v, non_negative=True)
        if t.bus_volume_m3 is not None and _is_number(t.bus_volume_m3) and t.bus_volume_m3 == 0:
            errors.append(FieldError("target.bus_volume_m3", "must be > 0"))
        if t.mli_fraction is not None and _check_number(errors, "target.mli_fraction", t.mli_fraction):
            if not 0.0 <= t.mli_fraction <= 1.0:
                errors.append(FieldError("target.mli_fraction", "must be in [0, 1]"))
        if t.launch_year is not None:
            _check_int(errors, "target.launch_year", t.launch_year)
        if t.norad_id is not None:
            _check_int(errors, "target.norad_id", t.norad_id)
        if t.bus_family is not None and not isinstance(t.bus_family, str):
            errors.append(FieldError("target.bus_family", "must be a string"))

    imp = event.impactor
    if event.event_type == "collision" and imp is None:
        errors.append(FieldError("impactor", "is required for collisions"))
    if event.event_type == "explosion" and imp is not None:
        errors.append(FieldError("impactor", "must be null for explosions"))
    if imp is not None:
        _check_number(errors, "impactor.mass_kg", imp.mass_kg, positive=True)
        if _check_number(errors, "impactor.v_rel_km_s", imp.v_rel_km_s):
            if not 0.0 < imp.v_rel_km_s <= MAX_V_REL_KM_S:
                errors.append(FieldError("impactor.v_rel_km_s", f"must be in (0, {MAX_V_REL_KM_S:g}]"))
        _check_enum(errors, "impactor.structure_material", imp.structure_material, MATERIALS)

    return errors


# JSON null on these defaulted fields means "use the default".
_NULL_TO_DEFAULT = {"propellant_mass_kg", "structure_material"}


def _required(cls: type) -> set[str]:
    return {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}


def _from_dict(cls: type, data: Any, loc: str, errors: list[FieldError]) -> Optional[dict]:
    if not isinstance(data, dict):
        errors.append(FieldError(loc or "body", "must be an object"))
        return None
    prefix = f"{loc}." if loc else ""
    names = {f.name for f in fields(cls)}
    n_before = len(errors)
    for key in data:
        if key not in names:
            errors.append(FieldError(f"{prefix}{key}", "unknown field"))
    for name in sorted(_required(cls) - data.keys()):
        errors.append(FieldError(f"{prefix}{name}", "is required"))
    if len(errors) > n_before:
        return None
    return {k: v for k, v in data.items() if v is not None or k not in _NULL_TO_DEFAULT}


def event_from_dict(data: Any) -> BreakupEvent:
    """Parse and validate a JSON-shaped dict; raises EventValidationError."""
    errors: list[FieldError] = []
    kwargs = _from_dict(BreakupEvent, data, "", errors)
    if kwargs is None:
        raise EventValidationError(errors)

    tk = _from_dict(Spacecraft, kwargs["target"], "target", errors)
    ik = None
    if kwargs.get("impactor") is not None:
        ik = _from_dict(Impactor, kwargs["impactor"], "impactor", errors)
    if errors:
        raise EventValidationError(errors)

    event = BreakupEvent(
        event_type=kwargs["event_type"],
        epoch=kwargs["epoch"],
        target=Spacecraft(**tk),
        impactor=Impactor(**ik) if ik is not None else None,
        explosion_cause=kwargs.get("explosion_cause"),
    )
    errors = validate(event)
    if errors:
        raise EventValidationError(errors)
    return event


def event_to_dict(event: BreakupEvent) -> dict:
    return asdict(event)


def params_to_dict(params: BreakupParameters) -> dict:
    return asdict(params)


def params_from_dict(data: dict) -> BreakupParameters:
    kwargs = dict(data)
    for name in BAND_FIELDS:
        kwargs[name] = Band(**data[name])
    kwargs["imputed_fields"] = list(data.get("imputed_fields", []))
    kwargs["warnings"] = list(data.get("warnings", []))
    return BreakupParameters(**kwargs)
