from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import sbm
from .contract import BOUNDS, LOG_SPACE
from .features import FEATURE_COLUMNS, build_features, has_construction_info, imputed_fields
from .heads import MONOTONE, QuantileHead, apply_offset
from .lab_rules import LabRules, combine
from .labels import LEARNED_HEADS
from .schema import (
    BAND_FIELDS,
    Band,
    BreakupEvent,
    BreakupParameters,
    EventValidationError,
    FieldError,
    validate,
)

MANIFEST_FORMAT = 1
OUTPUT_DECIMALS = 6


def _finalize(name: str, band: Band) -> Band:
    lo, hi = BOUNDS[name]
    values = sorted((band.p10, band.p50, band.p90))
    return Band(*(round(min(max(v, lo), hi), OUTPUT_DECIMALS) for v in values))


class BreakupParameterEstimator:
    def __init__(self, manifest: dict, heads: dict[str, QuantileHead], lab_rules: LabRules):
        if manifest.get("format_version") != MANIFEST_FORMAT:
            raise ValueError(f"unsupported manifest format {manifest.get('format_version')!r}")
        if manifest["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("model was trained with a different feature set; retrain it")
        self.manifest = manifest
        self.heads = heads
        self.lab_rules = lab_rules
        self.model_version: str = manifest["model_version"]
        self.categories: dict[str, list[str]] = manifest["categories"]

    @classmethod
    def load(cls, path: str) -> "BreakupParameterEstimator":
        d = Path(path)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        heads = {
            name: QuantileHead.load(d / "heads", name, MONOTONE.get(name))
            for name, h in manifest["heads"].items()
            if h["status"] == "shipped"
        }
        return cls(manifest, heads, LabRules.load(d / "lab_rules.json"))

    def predict(self, event: BreakupEvent) -> BreakupParameters:
        errors = validate(event)
        if errors:
            raise EventValidationError(errors)
        return self._predict_valid([event])[0]

    def predict_batch(self, events: Sequence[BreakupEvent]) -> list[BreakupParameters]:
        errors = [FieldError(f"[{i}].{e.loc}", e.msg) for i, ev in enumerate(events) for e in validate(ev)]
        if errors:
            raise EventValidationError(errors)
        return self._predict_valid(list(events)) if events else []

    def _predict_valid(self, events: list[BreakupEvent]) -> list[BreakupParameters]:
        X = build_features(events, self.categories)
        raw = {name: head.predict(X) for name, head in self.heads.items()}
        return [self._assemble(i, ev, raw) for i, ev in enumerate(events)]

    def _assemble(self, i: int, event: BreakupEvent, raw: dict[str, np.ndarray]) -> BreakupParameters:
        use_model = has_construction_info(event)
        fallback = False
        bands: dict[str, Band] = {}
        for name in LEARNED_HEADS:
            head = self.manifest["heads"][name]
            if name in raw and use_model:
                q = apply_offset(raw[name][i : i + 1], head["cqr_offset"])[0]
            else:
                lo, hi = head["fallback_spread"]
                q = np.array([lo, 0.0, hi])
                fallback = True
            if name in LOG_SPACE:
                q = np.exp(q)
            bands[name] = Band(*(float(v) for v in q))

        lab = self.lab_rules.for_material(event.target.structure_material)
        bands["am_mu_shift"] = combine(bands["am_mu_shift"], lab.am_mu_shift_lab)
        bands["slope_delta"] = lab.slope_delta

        d = sbm.derive(event)
        return BreakupParameters(
            **{name: _finalize(name, bands[name]) for name in BAND_FIELDS},
            emr_j_per_g=d.emr_j_per_g,
            is_catastrophic=d.is_catastrophic,
            sbm_mass_param=d.sbm_mass_param,
            model_version=self.model_version,
            imputed_fields=imputed_fields(event),
            fallback=fallback,
            warnings=self._warnings(event),
        )

    def _warnings(self, event: BreakupEvent) -> list[str]:
        ranges = self.manifest["training_ranges"]
        lo, hi = ranges["total_mass_kg"]
        out = []
        if not lo <= sbm.target_mass_kg(event) <= hi:
            out.append("ood_mass")
        if event.target.launch_year is not None and event.target.launch_year > ranges["max_epoch_year"]:
            out.append("ood_era")
        if event.event_type == "collision":
            out.append("low_support_collision")
        return out
