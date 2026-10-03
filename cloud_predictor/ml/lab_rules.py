from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .schema import MATERIALS, Band

DEFAULT_PATH = Path(__file__).with_name("lab_rules.json")


@dataclass(frozen=True)
class LabRule:
    slope_delta: Band
    am_mu_shift_lab: Band
    provisional: bool
    source: str


def _band(d: dict, where: str) -> Band:
    b = Band(float(d["p10"]), float(d["p50"]), float(d["p90"]))
    if not b.p10 <= b.p50 <= b.p90:
        raise ValueError(f"{where}: expected p10 <= p50 <= p90, got {b}")
    return b


class LabRules:
    def __init__(self, rules: dict[str, LabRule], raw: dict):
        self.rules = rules
        self.raw = raw

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "LabRules":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        materials = raw["materials"]
        missing = set(MATERIALS) - materials.keys()
        if missing:
            raise ValueError(f"{path}: missing materials {sorted(missing)}")
        rules = {
            m: LabRule(
                slope_delta=_band(r["slope_delta"], f"{m}.slope_delta"),
                am_mu_shift_lab=_band(r["am_mu_shift_lab"], f"{m}.am_mu_shift_lab"),
                provisional=bool(r["provisional"]),
                source=str(r.get("source", "")),
            )
            for m, r in materials.items()
        }
        return cls(rules, raw)

    def for_material(self, material: str) -> LabRule:
        return self.rules.get(material, self.rules["unknown"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.raw, indent=2) + "\n", encoding="utf-8")


def combine(learned: Band, lab: Band) -> Band:
    """Sum of two bands: p50s add, half-widths add in quadrature on each side."""
    lower = math.hypot(learned.p50 - learned.p10, lab.p50 - lab.p10)
    upper = math.hypot(learned.p90 - learned.p50, lab.p90 - lab.p50)
    p50 = learned.p50 + lab.p50
    return Band(p50 - lower, p50, p50 + upper)
