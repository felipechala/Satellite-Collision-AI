"""Synthetic training data with known ground-truth corrections, for tests and demos.

    python -m ml.synthetic --out /tmp/synth --n-events 400 --seed 0

Writes gunter_satellites.json, historical_breakups.csv, fragments.csv and truth.csv. Fragments are
sampled through contract.py, so the true corrections are exactly what the labels should recover.
dv_sigma_scale is neutral for every event, which gives training one head with no signal to learn.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import sbm
from .contract import BOUNDS, ParameterSet, am_params, dv_params
from .schema import BreakupEvent, Impactor, Spacecraft, parse_epoch

PAYLOAD_FAMILIES = [f"bus-{c}" for c in "ABCDEFGHIJKLMNOP"]
STAGE_FAMILIES = [f"stage-{i}" for i in range(1, 10)]
MATERIAL_CHOICES = ["aluminum", "cfrp", "mixed"]


def true_corrections(material: str, mli: float, total_mass: float, cause: str | None,
                     event_type: str) -> ParameterSet:
    cfrp, mixed = float(material == "cfrp"), float(material == "mixed")
    propulsion, battery = float(cause == "propulsion"), float(cause == "battery")
    ln_k = 0.35 * cfrp + 0.15 * mixed + 0.25 * (math.log10(total_mass) - 3.0) + 0.4 * (mli - 0.3) + 0.2 * propulsion
    raw = {
        "n_multiplier": math.exp(ln_k),
        "am_mu_shift": 0.3 * mli - 0.05 + 0.1 * cfrp,
        "am_sigma_scale": math.exp(0.15 * cfrp + 0.05 * mixed),
        "dv_mu_shift": 0.15 * propulsion - 0.1 * battery + 0.05 * float(event_type == "collision"),
    }
    return ParameterSet(**{k: min(max(v, BOUNDS[k][0]), BOUNDS[k][1]) for k, v in raw.items()})


def _maybe(rng: np.random.Generator, value, p_missing: float):
    return None if rng.random() < p_missing else value


def generate(n_events: int = 400, seed: int = 0, as_of: str = "2026-01-01T00:00:00Z",
             collision_share: float = 0.08, max_fragments: int = 60):
    rng = np.random.default_rng(seed)
    as_of_dt = parse_epoch(as_of)
    family_material = {f: MATERIAL_CHOICES[rng.integers(3)] for f in PAYLOAD_FAMILIES + STAGE_FAMILIES}
    family_mli = {f: float(rng.uniform(0.2, 0.9)) for f in PAYLOAD_FAMILIES}

    specs, events, fragments, truth = [], [], [], []
    for i in range(n_events):
        event_id = f"E{i:04d}"
        norad = 10000 + i
        u = rng.random()
        if u < 0.55:
            object_class, family = "payload", PAYLOAD_FAMILIES[rng.integers(len(PAYLOAD_FAMILIES))]
            dry = 10 ** rng.uniform(1.7, 3.8)
            prop = dry * rng.uniform(0.0, 0.15)
            mli = float(np.clip(family_mli[family] + rng.normal(0, 0.1), 0, 1))
            material = family_material[family]
        elif u < 0.95:
            object_class, family = "rocket_body", STAGE_FAMILIES[rng.integers(len(STAGE_FAMILIES))]
            dry = 10 ** rng.uniform(2.8, 3.9)
            prop = dry * rng.uniform(0.0, 0.3)
            mli = float(rng.uniform(0.0, 0.2))
            material = family_material[family]
        else:
            object_class, family = "debris", None
            dry = 10 ** rng.uniform(0.0, 1.5)
            prop = 0.0
            mli = float(rng.uniform(0.0, 0.5))
            material = MATERIAL_CHOICES[rng.integers(3)]
        total = dry + prop

        launch_year = int(rng.integers(1965, 2023))
        epoch_dt = min(parse_epoch(f"{launch_year}-01-01T00:00:00Z") + timedelta(days=float(rng.uniform(0, 12 * 365))),
                       as_of_dt - timedelta(days=1))
        epoch = epoch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        is_collision = object_class != "debris" and rng.random() < collision_share
        impactor = None
        cause = None
        if is_collision:
            impactor = Impactor(mass_kg=float(10 ** rng.uniform(-1, 3)), v_rel_km_s=float(rng.uniform(1, 15)),
                                structure_material=MATERIAL_CHOICES[rng.integers(3)])
        elif object_class == "rocket_body":
            cause = "propulsion" if rng.random() < 0.8 else "unknown"
        else:
            cause = str(rng.choice(["battery", "propulsion", "deliberate", "unknown"], p=[0.4, 0.3, 0.05, 0.25]))
        event_type = "collision" if is_collision else "explosion"

        truth_ps = true_corrections(material, mli, total, cause, event_type)
        spec = {
            "norad_id": norad,
            "object_class": object_class,
            "dry_mass_kg": round(dry, 3),
            "propellant_mass_kg": round(prop, 3),
            "structure_material": material if rng.random() > 0.25 else "unknown",
            "solar_array_area_m2": _maybe(rng, round(total / rng.uniform(30, 80), 3), 0.4)
            if object_class == "payload" else None,
            "bus_volume_m3": _maybe(rng, round(total / rng.uniform(100, 400), 3), 0.4),
            "mli_fraction": _maybe(rng, round(mli, 4), 0.3),
            "launch_year": _maybe(rng, launch_year, 0.15),
            "bus_family": _maybe(rng, family, 0.2) if family else None,
        }
        specs.append(spec)

        event = BreakupEvent(event_type=event_type, epoch=epoch, target=Spacecraft(**spec), impactor=impactor,
                             explosion_cause=cause if event_type == "explosion" else None)
        n_sbm = float(sbm.n_cum_sbm(sbm.LC_ANCHOR_M, event))
        n_cat = int(rng.poisson(truth_ps.n_multiplier * n_sbm * math.exp(rng.normal(0, 0.15))))
        events.append({
            "event_id": event_id,
            "event_type": event_type,
            "epoch": epoch,
            "parent_norad_id": norad,
            "n_cataloged": n_cat,
            "explosion_cause": event.explosion_cause,
            "impactor_mass_kg": impactor.mass_kg if impactor else None,
            "impactor_v_rel_km_s": impactor.v_rel_km_s if impactor else None,
            "impactor_material": impactor.structure_material if impactor else None,
            "group": family or f"debris-{event_id}",
        })
        truth.append({"event_id": event_id, **truth_ps.__dict__})

        if rng.random() < 0.85 and n_cat >= 8:
            n_frag = int(min(n_cat, rng.integers(8, max_fragments + 1)))
            alpha = sbm.ALPHA[event_type]
            lc = np.minimum(sbm.LC_ANCHOR_M * rng.random(n_frag) ** (-1.0 / alpha), 5.0)
            mu, sd = am_params(np.log10(lc), truth_ps)
            chi = rng.normal(mu, sd)
            dmu, dsd = dv_params(chi, truth_ps, event_type)
            log_dv = rng.normal(dmu, dsd)
            for j in range(n_frag):
                fragments.append({
                    "event_id": event_id,
                    "lc_m": round(float(lc[j]), 5),
                    "am_m2_kg": None if rng.random() < 0.05 else float(10 ** chi[j]),
                    "dv_m_s": None if rng.random() < 0.15 else float(10 ** log_dv[j]),
                    "tle_span_days": round(float(rng.uniform(20, 3000)), 1),
                })

    return specs, pd.DataFrame(events), pd.DataFrame(fragments), pd.DataFrame(truth)


def write(out_dir: str | Path, **kwargs) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs, events, fragments, truth = generate(**kwargs)
    (out / "gunter_satellites.json").write_text(json.dumps(specs, indent=1) + "\n", encoding="utf-8")
    events.to_csv(out / "historical_breakups.csv", index=False)
    fragments.to_csv(out / "fragments.csv", index=False)
    truth.to_csv(out / "truth.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ml.synthetic", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n-events", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--as-of", default="2026-01-01T00:00:00Z")
    a = p.parse_args(argv)
    out = write(a.out, n_events=a.n_events, seed=a.seed, as_of=a.as_of)
    print(f"wrote synthetic dataset to {out}")


if __name__ == "__main__":
    main()
