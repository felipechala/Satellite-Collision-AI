"""Train the breakup parameter estimator.

    python -m ml.train_breakup_scaler --events data/historical_breakups.csv \
        --specs data/gunter_satellites.json --fragments data/fragments.csv --out models/estimator-v1/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import sbm
from .contract import LOG_SPACE
from .data import build_training_rows, load_events, load_fragments, load_specs
from .estimator import MANIFEST_FORMAT
from .features import FEATURE_COLUMNS, build_features, fit_categories
from .heads import MONOTONE, QuantileHead, grouped_cv, label_spread
from .lab_rules import DEFAULT_PATH as LAB_RULES_PATH
from .lab_rules import LabRules
from .labels import LEARNED_HEADS, compute_labels
from .schema import parse_epoch


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def collision_gate(y_collision: np.ndarray, y_all: np.ndarray, min_rows: int) -> dict:
    """Whether collisions may use the learned n_multiplier head.

    With few labeled collisions the head is fit almost entirely to explosions and cannot be
    validated on collisions, so the estimator falls back to the unmodified SBM (x1) with the
    spread of the collision labels (all labels if fewer than 2 collisions) until min_rows exist.
    """
    spread = label_spread(y_collision if len(y_collision) >= 2 else y_all)
    return {"n_labeled": int(len(y_collision)), "min_rows": int(min_rows),
            "use_model": bool(len(y_collision) >= min_rows), "spread": list(spread)}


def train(events_path: str, specs_path: str, out_dir: str, fragments_path: Optional[str] = None,
          as_of: Optional[str] = None, model_version: Optional[str] = None, seed: int = 0,
          n_folds: int = 5, min_tle_span_days: float = 180.0, min_fragments: int = 5,
          min_rows: int = 30, min_improvement: float = 0.01, min_collision_rows: int = 30,
          log: Callable[[str], None] = print) -> dict:
    out = Path(out_dir)
    as_of_dt = parse_epoch(as_of) if as_of else datetime.now(timezone.utc)
    model_version = model_version or out.resolve().name

    rows, skipped = build_training_rows(load_events(events_path), load_specs(specs_path))
    for event_id, errs in skipped:
        log(f"skipped event {event_id}: " + "; ".join(f"{e.loc}: {e.msg}" for e in errs))
    if not rows:
        raise SystemExit("no valid events to train on")
    fragments = load_fragments(fragments_path) if fragments_path else None

    labels = compute_labels(rows, fragments, as_of_dt, min_fragments, min_tle_span_days)
    events = [r.event for r in rows]
    categories = fit_categories(events)
    X = build_features(events, categories)
    groups = np.array([r.group for r in rows])

    head_manifest: dict[str, dict] = {}
    head_report: dict[str, dict] = {}
    n_oof: Optional[np.ndarray] = None
    for name in LEARNED_HEADS:
        y_all = labels[name].to_numpy(float)
        mask = ~np.isnan(y_all)
        y = y_all[mask]
        spread = label_spread(y)
        entry = {"n_labeled": int(mask.sum()), "n_groups": int(len(np.unique(groups[mask])))}
        cv = None
        if entry["n_labeled"] < min_rows:
            status, reason = "fallback", f"only {entry['n_labeled']} labeled events (need {min_rows})"
        else:
            cv = grouped_cv(X[mask], y, groups[mask], n_folds, MONOTONE.get(name), seed)
            if cv is None:
                status, reason = "fallback", "fewer than 2 CV groups"
            elif cv.cv_loss < cv.baseline_loss * (1.0 - min_improvement):
                status, reason = "shipped", "beats neutral baseline in grouped CV"
            else:
                status, reason = "fallback", "did not beat neutral baseline in grouped CV"
        if cv is not None:
            entry.update(n_folds=cv.n_folds, cv_pinball=_r(cv.cv_loss), baseline_pinball=_r(cv.baseline_loss),
                         coverage_p10_p90=_r(cv.coverage, 3))
            if name == "n_multiplier":
                n_oof = np.full((len(rows), 3), np.nan)
                n_oof[mask] = cv.oof
        if status == "shipped":
            QuantileHead.fit(X[mask], y, MONOTONE.get(name), seed).save(out / "heads", name)
        entry.update(status=status, reason=reason)
        head_report[name] = entry
        head_manifest[name] = {
            "status": status,
            "cqr_offset": cv.cqr_offset if cv is not None else 0.0,
            "fallback_spread": list(spread),
            "space": "log" if name in LOG_SPACE else "linear",
        }
        log(f"{name}: {status} ({reason})")

    y_count = labels["n_multiplier"].to_numpy(float)
    is_collision = np.array([r.event.event_type == "collision" for r in rows])
    gate = collision_gate(y_count[is_collision & ~np.isnan(y_count)], y_count[~np.isnan(y_count)],
                          min_collision_rows)
    head_manifest["n_multiplier"]["collision_gate"] = gate
    log(f"collisions: {'learned head' if gate['use_model'] else 'SBM fallback'} "
        f"({gate['n_labeled']} labeled, need {min_collision_rows})")

    total_masses = [sbm.target_mass_kg(e) for e in events]
    manifest = {
        "format_version": MANIFEST_FORMAT,
        "model_version": model_version,
        "as_of": as_of_dt.isoformat(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLUMNS,
        "categories": categories,
        "heads": head_manifest,
        "training_ranges": {
            "total_mass_kg": [min(total_masses), max(total_masses)],
            "max_epoch_year": max(parse_epoch(e.epoch).year for e in events),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lab_rules = LabRules.load(LAB_RULES_PATH)
    lab_rules.save(out / "lab_rules.json")

    report = {
        "model_version": model_version,
        "as_of": manifest["as_of"],
        "n_events": len(rows),
        "n_skipped": len(skipped),
        "heads": head_report,
        "collision_gate": gate,
        "collisions": _collision_table(rows, n_oof, head_report["n_multiplier"]["status"], gate),
        "provisional_lab_rules": sorted(m for m, r in lab_rules.rules.items() if r.provisional),
        "skipped": [{"event_id": eid, "errors": [f"{e.loc}: {e.msg}" for e in errs]} for eid, errs in skipped],
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "eval_report.md").write_text(_markdown(report), encoding="utf-8")
    log(f"wrote {out}")
    return report


def _collision_table(rows, n_oof: Optional[np.ndarray], n_status: str, gate: dict) -> list[dict]:
    table = []
    for i, row in enumerate(rows):
        if row.event.event_type != "collision":
            continue
        n_sbm = float(sbm.n_cum_sbm(sbm.LC_ANCHOR_M, row.event))
        d = sbm.derive(row.event)
        entry = {
            "event_id": row.event_id,
            "epoch": row.event.epoch,
            "is_catastrophic": d.is_catastrophic,
            "n_cataloged": _r(row.n_cataloged, 0),
            "n_sbm": _r(n_sbm, 1),
        }
        if not gate["use_model"]:
            lo, hi = gate["spread"]
            entry.update(n_pred_p10=_r(n_sbm * math.exp(lo), 1), n_pred_p50=_r(n_sbm, 1),
                         n_pred_p90=_r(n_sbm * math.exp(hi), 1), source="SBM (collision gate)")
        elif n_status != "shipped":
            entry.update(n_pred_p10=None, n_pred_p50=_r(n_sbm, 1), n_pred_p90=None, source="fallback (SBM)")
        elif n_oof is not None and not np.isnan(n_oof[i, 1]):
            p10, p50, p90 = (float(np.exp(v)) * n_sbm for v in n_oof[i])
            entry.update(n_pred_p10=_r(p10, 1), n_pred_p50=_r(p50, 1), n_pred_p90=_r(p90, 1), source="held-out fold")
        else:
            entry.update(n_pred_p10=None, n_pred_p50=None, n_pred_p90=None, source="no count label")
        table.append(entry)
    return table


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def _markdown(report: dict) -> str:
    lines = [
        f"# Evaluation report: {report['model_version']}",
        "",
        f"As of {report['as_of']}. {report['n_events']} events used, {report['n_skipped']} skipped.",
        "",
        "## Learned heads (grouped cross-validation)",
        "",
        "Pinball loss is averaged over p10/p50/p90 in training space (ln for multipliers and scales). "
        "The baseline is the neutral SBM value with the training labels' spread. "
        "Coverage is the share of held-out labels inside the conformally adjusted p10-p90 band (target 80%).",
        "",
        "| Head | Status | Labeled | Groups | CV loss | Baseline loss | Coverage | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, h in report["heads"].items():
        lines.append(
            f"| `{name}` | {h['status']} | {h['n_labeled']} | {h['n_groups']} | {_fmt(h.get('cv_pinball'))} | "
            f"{_fmt(h.get('baseline_pinball'))} | {_fmt(h.get('coverage_p10_p90'))} | {h['reason']} |"
        )
    lines += ["", "## Collisions: predicted vs cataloged fragment count (>= 10 cm)", ""]
    gate = report["collision_gate"]
    if gate["use_model"]:
        lines += [f"Collisions use the learned head ({gate['n_labeled']} labeled collisions).", ""]
    else:
        lines += [f"Collision gate: only {gate['n_labeled']} labeled collisions (need {gate['min_rows']}), so "
                  "collisions use the unmodified SBM with the collision labels' spread, not the learned head.", ""]
    if report["collisions"]:
        lines += [
            "| Event | Epoch | Catastrophic | Cataloged | SBM | Pred p10 | Pred p50 | Pred p90 | Source |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in report["collisions"]:
            lines.append(
                f"| {c['event_id']} | {c['epoch']} | {c['is_catastrophic']} | {_fmt(c['n_cataloged'])} | "
                f"{c['n_sbm']} | {_fmt(c['n_pred_p10'])} | {_fmt(c['n_pred_p50'])} | {_fmt(c['n_pred_p90'])} | "
                f"{c['source']} |"
            )
    else:
        lines.append("No collision events in the training set.")
    if report["provisional_lab_rules"]:
        lines += ["", "## Provisional lab rules", "",
                  "These materials still use neutral DebriSat placeholders: "
                  + ", ".join(f"`{m}`" for m in report["provisional_lab_rules"]) + "."]
    if report["skipped"]:
        lines += ["", "## Skipped events", ""]
        lines += [f"- {s['event_id']}: {'; '.join(s['errors'])}" for s in report["skipped"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ml.train_breakup_scaler", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True, help="historical breakups CSV")
    p.add_argument("--specs", required=True, help="parent spacecraft specs JSON")
    p.add_argument("--out", required=True, help="output model directory")
    p.add_argument("--fragments", help="per-fragment CSV (enables A/M and delta-v labels)")
    p.add_argument("--as-of", help="ISO 8601 reference date for the catalog-lag cutoff (default: now)")
    p.add_argument("--model-version", help="default: name of the --out directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-tle-span-days", type=float, default=180.0)
    p.add_argument("--min-fragments", type=int, default=5)
    p.add_argument("--min-rows", type=int, default=30, help="labeled events a head needs to be trained")
    p.add_argument("--min-improvement", type=float, default=0.01,
                   help="relative CV loss improvement over the baseline a head needs to ship")
    p.add_argument("--min-collision-rows", type=int, default=30,
                   help="labeled collisions needed before collisions use the learned count head")
    a = p.parse_args(argv)
    try:
        train(a.events, a.specs, a.out, a.fragments, a.as_of, a.model_version, a.seed, a.n_folds,
              a.min_tle_span_days, a.min_fragments, a.min_rows, a.min_improvement, a.min_collision_rows,
              log=lambda m: print(m, file=sys.stderr))
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()