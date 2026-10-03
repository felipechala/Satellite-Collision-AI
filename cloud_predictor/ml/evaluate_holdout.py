"""Held-out accuracy: NASA SBM as-is vs SBM x learned n_multiplier, both scored against cataloged counts.

    python -m ml.evaluate_holdout --events data/historical_breakups.csv --specs data/gunter_satellites.json \
        --out data/reports/holdout --as-of 2026-10-03 [--test-size 0.2] [--repeats 5] [--seed 0]

Analytics only. Each repeat splits the labeled events by CV group (one bus family never sits on both
sides), trains the full estimator on the train split with train_breakup_scaler.train, and predicts the
test split. Repeats use different splits because one split of a few hundred events is noisy. The model
you ship is still trained on every event by train_breakup_scaler. Writes holdout_report.md and
holdout_predictions.csv to --out.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import sbm
from .data import build_training_rows, load_events, load_specs
from .estimator import BreakupParameterEstimator
from .labels import compute_labels
from .schema import parse_epoch
from .train_breakup_scaler import train

LN2, LN10 = math.log(2.0), math.log(10.0)


def _split_predictions(events: pd.DataFrame, test_rows: list, held_out_ids: set[str], specs_path: str,
                       as_of: str, seed: int, work: Path) -> tuple[pd.DataFrame, str]:
    """Train without held_out_ids, predict test_rows. Returns (predictions, head status)."""
    train_csv = work / "train_events.csv"
    events[~events["event_id"].astype(str).isin(held_out_ids)].to_csv(train_csv, index=False)
    report = train(str(train_csv), specs_path, str(work / "model"), as_of=as_of, seed=seed, log=lambda _: None)
    est = BreakupParameterEstimator.load(str(work / "model"))

    params = est.predict_batch([r.event for r in test_rows])
    out = []
    for r, p in zip(test_rows, params):
        n_sbm = float(sbm.n_cum_sbm(sbm.LC_ANCHOR_M, r.event))
        b = p.n_multiplier
        out.append({
            "event_id": r.event_id, "event_type": r.event.event_type, "explosion_cause": r.event.explosion_cause,
            "object_class": r.event.target.object_class, "group": r.group, "actual": r.n_cataloged,
            "nasa": n_sbm, "model_p10": n_sbm * b.p10, "model_p50": n_sbm * b.p50, "model_p90": n_sbm * b.p90,
        })
    return pd.DataFrame(out), report["heads"]["n_multiplier"]["status"]


def run(events_path: str, specs_path: str, as_of: str, test_size: float = 0.2, repeats: int = 5,
        seed: int = 0) -> tuple[pd.DataFrame, list[dict]]:
    events = load_events(events_path)
    rows, _ = build_training_rows(events, load_specs(specs_path))
    labels = compute_labels(rows, None, parse_epoch(as_of))["n_multiplier"]
    labeled = np.array([bool(np.isfinite(labels[r.event_id])) for r in rows])
    groups = np.array([r.group for r in rows])

    # Whole groups are held out, unlabeled events included, so nothing from a test bus family reaches
    # training; only the labeled ones can be scored.
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=test_size, random_state=seed)
    preds, splits = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, te) in enumerate(splitter.split(np.zeros(len(rows)), groups=groups)):
            held_out = {rows[j].event_id for j in te}
            test_rows = [rows[j] for j in te if labeled[j]]
            work = Path(tmp) / f"split{i}"
            work.mkdir()
            p, status = _split_predictions(events, test_rows, held_out, specs_path, as_of, seed + i, work)
            preds.append(p.assign(repeat=i))
            splits.append({"repeat": i, "n_train_labeled": int(labeled.sum()) - len(test_rows), "n_test": len(test_rows),
                           "test_collisions": int((p["event_type"] == "collision").sum()), "head_status": status})
    return pd.concat(preds, ignore_index=True), splits


def metrics(df: pd.DataFrame) -> dict:
    """Error of each predictor in log space: ln(predicted / actual)."""
    out = {"n": len(df)}
    for name, col in (("nasa", "nasa"), ("model", "model_p50")):
        err = np.log(df[col] / df["actual"])
        out[name] = {
            "typical_factor_off": float(np.exp(np.median(np.abs(err)))),
            "mean_abs_ln_error": float(np.mean(np.abs(err))),
            "rmse_ln": float(np.sqrt(np.mean(err ** 2))),
            "median_bias_factor": float(np.exp(np.median(err))),
            "within_2x": float(np.mean(np.abs(err) <= LN2)),
            "within_10x": float(np.mean(np.abs(err) <= LN10)),
        }
    out["model"]["p10_p90_coverage"] = float(((df["model_p10"] <= df["actual"]) & (df["actual"] <= df["model_p90"])).mean())
    return out


def _factor(x: float) -> str:
    return f"×{x:.2f}" if x < 10 else f"×{x:.0f}"


def _bias(x: float) -> str:
    """Median ratio pred/actual as words: 24 -> '24× too many', 0.5 -> '2.00× too few'."""
    ratio, word = (x, "too many") if x >= 1 else (1 / x, "too few")
    return f"{ratio:.0f}× {word}" if ratio >= 10 else f"{ratio:.2f}× {word}"


def markdown(preds: pd.DataFrame, splits: list[dict], test_size: float, as_of: str) -> str:
    pooled = metrics(preds)
    per_repeat = [metrics(g) for _, g in preds.groupby("repeat")]
    L = [
        "# Held-out accuracy: NASA SBM vs SBM × learned multiplier",
        "",
        f"As of {as_of}. {len(splits)} random splits, about {test_size:.0%} of bus-family groups held out each "
        f"time; each split trains the estimator on the rest and predicts the held-out events. "
        f"{pooled['n']} held-out predictions in total ({preds['event_id'].nunique()} distinct events). "
        "Target: fragments ≥10 cm ever cataloged. Analytics only: the shipped model trains on every event.",
        "",
        "Errors are ratios, because counts span 1 to 3,500: *typical factor off* is the median of "
        "max(pred/actual, actual/pred), so ×2 means half the events are within a factor of 2.",
        "",
        "## Overall (pooled over splits)",
        "",
        "| Metric | NASA SBM | SBM × model (p50) |",
        "| --- | --- | --- |",
        f"| Typical factor off | {_factor(pooled['nasa']['typical_factor_off'])} | "
        f"{_factor(pooled['model']['typical_factor_off'])} |",
        f"| Median bias | {_bias(pooled['nasa']['median_bias_factor'])} | {_bias(pooled['model']['median_bias_factor'])} |",
        f"| Within 2× of actual | {pooled['nasa']['within_2x']:.0%} | {pooled['model']['within_2x']:.0%} |",
        f"| Within 10× of actual | {pooled['nasa']['within_10x']:.0%} | {pooled['model']['within_10x']:.0%} |",
        f"| RMSE of ln(pred/actual) | {pooled['nasa']['rmse_ln']:.2f} | {pooled['model']['rmse_ln']:.2f} |",
        f"| Actual inside p10–p90 band | — | {pooled['model']['p10_p90_coverage']:.0%} (target 80%) |",
        "",
        "## Per split",
        "",
        "| Split | Train labeled | Test | Test collisions | Head | NASA factor off | Model factor off | Model within 2× |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s, m in zip(splits, per_repeat):
        L.append(f"| {s['repeat']} | {s['n_train_labeled']} | {s['n_test']} | {s['test_collisions']} | "
                 f"{s['head_status']} | {_factor(m['nasa']['typical_factor_off'])} | "
                 f"{_factor(m['model']['typical_factor_off'])} | {m['model']['within_2x']:.0%} |")

    L += ["", "## By event type and cause (pooled)", "",
          "| Type | Cause | n | NASA factor off | Model factor off | NASA bias | Model bias | Model coverage |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    keyed = preds.assign(cause=preds["explosion_cause"].fillna("—"))
    for (t, c), g in keyed.groupby(["event_type", "cause"]):
        m = metrics(g)
        L.append(f"| {t} | {c} | {m['n']} | {_factor(m['nasa']['typical_factor_off'])} | "
                 f"{_factor(m['model']['typical_factor_off'])} | {_bias(m['nasa']['median_bias_factor'])} | "
                 f"{_bias(m['model']['median_bias_factor'])} | {m['model']['p10_p90_coverage']:.0%} |")

    coll = preds[preds["event_type"] == "collision"]
    L += ["", "## Collisions held out", ""]
    if len(coll):
        L += ["| Split | Event | Actual | NASA | Model p50 | Model p10–p90 |", "| --- | --- | --- | --- | --- | --- |"]
        for r in coll.itertuples():
            L.append(f"| {r.repeat} | {r.event_id} | {r.actual:.0f} | {r.nasa:.0f} | {r.model_p50:.0f} | "
                     f"{r.model_p10:.0f}–{r.model_p90:.0f} |")
    else:
        L.append("No collision landed in a test split.")
    return "\n".join(L) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ml.evaluate_holdout", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True, help="historical breakups CSV")
    p.add_argument("--specs", required=True, help="parent spacecraft specs JSON")
    p.add_argument("--out", required=True, help="directory for holdout_report.md and holdout_predictions.csv")
    p.add_argument("--as-of", required=True, help="ISO 8601 reference date for the catalog-lag cutoff")
    p.add_argument("--test-size", type=float, default=0.2, help="share of bus-family groups held out per split")
    p.add_argument("--repeats", type=int, default=5, help="number of random splits")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    preds, splits = run(a.events, a.specs, a.as_of, a.test_size, a.repeats, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "holdout_predictions.csv", index=False)
    (out / "holdout_report.md").write_text(markdown(preds, splits, a.test_size, a.as_of), encoding="utf-8")
    m = metrics(preds)
    print(f"held-out: NASA typically off {_factor(m['nasa']['typical_factor_off'])}, "
          f"model {_factor(m['model']['typical_factor_off'])} -> {out / 'holdout_report.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
