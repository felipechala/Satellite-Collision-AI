"""Real breakups, NASA SBM vs our model, against the cataloged fragment counts.

    python -m ml.showcase [--out "../output data"] [--as-of 2026-10-03] [--model-dir models/estimator-v1]

Writes to --out (default: <repo>/output data):
  all_events_heldout.csv   every labeled event: real count, NASA prediction, our held-out p10/p50/p90
  accuracy_summary.csv     held-out accuracy of both models, overall and by event type / cause
  showcase_satellites.csv  three selected real satellites (see SELECTION below)
  showcase_adjustments.csv the shipped model's six adjustments to the NASA model for those three
  README.md                what the files mean, the selection rule and the caveats

Scored predictions are held out (ml.evaluate_holdout.run_kfold): each event is predicted by a model
trained without it and without its bus family. The showcase picks are illustrations chosen by a fixed
rule; accuracy_summary.csv is the fair comparison.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import sbm
from .contract import NEUTRAL, ParameterSet, corrected_count
from .data import build_training_rows, load_events, load_specs
from .evaluate_holdout import metrics, run_kfold

REPO = Path(__file__).resolve().parents[2]
CP = REPO / "cloud_predictor"
DEFAULT_OUT = REPO / "output data"

# SELECTION: explosions (NASA predicts the same 239 pieces >= 10 cm for every one) where NASA is at
# least 10x off, our held-out p50 is within 1.5x, and the real count is inside our p10-p90 band;
# ranked by improvement ln(NASA factor off) - ln(our factor off); one per explosion cause, payloads first.
NASA_MIN_FACTOR_OFF = 10.0
OURS_MAX_FACTOR_OFF = 1.5
N_SHOWCASE = 3

PARAM_SOURCES = {
    "n_multiplier": "learned (neural-net head, trained on {n} real breakups)",
    "slope_delta": "DebriSat lab rule by material: +0.24 (0.12-0.32) for cfrp/mixed, from DebriSat's excess of small "
                   "fragments over the SBM; neutral for aluminum/unknown (this satellite's material is unknown)",
    "am_mu_shift": "learned head (neutral until fragments.csv exists) + DebriSat lab rule",
    "am_sigma_scale": "learned head (neutral until fragments.csv exists)",
    "dv_mu_shift": "learned head (neutral until fragments.csv exists)",
    "dv_sigma_scale": "learned head (neutral until fragments.csv exists)",
}


def _factor_off(pred, actual):
    return np.exp(np.abs(np.log(np.asarray(pred, float) / np.asarray(actual, float))))


def _meaning(name: str, v: float) -> str:
    return {
        "n_multiplier": f"fragment count x{v:.3g} vs NASA (all sizes)",
        "slope_delta": f"size-distribution slope below 10 cm {v:+.2f} (0 = NASA)",
        "am_mu_shift": f"area-to-mass x{10 ** v:.2f} (log10 shift {v:+.2f})",
        "am_sigma_scale": f"area-to-mass spread x{v:.2f}",
        "dv_mu_shift": f"ejection speed x{10 ** v:.2f} (log10 shift {v:+.2f})",
        "dv_sigma_scale": f"ejection speed spread x{v:.2f}",
    }[name]


def heldout_table(events_csv: Path, specs_json: Path, provenance_csv: Path, as_of: str, n_folds: int,
                  seed: int) -> pd.DataFrame:
    preds, _ = run_kfold(str(events_csv), str(specs_json), as_of, n_folds=n_folds, seed=seed)
    prov = pd.read_csv(provenance_csv, dtype={"event_id": str})[["event_id", "parent_name", "epoch", "discos_event_type"]]
    specs = pd.DataFrame(load_specs(specs_json).values())[["norad_id", "dry_mass_kg", "bus_family", "launch_year"]]
    t = (preds.rename(columns={"repeat": "fold"})
         .merge(prov, on="event_id", how="left")
         .merge(specs, on="norad_id", how="left"))
    t["nasa_factor_off"] = _factor_off(t["nasa"], t["actual"])
    t["model_factor_off"] = _factor_off(t["model_p50"], t["actual"])
    t["actual_in_model_band"] = (t["model_p10"] <= t["actual"]) & (t["actual"] <= t["model_p90"])
    t["improvement_ln"] = np.log(t["nasa_factor_off"]) - np.log(t["model_factor_off"])
    t["better_model"] = np.where(t["model_factor_off"] < t["nasa_factor_off"], "ours",
                                 np.where(t["model_factor_off"] > t["nasa_factor_off"], "nasa", "tie"))
    cols = ["event_id", "parent_name", "norad_id", "epoch", "event_type", "explosion_cause", "discos_event_type",
            "object_class", "bus_family", "dry_mass_kg", "actual", "nasa", "model_p10", "model_p50", "model_p90",
            "nasa_factor_off", "model_factor_off", "actual_in_model_band", "improvement_ln", "better_model", "fold"]
    return t[cols].rename(columns={"actual": "real_cataloged_ge10cm", "nasa": "nasa_pred_ge10cm",
                                   "model_p10": "ours_p10_ge10cm", "model_p50": "ours_p50_ge10cm",
                                   "model_p90": "ours_p90_ge10cm"})


def accuracy_table(t: pd.DataFrame) -> pd.DataFrame:
    df = t.rename(columns={"real_cataloged_ge10cm": "actual", "nasa_pred_ge10cm": "nasa",
                           "ours_p10_ge10cm": "model_p10", "ours_p50_ge10cm": "model_p50",
                           "ours_p90_ge10cm": "model_p90"})
    scopes = [("all events", df)]
    scopes += [(f"type: {k}", g) for k, g in df.groupby("event_type")]
    scopes += [(f"explosion cause: {k}", g) for k, g in df[df["event_type"] == "explosion"].groupby("explosion_cause")]
    rows = []
    labels = [("typical_factor_off", "typical factor off (median of max(pred/real, real/pred))"),
              ("median_bias_factor", "median pred/real (1 = unbiased, >1 = too many)"),
              ("within_2x", "share within 2x of real"), ("within_10x", "share within 10x of real"),
              ("rmse_ln", "RMSE of ln(pred/real)"), ("mean_abs_ln_error", "mean |ln(pred/real)|")]
    for scope, g in scopes:
        m = metrics(g)
        for key, label in labels:
            rows.append({"scope": scope, "n_events": m["n"], "metric": label,
                         "nasa_sbm": round(m["nasa"][key], 4), "our_model": round(m["model"][key], 4)})
        rows.append({"scope": scope, "n_events": m["n"], "metric": "share of real counts inside our p10-p90 band",
                     "nasa_sbm": None, "our_model": round(m["model"]["p10_p90_coverage"], 4)})
    return pd.DataFrame(rows)


def select_showcase(t: pd.DataFrame, n: int = N_SHOWCASE) -> pd.DataFrame:
    c = t[(t["event_type"] == "explosion") & (t["nasa_factor_off"] >= NASA_MIN_FACTOR_OFF)
          & (t["model_factor_off"] <= OURS_MAX_FACTOR_OFF) & t["actual_in_model_band"]].copy()
    c["is_payload"] = c["object_class"] == "payload"
    c = c.sort_values(["is_payload", "improvement_ln"], ascending=[False, False])
    picks, causes = [], set()
    for _, r in c.iterrows():
        if r["explosion_cause"] in causes:
            continue
        picks.append(r)
        causes.add(r["explosion_cause"])
        if len(picks) == n:
            break
    return pd.DataFrame(picks).drop(columns="is_payload")


def showcase_details(picks: pd.DataFrame, events_csv: Path, specs_json: Path, model_dir: Path,
                     n_labeled: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .estimator import BreakupParameterEstimator  # imports torch

    est = BreakupParameterEstimator.load(str(model_dir))
    rows, _ = build_training_rows(load_events(events_csv), load_specs(specs_json))
    by_id = {r.event_id: r for r in rows}
    sats, adjustments = [], []
    for _, p in picks.iterrows():
        event = by_id[p["event_id"]].event
        bands = est.predict(event)
        ps = ParameterSet.from_p50(bands)
        nasa = lambda lc: float(sbm.n_cum_sbm(lc, event))  # noqa: E731
        ours = lambda lc: float(corrected_count(lc, ps, event))  # noqa: E731
        sats.append({
            **p.to_dict(),
            "shipped_n_multiplier_p50": bands.n_multiplier.p50,
            "nasa_expected_1cm_to_10cm": nasa(0.01) - nasa(0.1),
            "ours_expected_1cm_to_10cm": ours(0.01) - ours(0.1),
            "nasa_expected_2mm_to_10cm": nasa(0.002) - nasa(0.1),
            "ours_expected_2mm_to_10cm": ours(0.002) - ours(0.1),
            "model_warnings": ";".join(bands.warnings) or None,
        })
        for name in NEUTRAL:
            b = getattr(bands, name)
            adjustments.append({
                "event_id": p["event_id"], "parent_name": p["parent_name"], "parameter": name,
                "nasa_value": NEUTRAL[name], "ours_p10": b.p10, "ours_p50": b.p50, "ours_p90": b.p90,
                "meaning_of_p50": _meaning(name, b.p50),
                "source": PARAM_SOURCES[name].format(n=n_labeled),
            })
    return pd.DataFrame(sats), pd.DataFrame(adjustments)


def readme(t: pd.DataFrame, acc: pd.DataFrame, sats: pd.DataFrame, as_of: str, n_folds: int) -> str:
    overall = acc[acc["scope"] == "all events"].set_index("metric")
    tf = overall.loc["typical factor off (median of max(pred/real, real/pred))"]
    w2 = overall.loc["share within 2x of real"]
    w10 = overall.loc["share within 10x of real"]
    cov = overall.loc["share of real counts inside our p10-p90 band", "our_model"]
    lines = [
        "# Output data: our model vs the NASA Standard Breakup Model on real breakups",
        "",
        f"Generated by `python -m ml.showcase` (from `cloud_predictor/`), as of {as_of}.",
        "",
        "**What is compared:** the number of fragments **>= 10 cm ever cataloged** after each real breakup "
        "(ESA DISCOS), against the NASA Standard Breakup Model and our model (NASA model x our learned adjustments). "
        "This is the only fragment count with real ground truth; the untrackable 2 mm - 10 cm counts in "
        "`showcase_satellites.csv` are model outputs with no ground truth.",
        "",
        f"**Every scored prediction is held out:** {n_folds}-fold cross-validation grouped by spacecraft bus family, "
        "so each event is predicted by a model trained without it and without its sister spacecraft.",
        "",
        "## Headline accuracy (all events, held out)",
        "",
        "| | NASA model | Our model |",
        "| --- | --- | --- |",
        f"| Typical factor off | x{tf['nasa_sbm']:.1f} | x{tf['our_model']:.1f} |",
        f"| Within 2x of the real count | {w2['nasa_sbm']:.0%} | {w2['our_model']:.0%} |",
        f"| Within 10x of the real count | {w10['nasa_sbm']:.0%} | {w10['our_model']:.0%} |",
        f"| Real count inside our p10-p90 band | - | {cov:.0%} (target 80%) |",
        "",
        f"{len(t)} events. Ours is closer to the real count on {(t['better_model'] == 'ours').sum()}, NASA on "
        f"{(t['better_model'] == 'nasa').sum()}, tied on {(t['better_model'] == 'tie').sum()} (gated collisions). "
        "Per event type and cause in `accuracy_summary.csv`.",
        "",
        "## Showcase satellites (selected illustrations)",
        "",
        "Chosen by a fixed rule, not by hand: explosions (the NASA model predicts the same "
        f"{t.loc[t['event_type'] == 'explosion', 'nasa_pred_ge10cm'].iloc[0]:.0f} pieces for every explosion) where "
        f"NASA is at least {NASA_MIN_FACTOR_OFF:.0f}x off, our held-out prediction is within {OURS_MAX_FACTOR_OFF}x "
        "and the real count is inside our p10-p90 band; ranked by improvement, one per explosion cause, payloads first. "
        "They show what the model does when it works; they are not typical. The fair comparison is the table above.",
        "",
        "| Satellite | Date | Cause | Real (>= 10 cm) | NASA | Ours (held out) | NASA off | Ours off |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, s in sats.iterrows():
        lines.append(f"| {s['parent_name']} | {s['epoch']} | {s['explosion_cause']} | {s['real_cataloged_ge10cm']:.0f} | "
                     f"{s['nasa_pred_ge10cm']:.0f} | {s['ours_p50_ge10cm']:.1f} ({s['ours_p10_ge10cm']:.1f}-{s['ours_p90_ge10cm']:.1f}) | "
                     f"x{s['nasa_factor_off']:.0f} | x{s['model_factor_off']:.2f} |")
    lines += [
        "",
        "## Files",
        "",
        "- `showcase_satellites.csv`: the three satellites: specs, real count, NASA vs our held-out prediction "
        "(p10/p50/p90), how far off each is, and both models' expected untrackable fragment counts (1-10 cm, 2 mm-10 cm).",
        "- `showcase_adjustments.csv`: the shipped model's six adjustments to the NASA model for each (p10/p50/p90, "
        "the NASA neutral value, what the p50 means, and where it comes from).",
        "- `accuracy_summary.csv`: held-out accuracy of both models, overall and by event type and explosion cause.",
        "- `all_events_heldout.csv`: every scored event, so the selection and the summary can be checked.",
        "",
        "## Caveats",
        "",
        "- Explosions dominate the data. Collisions (4 scored) use the NASA model through the collision gate until "
        "30 labeled collisions exist, so ours equals NASA there.",
        "- Only the fragment-count adjustment is learned today. The size slope below 10 cm comes from DebriSat lab results for "
        "carbon-fibre and mixed construction (ml/lab_rules.json, with sources), but the structural material of real breakup "
        "parents is unknown in our sources, so it does not change these events. Area-to-mass and speed adjustments stay "
        "neutral until per-fragment data (fragments.csv) exists.",
        "- 'Ever cataloged' undercounts pieces that decayed before they could be tracked, and high-orbit breakups are "
        "tracked less completely.",
        "- `showcase_satellites.csv` scores the held-out prediction; its 'shipped' columns come from the model trained "
        "on all data (what the API serves), which has seen these events.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ml.showcase", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--events", type=Path, default=CP / "data/historical_breakups.csv")
    p.add_argument("--specs", type=Path, default=CP / "data/gunter_satellites.json")
    p.add_argument("--provenance", type=Path, default=CP / "data/reports/event_provenance.csv")
    p.add_argument("--model-dir", type=Path, default=CP / "models/estimator-v1")
    p.add_argument("--as-of", default="2026-10-03")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    t = heldout_table(a.events, a.specs, a.provenance, a.as_of, a.folds, a.seed)
    acc = accuracy_table(t)
    picks = select_showcase(t)
    if len(picks) < N_SHOWCASE:
        raise SystemExit(f"only {len(picks)} events meet the showcase rule")
    sats, adjustments = showcase_details(picks, a.events, a.specs, a.model_dir, n_labeled=len(t))

    a.out.mkdir(parents=True, exist_ok=True)
    t.sort_values(["event_type", "epoch"]).to_csv(a.out / "all_events_heldout.csv", index=False, float_format="%.6g")
    acc.to_csv(a.out / "accuracy_summary.csv", index=False)
    sats.to_csv(a.out / "showcase_satellites.csv", index=False, float_format="%.6g")
    adjustments.to_csv(a.out / "showcase_adjustments.csv", index=False, float_format="%.6g")
    (a.out / "README.md").write_text(readme(t, acc, sats, a.as_of, a.folds), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
