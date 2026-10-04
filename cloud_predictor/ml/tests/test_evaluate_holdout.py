import numpy as np
import pandas as pd
import pytest

from ml.data import build_training_rows, load_events, load_specs
from ml.evaluate_holdout import _bias, main, metrics
from ml.labels import compute_labels
from ml.schema import parse_epoch

from .conftest import AS_OF


def test_holdout_beats_sbm_on_synthetic_and_holds_out_whole_groups(synth_dir, tmp_path):
    out = tmp_path / "holdout"
    events_csv, specs_json = synth_dir / "historical_breakups.csv", synth_dir / "gunter_satellites.json"
    main(["--events", str(events_csv), "--specs", str(specs_json), "--out", str(out), "--as-of", AS_OF,
          "--repeats", "2"])
    preds = pd.read_csv(out / "holdout_predictions.csv", dtype={"event_id": str})
    assert set(preds["repeat"]) == {0, 1}

    # Every labeled event of a held-out group is scored in that split, i.e. none of them was trained on.
    rows, _ = build_training_rows(load_events(events_csv), load_specs(specs_json))
    labels = compute_labels(rows, None, parse_epoch(AS_OF))["n_multiplier"]
    labeled_by_group: dict[str, set[str]] = {}
    for r in rows:
        if np.isfinite(labels[r.event_id]):
            labeled_by_group.setdefault(r.group, set()).add(r.event_id)
    for _, g in preds.groupby("repeat"):
        scored = set(g["event_id"])
        for group in set(g["group"]):
            assert labeled_by_group[group] <= scored, group

    m = metrics(preds)
    assert m["model"]["rmse_ln"] < m["nasa"]["rmse_ln"]
    assert "## Collisions held out" in (out / "holdout_report.md").read_text(encoding="utf-8")


def test_metrics_and_bias_wording():
    df = pd.DataFrame({"actual": [10.0, 10.0], "nasa": [100.0, 1.0], "model_p10": [5.0, 5.0],
                       "model_p50": [10.0, 20.0], "model_p90": [15.0, 8.0]})
    m = metrics(df)
    assert m["nasa"]["typical_factor_off"] == pytest.approx(10.0)
    assert m["model"]["within_2x"] == 1.0
    assert m["model"]["p10_p90_coverage"] == 0.5
    assert _bias(24.0) == "24× too many"
    assert _bias(0.5) == "2.00× too few"
<<<<<<< HEAD
=======


def test_kfold_scores_every_labeled_event_once_with_groups_held_out(synth_dir):
    from ml.evaluate_holdout import run_kfold

    events_csv, specs_json = synth_dir / "historical_breakups.csv", synth_dir / "gunter_satellites.json"
    preds, splits = run_kfold(str(events_csv), str(specs_json), AS_OF, n_folds=3)
    rows, _ = build_training_rows(load_events(events_csv), load_specs(specs_json))
    labels = compute_labels(rows, None, parse_epoch(AS_OF))["n_multiplier"]
    labeled = {r.event_id for r in rows if np.isfinite(labels[r.event_id])}
    assert preds["event_id"].is_unique and set(preds["event_id"]) == labeled
    assert (preds.groupby("group")["repeat"].nunique() == 1).all()  # a group never straddles folds
    assert len(splits) == 3
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
