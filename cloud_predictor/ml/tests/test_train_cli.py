import json

import numpy as np
import pandas as pd

from ml.heads import QuantileHead, label_spread
from ml.labels import LEARNED_HEADS
from ml.synthetic import write
from ml.train_breakup_scaler import collision_gate, main

from .conftest import AS_OF


def test_output_folder_contents(train_report):
    out, report = train_report
    for name in ("manifest.json", "lab_rules.json", "eval_report.json", "eval_report.md"):
        assert (out / name).is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    for name in LEARNED_HEADS:
        shipped = manifest["heads"][name]["status"] == "shipped"
        assert all(p.is_file() == shipped for p in QuantileHead.paths(out / "heads", name))
    assert manifest["training_ranges"]["max_epoch_year"] == 2025


def test_shipped_heads_beat_baseline_with_calibrated_coverage(train_report):
    _, report = train_report
    shipped = {n: h for n, h in report["heads"].items() if h["status"] == "shipped"}
    assert "n_multiplier" in shipped
    for name, h in shipped.items():
        assert h["cv_pinball"] < h["baseline_pinball"], name
        assert 0.7 <= h["coverage_p10_p90"] <= 0.9, name


def test_manifest_records_collision_gate(train_report):
    out, report = train_report
    gate = json.loads((out / "manifest.json").read_text())["heads"]["n_multiplier"]["collision_gate"]
    assert gate == report["collision_gate"]
    assert gate["n_labeled"] == 30 and gate["min_rows"] == 30 and gate["use_model"] is True
    assert gate["spread"][0] < 0 < gate["spread"][1]


def test_collision_gate_thresholds_and_spread_fallback():
    y_all = np.array([-3.0, -1.0, 0.0, 1.0, 2.0])
    gate = collision_gate(np.array([0.5, -0.5, 1.0, 0.0]), y_all, min_rows=30)
    assert gate["use_model"] is False and gate["n_labeled"] == 4
    assert gate["spread"][0] < 0 < gate["spread"][1]
    assert collision_gate(np.zeros(30), y_all, min_rows=30)["use_model"] is True
    # Fewer than 2 collision labels: the band comes from every count label instead.
    assert collision_gate(np.array([0.3]), y_all, min_rows=30)["spread"] == list(label_spread(y_all))


def test_report_lists_collisions_against_catalog(train_report):
    out, report = train_report
    assert report["collisions"]
    held_out = [c for c in report["collisions"] if c["source"] == "held-out fold"]
    assert held_out
    assert all(c["n_pred_p10"] <= c["n_pred_p50"] <= c["n_pred_p90"] for c in held_out)
    md = (out / "eval_report.md").read_text(encoding="utf-8")
    assert "## Collisions" in md and "## Provisional lab rules" in md


def test_cli_reports_skipped_events(tmp_path):
    data = write(tmp_path / "data", n_events=120, seed=3, as_of=AS_OF)
    events = pd.read_csv(data / "historical_breakups.csv", dtype={"event_id": str})
    bad = events.iloc[[0]].copy()
    bad["event_id"] = "BAD"
    bad["parent_norad_id"] = 1
    pd.concat([events, bad]).to_csv(data / "historical_breakups.csv", index=False)

    out = tmp_path / "models" / "cli-test"
    main(["--events", str(data / "historical_breakups.csv"), "--specs", str(data / "gunter_satellites.json"),
          "--fragments", str(data / "fragments.csv"), "--out", str(out), "--as-of", AS_OF,
          "--model-version", "cli-v0", "--n-folds", "3"])
    report = json.loads((out / "eval_report.json").read_text())
    assert report["model_version"] == "cli-v0"
    assert report["n_skipped"] == 1
    assert report["skipped"][0]["event_id"] == "BAD"


def test_events_only_training_without_fragments(tmp_path):
    data = write(tmp_path / "data", n_events=120, seed=4, as_of=AS_OF)
    out = tmp_path / "models" / "events-only"
    main(["--events", str(data / "historical_breakups.csv"), "--specs", str(data / "gunter_satellites.json"),
          "--out", str(out), "--as-of", AS_OF])
    heads = json.loads((out / "manifest.json").read_text())["heads"]
    for name in ("am_mu_shift", "am_sigma_scale", "dv_mu_shift", "dv_sigma_scale"):
        assert heads[name]["status"] == "fallback"
        assert heads[name]["fallback_spread"] == [0.0, 0.0]
