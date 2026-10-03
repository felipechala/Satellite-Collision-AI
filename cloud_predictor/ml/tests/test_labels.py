import math

import numpy as np
import pandas as pd
import pytest

from ml.data import build_training_rows, load_events, load_fragments, load_specs
from ml.labels import am_labels, compute_labels
from ml.schema import parse_epoch

from .conftest import AS_OF


@pytest.fixture(scope="module")
def labelled(synth_dir):
    rows, skipped = build_training_rows(load_events(synth_dir / "historical_breakups.csv"),
                                        load_specs(synth_dir / "gunter_satellites.json"))
    assert not skipped
    labels = compute_labels(rows, load_fragments(synth_dir / "fragments.csv"), parse_epoch(AS_OF))
    truth = pd.read_csv(synth_dir / "truth.csv", dtype={"event_id": str}).set_index("event_id")
    return rows, labels, truth.loc[labels.index]


def test_count_label_is_unbiased_log_multiplier(labelled):
    _, labels, truth = labelled
    ok = labels["n_multiplier"].notna()
    assert ok.sum() > 300
    err = labels.loc[ok, "n_multiplier"] - np.log(truth.loc[ok, "n_multiplier"])
    # The generator adds 0.15 lognormal noise plus Poisson noise on top of the true multiplier.
    assert abs(err.mean()) < 0.03
    assert err.std() < 0.2


def test_recent_events_have_no_count_label(labelled):
    rows, labels, _ = labelled
    cutoff = parse_epoch(AS_OF).replace(year=2025)
    recent = [r.event_id for r in rows if parse_epoch(r.event.epoch) > cutoff]
    assert recent
    assert labels.loc[recent, "n_multiplier"].isna().all()


@pytest.mark.parametrize("head", ["am_mu_shift", "dv_mu_shift"])
def test_shift_labels_recover_truth(labelled, head):
    _, labels, truth = labelled
    ok = labels[head].notna()
    assert ok.sum() > 250
    err = labels.loc[ok, head] - truth.loc[ok, head]
    # Sampling error of a mean over 8-60 fragments with sigma ~0.4-0.6.
    assert abs(err.mean()) < 0.02
    assert err.abs().mean() < 0.15


@pytest.mark.parametrize("head", ["am_sigma_scale", "dv_sigma_scale"])
def test_scale_labels_are_unbiased_logs(labelled, head):
    _, labels, truth = labelled
    ok = labels[head].notna()
    err = labels.loc[ok, head] - np.log(truth.loc[ok, head])
    assert abs(err.mean()) < 0.03
    assert err.abs().mean() < 0.15


def _frags(n, span):
    lc = np.full(n, 0.2)
    return pd.DataFrame({"event_id": "x", "lc_m": lc, "am_m2_kg": np.full(n, 0.05),
                         "dv_m_s": np.full(n, 10.0), "tle_span_days": span})


def test_fragment_filters():
    assert all(math.isnan(v) for v in am_labels(_frags(4, 1000.0), 5, 180.0))
    assert all(math.isnan(v) for v in am_labels(_frags(10, 30.0), 5, 180.0))
    # Unknown TLE span is kept rather than dropped; identical fragments have no usable spread.
    shift, scale = am_labels(_frags(10, np.nan), 5, 180.0)
    assert not math.isnan(shift)
    assert math.isnan(scale)
