import numpy as np
import pytest

from ml.features import FEATURE_COLUMNS
from ml.heads import QuantileHead, conformal_offset, pinball_loss

MLI = FEATURE_COLUMNS.index("mli_fraction")
MASS = FEATURE_COLUMNS.index("log10_dry_mass")


@pytest.fixture(scope="module")
def anti_monotone_data():
    rng = np.random.default_rng(0)
    n = 400
    X = np.full((n, len(FEATURE_COLUMNS)), np.nan)
    X[:, MLI] = rng.random(n)
    X[:, MASS] = rng.uniform(1, 4, n)
    # The data say mli pushes the label DOWN, so the +1 constraint must bind.
    y = -0.5 * X[:, MLI] + 0.2 * X[:, MASS] + rng.normal(0, 0.1, n)
    return X, y


def test_monotone_envelope_is_non_decreasing(anti_monotone_data):
    X, y = anti_monotone_data
    head = QuantileHead.fit(X, y, {"mli_fraction": 1})
    grid = np.linspace(0, 1, 101)
    for row in X[:25]:
        Xg = np.tile(row, (len(grid), 1))
        Xg[:, MLI] = grid
        assert (np.diff(head.predict(Xg), axis=0) >= 0).all()


def test_envelope_matches_raw_model_when_already_monotone(anti_monotone_data):
    X, y = anti_monotone_data
    flipped = -y  # now mli pushes the label up, matching the constraint
    head = QuantileHead.fit(X, flipped, {"mli_fraction": 1})
    raw = head._raw(X)
    pred = head.predict(X)
    assert (pred >= raw).all()
    # The envelope's own-interval term reproduces the raw model exactly.
    assert np.mean(pred == raw) > 0.5


def test_envelope_skips_rows_with_missing_feature(anti_monotone_data):
    X, y = anti_monotone_data
    head = QuantileHead.fit(X, y, {"mli_fraction": 1})
    Xm = X[:10].copy()
    Xm[:, MLI] = np.nan
    np.testing.assert_array_equal(head.predict(Xm), head._raw(Xm))


def test_unconstrained_head_returns_raw_predictions(anti_monotone_data):
    X, y = anti_monotone_data
    head = QuantileHead.fit(X, y)
    np.testing.assert_array_equal(head.predict(X), head._raw(X))


def test_save_load_round_trip(anti_monotone_data, tmp_path):
    X, y = anti_monotone_data
    head = QuantileHead.fit(X, y, {"mli_fraction": 1})
    head.save(tmp_path, "h")
    loaded = QuantileHead.load(tmp_path, "h", {"mli_fraction": 1})
    np.testing.assert_array_equal(loaded.predict(X), head.predict(X))


def test_pinball_loss_and_conformal_offset():
    y = np.array([0.0, 1.0])
    perfect = np.column_stack([y, y, y])
    assert pinball_loss(y, perfect) == 0.0
    assert pinball_loss(y, np.zeros((2, 3))) == pytest.approx((0.1 + 0.5 + 0.9) / 3 / 2)
    scores = np.arange(1, 10, dtype=float)  # n = 9: ceil(10 * 0.8) / 9 -> 8th smallest
    assert conformal_offset(scores) == 8.0
    assert conformal_offset(np.array([])) == 0.0
