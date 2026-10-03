import math

import numpy as np
import pytest

from ml import sbm
from ml.contract import (
    BOUNDS,
    NEUTRAL,
    P10_P90_Z_SPAN,
    ParameterSet,
    am_params,
    corrected_count,
    dv_params,
    sample_parameter_set,
)
from ml.schema import BAND_FIELDS, Band, BreakupEvent, BreakupParameters, Impactor, Spacecraft

EXPLOSION = BreakupEvent("explosion", "2020-01-01T00:00:00Z", Spacecraft("rocket_body", 1400.0))
COLLISION = BreakupEvent("collision", "2020-01-01T00:00:00Z", Spacecraft("payload", 750.0), Impactor(5.0, 10.0))
LC = np.logspace(-3, 0, 61)


def _params(**bands):
    full = {n: Band(NEUTRAL[n], NEUTRAL[n], NEUTRAL[n]) for n in BAND_FIELDS}
    full.update(bands)
    return BreakupParameters(**full, emr_j_per_g=None, is_catastrophic=None, sbm_mass_param=None, model_version="t")


def test_neutral_parameter_set_matches_neutral_table():
    assert ParameterSet().__dict__ == NEUTRAL


@pytest.mark.parametrize("event", [EXPLOSION, COLLISION])
def test_neutral_corrections_reproduce_sbm(event):
    ps = ParameterSet()
    np.testing.assert_allclose(corrected_count(LC, ps, event), sbm.n_cum_sbm(LC, event), rtol=1e-12)
    lam = np.log10(LC)
    mu, sd = am_params(lam, ps)
    np.testing.assert_array_equal(mu, sbm.am_mu(lam))
    np.testing.assert_array_equal(sd, sbm.am_sigma(lam))
    chi = np.linspace(-3, 1, 9)
    dmu, dsd = dv_params(chi, ps, event.event_type)
    np.testing.assert_array_equal(dmu, sbm.dv_mu(chi, event.event_type))
    assert dsd == sbm.DV_SIGMA


def test_multiplier_anchors_at_10cm_and_slope_acts_only_below():
    ps = ParameterSet(n_multiplier=2.0, slope_delta=0.3)
    above = LC[LC >= 0.1]
    np.testing.assert_allclose(corrected_count(above, ps, COLLISION), 2.0 * sbm.n_cum_sbm(above, COLLISION))
    eps = 1e-9
    left, right = corrected_count([0.1 - eps, 0.1], ps, COLLISION)
    assert left == pytest.approx(right, rel=1e-6)
    lc = 0.01
    expected = 2.0 * sbm.n_cum_sbm(0.1, COLLISION) * (lc / 0.1) ** -(1.71 + 0.3)
    assert corrected_count(lc, ps, COLLISION) == pytest.approx(expected)


def test_shifts_and_scales_apply_as_specified():
    ps = ParameterSet(am_mu_shift=0.2, am_sigma_scale=1.5, dv_mu_shift=-0.1, dv_sigma_scale=0.5)
    mu, sd = am_params(-2.0, ps)
    assert mu == pytest.approx(sbm.am_mu(-2.0) + 0.2)
    assert sd == pytest.approx(1.5 * sbm.am_sigma(-2.0))
    dmu, dsd = dv_params(-1.0, ps, "explosion")
    assert dmu == pytest.approx(0.2 * -1.0 + 1.85 - 0.1)
    assert dsd == pytest.approx(0.4 * 0.5)


def test_sampling_uses_band_as_normal():
    params = _params(am_mu_shift=Band(-0.1, 0.1, 0.3), n_multiplier=Band(0.8, 1.2, 1.8))
    rng = np.random.default_rng(0)
    draws = [sample_parameter_set(params, rng) for _ in range(20_000)]
    shifts = np.array([d.am_mu_shift for d in draws])
    assert shifts.mean() == pytest.approx(0.1, abs=0.005)
    assert shifts.std() == pytest.approx(0.4 / P10_P90_Z_SPAN, rel=0.03)
    ks = np.array([d.n_multiplier for d in draws])
    assert (ks > 0).all()
    assert np.median(ks) == pytest.approx(1.2, rel=0.02)
    assert np.log(ks).std() == pytest.approx((math.log(1.8) - math.log(0.8)) / P10_P90_Z_SPAN, rel=0.03)


def test_zero_width_bands_sample_exactly_p50():
    params = _params(dv_mu_shift=Band(0.07, 0.07, 0.07))
    ps = sample_parameter_set(params, np.random.default_rng(1))
    assert ps.dv_mu_shift == 0.07
    assert ps.n_multiplier == 1.0


def test_samples_are_clipped_to_bounds():
    params = _params(am_mu_shift=Band(-0.4, 0.4, 1.2), am_sigma_scale=Band(0.6, 1.8, 4.0))
    rng = np.random.default_rng(2)
    for _ in range(2000):
        ps = sample_parameter_set(params, rng)
        for name in BAND_FIELDS:
            lo, hi = BOUNDS[name]
            assert lo <= getattr(ps, name) <= hi
