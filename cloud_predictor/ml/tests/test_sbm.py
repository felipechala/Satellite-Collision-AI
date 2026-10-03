import numpy as np
import pytest

from ml import sbm
from ml.schema import BreakupEvent, Impactor, Spacecraft, event_from_dict


def _collision(target_kg, imp_kg, v):
    return BreakupEvent("collision", "2020-01-01T00:00:00Z", Spacecraft("payload", target_kg), Impactor(imp_kg, v))


def test_spec_example_derived_values(spec_request):
    d = sbm.derive(event_from_dict(spec_request))
    assert d.emr_j_per_g == pytest.approx(333.3333333, rel=1e-9)
    assert d.is_catastrophic is True
    assert d.sbm_mass_param == 755.0
    assert isinstance(d.sbm_mass_param, float)


def test_non_catastrophic_mass_param_is_impactor_mass_times_v_squared():
    d = sbm.derive(_collision(750.0, 0.01, 10.0))
    assert d.emr_j_per_g == pytest.approx(0.5 * 0.01 * 1e8 / 750_000)
    assert d.is_catastrophic is False
    assert d.sbm_mass_param == pytest.approx(0.01 * 10.0 ** 2)


def test_catastrophic_threshold_is_inclusive():
    # 0.5 * m * (1000 v)^2 / (1000 * 1000 kg) == 40 J/g with m = 0.8 kg, v = 10 km/s
    d = sbm.derive(_collision(1000.0, 0.8, 10.0))
    assert d.emr_j_per_g == pytest.approx(40.0)
    assert d.is_catastrophic is True


def test_explosion_has_no_derived_values_and_no_mass_term():
    small = BreakupEvent("explosion", "2020-01-01T00:00:00Z", Spacecraft("rocket_body", 10.0))
    big = BreakupEvent("explosion", "2020-01-01T00:00:00Z", Spacecraft("rocket_body", 10_000.0))
    assert sbm.derive(small) == sbm.Derived(None, None, None)
    assert sbm.n_cum_sbm(0.1, small) == sbm.n_cum_sbm(0.1, big) == pytest.approx(6 * 0.1 ** -1.6)


def test_collision_law():
    event = _collision(750.0, 5.0, 10.0)
    assert sbm.n_cum_sbm(0.1, event) == pytest.approx(0.1 * 755.0 ** 0.75 * 0.1 ** -1.71)


def test_am_curves_are_continuous():
    eps = 1e-9
    for knot in (-1.75, -1.25):
        assert sbm.am_mu(knot - eps) == pytest.approx(sbm.am_mu(knot + eps), abs=1e-6)
    assert sbm.am_sigma(-3.5 - eps) == pytest.approx(sbm.am_sigma(-3.5 + eps), abs=1e-6)
    assert sbm.am_mu(-3.0) == -0.3 and sbm.am_mu(0.0) == -1.0
    np.testing.assert_allclose(sbm.am_sigma(np.array([-4.0, -1.0])), [0.2, 0.2 + 0.1333 * 2.5])


def test_dv_mean_coefficients():
    assert sbm.dv_mu(-1.0, "explosion") == pytest.approx(0.2 * -1.0 + 1.85)
    assert sbm.dv_mu(-1.0, "collision") == pytest.approx(0.9 * -1.0 + 2.9)
