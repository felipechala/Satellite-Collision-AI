import copy
import json
import math

import pytest

from ml.lab_rules import LabRules, combine
from ml.schema import MATERIALS, Band


def test_default_table_is_neutral_and_marks_placeholders():
    rules = LabRules.load()
    assert set(rules.rules) == set(MATERIALS)
    for m, r in rules.rules.items():
        assert r.slope_delta == Band(0.0, 0.0, 0.0)
        assert r.am_mu_shift_lab == Band(0.0, 0.0, 0.0)
        assert r.provisional == (m != "aluminum")


def test_combine_with_zero_band_is_identity():
    learned = Band(-0.1, 0.2, 0.35)
    out = combine(learned, Band(0.0, 0.0, 0.0))
    assert (out.p10, out.p50, out.p90) == pytest.approx((learned.p10, learned.p50, learned.p90))


def test_combine_adds_centres_and_widths_in_quadrature():
    out = combine(Band(-0.3, 0.0, 0.4), Band(0.1, 0.5, 0.8))
    assert out.p50 == pytest.approx(0.5)
    assert out.p50 - out.p10 == pytest.approx(math.hypot(0.3, 0.4))
    assert out.p90 - out.p50 == pytest.approx(math.hypot(0.4, 0.3))


def test_crossed_band_in_table_is_rejected(tmp_path):
    raw = copy.deepcopy(LabRules.load().raw)
    raw["materials"]["cfrp"]["slope_delta"] = {"p10": 0.3, "p50": 0.1, "p90": 0.2}
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cfrp.slope_delta"):
        LabRules.load(p)
