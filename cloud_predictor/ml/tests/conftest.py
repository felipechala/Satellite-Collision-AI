import copy

import pytest

from ml import BreakupParameterEstimator
from ml.synthetic import write
from ml.train_breakup_scaler import train

AS_OF = "2026-01-01T00:00:00Z"

SPEC_REQUEST = {
    "event_type": "collision",
    "epoch": "2026-10-03T12:00:00Z",
    "target": {
        "object_class": "payload",
        "dry_mass_kg": 700,
        "propellant_mass_kg": 50,
        "structure_material": "cfrp",
        "mli_fraction": 0.6,
        "launch_year": 2019,
    },
    "impactor": {"mass_kg": 5, "v_rel_km_s": 10},
}


@pytest.fixture
def spec_request():
    return copy.deepcopy(SPEC_REQUEST)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    return write(tmp_path_factory.mktemp("synth"), n_events=400, seed=0, as_of=AS_OF)


@pytest.fixture(scope="session")
def train_report(synth_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "estimator-test"
    report = train(
        str(synth_dir / "historical_breakups.csv"),
        str(synth_dir / "gunter_satellites.json"),
        str(out),
        fragments_path=str(synth_dir / "fragments.csv"),
        as_of=AS_OF,
        log=lambda _: None,
    )
    return out, report


@pytest.fixture(scope="session")
def model_dir(train_report):
    return train_report[0]


@pytest.fixture(scope="session")
def estimator(model_dir):
    return BreakupParameterEstimator.load(str(model_dir))
