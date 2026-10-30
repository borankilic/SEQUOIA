import math
import shutil
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.data import Curve
from sequoia.library.calibration import (
    CalibrationRecord,
    Uncertain,
    UncertainCurve,
    UncertaintyError,
    from_calibration,
    propagate_uncertainty,
    save_calibration,
)
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
PARTS = (
    "sources/laser_780_cw",
    "filters/bp_780_10nm",
    "detectors/photodiode_generic",
    "probes/power_meter",
)
X = [770.0, 775.0, 783.0, 790.0]
Y = [0.5, 0.7, 0.8, 0.6]
STD = [0.01, 0.02, 0.03, 0.01]
P0 = 5e-3
W = 5 / 8
T780 = Y[1] * (1 - W) + Y[2] * W


@pytest.fixture
def lib(tmp_path):
    for ref in PARTS:
        shutil.copytree(ROOT / "library" / ref, tmp_path / "lib" / ref)
    return Library(tmp_path / "lib")


@pytest.fixture
def setup():
    raw = setup_to_dict(load_setup(ROOT / "templates" / "laser_power_meter.yaml"))
    for item in raw["instances"]:
        if item["id"] == "nd":
            item["part"] = "filters/bp_780_10nm"
    return setup_from_dict(raw, source="test")


def curve(y=Y):
    return Curve(x=X, y=list(y), x_label="wavelength", x_unit="nm", y_label="transmission")


def calibrate(lib, **kw):
    record = CalibrationRecord(id="t_cal", parameter="transmission", curve=curve(), **kw)
    save_calibration(lib.part_dir("filters/bp_780_10nm"), [record])


def test_per_point_std_gives_the_interpolated_std(lib, setup):
    calibrate(lib, curve_std=STD)
    u = from_calibration(setup, lib, "nd.transmission")
    assert isinstance(u, UncertainCurve) and u.target == "overrides:nd.transmission"
    assert np.allclose(np.diag(u.covariance), np.square(STD))
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", [u], n=600, seed=11)
    std_t = math.hypot((1 - W) * STD[1], W * STD[2])
    assert r.nominal == pytest.approx(P0 * T780, rel=1e-9)
    assert r.contributions["overrides:nd.transmission"] == pytest.approx(P0 * std_t, rel=1e-6)
    assert "4 principal" in r.methods["overrides:nd.transmission"]
    assert r.std == pytest.approx(P0 * std_t, rel=0.1)
    assert r.mean == pytest.approx(P0 * T780, rel=0.01) and r.failures == 0
    again = propagate_uncertainty(
        setup, lib, "probe:pm.power_W", ["nd.transmission"], n=600, seed=11
    )
    assert np.array_equal(again.samples, r.samples)
    other = propagate_uncertainty(
        setup, lib, "probe:pm.power_W", ["nd.transmission"], n=600, seed=12
    )
    assert not np.array_equal(other.samples, r.samples)


def test_rank_one_covariance_is_a_correlated_scale(lib, setup):
    sigma = 0.05
    y = np.asarray(Y)
    calibrate(lib, covariance=(sigma**2 * np.outer(y, y)).tolist())
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", ["nd.transmission"], n=800, seed=5)
    analytic = P0 * T780 * sigma
    assert r.contributions["overrides:nd.transmission"] == pytest.approx(analytic, rel=1e-6)
    assert "1 principal" in r.methods["overrides:nd.transmission"]
    assert r.std == pytest.approx(analytic, rel=0.08)
    scale = r.samples / r.nominal - 1
    z = np.random.default_rng(5).standard_normal((800, 1))[:, 0]
    assert scale == pytest.approx(sigma * z, abs=1e-9)


def test_draws_are_clipped_to_the_physical_range(lib, setup):
    u = UncertainCurve.from_std(
        "overrides:nd.transmission", curve([0.99, 0.99, 0.99, 0.99]), [0.2] * 4
    )
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", [u], n=200, seed=1)
    assert r.samples.max() <= P0 * (1 + 1e-12)
    assert r.samples.min() >= 0.0
    assert np.mean(r.samples == r.samples.max()) > 0.2


def test_curves_combine_with_scalar_parameters(lib, setup):
    u = UncertainCurve.from_std("overrides:nd.transmission", curve(), STD, label="T(lambda)")
    power = Uncertain("overrides:laser.power", 0.05, "mW", label="power")
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", [u, power], n=600, seed=2)
    rel = math.hypot(0.05 / 5.0, math.hypot((1 - W) * STD[1], W * STD[2]) / T780)
    assert r.linear_std == pytest.approx(r.nominal * rel, rel=1e-6)
    assert r.std == pytest.approx(r.nominal * rel, rel=0.1)
    assert [name for name, _c, _f in r.ranked()] == ["T(lambda)", "power"]
    assert len(r.parameters) == len(r.means) == 2


def test_bad_curve_uncertainties_are_rejected(lib, setup):
    with pytest.raises(UncertaintyError, match="4 x 4"):
        propagate_uncertainty(
            setup,
            lib,
            "probe:pm.power_W",
            [UncertainCurve("overrides:nd.transmission", curve(), np.eye(3))],
        )
    bad = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
    bad[0, 1] = bad[1, 0] = 1e-3
    with pytest.raises(UncertaintyError, match="positive semi-definite"):
        propagate_uncertainty(
            setup,
            lib,
            "probe:pm.power_W",
            [UncertainCurve("overrides:nd.transmission", curve(), bad)],
        )
    with pytest.raises(UncertaintyError, match="placed part"):
        propagate_uncertainty(
            setup,
            lib,
            "probe:pm.power_W",
            [UncertainCurve.from_std("overrides:ghost.transmission", curve(), STD)],
        )


def test_many_point_curves_get_a_monte_carlo_contribution(lib, setup):
    x = np.linspace(700.0, 860.0, 21)
    many = Curve(x=x.tolist(), y=[0.8] * 21, x_label="wavelength", x_unit="nm")
    u = UncertainCurve.from_std("overrides:nd.transmission", many, [0.02] * 21)
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", [u], n=300, seed=3)
    assert "Monte Carlo" in r.methods["overrides:nd.transmission"]
    assert r.contributions["overrides:nd.transmission"] == pytest.approx(P0 * 0.02, rel=0.15)
    assert r.std == pytest.approx(P0 * 0.02, rel=0.15)
