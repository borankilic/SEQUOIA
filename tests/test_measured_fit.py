import json
import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis.fitting import fit, fringe, hom_dip, peak, record, scale_model, singles
from sequoia.analysis.measured import (
    POISSON,
    MeasuredDataError,
    compare,
    compare_sweep,
    from_arrays,
    load_measured,
    subtract_background,
    weights,
)
from sequoia.errors import SequoiaError
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(100, 260)


def _fringe_data(rng, x_unit="deg"):
    theta = np.linspace(0.0, 180.0, 37)
    mu = 500.0 + 400.0 * np.cos(2 * np.deg2rad(theta - 20.0))
    x = theta if x_unit == "deg" else np.deg2rad(theta)
    return from_arrays(x, rng.poisson(mu), x_unit=x_unit, y_unit="counts")


def _peak_data(rng, unit="ns", shape="gaussian"):
    t = np.linspace(-5.0, 5.0, 201)
    if shape == "gaussian":
        mu = 20.0 + 150.0 * np.exp(-((t - 0.3) ** 2) / (2 * 0.4**2)) / (
            0.4 * math.sqrt(2 * math.pi)
        )
    else:
        mu = peak("gaussian_tail").fn(t * 1e-9, np.array([0.3e-9, 0.3e-9, 0.6e-9, 20.0, 150e-9]))
    scale = {"ns": 1.0, "ps": 1e3}[unit]
    return from_arrays(t * scale, rng.poisson(mu), x_unit=unit, y_unit="counts")


def _hom_data(rng):
    x = np.linspace(-1.0, 1.0, 61)
    mu = 1000.0 * (1 - 0.9 * np.exp(-((x - 0.1) ** 2) / (2 * 0.15**2)))
    return from_arrays(x, rng.poisson(mu), x_unit="mm", y_unit="counts")


TRUTH = {
    "fringe": (
        lambda rng: _fringe_data(rng),
        lambda: fringe(2),
        {"amplitude": 400.0, "phase": math.radians(20.0), "offset": 500.0},
    ),
    "peak": (
        lambda rng: _peak_data(rng),
        lambda: peak(),
        {"center": 0.3e-9, "width": 0.4e-9, "background": 20.0, "area": 150e-9},
    ),
    "hom": (
        lambda rng: _hom_data(rng),
        lambda: hom_dip(),
        {"delay_offset": 0.1e-3, "visibility": 0.9, "width": 0.15e-3, "baseline": 1000.0},
    ),
}


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_synthetic_parameters_recovered_with_calibrated_errors(name):
    make, model, truth = TRUTH[name]
    pulls = {p: [] for p in truth}
    for seed in SEEDS:
        r = fit(make(np.random.default_rng(seed)), model())
        assert r.converged and not r.flags
        for p, value in truth.items():
            pulls[p].append((r.values[p] - value) / r.errors[p])
    for p, values in pulls.items():
        z = np.array(values)
        assert np.mean(np.abs(z) < 2) >= 0.9, (p, z)
        assert abs(np.mean(z)) < 0.3, (p, np.mean(z))
        assert 0.8 < np.std(z) < 1.2, (p, np.std(z))
    r = fit(make(np.random.default_rng(1)), model())
    for p, value in truth.items():
        assert abs(r.values[p] - value) < 2 * r.errors[p]


def test_fringe_visibility_and_tail_peak():
    r = fit(_fringe_data(np.random.default_rng(1)), fringe(2))
    v, se, unit = r.display("visibility")
    assert unit == "1" and abs(v - 0.8) < 2 * se and 0 < se < 0.02
    tail = fit(_peak_data(np.random.default_rng(4), shape="gaussian_tail"), peak("gaussian_tail"))
    assert tail.converged
    assert abs(tail.values["area"] - 150e-9) < 2 * tail.errors["area"]
    assert abs(tail.values["tail"] - 0.6e-9) < 3 * tail.errors["tail"]


def test_unit_changes_do_not_change_the_fit():
    ns = fit(_peak_data(np.random.default_rng(7), "ns"), peak())
    ps = fit(_peak_data(np.random.default_rng(7), "ps"), peak())
    for p in ns.values:
        assert ps.values[p] == pytest.approx(ns.values[p], rel=1e-6, abs=1e-15)
        assert ps.errors[p] == pytest.approx(ns.errors[p], rel=1e-5)
    assert ns.display("width")[2] == "ns" and ps.display("width")[2] == "ps"
    assert ps.display("width")[0] == pytest.approx(1e3 * ns.display("width")[0], rel=1e-6)
    deg = fit(_fringe_data(np.random.default_rng(2), "deg"), fringe(2))
    rad = fit(_fringe_data(np.random.default_rng(2), "rad"), fringe(2))
    assert rad.values["phase"] == pytest.approx(deg.values["phase"], rel=1e-6)
    theta = np.linspace(0, 180, 19)
    frac = 0.5 + 0.4 * np.cos(2 * np.deg2rad(theta - 10))
    sigma = np.full_like(theta, 0.01)
    noise = np.random.default_rng(5).normal(0, 0.01, theta.size)
    a = fit(from_arrays(theta, frac + noise, x_unit="deg", y_unit="1", sigma=sigma), fringe())
    b = fit(
        from_arrays(theta, 100 * (frac + noise), x_unit="deg", y_unit="%", sigma=100 * sigma),
        fringe(),
    )
    for p in a.values:
        assert b.values[p] == pytest.approx(a.values[p], rel=1e-6)
        assert b.errors[p] == pytest.approx(a.errors[p], rel=1e-5)
    assert b.display("amplitude")[0] == pytest.approx(100 * a.display("amplitude")[0], rel=1e-6)
    assert b.chi2 == pytest.approx(a.chi2, rel=1e-6)


def test_brightness_and_efficiency_from_singles_are_reported_unidentifiable():
    power = np.linspace(1, 20, 15)
    rate = 1e6 * 0.2 * power + 500
    rng = np.random.default_rng(0)
    data = from_arrays(
        power, rng.normal(rate, np.sqrt(rate)), x_unit="mW", y_unit="Hz", sigma=np.sqrt(rate)
    )
    r = fit(data, singles())
    assert not r.identifiable
    assert set(r.unidentifiable) == {"brightness", "efficiency"}
    assert any("cannot be separated from singles alone" in f for f in r.flags)
    assert math.isinf(r.errors["brightness"])
    product, se, _unit = r.display("brightness_x_efficiency")
    assert math.isfinite(se) and abs(product - 2e5) < 3 * se
    statuses = {row[0]: row[4] for row in r.rows()}
    assert statuses["brightness"] == "not identifiable"
    fixed = fit(data, singles(), fixed={"efficiency": 0.2})
    assert fixed.identifiable and fixed.free == ["brightness", "dark"]
    assert abs(fixed.values["brightness"] - 1e9) < 3 * fixed.errors["brightness"]
    assert "fixed" in {row[0]: row[4] for row in fixed.rows()}["efficiency"]


def test_strongly_correlated_parameters_are_flagged():
    r = fit(_peak_data(np.random.default_rng(0)), peak("gaussian_tail"))
    assert any("center and tail are strongly correlated" in f for f in r.flags)
    assert not r.identifiable
    no_tail = fit(_peak_data(np.random.default_rng(1)), peak("gaussian_tail"))
    assert set(no_tail.unidentifiable) == {"center", "tail"}
    assert any("center, tail cannot be separated" in f for f in no_tail.flags)
    assert math.isinf(no_tail.errors["tail"]) and math.isfinite(no_tail.errors["width"])


def test_parameter_with_large_relative_error_is_flagged():
    x = np.linspace(0.0, 1.0, 12)
    y = np.where(np.arange(12) % 2 == 0, 1.0, -1.0)
    data = from_arrays(x, y, x_unit="s", y_unit="Hz", sigma=np.ones(12))
    r = fit(data, scale_model(np.ones(12), label="flat model"))
    assert abs(r.values["scale"]) < 1e-9
    assert any("scale: standard error" in f and "barely constrain" in f for f in r.flags)


def test_csv_import_mapping_validation_and_provenance(tmp_path):
    path = tmp_path / "scan.csv"
    path.write_text(
        "angle (deg),coincidences,err,accidentals\n"
        "0,900,30,10\n"
        "45,500,22,10\n"
        "90,,10,10\n"
        "135,480,22,12\n"
        "180,910,30,11\n"
        "60,300,0,10\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    data = load_measured(
        path, "angle", "coincidences", sigma="err", background="accidentals", y_unit="counts"
    )
    assert len(data) == 4
    assert data.dropped == ((3, "missing or non-numeric y"), (6, "uncertainty is zero or negative"))
    assert "dropped 2 row(s)" in data.warnings[0]
    assert data.source["sha256"] and data.source["columns"]["x"] == "angle (deg)"
    assert path.read_bytes() == before
    with pytest.raises(MeasuredDataError, match="unit of y column"):
        load_measured(path, "angle", "coincidences")
    with pytest.raises(MeasuredDataError, match="must be of dimension time"):
        fit(data, peak())
    derived = subtract_background(data, sidebands=2)
    assert derived.original is data and np.all(data.y == [900, 500, 480, 910])
    assert list(derived.y) == [890, 490, 468, 899]
    assert derived.sigma[0] == pytest.approx(math.hypot(30, math.sqrt(10 / 2)))
    assert derived.operations[0]["operation"] == "subtract_background"
    rec = record(derived, fit(derived, fringe()))
    text = json.dumps(rec, allow_nan=False)
    assert "subtract_background" in text and rec["data"]["dropped_rows"][0]["row"] == 3
    with pytest.raises(SequoiaError, match="file not found"):
        load_measured(tmp_path / "missing.csv", 0, 1)


def test_weighting_choices():
    x = np.linspace(0, 1, 5)
    counts = from_arrays(x, [4, 9, 0, 16, 25], x_unit="s", y_unit="counts")
    sigma, kind, warning = weights(counts)
    assert kind == POISSON and warning is None
    assert list(sigma) == [2, 3, 1, 4, 5]
    rate = from_arrays(x, [100, 100, 100, 100, 100], x_unit="s", y_unit="kHz", time=2.0)
    sigma, kind, _ = weights(rate)
    assert kind == POISSON and sigma[0] == pytest.approx(math.sqrt(1e5 * 2) / 2)
    other = from_arrays(x, [1, 2, 3, 4, 5], x_unit="s", y_unit="mW")
    _, kind, warning = weights(other)
    assert kind == "unweighted" and "unweighted" in warning


def test_compare_with_model_curve_and_loss_scale():
    x = np.linspace(0, 10, 21)
    rng = np.random.default_rng(3)
    truth = 0.7 * (100 + 20 * x)
    data = from_arrays(x, rng.normal(truth, 2.0), x_unit="ns", y_unit="Hz", sigma=np.full(21, 2.0))
    model_x = np.linspace(-1e-9, 11e-9, 50)
    cmp = compare(data, model_x, 100 + 20 * model_x * 1e9, fit_scale=True)
    s = cmp.scale
    assert abs(s.values["scale"] - 0.7) < 2 * s.errors["scale"]
    assert cmp.reduced_chi2 == pytest.approx(1.0, abs=0.6)
    raw = compare(data, model_x, 100 + 20 * model_x * 1e9)
    assert raw.reduced_chi2 > 50
    partial = compare(data, model_x[model_x < 5e-9], (100 + 20 * model_x * 1e9)[model_x < 5e-9])
    assert any("outside the model range" in w for w in partial.warnings)


def test_compare_with_a_sequoia_sweep():
    library = Library(ROOT / "library")
    setup = load_setup(ROOT / "templates" / "malus_law.yaml")
    from sequoia.analysis.sweep import linspace_axis, run_sweep

    angles = np.linspace(0, 90, 19)
    model = run_sweep(
        setup,
        library,
        [linspace_axis("settings:hwp.angle", 0, 90, 19, "deg")],
        ["probe:pm.power_W"],
    )
    power_mw = 1e3 * model.column("probe:pm.power_W")
    rng = np.random.default_rng(11)
    sigma = 0.01 + 0.0 * angles
    data = from_arrays(
        angles,
        0.8 * power_mw + rng.normal(0, 0.01, angles.size),
        x_unit="deg",
        y_unit="mW",
        sigma=sigma,
    )
    cmp = compare_sweep(
        data, setup, library, "settings:hwp.angle", "deg", "probe:pm.power_W", fit_scale=True
    )
    assert abs(cmp.scale.values["scale"] - 0.8) < 2 * cmp.scale.errors["scale"]
    assert cmp.reduced_chi2 < 3
    assert cmp.details["metric_unit"] == "W"
    with pytest.raises(MeasuredDataError, match="frequency quantity"):
        compare_sweep(data, setup, library, "settings:hwp.angle", "deg", "readout:pd.rate_hz")
    with pytest.raises(MeasuredDataError, match="data x"):
        compare_sweep(data, setup, library, "link:s1.length", "mm", "probe:pm.power_W")
