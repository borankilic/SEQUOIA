import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis.budget import ideal_budget, power_budget, upstream_links
from sequoia.analysis.optimize import OptimizeResult, Variable, optimize
from sequoia.analysis.sweep import (
    Axis,
    SweepError,
    evaluate,
    get_value,
    linspace_axis,
    metrics_of,
    run_sweep,
    with_values,
)
from sequoia.engines.cache import Cancelled
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library():
    return Library(ROOT / "library")


def setup_of(name):
    return load_setup(ROOT / "setups" / f"{name}.yaml")


def test_get_and_set_values():
    setup = setup_of("laser_fiber_coupling")
    assert get_value(setup, "link:s2.length", "mm") == pytest.approx(150.0)
    assert get_value(setup, "link:s2.length", "m") == pytest.approx(0.15)
    changed = with_values(setup, [("link:s2.length", 0.2, "m")])
    assert get_value(changed, "link:s2.length", "mm") == pytest.approx(200.0)
    assert get_value(setup, "link:s2.length", "mm") == pytest.approx(150.0)
    assert get_value(setup, "settings:recv.mode") is None


def test_bad_targets_are_rejected():
    setup = setup_of("laser_fiber_coupling")
    for target in ("s2.length", "link:nope.length", "settings:ghost.angle", "foo:s2.length"):
        with pytest.raises(SweepError):
            with_values(setup, [(target, 1.0, "mm")])


def test_sweep_1d_matches_single_runs(library):
    setup = setup_of("laser_fiber_coupling")
    axis = linspace_axis("link:s2.length", 140, 160, 3, "mm")
    result = run_sweep(setup, library, [axis], ["link:f1.coupling_efficiency"])
    column = result.column("link:f1.coupling_efficiency")
    assert column.shape == (3,)
    for value, eta in zip(axis.values, column, strict=True):
        single = evaluate(with_values(setup, [("link:s2.length", value, "mm")]), library)
        assert metrics_of(single)["link:f1.coupling_efficiency"] == pytest.approx(eta)
    assert column[1] == column.max()
    assert result.to_csv().startswith("link:s2.length [mm],link:f1.coupling_efficiency")


def test_sweep_2d_shape_and_limits(library):
    setup = setup_of("laser_fiber_coupling")
    axes = [
        linspace_axis("link:s2.length", 140, 160, 2, "mm"),
        linspace_axis("link:s5.length", 90, 110, 3, "mm"),
    ]
    result = run_sweep(setup, library, axes, ["link:f1.power_W"])
    assert result.shape == (2, 3)
    with pytest.raises(SweepError):
        run_sweep(setup, library, [], None)
    with pytest.raises(SweepError):
        run_sweep(setup, library, [Axis("link:s2.length", np.arange(20_000.0), "mm")], None)


def test_failed_points_are_nan(library):
    setup = setup_of("laser_fiber_coupling")
    axis = Axis("link:s2.length", np.array([150.0, -5.0]), "mm")
    result = run_sweep(setup, library, [axis], ["link:f1.power_W"])
    assert math.isfinite(result.column("link:f1.power_W")[0])
    assert math.isnan(result.column("link:f1.power_W")[1])
    assert (1,) in result.failures


def test_sweep_can_be_cancelled(library):
    setup = setup_of("laser_fiber_coupling")
    axis = linspace_axis("link:s2.length", 140, 160, 5, "mm")
    with pytest.raises(Cancelled):
        run_sweep(setup, library, [axis], None, should_cancel=lambda: True)


def test_bell_hwp_sweep_gives_fringe_and_chsh(library):
    setup = setup_of("bell_test")
    axis = linspace_axis("settings:hwp_a.angle", 0, 90, 19, "deg")
    result = run_sweep(setup, library, [axis], ["analysis:tt.true_hz", "analysis:tr.true_hz"])
    tt = result.column("analysis:tt.true_hz")
    tr = result.column("analysis:tr.true_hz")
    total = tt + tr
    assert total.std() / total.mean() < 0.01
    visibility = (tt.max() - tt.min()) / (tt.max() + tt.min())
    assert visibility == pytest.approx(0.97, abs=0.03)
    expected = 0.5 * (1.0 + 0.97 * np.cos(4 * np.radians(axis.values) - 4 * np.radians(11.25)))
    assert np.allclose(tt / total, expected, atol=0.03)

    names = [f"analysis:{n}.true_hz" for n in ("tt", "tr", "rt", "rr")]
    a_axis = Axis("settings:hwp_a.angle", np.array([0.0, 22.5]), "deg")
    b_axis = Axis("settings:hwp_b.angle", np.array([11.25, 33.75]), "deg")
    grid = run_sweep(setup, library, [a_axis, b_axis], names)
    tt, tr, rt, rr = (grid.column(n) for n in names)
    e = (tt + rr - tr - rt) / (tt + rr + tr + rt)
    s = abs(e[0, 0] - e[0, 1] + e[1, 0] + e[1, 1])
    assert s == pytest.approx(2 * math.sqrt(2) * 0.97, abs=0.06)
    assert s > 2.0


def test_optimizer_recovers_telescope(library):
    setup = with_values(
        setup_of("laser_fiber_coupling"),
        [("link:s2.length", 190.0, "mm"), ("link:s5.length", 140.0, "mm")],
    )
    variables = [
        Variable("link:s2.length", 100, 250, "mm"),
        Variable("link:s3.length", 20, 300, "mm"),
        Variable("link:s5.length", 20, 300, "mm"),
    ]
    metric = "link:f1.coupling_efficiency"
    before = metrics_of(evaluate(setup, library))[metric]
    result = optimize(setup, library, variables, metric)
    assert isinstance(result, OptimizeResult)
    assert result.baseline == pytest.approx(before)
    assert result.objective >= 0.95
    assert result.objective > before
    assert result.values["link:s2.length"] == pytest.approx(150.0, abs=5.0)
    assert metrics_of(evaluate(result.setup, library))[metric] == pytest.approx(result.objective)


def test_optimizer_minimise_and_validation(library):
    setup = setup_of("laser_fiber_coupling")
    with pytest.raises(SweepError):
        optimize(setup, library, [], "link:f1.power_W")
    with pytest.raises(SweepError):
        optimize(setup, library, [Variable("link:s2.length", 5, 5, "mm")], "link:f1.power_W")
    result = optimize(
        setup,
        library,
        [Variable("link:s2.length", 120, 180, "mm")],
        "link:f1.coupling_efficiency",
        maximize=False,
        method="Nelder-Mead",
        max_evaluations=40,
    )
    assert result.objective <= result.baseline


def test_power_budget_follows_the_chain(library):
    outcome = evaluate(setup_of("laser_fiber_coupling"), library)
    chain = upstream_links(outcome.classical, "s6")
    assert chain == ["s1", "s2", "s3", "s4", "s5", "f1", "s6"]
    steps = power_budget(outcome.classical, "s6")
    assert steps[0].kind == "source"
    assert steps[-1].power_out == pytest.approx(outcome.classical.links["s6"].end.power)
    chain_t = math.prod(s.transmission for s in steps[1:])
    assert chain_t == pytest.approx(steps[-1].power_out / steps[0].power_out)


def test_ideal_budget_ranks_the_detectors(library):
    setup = setup_of("hbt_heralded")
    budget = ideal_budget(setup, library, "analysis:herald_a.true_hz")
    assert budget.all_ideal > budget.baseline
    gains = {e.instance: e.gain for e in budget.entries}
    assert gains["spad_a"] > 0 and gains["spad_h"] > 0
    assert budget.entries[0].gain >= budget.entries[-1].gain
    assert gains["pump_lens"] == pytest.approx(0.0, abs=1e-6 * budget.baseline)
