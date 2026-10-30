import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis.sweep import (
    Axis,
    SweepError,
    candidate_targets,
    check_bounds,
    coherence_length,
    default_plan,
    evaluate,
    metrics_of,
    run_sweep,
)
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library():
    return Library(ROOT / "library")


def template(name):
    return load_setup(ROOT / "templates" / f"{name}.yaml")


def by_target(setup, library, outcome=None):
    return {c.target: c for c in candidate_targets(setup, library, outcome)}


def test_bounds_follow_the_parameter_type(library):
    c = by_target(template("hom"), library)
    hwp = c["settings:hwp_idler.angle"]
    assert (hwp.lower, hwp.upper, hwp.period, hwp.unit) == (0.0, 180.0, 180.0, "deg")
    assert hwp.default == (0.0, 90.0, 46, False)
    assert c["settings:pbs.angle"].default == (0.0, 180.0, 91, False)
    for name in ("overrides:fbs.split_ratio", "overrides:cpl_signal.transmission"):
        assert (c[name].lower, c[name].upper) == (0.0, 1.0)
    length = c["link:s_pump.length"]
    assert (length.lower, length.upper, length.unit) == (0.0, None, "mm")
    assert length.default == pytest.approx((80.0, 120.0, 21, False))
    assert c["overrides:crystal.length"].lower == 0.0
    temperature = c["settings:crystal.temperature"]
    assert (temperature.lower, temperature.upper) == pytest.approx((-20.0, 250.0))
    power = c["overrides:pump.power"]
    assert (power.lower, power.upper, power.unit) == (0.0, None, "mW")
    assert power.default == pytest.approx((1.0, 100.0, 21, True))
    waist = c["overrides:pump.waist_radius"]
    assert waist.lower == 0.0 and waist.open_lower
    assert c["overrides:pump.m_squared"].lower == 1.0
    assert c["overrides:pump_lens.focal_length"].lower is None
    assert c["link:c_1.delay"].lower is None
    assert c["link:f_1.connector_loss_db"].lower == 0.0
    fock = by_target(template("fock_hom"), library)
    phase = next(v for k, v in fock.items() if k.endswith(".phase") and k.startswith("settings"))
    assert (phase.lower, phase.upper, phase.period) == (0.0, 360.0, 360.0)
    assert phase.default == (0.0, 360.0, 73, False)
    rate = next(v for k, v in fock.items() if k.endswith(".repetition_rate"))
    assert rate.lower == 0.0 and rate.open_lower
    assert by_target(template("laser_power_meter"), library)[
        "overrides:nd.optical_density"
    ].lower == (0.0)
    bell = by_target(template("bell_test"), library)
    assert bell["overrides:source.phase"].upper == pytest.approx(2 * math.pi)
    assert bell["overrides:source.visibility"].upper == 1.0
    assert by_target(template("malus_law"), library)["settings:pol.angle"].default == (
        0.0,
        180.0,
        91,
        False,
    )


def test_out_of_bound_values_are_refused(library):
    setup = template("hom")
    with pytest.raises(SweepError, match=r"overrides:fbs.split_ratio: 1.2 is outside .*0 to 1"):
        check_bounds(setup, library, [Axis("overrides:fbs.split_ratio", [0.5, 1.2])])
    with pytest.raises(SweepError, match=r"link:s_pump.length: -1 mm .*>= 0 mm"):
        check_bounds(setup, library, [Axis("link:s_pump.length", [0.05, -0.001], "m")])
    with pytest.raises(SweepError, match="crystal.temperature"):
        check_bounds(setup, library, [Axis("settings:crystal.temperature", [300, 200], "K")])
    with pytest.raises(SweepError, match="pump.waist_radius: 0 mm .*> 0 mm"):
        check_bounds(setup, library, [Axis("overrides:pump.waist_radius", [0.0], "mm")])
    check_bounds(setup, library, [Axis("settings:hwp_idler.angle", [-30.0, 400.0], "deg")])
    check_bounds(setup, library, [Axis("settings:crystal.temperature", [300.0, 320.0], "K")])
    malus = template("malus_law")
    result = run_sweep(
        malus, library, [Axis("overrides:pol.transmission", [0.9, 1.5])], ["probe:pm.power_W"]
    )
    assert math.isfinite(result.column("probe:pm.power_W")[0])
    assert math.isnan(result.column("probe:pm.power_W")[1])
    assert "physical range (0 to 1)" in result.failures[(1,)]


def test_hom_default_sweep_traces_the_dip(library):
    setup = template("hom")
    outcome = evaluate(setup, library)
    nominal = metrics_of(outcome)
    plan = default_plan(setup, library, outcome)
    assert plan.target == "link:delay_arm.length" and plan.output == "analysis:hom.true_hz"
    l_c = coherence_length(outcome, "delay_arm")
    assert 0.3e-3 < l_c < 2e-3
    assert (plan.start, plan.stop) == pytest.approx((100.66 - 3e3 * l_c, 100.66 + 3e3 * l_c))
    assert plan.points % 2 == 1
    axis = plan.axis()
    result = run_sweep(
        setup, library, [axis], [plan.output, "analysis:hom.distinguishable_rate_hz"]
    )
    true = result.column(plan.output)
    assert not result.failures
    step = axis.values[1] - axis.values[0]
    assert abs(axis.values[np.argmin(true)] - 100.66) <= step / 2
    distinguishable = result.column("analysis:hom.distinguishable_rate_hz")
    visibility = 1 - true.min() / distinguishable[np.argmin(true)]
    assert visibility == pytest.approx(nominal["analysis:hom.two_photon_visibility"], abs=2e-3)
    assert visibility > 0.98
    assert true[0] == pytest.approx(distinguishable[0], rel=0.02)
    assert true[-1] == pytest.approx(distinguishable[-1], rel=0.02)


def test_malus_default_sweep_gives_cos_squared(library):
    setup = template("malus_law")
    plan = default_plan(setup, library)
    assert (plan.target, plan.output) == ("settings:hwp.angle", "probe:pm.power_W")
    assert (plan.start, plan.stop, plan.points, plan.log) == (0.0, 90.0, 46, False)
    axis = plan.axis()
    power = run_sweep(setup, library, [axis], [plan.output]).column(plan.output)
    design = np.column_stack([np.ones_like(power), np.cos(np.radians(2 * axis.values)) ** 2])
    (floor, amplitude), *_ = np.linalg.lstsq(design, power, rcond=None)
    assert design @ [floor, amplitude] == pytest.approx(power, rel=1e-6, abs=1e-12 * power.max())
    assert amplitude > 0 and 0 <= floor / (floor + amplitude) < 1e-2
    assert np.argmax(power) == 0 and abs(axis.values[np.argmin(power)] - 45.0) <= 1.0


def test_task_defaults_for_bell_hbt_and_coupling(library):
    bell = default_plan(template("bell_test"), library)
    assert (bell.target, bell.output) == ("settings:hwp_a.angle", "analysis:tt.true_hz")
    assert (bell.start, bell.stop, bell.points) == (0.0, 90.0, 46)
    hbt = default_plan(template("hbt_heralded"), library)
    assert (hbt.target, hbt.output, hbt.log) == (
        "overrides:pump.power",
        "analysis:hbt.g2_heralded",
        True,
    )
    assert hbt.axis().values[[0, -1]] == pytest.approx([1.0, 100.0])
    coupling = default_plan(template("laser_fiber_coupling"), library)
    assert coupling.output == "link:f1.coupling_efficiency"
    assert default_plan(template("laser_power_meter"), library) is None
