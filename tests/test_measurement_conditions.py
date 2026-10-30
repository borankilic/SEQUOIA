import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis.correlation import CW_OFF_PEAK, CorrelationError, analyse_tags
from sequoia.fock.analysis import condition
from sequoia.fock.circuit import FockError
from sequoia.fock.from_setup import (
    build_experiment_from_setup,
    measurement_conditions,
    run_setup_experiment,
)
from sequoia.fock.run import SetupRequest, run_setup_request, setup_views
from sequoia.library.store import Library
from sequoia.measurement import analyse_measured_tags, check_measurement
from sequoia.setup.io import load_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


def _with(path: Path, measurement: dict, analyses=None):
    raw = setup_to_dict(load_setup(path))
    raw["measurement"] = measurement
    if analyses is not None:
        raw["analyses"] = analyses
    return setup_from_dict(raw)


@pytest.fixture(scope="module")
def hbt():
    setup = _with(
        ROOT / "setups" / "hbt_heralded.yaml",
        {
            "conditions": [
                {"name": "H", "herald": ["tagger.ch1"], "description": "herald clicked"},
                {"name": "H_not_b", "herald": ["spad_h"], "veto": ["tagger.ch3"]},
                {"name": "HAB", "herald": ["tagger.ch1", "spad_a", "spad_b"]},
            ]
        },
    )
    res = run_setup_experiment(build_experiment_from_setup(setup, LIB))
    return setup, res


def test_fock_named_conditions_equal_manual_conditioning(hbt):
    _setup, res = hbt
    ix = {d: i for i, d in enumerate(res.detector_ids)}
    views = setup_views(res)
    assert views.condition_error == ""
    named = {nc.name: nc for nc in views.conditions}
    manual = {
        "H": {ix["spad_h"]: (">=", 1)},
        "H_not_b": {ix["spad_h"]: (">=", 1), ix["spad_b"]: 0},
        "HAB": {ix["spad_h"]: (">=", 1), ix["spad_a"]: (">=", 1), ix["spad_b"]: (">=", 1)},
    }
    for name, spec in manual.items():
        ref = condition(res.detected, spec)
        got = named[name].conditional
        assert got.success_probability == pytest.approx(ref.success_probability, rel=1e-12)
        assert got.success_probability > 0
        assert set(got.probs) == set(ref.probs)
        for s, p in ref.probs.items():
            assert got.probs[s] == pytest.approx(p, rel=1e-12)
        assert sum(got.probs.values()) == pytest.approx(1.0)
    assert named["H_not_b"].rule == "click on spad_h, none on tagger.ch3"
    assert named["H"].description == "herald clicked"
    p_h = sum(p for s, p in res.detected.items() if s[ix["spad_h"]] > 0)
    assert named["H"].conditional.success_probability == pytest.approx(p_h, rel=1e-12)


def test_fock_hom_condition_on_one_output():
    setup = _with(
        ROOT / "templates" / "fock_hom.yaml",
        {"conditions": [{"name": "only1", "herald": ["tagger.ch1"], "veto": ["spad_2"]}]},
    )
    out = run_setup_request(SetupRequest("run", setup, LIB))
    assert out.error == ""
    res = out.result
    ix = {d: i for i, d in enumerate(res.detector_ids)}
    ref = condition(res.detected, {ix["spad_1"]: (">=", 1), ix["spad_2"]: 0})
    nc = out.views.conditions[0]
    assert nc.conditional.success_probability == pytest.approx(ref.success_probability)
    assert nc.conditional.probs == pytest.approx(ref.probs)


def test_fock_unknown_condition_channel_is_a_clear_error(hbt):
    setup, res = hbt
    bad = setup.model_copy(
        update={
            "measurement": setup.measurement.model_copy(
                update={
                    "conditions": [
                        setup.measurement.conditions[0].model_copy(
                            update={"herald": ["tagger.ch9"]}
                        )
                    ]
                }
            )
        }
    )
    with pytest.raises(FockError, match="'tagger.ch9' is neither a detector nor a tagger channel"):
        measurement_conditions(bad, res.detector_ids)
    with pytest.raises(FockError, match="'tagger' .* is not a single-photon detector"):
        measurement_conditions(
            bad.model_copy(
                update={
                    "measurement": bad.measurement.model_copy(
                        update={
                            "conditions": [
                                bad.measurement.conditions[0].model_copy(
                                    update={"herald": ["tagger"]}
                                )
                            ]
                        }
                    )
                }
            ),
            res.detector_ids,
        )
    res.setup_experiment.setup = bad
    try:
        views = setup_views(res)
    finally:
        res.setup_experiment.setup = setup
    assert views.conditions == () and "tagger.ch9" in views.condition_error


SLOT = 1e-6
N = 100


def _structured_tags():
    t = (np.arange(N) + 1) * SLOT
    k = np.arange(N)
    return {
        "tagger.ch1": t.copy(),
        "tagger.ch2": t + 0.2e-9,
        "tagger.ch3": t[k % 2 == 0] + 0.1e-9,
        "tagger.ch4": t[k % 5 == 0] - 0.25e-9,
    }, (N + 1) * SLOT


def _spdc(measurement):
    return _with(ROOT / "templates" / "spdc_car.yaml", measurement)


def _coinc(name, a, b, cond=None):
    spec = {"name": name, "type": "coincidences", "channels": [a, b], "window": "1 ns"}
    spec["center"] = "0 ns"
    if cond:
        spec["condition"] = cond
    return spec


def test_tag_conditions_gate_singles_and_coincidences_exactly():
    tags, duration = _structured_tags()
    setup = _spdc(
        {
            "conditions": [
                {"name": "HV", "herald": ["tagger.ch3"], "veto": ["tagger.ch4"]},
                {"name": "H", "herald": ["tagger.ch3"]},
                {"name": "V", "herald": ["tagger.ch1"], "veto": ["tagger.ch4"]},
                {"name": "S", "herald": ["spad_s"]},
            ]
        }
    )
    specs = [
        _coinc("plain", "tagger.ch1", "tagger.ch2"),
        _coinc("hv", "tagger.ch1", "tagger.ch2", "HV"),
        _coinc("h", "tagger.ch1", "tagger.ch2", "H"),
        _coinc("v", "tagger.ch1", "tagger.ch2", "V"),
        _coinc("s", "tagger.ch2", "tagger.ch3", "S"),
    ]
    out = analyse_measured_tags(setup, tags, duration, {}, None, specs)
    even_not_5 = sum(1 for k in range(N) if k % 2 == 0 and k % 5)
    assert even_not_5 == 40
    assert out["plain"]["counts"] == N and "condition" not in out["plain"]
    hv = out["hv"]
    assert hv["counts"] == 40
    assert hv["singles_a_hz"] == pytest.approx(40 / duration)
    assert hv["singles_b_hz"] == pytest.approx(40 / duration)
    assert hv["condition"]["name"] == "HV"
    assert hv["condition"]["kept"] == {"tagger.ch1": 40, "tagger.ch2": 40}
    assert hv["condition"]["events"] == {"tagger.ch1": N, "tagger.ch2": N}
    assert out["h"]["counts"] == 50
    assert out["v"]["counts"] == 80
    assert out["v"]["condition"]["kept"] == {"tagger.ch1": 80, "tagger.ch2": 80}
    s = out["s"]
    assert s["condition"]["herald"] == ["tagger.ch1"]
    assert s["condition"]["kept"] == {"tagger.ch2": N, "tagger.ch3": 50}
    assert s["counts"] == 50


def test_tag_condition_errors_and_check():
    tags, duration = _structured_tags()
    with pytest.raises(CorrelationError, match="unknown condition 'X'"):
        analyse_tags(tags, duration, [_coinc("a", "tagger.ch1", "tagger.ch2", "X")])
    with pytest.raises(CorrelationError, match="condition 'B': unknown channel 'tagger.ch7'"):
        analyse_tags(
            tags,
            duration,
            [_coinc("a", "tagger.ch1", "tagger.ch2", "B")],
            conditions={"B": (["tagger.ch7"], [])},
        )
    setup = _spdc({"conditions": [{"name": "H", "herald": ["tagger.ch1"]}]})
    raw = setup_to_dict(setup)
    raw["analyses"][0]["condition"] = "nope"
    issues = check_measurement(setup_from_dict(raw))
    assert any("condition 'nope'" in i.message for i in issues)


def _cw_tags(seed=7, duration=1.0, bg=1.0e6, pairs=5.0e4, jitter=0.1e-9):
    rng = np.random.default_rng(seed)
    n_pairs = rng.poisson(pairs * duration)
    t_pair = rng.uniform(0, duration, n_pairs)
    d = rng.normal(0.0, jitter, n_pairs)
    a = np.concatenate([rng.uniform(0, duration, rng.poisson(bg * duration)), t_pair])
    b = np.concatenate([rng.uniform(0, duration, rng.poisson(bg * duration)), t_pair + d])
    inside = int(np.sum(np.abs(d) <= 0.5e-9))
    return {"tagger.ch1": np.sort(a), "tagger.ch2": np.sort(b)}, duration, inside


def test_cw_sidebands_agree_with_singles_and_recover_true_counts():
    tags, duration, inside = _cw_tags()
    spec = _coinc("pairs", "tagger.ch1", "tagger.ch2")
    clock = {c: 0.0 for c in tags}
    side = analyse_tags(tags, duration, [spec], clock, None, "sideband")["pairs"]
    single = analyse_tags(tags, duration, [spec], clock, None, "auto")["pairs"]
    assert side["accidental_model"] == CW_OFF_PEAK and single["accidental_model"] == "singles"
    assert side["sideband_offsets_s"] == pytest.approx([-20e-9, 20e-9])
    assert "off-peak" in side["timing_note"]
    acc_side = side["accidental_hz"] * duration
    acc_single = single["accidental_hz"] * duration
    sigma_acc = math.sqrt(2 * acc_side) / 2
    assert acc_single > 1000
    assert abs(acc_side - acc_single) < 4 * sigma_acc
    for r in (side, single):
        true = r["true_hz"] * duration
        assert abs(true - inside) < 4 * r["true_sigma_hz"] * duration
    assert side["true_sigma_hz"] > single["true_sigma_hz"]


def test_cw_sideband_offsets_follow_width_and_short_records_fall_back():
    tags, duration, _ = _cw_tags(duration=0.05)
    clock = {c: 0.0 for c in tags}
    spec = {**_coinc("pairs", "tagger.ch1", "tagger.ch2"), "correlation_width": "10 ns"}
    r = analyse_tags(tags, duration, [spec], clock, None, "sideband")["pairs"]
    assert r["sideband_offsets_s"] == pytest.approx([-50e-9, 50e-9])
    wide = {**spec, "window": "5 ns", "correlation_width": "1 ns"}
    r = analyse_tags(tags, duration, [wide], clock, None, "sideband")["pairs"]
    assert r["sideband_offsets_s"] == pytest.approx([-50e-9, 50e-9])
    short = {c: v[v < 1e-6] for c, v in tags.items()}
    r = analyse_tags(short, 1e-6, [spec], clock, None, "sideband")["pairs"]
    assert r["accidental_model"] == "singles" and "shorter than" in r["timing_note"]
    assert "sideband_offsets_s" not in r


def test_measured_cw_sidebands_through_the_measurement():
    tags, duration, inside = _cw_tags(duration=0.2)
    setup = _spdc({"accidentals": "sideband", "pulse_clock": "cw"})
    spec = _coinc("pairs", "tagger.ch1", "tagger.ch2")
    r = analyse_measured_tags(setup, tags, duration, {}, None, [spec])["pairs"]
    assert r["accidental_model"] == CW_OFF_PEAK and r["accidental_choice"] == "sideband"
    assert "no common pulse clock" not in r["timing_note"]
    assert abs(r["true_hz"] * duration - inside) < 4 * r["true_sigma_hz"] * duration
    pulsed = _spdc({"accidentals": "sideband", "pulse_clock": {"kind": "rate", "rate": "80 MHz"}})
    r = analyse_measured_tags(pulsed, tags, duration, {}, None, [spec])["pairs"]
    assert r["accidental_model"] == "sideband" and "sideband_offsets_s" not in r
