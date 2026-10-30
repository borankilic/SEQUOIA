import dataclasses
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia import runs
from sequoia.analysis.chsh import run_chsh
from sequoia.analysis.planning import PlanningError, plan_chsh, plan_probability, plan_rate
from sequoia.analysis.sweep import Axis
from sequoia.engines.cache import RunCache
from sequoia.engines.outcome import compute_outcome
from sequoia.fock.analysis import expected_counts, outcome_table
from sequoia.fock.engine import Distribution
from sequoia.library.store import Library
from sequoia.measurement import (
    Measurement,
    analyse_measured_tags,
    channel_map,
    check_measurement,
    effective_analyses,
    measurement_card,
    resolve_setup,
)
from sequoia.setup.io import load_setup, save_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")

MEASUREMENT = {
    "purpose": "heralding efficiency of the H/V pair",
    "channels": {"spad_s": "tagger.ch1", "spad_i": "tagger.ch2"},
    "coincidence_window": "2 ns",
    "centers": {"pairs": "0.1 ns"},
    "pulse_clock": {"kind": "rate", "rate": "80 MHz"},
    "acquisition_duration": "3 s",
    "repeats": 2,
    "accidentals": "sideband",
    "dead_time": "non-paralyzable",
    "conditions": [{"name": "H", "herald": ["tagger.ch1"], "veto": ["spad_i"]}],
}


def _spdc(measurement=None, **analysis):
    raw = setup_to_dict(load_setup(ROOT / "templates" / "spdc_car.yaml"))
    if analysis:
        raw["analyses"] = [{"name": "pairs", "type": "coincidences", **analysis}]
    if measurement is not None:
        raw["measurement"] = measurement
    return setup_from_dict(raw)


def _no_nan(text: str) -> None:
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


# U02: the measurement definition is a saved object


def test_saved_measurement_reopens_with_all_settings(tmp_path):
    setup = _spdc(MEASUREMENT)
    save_setup(setup, tmp_path / "exp.yaml")
    again = load_setup(tmp_path / "exp.yaml")
    assert again.measurement == setup.measurement
    m = again.measurement
    assert m.coincidence_window.si == pytest.approx(2e-9)
    assert m.window_convention == "full_width"
    assert m.pulse_clock.rate.si == pytest.approx(80e6)
    assert (m.acquisition_duration.si, m.repeats, m.accidentals) == (3.0, 2, "sideband")
    assert m.conditions[0].veto == ["spad_i"]
    assert channel_map(again)["spad_s"] == ("tagger.ch1", "measurement")


def test_old_files_load_and_save_unchanged(tmp_path):
    setup = load_setup(ROOT / "templates" / "spdc_car.yaml")
    assert setup.measurement is None
    assert "measurement" not in setup_to_dict(setup)
    assert resolve_setup(setup, LIB) is setup


@pytest.mark.parametrize(
    "bad",
    [
        {"coincidence_window": "0 ns"},
        {"acquisition_duration": "-1 s"},
        {"repeats": 0},
        {"pulse_clock": {"kind": "rate"}},
        {"pulse_clock": {"kind": "source"}},
        {"accidentals": "fit"},
        {"window_convention": "half_width"},
        {"conditions": [{"name": "H", "herald": ["a"]}, {"name": "H", "herald": ["b"]}]},
    ],
)
def test_measurement_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        Measurement.model_validate(bad)


def test_measurement_fills_only_what_an_analysis_leaves_out():
    setup = _spdc(MEASUREMENT, channels=["spad_s", "spad_i"])
    spec = effective_analyses(setup)[0]
    assert spec["channels"] == ["tagger.ch1", "tagger.ch2"]
    assert spec["window"] == pytest.approx(2e-9)
    assert spec["center"] == pytest.approx(0.1e-9)
    explicit = _spdc(MEASUREMENT, channels=["tagger.ch1", "tagger.ch2"], window="1 ns", center=0.0)
    spec = effective_analyses(explicit)[0]
    assert spec["window"] == "1 ns" and spec["center"] == 0.0


def test_rate_engine_reads_the_measurement_window():
    setup = _spdc(MEASUREMENT, channels=["spad_s", "spad_i"])
    out = compute_outcome(setup, LIB, RunCache(), None)
    assert not out.has_errors
    a = out.rates.analyses["pairs"]
    assert a["window_s"] == pytest.approx(2e-9)
    assert a["channels"] == ["tagger.ch1", "tagger.ch2"]
    narrow = dict(MEASUREMENT, coincidence_window="0.5 ns")
    b = compute_outcome(_spdc(narrow, channels=["spad_s", "spad_i"]), LIB, RunCache(), None)
    assert b.rates.analyses["pairs"]["accidental_hz"] < a["accidental_hz"]


def test_no_dead_time_model_zeroes_dead_times():
    m = dict(MEASUREMENT, dead_time="none")
    out = compute_outcome(_spdc(m, channels=["spad_s", "spad_i"]), LIB, RunCache(), None)
    ch = out.rates.channels["tagger.ch1"]
    assert ch["dead_time_detector_s"] == 0.0
    base = compute_outcome(_spdc(MEASUREMENT), LIB, RunCache(), None)
    assert base.rates.channels["tagger.ch1"]["dead_time_detector_s"] > 0


def test_tag_analysis_uses_the_measurement_estimator():
    setup = _spdc({"accidentals": "singles"})
    tags = runs.run_montecarlo(setup, LIB, duration_s=0.05, seed=1).tags
    pulsed = {c: 80e6 for c in tags.channels}
    ids = {c: "pump" for c in tags.channels}
    out = analyse_measured_tags(setup, tags.channels, tags.duration, pulsed, ids)["pairs"]
    assert out["accidental_model"] == "singles" and "not valid for pulsed" in out["timing_note"]
    side = _spdc({"accidentals": "sideband", "pulse_clock": {"kind": "rate", "rate": "80 MHz"}})
    out = analyse_measured_tags(side, tags.channels, tags.duration, {}, None)["pairs"]
    assert out["accidental_model"] == "sideband" and "80" in out["timing_note"]
    cw = _spdc({"accidentals": "sideband"})
    out = analyse_measured_tags(cw, tags.channels, tags.duration, {}, None)["pairs"]
    assert out["accidental_model"] == "singles" and "no common pulse clock" in out["timing_note"]


def test_measurement_references_are_checked():
    m = {
        "channels": {"nobody": "tagger.ch1", "spad_s": "tagger.ch4"},
        "pulse_clock": {"kind": "source", "source": "ghost"},
        "centers": {"missing": "1 ns"},
        "conditions": [{"name": "H", "herald": ["tagger.ch9"]}],
    }
    issues = check_measurement(_spdc(m))
    text = " ".join(str(i) for i in issues)
    assert {i.severity for i in issues} == {"warning"}
    for word in ("nobody", "tagger.ch4", "ghost", "missing", "tagger.ch9"):
        assert word in text


def test_measurement_card_names_every_time():
    setup = _spdc(MEASUREMENT)
    out = compute_outcome(setup, LIB, RunCache(), None)
    card = dict(measurement_card(setup, out))
    assert "full width" in card["Coincidence window"]
    assert (
        "per setting" in card["Acquisition duration"] and "2 repeat" in card["Acquisition duration"]
    )
    assert "CW" in card["Repetition period (sources)"]
    assert "80 MHz" in card["Pulse clock (tag analysis)"]
    assert "H: click on tagger.ch1" in card["Conditions"]
    assert not any(label.lower().strip() == "window" for label in card)


def test_mc_and_chsh_read_the_acquisition_duration(tmp_path, monkeypatch, capsys):
    from sequoia import cli

    path = tmp_path / "exp.yaml"
    save_setup(_spdc({"acquisition_duration": "0.05 s"}), path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sequoia",
            "run",
            str(path),
            "--library",
            str(ROOT / "library"),
            "--engine",
            "mc",
            "--seed",
            "3",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "0.05 s" in capsys.readouterr().out


# U09: run records and bundles


def test_json_has_no_nan_and_says_why():
    data = runs.json_safe({"car": math.nan, "g2": math.inf, "x": [1.0, math.nan], "ok": 1.0})
    assert data["car"] is None and data["car_status"] == "undefined: no accidental estimate"
    assert data["g2"] is None and data["g2_status"] == "infinite"
    assert data["x"] == [1.0, None] and "x_status" in data
    _no_nan(json.dumps(data, allow_nan=False))


def test_zero_accidentals_record_is_valid_json(tmp_path):
    setup = _spdc()
    out = compute_outcome(setup, LIB, RunCache(), None)
    rec = runs.record_from_outcome(out, setup, LIB)
    outputs = dict(rec.outputs)
    outputs["analysis:pairs.car"] = math.nan
    patched = runs.RunRecord(**{**rec.__dict__, "outputs": outputs})
    text = patched.to_json_text()
    _no_nan(text)
    data = json.loads(text)["outputs"]
    assert data["analysis:pairs.car"] is None
    assert data["analysis:pairs.car_status"] == "undefined: no accidental estimate"


def test_record_contents_and_issue_binding():
    setup = _spdc(MEASUREMENT)
    out, rec = runs.run_rates(setup, LIB)
    assert rec.engine == "rates" and rec.run_id == out.run_id
    assert rec.setup_sha256 and rec.library_fingerprint
    assert set(rec.parts) == set(runs.used_parts(setup))
    assert rec.measurement["coincidence_window"] == "2 ns"
    assert {"sequoia", "numpy", "scipy"} <= set(rec.versions)
    assert all(i.engine == "rates" and i.run_id == rec.run_id for i in rec.issues)
    assert all(i.engine == "rates" and i.run_id == out.run_id for i in out.issues)
    assert any(d["type"] for d in rec.detectors.values())
    with pytest.raises(TypeError):
        rec.settings["revision"] = 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.run_id = "x"


def _portable(bundle: Path, tmp_path: Path) -> Path:
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(bundle, moved)
    return moved


@pytest.mark.parametrize("engine", ["rates", "gaussian", "montecarlo", "sweep", "chsh"])
def test_bundle_reruns_without_the_author_library(engine, tmp_path):
    if engine == "chsh":
        setup = load_setup(ROOT / "setups" / "bell_test.yaml")
        rec = runs.run_chsh_record(
            setup,
            LIB,
            target_a="settings:hwp_a.angle",
            target_b="settings:hwp_b.angle",
            settings_a=[0.0, 22.5],
            settings_b=[11.25, 33.75],
            integration_time_s=2.0,
        ).record
    else:
        setup = _spdc(MEASUREMENT)
        if engine == "rates":
            rec = runs.run_rates(setup, LIB)[1]
        elif engine == "gaussian":
            rec = runs.run_gaussian(setup, LIB, pump_scale=0.5, sweep=True).record
        elif engine == "montecarlo":
            rec = runs.run_montecarlo(setup, LIB, duration_s=0.05).record
        else:
            axis = Axis("overrides:pump.power", np.linspace(5, 10, 3), "mW")
            rec = runs.run_sweep_record(setup, LIB, [axis], ["analysis:pairs.true_hz"]).record
    bundle = runs.write_bundle(rec, tmp_path / "bundle")
    for name in ("run.json", "setup.yaml", "summary.md", "library", "data"):
        assert (bundle / name).exists()
    _no_nan((bundle / "run.json").read_text())
    moved = _portable(bundle, tmp_path)
    again = runs.rerun_bundle(moved)
    assert again.comparison.matches, again.comparison.text()
    assert again.comparison.input_differences == ()
    assert dict(again.record.parts) == dict(rec.parts)
    assert Library(moved / "library").root == moved / "library"


def test_bundle_keeps_parts_as_they_were_at_the_run(tmp_path):
    lib_dir = tmp_path / "lib"
    shutil.copytree(ROOT / "library", lib_dir)
    library = Library(lib_dir)
    setup = _spdc()
    rec = runs.run_rates(setup, library)[1]
    part = lib_dir / "detectors" / "spad_si_generic" / "part.yaml"
    original = part.read_text()
    part.write_text(original.replace("efficiency", "efficiency", 1) + "\n# edited later\n")
    bundle = runs.write_bundle(rec, tmp_path / "bundle")
    copied = bundle / "library" / "detectors" / "spad_si_generic" / "part.yaml"
    assert copied.read_text() == original


def test_comparison_lists_inputs_that_differ():
    a = runs.run_rates(_spdc(), LIB)[1]
    raw = setup_to_dict(_spdc(MEASUREMENT))
    raw["instances"][0].setdefault("overrides", {})["power"] = "20 mW"
    b = runs.run_rates(setup_from_dict(raw), LIB)[1]
    c = runs.compare_records(a, b)
    text = "\n".join(c.input_differences)
    assert "instances[pump].overrides.power" in text
    assert "measurement coincidence_window" in text
    assert not c.matches
    row = {k: (x, y, d) for k, x, y, d in c.rows}["analysis:pairs.true_hz"]
    assert row[2] == pytest.approx(abs(row[1] - row[0]))
    same = runs.compare_records(a, runs.run_rates(_spdc(), LIB)[1])
    assert same.matches and same.input_differences == ()


def test_cli_export_and_rerun(tmp_path, monkeypatch, capsys):
    from sequoia import cli

    setup = tmp_path / "exp.yaml"
    save_setup(_spdc(MEASUREMENT), setup)
    for engine in ("rates", "gaussian", "mc"):
        out = tmp_path / f"run_{engine}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sequoia",
                "run",
                str(setup),
                "--library",
                str(ROOT / "library"),
                "--engine",
                engine,
                "--duration",
                "0.05",
                "--export-run",
                str(out),
            ],
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert "exported run" in capsys.readouterr().out
        monkeypatch.setattr(
            sys, "argv", ["sequoia", "rerun", str(_portable(out, tmp_path / engine))]
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0, capsys.readouterr().out
        assert "reproduced" in capsys.readouterr().out


# U08: acquisition planning


def test_halving_success_doubles_the_time():
    full = plan_probability(0.2, 1e6, success_probability=0.1, target_events=1e4)
    half = plan_probability(0.2, 1e6, success_probability=0.05, target_events=1e4)
    assert half.time_per_setting_s == pytest.approx(2 * full.time_per_setting_s)
    assert half.events_per_setting == pytest.approx(1e4)
    r1 = plan_rate(1e5, success_probability=0.1, target_relative=0.01)
    r2 = plan_rate(1e5, success_probability=0.05, target_relative=0.01)
    assert r2.time_per_setting_s == pytest.approx(2 * r1.time_per_setting_s)
    assert r1.events_per_setting == pytest.approx(1e4)


def test_fixed_pulse_uncertainty_agrees_with_fock_analysis():
    n = 1000
    for p in (0.0, 0.3, 0.5, 1.0):
        plan = plan_probability(p, n, time_s=1.0)
        counts_sigma = plan.sigma * plan.accepted_per_setting
        ec = expected_counts({(0,): p}, rate_hz=n, duration_s=1.0)
        row = outcome_table(Distribution({(0,): p}, 1), n_windows=n)[0]
        assert counts_sigma == pytest.approx(ec.sigma[(0,)], abs=1e-12)
        assert counts_sigma == pytest.approx(row["sigma"], abs=1e-12)
    poisson = plan_probability(1.0, n, time_s=1.0, model="poisson")
    assert poisson.sigma * n == pytest.approx(math.sqrt(n))
    inv = plan_probability(0.3, n, target_relative=0.05)
    assert inv.relative_precision == pytest.approx(0.05)


def test_plan_settings_and_split_are_explicit():
    per = plan_rate(100.0, time_s=10.0, settings=4, repeats=2)
    assert per.time_per_setting_s == 10.0 and per.total_time_s == 80.0
    tot = plan_rate(100.0, time_s=80.0, settings=4, repeats=2, split="total")
    assert tot.time_per_setting_s == 10.0
    with pytest.raises(PlanningError):
        plan_rate(0.0, target_events=10)
    with pytest.raises(PlanningError):
        plan_rate(10.0, time_s=1.0, target_events=10)


def _bell(t):
    return run_chsh(
        load_setup(ROOT / "setups" / "bell_test.yaml"),
        LIB,
        "settings:hwp_a.angle",
        "settings:hwp_b.angle",
        [0.0, 22.5],
        [11.25, 33.75],
        integration_time=t,
    )


def test_chsh_sigma_follows_inverse_sqrt_time_and_matches_run_chsh():
    base = _bell(1.0)
    for t in (4.0, 25.0):
        assert plan_chsh(base, time_s=t).s_sigma == pytest.approx(base.s_sigma / math.sqrt(t))
    plan = plan_chsh(base, target_sigma=base.s_sigma / 3)
    assert plan.time_per_setting_s == pytest.approx(9.0)
    assert plan.total_time_s == pytest.approx(36.0)
    assert _bell(plan.time_per_setting_s).s_sigma == pytest.approx(plan.s_sigma, rel=1e-9)
    z = plan_chsh(base, target_significance=10.0)
    assert _bell(z.time_per_setting_s).significance == pytest.approx(10.0, rel=1e-9)
    total = plan_chsh(base, time_s=40.0, split="total")
    assert total.time_per_setting_s == pytest.approx(10.0)
