import json
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from sequoia.analysis.chsh import run_chsh
from sequoia.gui.document import SetupDocument
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[2]
LIB = Library(ROOT / "library")
WAIT = 120000


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(LIB, settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


def open_template(qtbot, w, name, folder="templates"):
    assert w.open_path(ROOT / folder / f"{name}.yaml")
    qtbot.waitUntil(lambda: w.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


def settle(qtbot, t):
    qtbot.waitUntil(lambda: not t.runner.busy, timeout=WAIT)
    qtbot.wait(30)


def doc_of(w) -> SetupDocument:
    return w.controller.document


def test_measurement_edit_is_saved_undone_and_invalidates(window, qtbot, tmp_path):
    open_template(qtbot, window, "spdc_car")
    doc = doc_of(window)
    g = window.results.gaussian
    g.sweep.setChecked(False)
    g.run()
    settle(qtbot, g)
    assert "outdated" not in g.status.text()
    key = doc.key()
    doc.set_measurement({"coincidence_window": "2 ns", "acquisition_duration": "4 s"})
    assert doc.key() != key
    assert "outdated" in g.status.text()
    with pytest.raises(ValueError):
        doc.set_measurement({"coincidence_window": "-1 ns"})
    doc.save(tmp_path / "exp.yaml")
    assert load_setup(tmp_path / "exp.yaml").measurement == doc.setup.measurement
    doc.undo_stack.undo()
    assert doc.setup.measurement is None


def test_analyses_may_omit_the_window_and_name_detectors(doc):
    with pytest.raises(ValueError):
        doc.set_analyses([{"name": "c", "type": "coincidences", "channels": ["a.x", "b.y"]}])
    doc.set_measurement({"coincidence_window": "1 ns", "channels": {"det": "tagger.ch1"}})
    doc.set_analyses([{"name": "c", "type": "coincidences", "channels": ["det", "b.y"]}])
    with pytest.raises(ValueError):
        doc.set_measurement(None)


def test_rename_keeps_measurement_references(window, qtbot):
    open_template(qtbot, window, "spdc_car")
    doc = doc_of(window)
    doc.set_measurement(
        {"channels": {"spad_s": "tagger.ch1"}, "conditions": [{"name": "H", "herald": ["spad_s"]}]}
    )
    doc.rename_instance("spad_s", "spad_x")
    doc.rename_instance("tagger", "tt")
    m = doc.setup.measurement
    assert m.channels == {"spad_x": "tt.ch1"} and m.conditions[0].herald == ["spad_x"]


def test_base_simulation_issues_are_bound_to_their_run(window, qtbot):
    open_template(qtbot, window, "fock_hom")
    out = window.controller.last_outcome
    assert out.issues and out.run_id.startswith(f"rates-r{out.revision}-")
    assert all(i.engine == "rates" and i.run_id == out.run_id for i in out.issues)


def test_gaussian_card_record_and_tagged_issues(window, qtbot):
    open_template(qtbot, window, "spdc_car")
    g = window.results.gaussian
    seen = []
    g.on_issues = seen.append
    g.sweep.setChecked(False)
    g.run()
    settle(qtbot, g)
    card = g.model_card.text()
    assert g.model_card.isVisibleTo(g)
    for word in ("Statistics window", "Coincidence", "dead time", "Normalization", "placeholder"):
        assert word in card
    assert "exact photon statistics" not in card.lower()
    quantities = [g.table.item(r, 1).text() for r in range(g.table.rowCount())]
    assert any("probability per" in q for q in quantities)
    assert any("[Hz]" in q for q in quantities)
    assert g.record is not None and g.record.engine == "gaussian"
    assert all(i.engine == "gaussian" and i.run_id == g.record.run_id for i in seen[-1])


def test_export_after_edit_exports_the_old_run(window, qtbot, tmp_path):
    open_template(qtbot, window, "spdc_car")
    mc = window.results.montecarlo
    mc.duration.setValue(0.05)
    mc.run()
    settle(qtbot, mc)
    old = mc.record
    doc_of(window).set_override("pump", "power", "40 mW")
    assert mc.results_outdated()
    assert mc.export_run(tmp_path / "bundle")
    assert "predates later edits" in mc.status.text()
    saved = (tmp_path / "bundle" / "setup.yaml").read_text()
    assert "40 mW" not in saved
    record = json.loads((tmp_path / "bundle" / "run.json").read_text())
    assert record["run_id"] == old.run_id
    assert record["settings"]["duration_s"] == pytest.approx(0.05)
    assert record["settings"]["seed"] is not None


def test_pin_and_compare_lists_changed_inputs(window, qtbot):
    open_template(qtbot, window, "spdc_car")
    g = window.results.gaussian
    g.sweep.setChecked(False)
    assert not g.export_run_button.isEnabled()
    g.run()
    settle(qtbot, g)
    g.pin_baseline()
    assert g.compare_button.isEnabled()
    g.scale.setValue(2.0)
    g.run()
    settle(qtbot, g)
    c = g.compare_with_baseline()
    assert any("pump_scale" in d for d in c.input_differences)
    assert not c.matches
    assert g.compare_dialog is not None and g.compare_dialog.isVisible()
    g.compare_dialog.close()


def test_mc_tab_reads_duration_and_plans(window, qtbot):
    open_template(qtbot, window, "spdc_car")
    doc_of(window).set_measurement({"acquisition_duration": "0.05 s", "repeats": 3})
    mc = window.results.montecarlo
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    mc.set_outcome(None)
    assert mc.duration.value() == pytest.approx(0.05)
    mc.run()
    settle(qtbot, mc)
    assert "full width" in mc.model_card.text()
    mc.plan_target.setCurrentIndex(1)
    mc.plan_value.setValue(1000)
    plan = mc.plan()
    rate = mc.outcome.analyses["pairs"]["total_hz"]
    assert plan.time_per_setting_s == pytest.approx(1000 / rate)
    assert plan.total_time_s == pytest.approx(3 * plan.time_per_setting_s)
    assert "3 repeat" in mc.plan_label.text()


def test_chsh_card_plan_and_record(window, qtbot):
    open_template(qtbot, window, "bell_test", "setups")
    c = window.results.chsh
    c.set_outcome(window.controller.last_outcome)
    c.time.setValue(1.0)
    c.run()
    settle(qtbot, c)
    assert "per setting pair" in c.model_card.text()
    assert "full width" in c.model_card.text()
    assert "raw coincidences in 1 s" in c.table.horizontalHeaderItem(3).text()
    assert c.record.outputs["S"] == pytest.approx(c.result.s)
    c.plan_target.setCurrentIndex(0)
    c.plan_value.setValue(c.result.s_sigma / 2)
    plan = c.plan()
    expected = (c.result.s_sigma / c.plan_value.value()) ** 2
    assert plan.time_per_setting_s == pytest.approx(expected)
    again = run_chsh(
        load_setup(ROOT / "setups" / "bell_test.yaml"),
        LIB,
        "settings:hwp_a.angle",
        "settings:hwp_b.angle",
        list(c.result.settings_a),
        list(c.result.settings_b),
        integration_time=plan.time_per_setting_s,
    )
    assert again.s_sigma == pytest.approx(plan.s_sigma, rel=1e-9)
    c.plan_target.setCurrentIndex(3)
    c.plan_value.setValue(40.0)
    assert c.plan().time_per_setting_s == pytest.approx(10.0)


def test_sweep_record_and_labels(window, qtbot, tmp_path):
    open_template(qtbot, window, "spdc_car")
    s = window.results.sweeps
    s.set_outcome(window.controller.last_outcome)
    row = s.rows[0]
    row.target.setCurrentText("overrides:pump.power")
    row.points.setValue(3)
    s.run()
    settle(qtbot, s)
    assert s.record is not None and s.record.engine == "sweep"
    assert "rate-engine simulation" in s.model_card.text()
    s.metric.setCurrentText("analysis:pairs.true_hz")
    assert "[Hz]" in s.plot.getPlotItem().getAxis("left").labelText
    assert s.export_run(tmp_path / "sweep")
    assert (tmp_path / "sweep" / "data" / "sweep.csv").exists()
