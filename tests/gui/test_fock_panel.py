from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from sequoia.fock import Circuit, Experiment, save_experiment
from sequoia.gui.fock_panel import FockTab, parse_filter

GUI_DIR = Path(__file__).resolve().parents[3] / "percevalGUI-v0 0 11"
WAIT = 30000


@pytest.fixture
def tab(qtbot):
    t = FockTab()
    t.confirm_replace = lambda _t: "discard"
    qtbot.addWidget(t)
    t.show()
    yield t
    t.shutdown()


def run_and_wait(tab, qtbot):
    tab.run()
    qtbot.waitUntil(lambda: not tab.runner.busy, timeout=WAIT)


def test_default_run_fills_table_and_plot(tab, qtbot):
    run_and_wait(tab, qtbot)
    assert tab.result is not None
    assert tab.results_table.rowCount() == 2
    assert tab.bars is not None
    assert "total probability" in tab.success_label.text()
    assert tab.run_button.isEnabled() and not tab.cancel_button.isEnabled()


def test_tags_distinguish_photons(tab, qtbot):
    tab.input_table.item(0, 2).setText("a")
    tab.input_table.item(1, 2).setText("b")
    run_and_wait(tab, qtbot)
    probs = tab.result.probabilities
    assert probs[(1, 1)] == pytest.approx(0.5)


def test_edit_circuit_rows(tab, qtbot):
    tab.circuit_table.item(0, 2).setText("0")
    run_and_wait(tab, qtbot)
    assert tab.result.probabilities == {(1, 1): pytest.approx(1.0)}
    tab.new_type.setCurrentText("loss")
    tab.add_element()
    assert tab.circuit_table.rowCount() == 2
    tab.circuit_table.item(1, 2).setText("0.5")
    tab.circuit_table.selectRow(1)
    tab.move_element(-1)
    assert tab.circuit_table.item(0, 0).text() == "loss"
    tab.remove_element()
    assert tab.circuit_table.rowCount() == 1


def test_conditional_filter_reports_success_probability(tab, qtbot):
    run_and_wait(tab, qtbot)
    tab.filter_edit.setText("0=2")
    assert "success probability 0.5" in tab.success_label.text()
    texts = [tab.results_table.item(r, 3).text() for r in range(tab.results_table.rowCount())]
    assert sorted(texts) == ["", "1"]
    assert parse_filter("0=1, 1>=1") == {0: ("=", 1), 1: (">=", 1)}


def test_invalid_input_reported_not_raised(tab, qtbot):
    tab.circuit_table.item(0, 1).setText("0,zero")
    tab.run()
    assert not tab.runner.busy
    assert "invalid input" in tab.status.text()


def test_resource_limit_message(tab, qtbot):
    tab.n_modes.setValue(3)
    for r in range(3):
        tab.input_table.item(r, 1).setText("4")
    tab.max_photons.setValue(3)
    run_and_wait(tab, qtbot)
    assert tab.status.text().startswith("failed")
    assert "photons exceed" in tab.message_label.text()


def test_save_open_roundtrip(tab, tmp_path, qtbot, monkeypatch):
    tab.input_table.item(0, 1).setText("1")
    tab.input_table.item(1, 1).setText("1")
    tab.shots.setValue(100)
    path = tmp_path / "c.yaml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    tab.save_dialog()
    assert path.exists()
    tab.input_table.item(0, 1).setText("0")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    tab.open_dialog()
    assert tab.input_table.item(0, 1).text() == "1"
    assert tab.shots.value() == 100
    run_and_wait(tab, qtbot)
    assert sum(tab.result.samples.counts.values()) == 100


def test_io_errors_become_messages(tab, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert tab.open_path(bad) is False
    assert tab.import_path(bad) is False
    assert tab.save_path(tmp_path / "no" / "dir" / "x.yaml") is False
    assert len(shown) == 3


def test_import_perceval_circuit_and_run(tab, qtbot, monkeypatch):
    if not GUI_DIR.is_dir():
        pytest.skip("Perceval GUI folder not found")
    assert tab.import_path(GUI_DIR / "4p5m.json")
    assert tab.n_modes.value() == 5
    assert tab.circuit_table.rowCount() > 0
    for r, k in enumerate([1, 1, 1, 1, 0]):
        tab.input_table.item(r, 1).setText(str(k))
    run_and_wait(tab, qtbot)
    assert tab.result.mass == pytest.approx(1.0)
    assert tab.results_table.rowCount() == 50


def test_set_outcome_is_noop_and_shutdown(tab):
    tab.set_outcome(None)
    tab.set_outcome(object())
    tab.shutdown()


def test_cancel_during_run(tab, qtbot):
    tab.n_modes.setValue(8)
    for r in range(8):
        tab.input_table.item(r, 1).setText("1")
    tab.max_photons.setValue(10)
    tab.shots.setValue(0)
    tab.run()
    tab.cancel_button.click()
    qtbot.waitUntil(lambda: not tab.runner.busy, timeout=WAIT)
    assert tab.run_button.isEnabled()


def test_loads_experiment_object(tab, tmp_path, qtbot):
    exp = Experiment(
        Circuit(3).beam_splitter(0, 1).beam_splitter(1, 2), [1, 0, 1], name="three modes"
    )
    path = tmp_path / "e.json"
    save_experiment(exp, path)
    assert tab.open_path(path)
    assert tab.n_modes.value() == 3
    run_and_wait(tab, qtbot)
    assert tab.result.mass == pytest.approx(1.0)


def test_labels_and_detector_groups_survive_the_editor(tab):
    from sequoia.fock import DetectorSpec, PhaseShifter, Threshold

    c = Circuit(2).beam_splitter(0, 1)
    c.add(PhaseShifter(0, 0.5, "arm"))
    exp = Experiment(
        c, [1, 1], detectors=[DetectorSpec((0, 1), Threshold(efficiency=0.7))], name="grouped"
    )
    tab.set_experiment(exp)
    assert tab.circuit_table.item(1, 4).text() == "arm"
    out = tab.experiment()
    assert out.circuit.elements[1].label == "arm"
    assert [s.modes for s in out.detectors] == [(0, 1)]
    assert isinstance(out.detectors[0].model, Threshold)
    assert out.detectors[0].model.efficiency == pytest.approx(0.7)


def test_open_and_save_keep_element_labels(tab, tmp_path):
    from sequoia.fock import PhaseShifter

    c = Circuit(2).beam_splitter(0, 1)
    c.add(PhaseShifter(1, 0.25, "link_a"))
    tab.set_experiment(Experiment(c, [1, 0], name="labelled"))
    path = tmp_path / "c.yaml"
    assert tab.save_path(path)
    tab.set_experiment(Experiment(Circuit(2).beam_splitter(0, 1), [1, 1]))
    assert tab.open_path(path)
    assert tab.circuit_table.item(1, 4).text() == "link_a"
