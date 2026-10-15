from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
WAIT = 120000


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(Library(ROOT / "library"), settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


def open_template(qtbot, w, path):
    assert w.open_path(ROOT / path)
    qtbot.waitUntil(lambda: w.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


def settle(qtbot, tab):
    qtbot.waitUntil(lambda: not tab.runner.busy, timeout=WAIT)
    qtbot.wait(30)


def test_tab_is_named_photon_counts_and_defaults_to_setup_mode(window):
    r = window.results
    titles = [r.tabs.tabText(i) for i in range(r.tabs.count())]
    assert "Photon counts" in titles
    assert "Photon states" not in titles
    assert r.fock.mode_box.currentText() == "From setup"
    assert "Photon counts" in window.act_fock.text()
    window.show_fock()
    assert r.tabs.currentWidget() is r.fock


def test_setup_without_photon_source_is_reported_cleanly(window, qtbot):
    open_template(qtbot, window, "templates/malus_law.yaml")
    tab = window.results.fock
    tab.import_from_setup()
    settle(qtbot, tab)
    assert tab.components_table.rowCount() > 0
    assert "cannot use the drawn circuit" in tab.status.text()
    assert "photon" in tab.notes_view.toPlainText()
    tab.run()
    settle(qtbot, tab)
    assert tab.setup_result is None
    assert tab.states_table.rowCount() == 0


def test_import_and_run_hom_fills_every_table(window, qtbot):
    open_template(qtbot, window, "templates/hom.yaml")
    tab = window.results.fock
    tab.shots.setValue(500)
    tab.import_from_setup()
    settle(qtbot, tab)
    n_expected = len(window.document.setup.instances) + len(window.document.setup.links)
    assert tab.components_table.rowCount() == n_expected
    ids = {tab.components_table.item(r, 0).text() for r in range(tab.components_table.rowCount())}
    assert {"crystal", "pbs", "fbs", "spad_1", "spad_2"} <= ids
    assert tab.inputs_table.rowCount() == 1
    assert tab.inputs_table.item(0, 1).text() == "pair"
    assert float(tab.inputs_table.item(0, 3).text()) > 0
    tab.run()
    settle(qtbot, tab)
    res = tab.setup_result
    assert res is not None
    assert res.mass == pytest.approx(1.0, abs=1e-3)
    assert tab.states_table.rowCount() == len(res.detected)
    assert tab.detector_table.rowCount() == 2
    assert tab.clicks_table.rowCount() >= 2
    assert tab.marginals_table.rowCount() > 0
    assert tab.samples_table.rowCount() > 0
    assert tab.cond_table.rowCount() == 2
    assert tab.setup_bars is not None
    assert "window" in tab.notes_view.toPlainText()
    tab.state_filter.setText("spad_1=1")
    assert "success probability" in tab.state_summary.text()
    tab.cond_table.cellWidget(0, 1).setCurrentText("=")
    assert "success probability" in tab.cond_summary.text()
    assert tab.cond_result.rowCount() > 0


def test_extract_bell_test_to_editor_and_run_there(window, qtbot):
    open_template(qtbot, window, "templates/bell_test.yaml")
    tab = window.results.fock
    tab.extract_to_editor()
    settle(qtbot, tab)
    assert tab.mode_box.currentText() == "Circuit editor"
    assert tab.circuit_table.rowCount() > 10
    labels = {tab.circuit_table.item(r, 4).text() for r in range(tab.circuit_table.rowCount())}
    assert {"hwp_a", "pbs_a", "f_at"} <= labels
    assert not tab.mixture_label.isHidden()
    assert tab.extraction.verified
    tab.run()
    settle(qtbot, tab)
    assert tab.result is not None
    assert tab.results_table.rowCount() > 0
    assert tab.result.mass == pytest.approx(1.0, abs=1e-3)
    tab.clear_mixture()
    assert tab.mixture_label.isHidden()


def test_fock_template_runs_and_counts_scale_with_duration(window, qtbot):
    open_template(qtbot, window, "templates/fock_hom.yaml")
    tab = window.results.fock
    tab.duration.setValue(2.0)
    tab.run()
    settle(qtbot, tab)
    res = tab.setup_result
    assert res is not None
    assert res.setup_experiment.rate_hz == pytest.approx(80e6)
    s = max(res.detected, key=res.detected.get)
    assert res.counts.total[s] == pytest.approx(res.detected[s] * 80e6 * 2.0)
    assert tab.inputs_table.rowCount() == 2


def test_project_switch_clears_everything(window, qtbot):
    open_template(qtbot, window, "templates/fock_hom.yaml")
    tab = window.results.fock
    tab.run()
    settle(qtbot, tab)
    assert tab.states_table.rowCount() > 0
    open_template(qtbot, window, "templates/malus_law.yaml")
    assert tab.setup_result is None
    assert tab.setup_exp is None
    for table in (
        tab.components_table,
        tab.inputs_table,
        tab.states_table,
        tab.clicks_table,
        tab.detector_table,
        tab.marginals_table,
        tab.cond_result,
        tab.samples_table,
    ):
        assert table.rowCount() == 0
    assert tab.notes_view.toPlainText() == ""
    assert tab.status.text() == ""
