from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from sequoia.errors import Issue
from sequoia.gui.analysis_tabs import save_text
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
WAIT = 90000


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(Library(ROOT / "library"), settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


def open_template(qtbot, w, name):
    assert w.open_path(ROOT / "templates" / f"{name}.yaml")
    qtbot.waitUntil(lambda: w.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


def sweep_done(qtbot, tab):
    qtbot.waitUntil(lambda: not tab.runner.busy and tab.result is not None, timeout=WAIT)


def test_first_sweep_target_has_a_unit_and_runs_clean(window, qtbot):
    open_template(qtbot, window, "malus_law")
    tab = window.results.sweeps
    row = tab.rows[0]
    assert row.target.count() > 0
    cand = row.candidates[row.target.currentIndex()]
    assert row.unit.text() == cand.unit
    assert row.start.value() != row.stop.value()
    tab.run()
    sweep_done(qtbot, tab)
    assert not tab.result.failures


def test_mismatched_unit_is_rejected_before_any_point_runs(window, qtbot):
    open_template(qtbot, window, "malus_law")
    tab = window.results.sweeps
    row = tab.rows[0]
    row.target.setCurrentText("settings:hwp.angle")
    row.unit.setText("mm")
    tab.run()
    assert tab.result is None
    assert not tab.runner.busy
    assert tab.status.text()


def test_switching_project_clears_every_analysis_tab(window, qtbot):
    open_template(qtbot, window, "malus_law")
    r = window.results
    r.sweeps.run()
    sweep_done(qtbot, r.sweeps)
    old_targets = {c.target for c in r.sweeps.rows[0].candidates}
    assert old_targets
    open_template(qtbot, window, "bell_test")
    assert r.sweeps.result is None
    new_targets = {c.target for c in r.sweeps.rows[0].candidates}
    assert new_targets and new_targets != old_targets
    assert r.sweeps.rows[0].target.currentText() in new_targets
    assert r.sweeps.metric.count() == 0 or all(
        "malus" not in r.sweeps.metric.itemText(i) for i in range(r.sweeps.metric.count())
    )
    r.sweeps.run()
    sweep_done(qtbot, r.sweeps)
    assert not r.sweeps.result.failures


def test_monte_carlo_and_gaussian_results_do_not_survive_a_project_switch(window, qtbot):
    open_template(qtbot, window, "bell_test")
    r = window.results
    r.montecarlo.duration.setValue(0.05)
    r.montecarlo.seed.setValue(7)
    r.montecarlo.run()
    qtbot.waitUntil(lambda: r.montecarlo.outcome is not None, timeout=WAIT)
    r.gaussian.sweep.setChecked(False)
    r.gaussian.run()
    qtbot.waitUntil(lambda: r.gaussian.outcome is not None, timeout=WAIT)
    assert r.montecarlo.table.rowCount() > 0 and r.gaussian.table.rowCount() > 0
    open_template(qtbot, window, "malus_law")
    assert r.montecarlo.outcome is None and r.gaussian.outcome is None
    assert r.montecarlo.table.rowCount() == 0 and r.gaussian.table.rowCount() == 0
    assert r.montecarlo.analysis.count() == 0


def test_result_finishing_after_a_project_switch_is_dropped(window, qtbot):
    open_template(qtbot, window, "bell_test")
    r = window.results
    r.montecarlo.duration.setValue(0.5)
    r.montecarlo.run()
    open_template(qtbot, window, "malus_law")
    qtbot.waitUntil(lambda: not r.montecarlo.runner.busy, timeout=WAIT)
    qtbot.wait(50)
    assert r.montecarlo.outcome is None
    assert r.montecarlo.table.rowCount() == 0


def test_results_are_marked_outdated_after_an_edit(window, qtbot):
    open_template(qtbot, window, "bell_test")
    r = window.results
    r.gaussian.sweep.setChecked(False)
    r.gaussian.run()
    qtbot.waitUntil(lambda: r.gaussian.outcome is not None, timeout=WAIT)
    fresh = r.gaussian.status.text()
    assert "outdated" not in fresh
    window.document.set_setting("hwp_a", "angle", "10 deg")
    qtbot.waitUntil(lambda: "outdated" in r.gaussian.status.text(), timeout=WAIT)


def test_photon_populations_in_polarization_and_spectra_tabs(window, qtbot):
    open_template(qtbot, window, "bell_test")
    r = window.results
    pol = r.polarization
    assert pol.combo.count() > 0
    photon_links = [
        pol.combo.itemText(i)
        for i in range(pol.combo.count())
        if pol.photon_entries(pol.outcome, pol.combo.itemText(i))
    ]
    assert photon_links
    pol.select_link(photon_links[0])
    options = [pol.population.itemText(i) for i in range(pol.population.count())]
    assert any(o.startswith("photons: ") for o in options)
    pol.population.setCurrentText(next(o for o in options if o.startswith("photons: ")))
    assert pol.ellipse.stokes is not None and pol.ellipse.stokes[0] > 0
    assert "single-photon DOP" in pol.note.text()
    spec = r.spectra
    spec.select_link(photon_links[0])
    options = [spec.population.itemText(i) for i in range(spec.population.count())]
    spec.population.setCurrentText(next(o for o in options if o.startswith("photons: ")))
    assert spec.photon_curves
    x, y = spec.photon_curves[0].getData()
    assert len(x) > 1 and np.nanmax(y) > 0
    assert not r.tab_errors


def test_engine_issues_reach_the_diagnostics_table(window, qtbot):
    open_template(qtbot, window, "hbt_heralded")
    r = window.results
    before = r.diagnostics.rowCount()
    r.set_engine_issues(
        "Gaussian", [Issue("warning", "gaussian_truncated", "'crystal'", "dropped 1% of pairs")]
    )
    assert r.diagnostics.rowCount() == before + 1
    rows = [
        (r.diagnostics.item(i, 1).text(), r.diagnostics.item(i, 5).text())
        for i in range(r.diagnostics.rowCount())
    ]
    assert ("Gaussian", "dropped 1% of pairs") in rows
    r.set_engine_issues("Gaussian", [])
    assert r.diagnostics.rowCount() == before


def test_export_failures_show_a_message_and_keep_the_app_alive(qtbot, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "missing" / "out.csv"), "")),
    )
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1])))
    parent = QWidget()
    qtbot.addWidget(parent)
    assert save_text(parent, "Export", "x.csv", "a,b\n") is False
    assert seen == ["Cannot save"]
    good = tmp_path / "ok.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(good), ""))
    )
    assert save_text(parent, "Export", "x.csv", "a,b\n") is True
    assert good.read_text() == "a,b\n"
