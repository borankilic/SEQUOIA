from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

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


def test_hom_template_selects_the_delay_scan(window, qtbot):
    open_template(qtbot, window, "hom")
    tab = window.results.sweeps
    row = tab.rows[0]
    assert row.target.currentText() == "link:delay_arm.length"
    assert row.unit.text() == "mm" and row.points.value() == 41 and not row.log.isChecked()
    assert row.start.value() < 100.66 < row.stop.value()
    assert (row.start.value() + row.stop.value()) / 2 == pytest.approx(100.66, abs=1e-6)
    assert tab.metric.currentText() == "analysis:hom.true_hz"
    assert "HOM dip" in tab.status.text()
    assert "physical: >= 0 mm" in row.bounds.text()


def test_default_sweep_button_restores_the_task_sweep(window, qtbot):
    open_template(qtbot, window, "malus_law")
    tab = window.results.sweeps
    row = tab.rows[0]
    assert row.target.currentText() == "settings:hwp.angle"
    assert (row.start.value(), row.stop.value(), row.points.value()) == (0.0, 90.0, 46)
    assert tab.metric.currentText() == "probe:pm.power_W"
    row.target.setCurrentText("settings:pol.angle")
    assert (row.start.value(), row.stop.value(), row.points.value()) == (0.0, 180.0, 91)
    row.target.setCurrentText("overrides:laser.power")
    assert row.log.isChecked() and row.start.value() == pytest.approx(0.5)
    tab.default_button.click()
    assert row.target.currentText() == "settings:hwp.angle"
    assert not row.log.isChecked() and row.stop.value() == 90.0


def test_out_of_bound_sweep_is_refused_before_running(window, qtbot):
    open_template(qtbot, window, "malus_law")
    tab = window.results.sweeps
    row = tab.rows[0]
    row.target.setCurrentText("overrides:pol.transmission")
    row.stop.setValue(1.5)
    tab.run()
    assert not tab.runner.busy and tab.result is None
    assert "physical range (0 to 1)" in tab.status.text()
    row.target.setCurrentText("overrides:laser.power")
    row.start.setValue(0.0)
    tab.run()
    assert "log axis needs positive limits" in tab.status.text()


def test_optimizer_starts_at_the_physical_bounds(window, qtbot):
    open_template(qtbot, window, "malus_law")
    tab = window.results.sweeps
    tab.open_optimizer()
    dialog = tab.optimizer
    row = dialog.rows[0]
    row.target.setCurrentText("overrides:pol.transmission")
    assert (row.start.value(), row.stop.value()) == (0.0, 1.0)
    row.target.setCurrentText("settings:hwp.angle")
    assert (row.start.value(), row.stop.value()) == (0.0, 180.0)
    row.target.setCurrentText("overrides:pol.transmission")
    row.stop.setValue(2.0)
    dialog.run()
    assert not dialog.runner.busy
    assert "physical range" in dialog.summary.text()
