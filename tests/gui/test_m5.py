from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from sequoia.gui.beamcard import build_link_infos
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
WAIT = 60000


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(
        Library(ROOT / "library"), ROOT / "setups" / "laser_fiber_coupling.yaml", settings=settings
    )
    w.confirm_discard = lambda _w: "discard"
    qtbot.addWidget(w)
    w.show()
    qtbot.waitUntil(lambda: w.controller.last_outcome is not None, timeout=WAIT)
    return w


def test_link_infos_and_wavelength_colour(window, qtbot):
    qtbot.waitUntil(lambda: bool(window.scene.link_infos), timeout=WAIT)
    infos = window.scene.link_infos
    assert infos["s1"].power_w == pytest.approx(5e-3)
    assert infos["s1"].color is not None
    assert infos["s1"].wavelength_m == pytest.approx(780e-9, rel=1e-3)
    assert infos["f1"].coupling is not None
    assert any(k == "power" for k, _ in infos["s1"].lines())
    assert build_link_infos(None) == {}


def test_hover_card(window, qtbot):
    qtbot.waitUntil(lambda: bool(window.scene.link_infos), timeout=WAIT)
    from PySide6.QtCore import QPoint

    window.scene.show_link_card("s2", QPoint(100, 100))
    assert window.scene._card is not None and window.scene._card.isVisible()
    window.scene.hide_link_card()
    assert not window.scene._card.isVisible()


def test_plot_tabs_fill(window, qtbot):
    r = window.results
    assert not r.tab_errors
    c = r.caustic
    c.select_link("s5")
    assert c.data is not None and len(c.data.z) > 0
    assert np.all(c.data.w_x > 0)
    r.polarization.select_link("s3")
    assert r.polarization.ellipse.stokes is not None
    assert r.spectra.combo.count() > 0
    assert r.budget.steps.rowCount() >= 3
    assert "Caustic" in [r.tabs.tabText(i) for i in range(r.tabs.count())]


def test_sweep_tab_runs(window, qtbot):
    sweeps = window.results.sweeps
    sweeps.reload_candidates()
    row = sweeps.rows[0]
    row.target.setCurrentText("link:s2.length")
    row.start.setValue(140)
    row.stop.setValue(160)
    row.points.setValue(5)
    sweeps.run()
    qtbot.waitUntil(lambda: sweeps.result is not None, timeout=WAIT)
    assert sweeps.result.shape == (5,)
    sweeps.metric.setCurrentText("link:f1.coupling_efficiency")
    assert "link:f1.coupling_efficiency" in sweeps.result.data
    assert sweeps.result.to_csv().count("\n") == 6


def test_optimize_apply_edits_document(window, qtbot):
    window.document.set_link_field("s2", "length", "190 mm")
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    window.show_optimizer()
    dialog = window.results.sweeps.optimizer
    dialog.objective.setCurrentText("link:f1.coupling_efficiency")
    row = dialog.rows[0]
    row.target.setCurrentText("link:s2.length")
    row.start.setValue(100)
    row.stop.setValue(250)
    dialog.run()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=WAIT)
    assert dialog.result.objective > 0.95
    dialog.apply()
    length = window.document.setup.links[1].length.si
    assert length == pytest.approx(0.150, abs=0.005)
    assert window.document.undo_stack.count() >= 1
