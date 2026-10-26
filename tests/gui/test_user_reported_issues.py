from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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


def settle(qtbot, w):
    qtbot.wait(250)
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


class WindowSpy(QObject):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.shown = []

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.Type.Show
            and hasattr(obj, "isWindow")
            and obj.isWindow()
            and obj is not self.main
            and not (int(obj.windowFlags()) & int(Qt.WindowType.Popup) == int(Qt.WindowType.Popup))
        ):
            self.shown.append(type(obj).__name__)
        return False


@pytest.mark.parametrize("name", ["hom", "bell_test", "laser_fiber_coupling", "malus_law"])
def test_selecting_parts_opens_no_stray_windows(window, qtbot, name):
    open_template(qtbot, window, name)
    spy = WindowSpy(window)
    QApplication.instance().installEventFilter(spy)
    try:
        for item in list(window.scene.component_items.values()):
            pos = window.view.mapFromScene(item.sceneBoundingRect().center())
            QTest.mouseClick(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
            qtbot.wait(5)
        for item in list(window.scene.link_items.values())[:4]:
            window.scene.select(item.link_id)
            qtbot.wait(5)
    finally:
        QApplication.instance().removeEventFilter(spy)
    assert spy.shown == []


def test_adding_parts_from_the_palette_opens_no_stray_windows(window, qtbot):
    open_template(qtbot, window, "hom")
    spy = WindowSpy(window)
    QApplication.instance().installEventFilter(spy)
    try:
        for ref in ("freespace/lens_f200mm", "polarization/hwp_810", "detectors/spad_si_generic"):
            assert window._add_at_center(ref)
            qtbot.wait(10)
    finally:
        QApplication.instance().removeEventFilter(spy)
    assert spy.shown == []


def test_probes_attach_from_the_palette(window, qtbot):
    open_template(qtbot, window, "laser_fiber_coupling")
    scene = window.scene
    link = next(iter(scene.link_items))
    scene.select(link)
    probe = window._add_at_center("probes/power_meter")
    assert probe in {p.id for p in window.document.setup.probes}
    assert next(p for p in window.document.setup.probes if p.id == probe).link == link
    laser = next(i.id for i in window.document.setup.instances if i.part.startswith("sources/"))
    scene.select(laser)
    second = window._add_at_center("probes/beam_profiler")
    outgoing = next(lk.id for lk in window.document.setup.links if lk.src.startswith(laser + "."))
    assert next(p for p in window.document.setup.probes if p.id == second).link == outgoing
    scene.select(None)
    assert window._add_at_center("probes/polarimeter") is None
    assert scene.pending_probe == "probes/polarimeter"
    target = scene.link_items[link]
    mid = target.shape().pointAtPercent(0.5)
    pos = window.view.mapFromScene(target.mapToScene(mid))
    QTest.mouseClick(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    assert scene.pending_probe is None
    kinds = {p.part for p in window.document.setup.probes}
    assert {"probes/power_meter", "probes/beam_profiler", "probes/polarimeter"} <= kinds
    point = target.mapToScene(mid)
    dropped = scene.add_part("probes/spectrometer", (0, 0), point)
    assert dropped is not None
    settle(qtbot, window)
    assert window.controller.last_outcome.ok


def test_single_line_spectrum_is_visible(window, qtbot):
    open_template(qtbot, window, "hom")
    spectra = window.results.spectra
    spectra.select_link("s_pump")
    if spectra.population.count():
        spectra.population.setCurrentIndex(0)
    assert "single-frequency line" in spectra.note.text()
    assert spectra.photon_curves
    x, y = spectra.photon_curves[0].getData()
    assert np.nanmax(y) > 0 and np.allclose(x, 405.0, atol=0.5)


def test_pump_divergence_changes_the_caustic(window, qtbot):
    open_template(qtbot, window, "hom")
    caustic = window.results.caustic
    caustic.select_link("s_collect")
    before = np.max(caustic.data.w_x)
    pump = next(i.id for i in window.document.setup.instances if i.part.startswith("sources/"))
    window.document.set_override(pump, "divergence", "3 mrad")
    settle(qtbot, window)
    caustic.select_link("s_collect")
    after = np.max(caustic.data.w_x)
    assert after > 1.5 * before
    window.document.set_override(pump, "divergence", "0.05 mrad")
    settle(qtbot, window)
    errors = [i for i in window.controller.last_outcome.issues if i.severity == "error"]
    assert errors and "M^2" in errors[0].message


def test_key_metrics_are_listed_individually(window, qtbot):
    open_template(qtbot, window, "hom")
    table = window.results.metrics
    rows = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(table.rowCount())}
    assert "CAR" in rows and "two-photon interference visibility" in rows
    assert float(rows["two-photon interference visibility"]) > 0.95
    open_template(qtbot, window, "hbt_heralded")
    names = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert "heralded g2(0)" in names and "heralding efficiency" in names
    open_template(qtbot, window, "bell_test")
    names = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert "correlation E" in names
    g = window.results.gaussian
    assert g.g2_plot is not g.plot


def test_simulation_run_exports_and_reruns(window, qtbot, tmp_path):
    from sequoia.runs import rerun_bundle

    open_template(qtbot, window, "hbt_heralded")
    assert "Model" in window.results.rates_card.text()
    assert window.export_run(str(tmp_path / "bundle"))
    bundle = next((tmp_path / "bundle").rglob("run.json")).parent
    result = rerun_bundle(bundle)
    assert result.comparison.matches, result.comparison.text()
