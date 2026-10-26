import math
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPalette

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis.sweep import evaluate
from sequoia.gui.analysis_tabs import (
    LINE_CONTRAST,
    TEXT_CONTRAST,
    PolarizationTab,
    _Ellipse,
    _Poincare,
    contrast_ratio,
)
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[2]
LIB = Library(ROOT / "library")


@pytest.fixture(scope="module")
def bell_outcome():
    return evaluate(load_setup(ROOT / "setups" / "bell_test.yaml"), LIB)


def _palette(dark: bool) -> QPalette:
    pal = QPalette()
    window, text = (
        (QColor("#2b2b2b"), QColor("#e6e6e6")) if dark else (QColor("#f4f4f4"), QColor("#1a1a1a"))
    )
    pal.setColor(QPalette.ColorRole.Window, window)
    pal.setColor(QPalette.ColorRole.Base, window.darker(110) if dark else QColor("white"))
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Text, text)
    muted = QColor("#7f7f7f")
    pal.setColor(QPalette.ColorRole.PlaceholderText, muted)
    return pal


def _render(widget, size, ratio):
    widget.resize(*size)
    image = QImage(int(size[0] * ratio), int(size[1] * ratio), QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(ratio)
    image.fill(0)
    widget.render(image)
    return image


STATES = {
    "near_linear": np.array([1.0, math.cos(0.6), math.sin(0.6), 2e-4]),
    "near_circular": np.array([1.0, 3e-4, -2e-4, -0.99999]),
    "elliptical": np.array([1.0, 0.5, 0.5, -0.7071]),
}


@pytest.mark.parametrize("size", [(160, 140), (320, 260), (900, 700)])
@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("state", list(STATES))
def test_polarization_plots_fit_and_contrast(qtbot, size, dark, ratio, state):
    for cls in (_Ellipse, _Poincare):
        w = cls()
        qtbot.addWidget(w)
        w.setPalette(_palette(dark))
        s = STATES[state]
        if cls is _Ellipse:
            w.stokes = s
        else:
            w.points = {"link with a fairly long name": s, "other": np.array([1.0, 1, 0, 0])}
            w.selected = "link with a fairly long name"
        image = _render(w, size, ratio)
        assert image.width() == size[0] * ratio
        bounds = QRectF(0, 0, w.width(), w.height())
        assert w.text_rects
        for rect, text in w.text_rects:
            assert bounds.contains(rect), (cls.__name__, size, text, rect)
        bg = w.colors["background"]
        assert bg == w.palette().window().color()
        for key in ("text", "muted_text"):
            assert contrast_ratio(w.colors[key], bg) >= TEXT_CONTRAST
        for key in ("axis", "ring", "data", "selected"):
            assert contrast_ratio(w.colors[key], bg) >= LINE_CONTRAST
        corner = image.pixelColor(1, 1)
        assert corner.rgb() == bg.rgb()


def test_ellipse_captions_at_the_linear_and_circular_limits(qtbot):
    w = _Ellipse()
    qtbot.addWidget(w)
    for s, want, has_azimuth in (
        (STATES["near_linear"], "linear", True),
        (STATES["near_circular"], "circular R", False),
        (np.array([1.0, 0, 0, 1.0]), "circular L", False),
        (np.array([1.0, 0.0, 0.0, 0.0]), "unpolarized", False),
    ):
        w.stokes = s
        _render(w, (200, 180), 1)
        assert want in w.caption
        assert ("ψ" in w.caption) is has_azimuth
    w.stokes = STATES["elliptical"]
    _render(w, (200, 180), 1)
    assert "χ" in w.caption and w.caption.endswith("R")


def test_joint_view_shows_bell_pair_and_conditional_state(qtbot, bell_outcome):
    tab = PolarizationTab()
    qtbot.addWidget(tab)
    tab.set_outcome(bell_outcome)
    tab.views.setCurrentIndex(1)
    assert not tab.combo.isEnabled()
    tab.select_pair("s_a1", "s_b1")
    state = tab.selected_pair()
    assert state is not None and state["source"] == "source"
    assert tab.rho_table.item(0, 3).text().startswith("+0.485")
    text = tab.joint_metrics.text()
    assert "concurrence 0.955" in text and "Φ+ 0.977" in text
    assert "single-photon reduced state (entanglement not visible here)" in text
    info = tab.joint_info.text()
    assert "signal on s_a1" in info and "idler on s_b1" in info and "HH,HV,VH,VV" in info
    tab.analyser.setCurrentText("H")
    assert "p(a passes |H>) = 0.5000" in tab.cond_note.text()
    assert tab.cond_ellipse.stokes[1] == pytest.approx(0.97, abs=1e-6)
    tab.analyser.setCurrentText("D")
    assert tab.cond_sphere.points["b | a=D"][2] == pytest.approx(0.97, abs=1e-6)
    detector_items = [tab.pair_b.itemText(i) for i in range(tab.pair_b.count())]
    assert "f_bt → snspd_bt" in detector_items
    tab.select_pair("s_a1", "f_bt")
    tab.analyser.setCurrentText("V")
    assert "undefined" in tab.cond_note.text() or "p(a passes |V>)" in tab.cond_note.text()
    tab.resize(700, 500)
    tab.views.setCurrentIndex(0)
    assert tab.combo.isEnabled()


def test_single_view_is_labelled_as_reduced_state(qtbot, bell_outcome):
    tab = PolarizationTab()
    qtbot.addWidget(tab)
    tab.set_outcome(bell_outcome)
    tab.select_link("s_a1")
    options = [tab.population.itemText(i) for i in range(tab.population.count())]
    tab.population.setCurrentText(next(o for o in options if o.startswith("photons: ")))
    note = tab.note.text()
    assert "single-photon reduced state (entanglement not visible here)" in note
    assert "single-photon DOP 0.000" in note


def test_joint_view_without_pairs(qtbot):
    outcome = evaluate(load_setup(ROOT / "setups" / "mzi.yaml"), LIB)
    tab = PolarizationTab()
    qtbot.addWidget(tab)
    tab.set_outcome(outcome)
    assert tab.pair_a.count() == 0 and tab.selected_pair() is None
    assert "no two-photon state" in tab.joint_info.text()
    tab.set_outcome(None)
    assert tab.joint_info.text() == ""
