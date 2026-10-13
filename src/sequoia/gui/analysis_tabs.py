import math
import os
import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from scipy.constants import c as C_LIGHT

from sequoia.analysis.budget import ideal_budget, photon_budget, power_budget
from sequoia.analysis.profile import caustic
from sequoia.analysis.sweep import metrics_of
from sequoia.engines.cache import RunCache
from sequoia.engines.outcome import SimulationOutcome
from sequoia.gui.beamcard import build_link_infos
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, eng, percent, plain

pg.setConfigOptions(antialias=True)

Provider = Callable[[], tuple]


def style_plot(plot: pg.PlotWidget, x: str, y: str) -> None:
    pal = QApplication.palette()
    plot.setBackground(pal.base().color())
    item = plot.getPlotItem()
    fg = pal.text().color()
    for axis in ("left", "bottom"):
        item.getAxis(axis).setPen(fg)
        item.getAxis(axis).setTextPen(fg)
    for axis in ("left", "bottom"):
        item.getAxis(axis).enableAutoSIPrefix(False)
    item.setLabel("bottom", x)
    item.setLabel("left", y)
    item.showGrid(x=True, y=True, alpha=0.25)


CLASSICAL_FIELD = "classical field"


def photon_stokes(coherency: np.ndarray) -> np.ndarray:
    w = np.asarray(coherency).sum(axis=0)
    return np.array(
        [
            (w[0, 0] + w[1, 1]).real,
            (w[0, 0] - w[1, 1]).real,
            2.0 * w[0, 1].real,
            -2.0 * w[0, 1].imag,
        ]
    )


class LinkTab(QWidget):
    """A tab with a link selector above a plot; subclasses implement ``redraw``.

    With ``photon_views`` the tab can also show the pair photons of a link (a reduced
    single-photon state per source and role) instead of the classical field.
    """

    photon_views = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.outcome: SimulationOutcome | None = None
        self.combo = QComboBox()
        self.combo.setMinimumWidth(160)
        self.population = QComboBox()
        self.population.setMinimumWidth(150)
        self.population.setVisible(self.photon_views)
        self.note = QLabel("")
        self.note.setStyleSheet("color: palette(placeholder-text);")
        top = QHBoxLayout()
        top.setContentsMargins(6, 4, 6, 0)
        top.addWidget(QLabel("Link"))
        top.addWidget(self.combo)
        top.addWidget(self.population)
        top.addWidget(self.note, 1)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.addLayout(top)
        self.combo.currentIndexChanged.connect(self._link_changed)
        self.population.currentIndexChanged.connect(lambda _i: self._redraw_current())

    def classical_links(self, outcome: SimulationOutcome) -> list[str]:
        classical = outcome.classical
        if classical is None:
            return []
        return [
            lid
            for lid, s in classical.links.items()
            if s.kind != "cable"
            and s.end is not None
            and getattr(s.end, "mode", None) is not None
            and hasattr(s.end, "wavelengths")
        ]

    def photon_entries(self, outcome: SimulationOutcome | None, link_id: str) -> dict[str, Any]:
        rates = getattr(outcome, "rates", None)
        if rates is None:
            return {}
        return dict(rates.links.get(link_id, {}).get("quantum", {}))

    def link_choices(self, outcome: SimulationOutcome) -> list[str]:
        choices = self.classical_links(outcome)
        if self.photon_views and outcome.rates is not None:
            for lid, entry in outcome.rates.links.items():
                if entry.get("quantum") and lid not in choices:
                    choices.append(lid)
        return choices

    def select_link(self, link_id: str) -> None:
        index = self.combo.findText(link_id)
        if index >= 0:
            self.combo.setCurrentIndex(index)

    def set_outcome(self, outcome: SimulationOutcome | None) -> None:
        self.outcome = outcome
        previous = self.combo.currentText()
        choices = self.link_choices(outcome) if outcome is not None else []
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(choices)
        if previous in choices:
            self.combo.setCurrentText(previous)
        elif choices:
            self.combo.setCurrentIndex(len(choices) - 1)
        self.combo.blockSignals(False)
        self._refresh_population()
        self._safe_redraw()

    def _link_changed(self, _index: int) -> None:
        self._refresh_population()
        self._safe_redraw()

    def _refresh_population(self) -> None:
        if not self.photon_views:
            return
        previous = self.population.currentText()
        options: list[str] = []
        link = self.combo.currentText()
        if self.outcome is not None and link:
            if link in self.classical_links(self.outcome):
                options.append(CLASSICAL_FIELD)
            entries = list(self.photon_entries(self.outcome, link))
            options += [f"photons: {k}" for k in entries]
            if len(entries) > 1:
                options.append("photons: all")
        self.population.blockSignals(True)
        self.population.clear()
        self.population.addItems(options)
        if previous in options:
            self.population.setCurrentText(previous)
        self.population.blockSignals(False)

    def _redraw_current(self) -> None:
        self._safe_redraw()

    def photon_selection(self) -> dict[str, Any] | None:
        text = self.population.currentText()
        if not self.photon_views or not text.startswith("photons: "):
            return None
        entries = self.photon_entries(self.outcome, self.combo.currentText())
        key = text.removeprefix("photons: ")
        if key == "all":
            return entries
        return {key: entries[key]} if key in entries else {}

    def _safe_redraw(self) -> None:
        self.note.setText("")
        if self.outcome is None or not self.combo.currentText():
            self.clear()
            return
        photons = self.photon_selection()
        if photons is not None:
            self.redraw_photons(self.outcome, self.combo.currentText(), photons)
            return
        self.redraw(self.outcome, self.combo.currentText())

    def clear(self) -> None:
        pass

    def redraw_photons(self, outcome: SimulationOutcome, link_id: str, entries: dict) -> None:
        self.clear()

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        raise NotImplementedError


class CausticTab(LinkTab):
    """Beam envelope (1/e^2 radius, mirrored) along the optical path with apertures and waists."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "distance along the path [mm]", "beam radius [mm]")
        self.plot.addLegend(offset=(-10, 10))
        blue, red = pg.mkPen("#1f77b4", width=2), pg.mkPen("#d62728", width=1.5)
        self.upper_x = self.plot.plot(pen=blue, name="w (x)")
        self.lower_x = self.plot.plot(pen=blue)
        self.upper_y = self.plot.plot(
            pen=pg.mkPen("#d62728", width=1.5, style=Qt.PenStyle.DashLine), name="w (y)"
        )
        self.lower_y = self.plot.plot(
            pen=pg.mkPen("#d62728", width=1.5, style=Qt.PenStyle.DashLine)
        )
        self.fill = pg.FillBetweenItem(self.upper_x, self.lower_x, brush=(31, 119, 180, 45))
        self.plot.addItem(self.fill)
        self._red = red
        self.items: list = []
        self.layout_.addWidget(self.plot, 1)
        self.data = None

    def clear(self) -> None:
        for curve in (self.upper_x, self.lower_x, self.upper_y, self.lower_y):
            curve.setData([], [])
        self._clear_items()
        self.data = None

    def _clear_items(self) -> None:
        for item in self.items:
            self.plot.removeItem(item)
        self.items = []

    def _add(self, item) -> None:
        self.plot.addItem(item)
        self.items.append(item)

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        self._clear_items()
        data = caustic(outcome.classical, link_id)
        self.data = data
        if len(data.z) == 0:
            self.clear()
            self.note.setText("no free-space path ends at this link")
            return
        z = data.z * 1e3
        for curve, y in (
            (self.upper_x, data.w_x * 1e3),
            (self.lower_x, -data.w_x * 1e3),
            (self.upper_y, data.w_y * 1e3),
            (self.lower_y, -data.w_y * 1e3),
        ):
            curve.setData(z, y)
        for zz, label in data.markers:
            line = pg.InfiniteLine(
                zz * 1e3,
                angle=90,
                pen=pg.mkPen((128, 128, 128, 120), style=Qt.PenStyle.DotLine),
                label=label,
                labelOpts={"position": 0.97, "rotateAxis": (1, 0), "anchors": [(1, 1), (1, 1)]},
            )
            self._add(line)
        for zz, radius, _name in data.apertures:
            half = radius * 1e3
            xs = [zz * 1e3] * 2
            self._add(pg.PlotCurveItem(xs, [-half, half], pen=pg.mkPen("#444", width=4), name=None))
            for sign in (-1, 1):
                self._add(
                    pg.PlotCurveItem(
                        [zz * 1e3 - 4, zz * 1e3 + 4],
                        [sign * half] * 2,
                        pen=pg.mkPen("#444", width=3),
                    )
                )
        for zz, w0 in data.waists:
            text = pg.TextItem(f"w0 = {eng(w0, 'm', 3)}", color="#1f77b4", anchor=(0.5, 1.2))
            text.setPos(zz * 1e3, w0 * 1e3)
            self._add(text)
            dot = pg.ScatterPlotItem(
                [zz * 1e3, zz * 1e3], [w0 * 1e3, -w0 * 1e3], size=7, brush="#1f77b4"
            )
            self._add(dot)
        top = float(max(data.w_x.max(), data.w_y.max())) * 1e3
        top = max(top * 1.6, 1e-6)
        self.plot.setXRange(float(z.min()), float(z.max()), padding=0.02)
        self.plot.setYRange(-top, top, padding=0)
        self.note.setText(
            " → ".join(data.links) + "   (bars: clear apertures; envelope = 1/e² beam radius)"
        )


class _Ellipse(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.stokes: np.ndarray | None = None
        self.setMinimumSize(150, 150)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = QApplication.palette()
        pad = self.fontMetrics().height() + 8
        side = min(self.width(), self.height() - pad) - 12
        c = QPointF(self.width() / 2, (self.height() - pad) / 2 + 4)
        r = max(side / 2, 10.0)
        axis = QColor(pal.text().color())
        axis.setAlpha(150)
        p.setPen(QPen(axis, 1.2))
        p.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
        p.drawText(QPointF(c.x() + r - 12, c.y() - 4), "H")
        p.drawText(QPointF(c.x() + 4, c.y() - r + 12), "V")
        s = self.stokes
        if s is None or s[0] <= 0:
            return
        pol = float(np.linalg.norm(s[1:]))
        if pol <= 1e-9 * s[0]:
            p.setPen(pal.text().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "unpolarized")
            return
        psi = 0.5 * math.atan2(s[2], s[1])
        chi = 0.5 * math.asin(max(-1.0, min(1.0, s[3] / pol)))
        a, b = 0.9 * r * math.cos(chi), 0.9 * r * abs(math.sin(chi))
        p.save()
        p.translate(c)
        p.rotate(-math.degrees(psi))
        p.setPen(QPen(QColor("#1f77b4"), 2))
        p.drawEllipse(QRectF(-a, -b, 2 * a, 2 * b))
        p.restore()
        p.setPen(pal.text().color())
        circ = s[3] / pol
        hand = "right" if circ > 1e-3 else "left" if circ < -1e-3 else "linear"
        p.drawText(
            QPointF(8, self.height() - 8),
            f"ψ {math.degrees(psi):.1f}°  χ {math.degrees(chi):.1f}°  {hand}",
        )


class _Poincare(QWidget):
    DEFAULT_VIEW = (-35.0, 22.0)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points: dict[str, np.ndarray] = {}
        self.selected = ""
        self.azimuth, self.elevation = self.DEFAULT_VIEW
        self._drag: QPointF | None = None
        self.setMinimumSize(190, 170)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to rotate, double-click to reset the view")

    def mousePressEvent(self, event) -> None:
        self._drag = event.position()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            return
        pos = event.position()
        self.azimuth += (pos.x() - self._drag.x()) * 0.7
        self.elevation = max(-90.0, min(90.0, self.elevation + (pos.y() - self._drag.y()) * 0.7))
        self._drag = pos
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        self.azimuth, self.elevation = self.DEFAULT_VIEW
        self.update()

    def _project(self, v: np.ndarray) -> tuple[float, float, float]:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        x, y, z = v
        x1 = x * math.cos(az) - y * math.sin(az)
        y1 = x * math.sin(az) + y * math.cos(az)
        return x1, z * math.cos(el) - y1 * math.sin(el), y1 * math.cos(el) + z * math.sin(el)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = QApplication.palette()
        label = self.fontMetrics().height()
        r = max(min(self.width() - 3 * label, self.height() - 4 * label) / 2, 20.0)
        c = QPointF(self.width() / 2, (self.height() - label) / 2)

        def pt(v):
            x, y, _z = self._project(np.asarray(v, dtype=float))
            return QPointF(c.x() + r * x, c.y() - r * y)

        ring_color = QColor(pal.text().color())
        ring_color.setAlpha(110)
        p.setPen(QPen(ring_color, 1))
        p.drawEllipse(c, r, r)
        t = np.linspace(0, 2 * np.pi, 73)
        for ring in (
            lambda a: (np.cos(a), np.sin(a), 0 * a),
            lambda a: (np.cos(a), 0 * a, np.sin(a)),
            lambda a: (0 * a, np.cos(a), np.sin(a)),
        ):
            xs, ys, zs = ring(t)
            for i in range(len(t) - 1):
                p.drawLine(pt((xs[i], ys[i], zs[i])), pt((xs[i + 1], ys[i + 1], zs[i + 1])))
        p.setPen(pal.text().color())
        for name, v in (("S1", (1.12, 0, 0)), ("S2", (0, 1.12, 0)), ("S3", (0, 0, 1.12))):
            q = pt(v)
            p.drawText(QPointF(q.x() - 8, q.y() + 4), name)
        axis_pen = QPen(pal.text().color(), 1.2)
        for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            p.setPen(axis_pen)
            p.drawLine(pt((-v[0], -v[1], -v[2])), pt(v))
        p.setPen(pal.placeholderText().color())
        p.drawText(QPointF(8, self.height() - 8), "H/V: S1   D/A: S2   R/L: S3   (drag to rotate)")
        for name, s in self.points.items():
            if s[0] <= 0:
                continue
            q = pt(s[1:] / s[0])
            chosen = name == self.selected
            p.setBrush(QColor("#d62728") if chosen else QColor("#1f77b4"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(q, 5 if chosen else 3, 5 if chosen else 3)
            if chosen:
                p.setPen(pal.text().color())
                p.drawText(QPointF(q.x() + 7, q.y() - 4), name)


class PolarizationTab(LinkTab):
    photon_views = True

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ellipse = _Ellipse()
        self.sphere = _Poincare()
        row = QHBoxLayout()
        row.addWidget(self.ellipse, 1)
        row.addWidget(self.sphere, 1)
        self.layout_.addLayout(row, 1)

    def clear(self) -> None:
        self.ellipse.stokes = None
        self.sphere.points = {}
        self.ellipse.update()
        self.sphere.update()

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        infos = build_link_infos(outcome)
        info = infos.get(link_id)
        self.ellipse.stokes = None if info is None else info.stokes
        self.sphere.points = {k: v.stokes for k, v in infos.items() if v.kind != "cable"}
        self.sphere.selected = link_id
        if info is not None:
            self.note.setText(f"DOP {info.dop:.3f}" if info.dop is not None else "")
        self.ellipse.update()
        self.sphere.update()

    def redraw_photons(self, outcome: SimulationOutcome, link_id: str, entries: dict) -> None:
        total = np.zeros(4)
        for data in entries.values():
            total += photon_stokes(data["coherency"])
        points = {}
        for lid in outcome.rates.links:
            for key, data in self.photon_entries(outcome, lid).items():
                points[f"{lid} {key}"] = photon_stokes(data["coherency"])
        self.ellipse.stokes = total if total[0] > 0 else None
        self.sphere.points = points
        self.sphere.selected = ""
        dop = float(np.linalg.norm(total[1:]) / total[0]) if total[0] > 0 else 0.0
        self.note.setText(
            f"single-photon DOP {dop:.3f}; {eng(total[0], 'Hz')} photons "
            "(a pair partner of an entangled photon looks unpolarized)"
        )
        self.ellipse.update()
        self.sphere.update()


class SpectraTab(LinkTab):
    photon_views = True
    PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "wavelength [nm]", "power [mW per sample]")
        self.curve = self.plot.plot(
            pen=pg.mkPen("#1f77b4", width=2), fillLevel=0, brush=(31, 119, 180, 60)
        )
        self.photon_curves: list = []
        self.layout_.addWidget(self.plot, 1)

    def _drop_photon_curves(self) -> None:
        for item in self.photon_curves:
            self.plot.removeItem(item)
        self.photon_curves = []

    def clear(self) -> None:
        self._drop_photon_curves()
        self.curve.setData([], [])

    def redraw_photons(self, outcome: SimulationOutcome, link_id: str, entries: dict) -> None:
        self._drop_photon_curves()
        self.curve.setData([], [])
        style_plot(self.plot, "wavelength [nm]", "photon rate [Hz per sample]")
        total = 0.0
        for n, (key, data) in enumerate(entries.items()):
            nu = np.asarray(data["nu"])
            valid = nu > 0
            wl = C_LIGHT / nu[valid] * 1e9
            rate = np.einsum("nii->n", np.asarray(data["coherency"])).real[valid]
            order = np.argsort(wl)
            color = self.PALETTE[n % len(self.PALETTE)]
            item = self.plot.plot(
                wl[order],
                rate[order],
                pen=pg.mkPen(color, width=2),
                name=key,
                fillLevel=0,
                brush=pg.mkBrush(color + "40"),
            )
            self.photon_curves.append(item)
            total += float(rate.sum())
        self.note.setText(f"{len(entries)} population(s), total {eng(total, 'Hz')}")

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        self._drop_photon_curves()
        style_plot(self.plot, "wavelength [nm]", "power [mW per sample]")
        state = outcome.classical.links[link_id]
        light = state.end
        if light is None:
            self.clear()
            return
        wl = np.asarray(light.wavelengths) * 1e9
        order = np.argsort(wl)
        self.curve.setData(wl[order], np.asarray(light.power_spectrum)[order] * 1e3)
        self.note.setText(f"{len(wl)} samples, total {eng(light.power, 'W')}")


class HistogramTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.outcome: SimulationOutcome | None = None
        self.combo = QComboBox()
        self.note = QLabel("")
        self.note.setStyleSheet("color: palette(placeholder-text);")
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "delay [ns]", "coincidence rate per bin [Hz]")
        self.curve = self.plot.plot(
            pen=pg.mkPen("#1f77b4", width=1),
            stepMode="center",
            fillLevel=0,
            brush=(31, 119, 180, 80),
        )
        top = QHBoxLayout()
        top.setContentsMargins(6, 4, 6, 0)
        top.addWidget(QLabel("Analysis"))
        top.addWidget(self.combo)
        top.addWidget(self.note, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        self.combo.currentIndexChanged.connect(lambda _i: self.redraw())

    def set_outcome(self, outcome: SimulationOutcome | None) -> None:
        self.outcome = outcome
        previous = self.combo.currentText()
        names = []
        if outcome is not None and outcome.rates is not None:
            names = [
                n
                for n, a in outcome.rates.analyses.items()
                if "tau_s" in a and "rate_hz" in a and len(np.atleast_1d(a["tau_s"])) > 1
            ]
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(names)
        if previous in names:
            self.combo.setCurrentText(previous)
        self.combo.blockSignals(False)
        self.redraw()

    def redraw(self) -> None:
        name = self.combo.currentText()
        if self.outcome is None or self.outcome.rates is None or not name:
            self.curve.setData([0, 1], [0])
            self.note.setText("add a 'g2' or 'histogram' analysis to see one")
            return
        a = self.outcome.rates.analyses[name]
        tau = np.asarray(a["tau_s"], dtype=float) * 1e9
        rate = np.asarray(a["rate_hz"], dtype=float)
        width = float(a.get("bin_width_s", (tau[1] - tau[0]) * 1e-9)) * 1e9
        edges = np.concatenate([tau - width / 2, [tau[-1] + width / 2]])
        self.curve.setData(edges, rate)
        peak = a.get("g2_peak")
        self.note.setText(f"g2 peak {plain(peak)}" if peak is not None else "")


class BudgetTab(LinkTab):
    def __init__(self, provider: Provider, parent=None) -> None:
        super().__init__(parent)
        self.provider = provider
        self.runner = JobRunner(self)
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "element along the path", "power [mW]")
        self.light = QComboBox()
        self.light.addItems(["classical power", "pair photons"])
        self.light.currentIndexChanged.connect(lambda _i: self._safe_redraw())
        self.steps = _make_table(["Element", "Kind", "In", "Out", "Transmission"])
        self.ideal = _make_table(["Part", "Type", "Metric if ideal", "Gain", "Gain / baseline"])
        self.metric = QComboBox()
        self.metric.setMinimumWidth(220)
        self.compute = QPushButton("Ideal-parts budget")
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.status = QLabel("")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(120)
        self.progress.setVisible(False)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 0, 6, 0)
        bar.addWidget(QLabel("Metric"))
        bar.addWidget(self.metric)
        bar.addWidget(self.compute)
        bar.addWidget(self.cancel)
        bar.addWidget(self.progress)
        bar.addWidget(self.status, 1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(bar)
        rl.addWidget(self.ideal, 1)
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.plot)
        left.addWidget(self.steps)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        self.layout_.addWidget(split, 1)
        self.layout_.itemAt(0).layout().insertWidget(2, self.light)
        self.compute.clicked.connect(self.run_ideal)
        self.cancel.clicked.connect(self.runner.cancel)
        self.runner.done.connect(self._ideal_done)
        self.runner.failed.connect(lambda m: self._ideal_end(f"failed: {m.splitlines()[0]}"))
        self.runner.cancelled.connect(lambda: self._ideal_end("cancelled"))
        self.runner.progress.connect(self._progress)
        self.last_budget = None

    def set_outcome(self, outcome) -> None:
        previous = self.metric.currentText()
        names = sorted(metrics_of(outcome)) if outcome is not None and outcome.classical else []
        self.metric.clear()
        self.metric.addItems(names)
        if previous in names:
            self.metric.setCurrentText(previous)
        else:
            for preferred in ("true_hz", "photocurrent_A", "coupling_efficiency", "power_W"):
                match = next((n for n in names if n.endswith("." + preferred)), None)
                if match:
                    self.metric.setCurrentText(match)
                    break
        super().set_outcome(outcome)

    def clear(self) -> None:
        self.plot.clear()
        _fill(self.steps, [])

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        pairs = self.light.currentIndex() == 1
        steps = (
            photon_budget(outcome, link_id) if pairs else power_budget(outcome.classical, link_id)
        )
        self.plot.clear()
        unit, scale = ("photons/s", 1.0) if pairs else ("W", 1.0)
        rows = []
        cumulative = []
        for step in steps:
            rows.append(
                [
                    _NumItem(step.element),
                    _NumItem(step.kind),
                    _NumItem(eng(step.power_in * scale, unit), step.power_in),
                    _NumItem(eng(step.power_out * scale, unit), step.power_out),
                    _NumItem(percent(step.transmission), step.transmission),
                ]
            )
        first = next((s.power_in for s in steps if s.power_in > 0), 0.0)
        for step in steps:
            cumulative.append(step.power_out / first if first > 0 else 0.0)
        positive = [c for c in cumulative if c > 0]
        if positive:
            x = np.arange(len(cumulative), dtype=float)
            y = np.log10(np.maximum(cumulative, min(positive) * 1e-3))
            self.plot.plot(
                x, y, pen=pg.mkPen("#1f77b4", width=2), symbol="o", symbolBrush="#d62728"
            )
        item = self.plot.getPlotItem()
        item.setLabel("left", "log10 (remaining fraction)")
        item.getAxis("bottom").setTicks([[(i, s.element) for i, s in enumerate(steps)]])
        _fill(self.steps, rows)
        if steps and first > 0:
            self.note.setText(f"end-to-end {percent(steps[-1].power_out / first)}")

    def run_ideal(self) -> None:
        metric = self.metric.currentText()
        setup, library = self.provider()
        if not metric or setup is None or self.runner.busy:
            return

        def job(should_cancel, progress):
            progress(0, 0)
            return ideal_budget(
                setup, library, metric, cache=RunCache(), should_cancel=should_cancel
            )

        self.compute.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("running...")
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _ideal_end(self, text: str) -> None:
        self.compute.setEnabled(True)
        self.cancel.setEnabled(False)
        self.progress.setVisible(False)
        self.status.setText(text)

    def _ideal_done(self, budget) -> None:
        self.last_budget = budget
        rows = []
        base = budget.baseline
        for e in budget.entries:
            rel = e.gain / base if base else float("nan")
            rows.append(
                [
                    _NumItem(e.instance),
                    _NumItem(e.part_type),
                    _NumItem(plain(e.value), e.value),
                    _NumItem(plain(e.gain), e.gain),
                    _NumItem(percent(rel), rel),
                ]
            )
        rows.append(
            [
                _NumItem("everything ideal"),
                _NumItem(""),
                _NumItem(plain(budget.all_ideal), budget.all_ideal),
                _NumItem(plain(budget.all_ideal - base), budget.all_ideal - base),
                _NumItem(
                    percent((budget.all_ideal - base) / base if base else float("nan")),
                ),
            ]
        )
        _fill(self.ideal, rows)
        self._ideal_end(f"baseline {plain(base)}")


def save_text(parent: QWidget, title: str, default: str, text: str) -> bool:
    path, _ = QFileDialog.getSaveFileName(parent, title, default)
    if not path:
        return False
    try:
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".sequoia-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except OSError as exc:
        QMessageBox.warning(parent, "Cannot save", f"{path}\n\n{exc.strerror or exc}")
        return False
    return True
