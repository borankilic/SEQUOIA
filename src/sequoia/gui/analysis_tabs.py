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
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.constants import c as C_LIGHT

from sequoia.analysis.budget import ideal_budget, photon_budget, power_budget
from sequoia.analysis.metrics import KETS
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


TEXT_CONTRAST = 4.5
LINE_CONTRAST = 3.0
NEAR_POLE = 1e-2


def _luminance(c: QColor) -> float:
    ch = [x / 255.0 for x in (c.red(), c.green(), c.blue())]
    ch = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def contrast_ratio(a: QColor, b: QColor) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    pairs = zip(a.getRgbF()[:3], b.getRgbF()[:3], strict=True)
    return QColor.fromRgbF(*[(1 - t) * x + t * y for x, y in pairs])


def _ink(color: QColor, bg: QColor, minimum: float) -> QColor:
    target = QColor("black") if _luminance(bg) > 0.18 else QColor("white")
    for i in range(21):
        c = _mix(color, target, i / 20)
        if contrast_ratio(c, bg) >= minimum:
            return c
    return target


def _plot_colors(widget: QWidget) -> dict[str, QColor]:
    pal = widget.palette()
    bg = pal.window().color()
    fg = _ink(pal.windowText().color(), bg, TEXT_CONTRAST)
    return {
        "background": bg,
        "text": fg,
        "muted_text": _ink(pal.placeholderText().color(), bg, TEXT_CONTRAST),
        "axis": _ink(_mix(fg, bg, 0.35), bg, LINE_CONTRAST),
        "ring": _ink(_mix(fg, bg, 0.55), bg, LINE_CONTRAST),
        "data": _ink(QColor("#1f77b4"), bg, LINE_CONTRAST),
        "selected": _ink(QColor("#d62728"), bg, LINE_CONTRAST),
    }


def _label(p: QPainter, widget: QWidget, rects: list, x: float, y: float, text: str) -> None:
    fm = widget.fontMetrics()
    room = max(widget.width() - 4, 1)
    if fm.horizontalAdvance(text) > room:
        text = fm.elidedText(text, Qt.TextElideMode.ElideRight, room)
    w, h = fm.horizontalAdvance(text), fm.height()
    box = QRectF(
        max(2.0, min(x, widget.width() - 2.0 - w)),
        max(2.0, min(y, widget.height() - 2.0 - h)),
        w,
        h,
    )
    flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    p.drawText(box, flags, text)
    rects.append((p.boundingRect(box, flags, text), text))


class _Ellipse(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.stokes: np.ndarray | None = None
        self.caption = ""
        self.text_rects: list[tuple[QRectF, str]] = []
        self.colors: dict[str, QColor] = {}
        self.setAutoFillBackground(True)
        self.setMinimumSize(120, 110)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        colors = self.colors = _plot_colors(self)
        self.text_rects = []
        self.caption = ""
        fm = self.fontMetrics()
        th = fm.height()
        top, bottom = th + 4, self.height() - th - 6
        side = fm.horizontalAdvance("H") + 8
        r = max(min((self.width() - 2 * side) / 2, (bottom - top) / 2), 10.0)
        c = QPointF(self.width() / 2, (top + bottom) / 2)
        p.setPen(QPen(colors["axis"], 1.2))
        p.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
        p.setPen(colors["text"])
        _label(p, self, self.text_rects, c.x() + r + 3, c.y() - th / 2, "H")
        _label(p, self, self.text_rects, c.x() + 4, c.y() - r - th + 2, "V")
        s = self.stokes
        if s is None or not s[0] > 0 or not np.all(np.isfinite(s)):
            return
        pol = float(np.linalg.norm(s[1:]))
        if pol <= 1e-9 * s[0]:
            self.caption = "unpolarized"
            _label(p, self, self.text_rects, 6, self.height() - th - 4, self.caption)
            return
        circ = max(-1.0, min(1.0, s[3] / pol))
        psi = 0.5 * math.atan2(s[2], s[1])
        chi = 0.5 * math.asin(circ)
        a, b = 0.9 * r * math.cos(chi), 0.9 * r * abs(math.sin(chi))
        p.save()
        p.translate(c)
        p.rotate(-math.degrees(psi))
        p.setPen(QPen(colors["data"], 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(-a, -b, 2 * a, 2 * b))
        p.restore()
        # R = (H - iV)/sqrt2 as in analysis/metrics.py: clockwise seen towards the source, S3 < 0
        hand = "R" if circ < 0 else "L"
        if abs(circ) < NEAR_POLE:
            self.caption = f"linear  ψ {math.degrees(psi):.1f}°"
        elif math.hypot(s[1], s[2]) / pol < NEAR_POLE:
            self.caption = f"circular {hand}"
        else:
            self.caption = f"ψ {math.degrees(psi):.1f}°  χ {math.degrees(chi):.1f}°  {hand}"
        p.setPen(colors["text"])
        _label(p, self, self.text_rects, 6, self.height() - th - 4, self.caption)


class _Poincare(QWidget):
    DEFAULT_VIEW = (-35.0, 22.0)
    FOOTER = "S1 H/V   S2 D/A   S3 L/R"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points: dict[str, np.ndarray] = {}
        self.selected = ""
        self.azimuth, self.elevation = self.DEFAULT_VIEW
        self._drag: QPointF | None = None
        self.text_rects: list[tuple[QRectF, str]] = []
        self.colors: dict[str, QColor] = {}
        self.setAutoFillBackground(True)
        self.setMinimumSize(120, 110)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(
            "Drag to rotate, double-click to reset the view. +S3 is L = (H + iV)/√2, -S3 is "
            "R = (H - iV)/√2 (clockwise seen looking towards the source)"
        )

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
        colors = self.colors = _plot_colors(self)
        self.text_rects = []
        fm = self.fontMetrics()
        th = fm.height()
        side = fm.horizontalAdvance("S1") + 6
        r = max(min((self.width() - 2 * side) / 2.24, (self.height() - 3 * th - 8) / 2.24), 12.0)
        c = QPointF(self.width() / 2, (self.height() - th - 4) / 2)

        def pt(v):
            x, y, _z = self._project(np.asarray(v, dtype=float))
            return QPointF(c.x() + r * x, c.y() - r * y)

        p.setPen(QPen(colors["ring"], 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
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
        p.setPen(QPen(colors["axis"], 1.2))
        for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            p.drawLine(pt((-v[0], -v[1], -v[2])), pt(v))
        p.setPen(colors["text"])
        for name, v in (("S1", (1.12, 0, 0)), ("S2", (0, 1.12, 0)), ("S3", (0, 0, 1.12))):
            q = pt(v)
            w = fm.horizontalAdvance(name)
            _label(p, self, self.text_rects, q.x() - w / 2, q.y() - th / 2, name)
        p.setPen(colors["muted_text"])
        _label(p, self, self.text_rects, 6, self.height() - th - 4, self.FOOTER)
        for name, s in self.points.items():
            if not s[0] > 0 or not np.all(np.isfinite(s)):
                continue
            q = pt(s[1:] / s[0])
            chosen = name == self.selected
            p.setBrush(colors["selected"] if chosen else colors["data"])
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(q, 5 if chosen else 3, 5 if chosen else 3)
            if chosen:
                p.setPen(colors["text"])
                _label(p, self, self.text_rects, q.x() + 7, q.y() - th, name)


BASIS = ["HH", "HV", "VH", "VV"]
BELL_NAMES = {"phi+": "Φ+", "phi-": "Φ−", "psi+": "Ψ+", "psi-": "Ψ−"}
REDUCED_NOTE = "single-photon reduced state (entanglement not visible here)"


class PolarizationTab(LinkTab):
    photon_views = True

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ellipse = _Ellipse()
        self.sphere = _Poincare()
        single = QWidget()
        row = QHBoxLayout(single)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.ellipse, 1)
        row.addWidget(self.sphere, 1)
        self.pair_a = QComboBox()
        self.pair_b = QComboBox()
        self.analyser = QComboBox()
        for label, ket in KETS.items():
            self.analyser.addItem(label)
            self.analyser.setItemData(
                self.analyser.count() - 1,
                f"photon a passes the analyser state |{label}> = "
                f"({ket[0].real:.3g}, {ket[1].real:+.3g}{ket[1].imag:+.3g}i)",
                Qt.ItemDataRole.ToolTipRole,
            )
        self.joint_info = QLabel("")
        self.joint_info.setWordWrap(True)
        self.joint_info.setStyleSheet("color: palette(placeholder-text);")
        self.rho_table = QTableWidget(4, 4)
        self.rho_table.setHorizontalHeaderLabels(BASIS)
        self.rho_table.setVerticalHeaderLabels(BASIS)
        self.rho_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.rho_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rho_table.setToolTip("two-photon polarization density matrix (first letter: photon a)")
        self.rho_table.setMaximumHeight(
            self.rho_table.horizontalHeader().sizeHint().height()
            + 4 * self.rho_table.verticalHeader().defaultSectionSize()
            + 2 * self.rho_table.frameWidth()
        )
        self.joint_metrics = QLabel("")
        self.joint_metrics.setWordWrap(True)
        self.joint_metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cond_note = QLabel("")
        self.cond_note.setWordWrap(True)
        self.cond_ellipse = _Ellipse()
        self.cond_sphere = _Poincare()
        joint = QWidget()
        jl = QVBoxLayout(joint)
        jl.setContentsMargins(6, 4, 6, 4)
        pick = QHBoxLayout()
        pick.addWidget(QLabel("Photon a"))
        pick.addWidget(self.pair_a, 1)
        pick.addWidget(QLabel("Photon b"))
        pick.addWidget(self.pair_b, 1)
        jl.addLayout(pick)
        jl.addWidget(self.joint_info)
        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Pair density matrix ρ (normalised on one photon per link)"))
        left.addWidget(self.rho_table)
        left.addWidget(self.joint_metrics)
        left.addStretch(1)
        right = QVBoxLayout()
        analyse = QHBoxLayout()
        analyse.addWidget(QLabel("Photon a analysed in"))
        analyse.addWidget(self.analyser)
        analyse.addStretch(1)
        right.addLayout(analyse)
        right.addWidget(self.cond_note)
        plots = QHBoxLayout()
        plots.addWidget(self.cond_ellipse, 1)
        plots.addWidget(self.cond_sphere, 1)
        right.addLayout(plots, 1)
        body.addLayout(left, 1)
        body.addLayout(right, 1)
        jl.addLayout(body, 1)
        self.views = QTabWidget()
        self.views.setDocumentMode(True)
        self.views.addTab(single, "Single link")
        self.views.addTab(joint, "Joint (pairs)")
        self.views.setTabToolTip(0, f"classical field or {REDUCED_NOTE}")
        self.views.setTabToolTip(1, "two-photon polarization state of a pair: entanglement")
        self.layout_.addWidget(self.views, 1)
        self.views.currentChanged.connect(self._view_changed)
        self.pair_a.currentIndexChanged.connect(lambda _i: self._refresh_partners())
        self.pair_b.currentIndexChanged.connect(lambda _i: self.redraw_joint())
        self.analyser.currentIndexChanged.connect(lambda _i: self.redraw_joint())

    def set_outcome(self, outcome: SimulationOutcome | None) -> None:
        super().set_outcome(outcome)
        self._refresh_pairs()

    def _view_changed(self, index: int) -> None:
        self.combo.setEnabled(index == 0)
        self.population.setEnabled(index == 0)

    def pair_states(self) -> dict:
        rates = getattr(self.outcome, "rates", None)
        return {} if rates is None else getattr(rates, "pair_states", {})

    def _link_text(self, link_id: str) -> str:
        classical = getattr(self.outcome, "classical", None)
        state = None if classical is None else classical.links.get(link_id)
        inst = None if state is None else state.dst.split(".")[0]
        rates = getattr(self.outcome, "rates", None)
        detectors = {} if rates is None else rates.detectors
        return f"{link_id} → {inst}" if inst in detectors else link_id

    def _fill_links(self, box: QComboBox, links: list[str]) -> None:
        previous = box.currentData()
        box.blockSignals(True)
        box.clear()
        for lid in links:
            box.addItem(self._link_text(lid), lid)
        if previous in links:
            box.setCurrentIndex(links.index(previous))
        box.blockSignals(False)

    def _refresh_pairs(self) -> None:
        firsts = list(dict.fromkeys(a for a, _b in self.pair_states()))
        self._fill_links(self.pair_a, firsts)
        self._refresh_partners()

    def _refresh_partners(self) -> None:
        a = self.pair_a.currentData()
        self._fill_links(self.pair_b, [y for x, y in self.pair_states() if x == a])
        self.redraw_joint()

    def selected_pair(self) -> dict | None:
        key = (self.pair_a.currentData(), self.pair_b.currentData())
        return self.pair_states().get(key)

    def select_pair(self, a: str, b: str) -> None:
        index = self.pair_a.findData(a)
        if index >= 0:
            self.pair_a.setCurrentIndex(index)
        index = self.pair_b.findData(b)
        if index >= 0:
            self.pair_b.setCurrentIndex(index)

    def redraw_joint(self) -> None:
        state = self.selected_pair()
        self.cond_ellipse.stokes = None
        self.cond_sphere.points = {}
        if state is None:
            for i in range(4):
                for j in range(4):
                    self.rho_table.setItem(i, j, QTableWidgetItem(""))
            self.joint_metrics.setText("")
            self.cond_note.setText("")
            self.joint_info.setText(
                "no two-photon state: the rate engine builds one for every two links that "
                "carry the two photons of a pair (not in series) once a pair source is present"
                if self.outcome is not None
                else ""
            )
            self.cond_ellipse.update()
            self.cond_sphere.update()
            return
        rho = state["rho"]
        for i in range(4):
            for j in range(4):
                z = rho[i, j]
                text = f"{z.real:+.4f}" + (f" {z.imag:+.4f}i" if abs(z.imag) >= 5e-5 else "")
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.rho_table.setItem(i, j, item)
        others = ", ".join(
            f"{k} {eng(v, 'Hz')}" for k, v in state["by_source_hz"].items() if k != state["source"]
        )
        self.joint_info.setText(
            f"source '{state['source']}': photon a = {state['photon_a']}, photon b = "
            f"{state['photon_b']}; {eng(state['rate_hz'], 'Hz')} pairs with one photon on each "
            f"link (rate engine, mean rates){'; also ' + others if others else ''}. Basis "
            f"{state['basis']}. Frame: {state['frame']}."
        )
        fid = "  ".join(f"{BELL_NAMES[k]} {v:.4f}" for k, v in state["bell_fidelity"].items())
        dop_a = float(np.linalg.norm(state["stokes_a"][1:]))
        dop_b = float(np.linalg.norm(state["stokes_b"][1:]))
        self.joint_metrics.setText(
            f"purity {state['purity']:.4f}   concurrence {state['concurrence']:.4f}\n"
            f"Bell-state fidelity  {fid}\n"
            f"{REDUCED_NOTE}: DOP a {dop_a:.3f}, DOP b {dop_b:.3f}"
        )
        label = self.analyser.currentText()
        cond = state["conditional"][label]
        prob = cond["probability"]
        s = np.round(cond["stokes_b"], 3) + 0.0
        if not np.all(np.isfinite(s)):
            self.cond_note.setText(
                f"p(a passes |{label}>) = {prob:.4f}: undefined, photon a never passes this "
                "analyser"
            )
        else:
            dop = float(np.linalg.norm(s[1:]))
            self.cond_note.setText(
                f"p(a passes |{label}>) = {prob:.4f} ({eng(prob * state['rate_hz'], 'Hz')} "
                f"heralds). Photon b given that: S1 {s[1]:+.3f}  S2 {s[2]:+.3f}  S3 {s[3]:+.3f}, "
                f"DOP {dop:.3f}"
            )
            self.cond_ellipse.stokes = s
            self.cond_sphere.points = {f"b | a={label}": s, "b reduced": state["stokes_b"]}
            self.cond_sphere.selected = f"b | a={label}"
        self.cond_ellipse.update()
        self.cond_sphere.update()

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
            f"{REDUCED_NOTE}: single-photon DOP {dop:.3f}; {eng(total[0], 'Hz')} photons. "
            "Pair correlations: Joint (pairs)"
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
