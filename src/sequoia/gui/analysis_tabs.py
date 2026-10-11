import math
from collections.abc import Callable

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
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

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


class LinkTab(QWidget):
    """A tab with a link selector above a plot; subclasses implement ``redraw``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.outcome: SimulationOutcome | None = None
        self.combo = QComboBox()
        self.combo.setMinimumWidth(160)
        self.note = QLabel("")
        self.note.setStyleSheet("color: palette(placeholder-text);")
        top = QHBoxLayout()
        top.setContentsMargins(6, 4, 6, 0)
        top.addWidget(QLabel("Link"))
        top.addWidget(self.combo)
        top.addWidget(self.note, 1)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.addLayout(top)
        self.combo.currentIndexChanged.connect(lambda _i: self._safe_redraw())

    def link_choices(self, outcome: SimulationOutcome) -> list[str]:
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
        self._safe_redraw()

    def _safe_redraw(self) -> None:
        self.note.setText("")
        if self.outcome is None or not self.combo.currentText():
            self.clear()
            return
        self.redraw(self.outcome, self.combo.currentText())

    def clear(self) -> None:
        pass

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        raise NotImplementedError


class CausticTab(LinkTab):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "distance along the path [mm]", "beam radius w [mm]")
        self.plot.addLegend(offset=(-10, 10))
        self.curve_x = self.plot.plot(pen=pg.mkPen("#1f77b4", width=2), name="w_x")
        self.curve_y = self.plot.plot(pen=pg.mkPen("#d62728", width=2, style=Qt.PenStyle.DashLine))
        self.curve_y.opts["name"] = "w_y"
        self.plot.getPlotItem().legend.addItem(self.curve_y, "w_y")
        self.markers: list[pg.InfiniteLine] = []
        self.layout_.addWidget(self.plot, 1)
        self.data = None

    def clear(self) -> None:
        self.curve_x.setData([], [])
        self.curve_y.setData([], [])
        self._clear_markers()
        self.data = None

    def _clear_markers(self) -> None:
        for line in self.markers:
            self.plot.removeItem(line)
        self.markers = []

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
        self._clear_markers()
        data = caustic(outcome.classical, link_id)
        self.data = data
        if len(data.z) == 0:
            self.clear()
            self.note.setText("no free-space path ends at this link")
            return
        self.curve_x.setData(data.z * 1e3, data.w_x * 1e3)
        self.curve_y.setData(data.z * 1e3, data.w_y * 1e3)
        for z, label in data.markers:
            line = pg.InfiniteLine(
                z * 1e3,
                angle=90,
                pen=pg.mkPen((128, 128, 128, 120), style=Qt.PenStyle.DotLine),
                label=label,
                labelOpts={"position": 0.97, "rotateAxis": (1, 0), "anchors": [(1, 1), (1, 1)]},
            )
            self.plot.addItem(line)
            self.markers.append(line)
        self.note.setText(" → ".join(data.links))


class _Ellipse(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.stokes: np.ndarray | None = None
        self.setMinimumSize(150, 150)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = QApplication.palette()
        side = min(self.width(), self.height()) - 12
        c = QPointF(self.width() / 2, self.height() / 2)
        r = side / 2
        p.setPen(QPen(pal.mid().color(), 1))
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
        hand = "right" if s[3] > 0 else "left" if s[3] < 0 else "linear"
        p.drawText(
            QPointF(8, self.height() - 8),
            f"ψ {math.degrees(psi):.1f}°  χ {math.degrees(chi):.1f}°  {hand}",
        )


class _Poincare(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points: dict[str, np.ndarray] = {}
        self.selected = ""
        self.setMinimumSize(190, 170)

    @staticmethod
    def _project(v: np.ndarray) -> tuple[float, float, float]:
        az, el = math.radians(-35), math.radians(22)
        x, y, z = v
        x1 = x * math.cos(az) - y * math.sin(az)
        y1 = x * math.sin(az) + y * math.cos(az)
        return x1, z * math.cos(el) - y1 * math.sin(el), y1 * math.cos(el) + z * math.sin(el)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = QApplication.palette()
        r = min(self.width(), self.height()) / 2 - 14
        c = QPointF(self.width() / 2, self.height() / 2)

        def pt(v):
            x, y, _z = self._project(np.asarray(v, dtype=float))
            return QPointF(c.x() + r * x, c.y() - r * y)

        p.setPen(QPen(pal.mid().color(), 1))
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
        for label, v in (("S1", (1.2, 0, 0)), ("S2", (0, 1.2, 0)), ("S3", (0, 0, 1.2))):
            p.drawText(pt(v), label)
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


class SpectraTab(LinkTab):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "wavelength [nm]", "power [mW per sample]")
        self.curve = self.plot.plot(
            pen=pg.mkPen("#1f77b4", width=2), fillLevel=0, brush=(31, 119, 180, 60)
        )
        self.layout_.addWidget(self.plot, 1)

    def clear(self) -> None:
        self.curve.setData([], [])

    def redraw(self, outcome: SimulationOutcome, link_id: str) -> None:
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
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return True
