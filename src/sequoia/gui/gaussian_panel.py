from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sequoia.engines.gaussian import GaussianResult, prepare_gaussian
from sequoia.gui.analysis_tabs import style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, eng, plain
from sequoia.gui.runguard import RunGuard

SWEEP_POINTS = 13


@dataclass
class GaussianOutcome:
    result: GaussianResult
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    pairs: np.ndarray = field(default_factory=lambda: np.array([]))
    car: np.ndarray = field(default_factory=lambda: np.array([]))
    g2: np.ndarray = field(default_factory=lambda: np.array([]))


class GaussianTab(RunGuard, QWidget):
    """Exact multimode photon statistics, and CAR / heralded g2 against pump power."""

    def __init__(self, provider, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard_init()
        self.provider = provider
        self.runner = JobRunner(self)
        self.outcome: GaussianOutcome | None = None
        self.on_issues = None
        self.note = ""
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1e-6, 1e6)
        self.scale.setDecimals(4)
        self.scale.setValue(1.0)
        self.scale.setPrefix("pump x ")
        self.scale.setToolTip(
            "Multiplies the pump power: pair gain and classical leakage scale with it, "
            "detector dark counts stay fixed."
        )
        self.sweep = QCheckBox("sweep pump power")
        self.sweep.setChecked(True)
        self.run_button = QPushButton("Run Gaussian engine")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setStyleSheet("color: palette(placeholder-text);")
        self.table = _make_table(["Item", "Quantity", "Value"])
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "pairs per window (low-gain)", "CAR / heralded g2")
        self.plot.setLogMode(x=True, y=True)
        self.car_curve = self.plot.plot(
            pen=pg.mkPen("#1f77b4", width=2), symbol="o", symbolSize=5, name="CAR"
        )
        self.g2_curve = self.plot.plot(
            pen=pg.mkPen("#d62728", width=2), symbol="s", symbolSize=5, name="heralded g2(0)"
        )
        self.plot.addLegend()
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        for w in (self.scale, self.sweep, self.run_button, self.cancel_button, self.progress):
            bar.addWidget(w)
        bar.addWidget(self.status, 1)
        split = QSplitter()
        split.addWidget(self.table)
        split.addWidget(self.plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(split, 1)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))

    def set_outcome(self, outcome) -> None:
        if self.track_revision(outcome) and self.outcome is not None:
            self.status.setText(self.stale_text(self.note))

    def reset(self) -> None:
        self.invalidate()
        self.outcome = None
        self.note = ""
        if self.on_issues is not None:
            self.on_issues([])
        _fill(self.table, [])
        self.car_curve.setData([], [])
        self.g2_curve.setData([], [])
        self.status.setText("")

    def _failed(self, message: str) -> None:
        if self.run_is_current():
            self._end(f"failed: {message.splitlines()[0]}")
        else:
            self._end("")

    def run(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return
        scale = self.scale.value()
        sweep = self.sweep.isChecked()
        self.begin_run()

        def job(should_cancel, progress):
            progress(0, 0)
            prepared = prepare_gaussian(setup, library, should_cancel=should_cancel)
            result = prepared.evaluate(scale)
            outcome = GaussianOutcome(result)
            if not sweep or not prepared.source_info:
                return outcome
            mu = next(iter(prepared.source_info.values()))["pairs_per_window"]
            grid = scale * np.logspace(-1, 3, SWEEP_POINTS)
            rows = []
            for i, s in enumerate(grid):
                progress(i, len(grid))
                if should_cancel():
                    break
                r = prepared.evaluate(float(s))
                rows.append((s, r))
            names = [n for n, a in result.analyses.items() if a["type"] == "coincidences"]
            hnames = [n for n, a in result.analyses.items() if a["type"] == "heralded_g2"]
            outcome.scales = np.array([s for s, _ in rows])
            outcome.pairs = outcome.scales * mu
            if names:
                outcome.car = np.array([r.analyses[names[0]]["car"] for _, r in rows])
            if hnames:
                outcome.g2 = np.array([r.analyses[hnames[0]]["g2_heralded"] for _, r in rows])
            return outcome

        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("running...")
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self.status.setText(text)

    def _done(self, outcome: GaussianOutcome) -> None:
        if not self.run_is_current():
            self._end("")
            return
        self.outcome = outcome
        r = outcome.result
        rows = []
        for ch in r.channels:
            p = r.probability(ch)
            rows += [
                [_NumItem(ch), _NumItem("P(click)"), _NumItem(plain(p), p)],
                [_NumItem(ch), _NumItem("mean photons"), _NumItem(plain(r.mean_photons[ch]))],
                [_NumItem(ch), _NumItem("rate"), _NumItem(eng(p / r.window, "Hz"), p / r.window)],
            ]
        for name, a in r.analyses.items():
            if a["type"] == "coincidences":
                rows.append([_NumItem(name), _NumItem("CAR"), _NumItem(plain(a["car"]))])
                rows.append(
                    [
                        _NumItem(name),
                        _NumItem("true coincidences"),
                        _NumItem(eng(a["true_hz"], "Hz"), a["true_hz"]),
                    ]
                )
            elif a["type"] == "heralded_g2":
                rows.append([_NumItem(name), _NumItem("g2(0)"), _NumItem(plain(a["g2_heralded"]))])
        _fill(self.table, rows)
        good = np.isfinite(outcome.car) & (outcome.car > 0)
        if len(outcome.pairs) and good.any():
            self.car_curve.setData(outcome.pairs[good], outcome.car[good])
        else:
            self.car_curve.setData([], [])
        g_good = np.isfinite(outcome.g2) & (outcome.g2 > 0)
        if len(outcome.g2) and g_good.any():
            self.g2_curve.setData(outcome.pairs[g_good], outcome.g2[g_good])
        else:
            self.g2_curve.setData([], [])
        modes = sum(v["modes"] for v in r.sources.values())
        note = f"window {eng(r.window, 's')}, {modes} Schmidt modes"
        warnings = [i for i in r.issues if i.severity in ("warning", "error")]
        if warnings:
            note += f"; {len(warnings)} note(s): " + "; ".join(str(i) for i in warnings[:2])
        self.note = note
        if self.on_issues is not None:
            self.on_issues(r.issues)
        self.status.setToolTip("\n".join(str(i) for i in r.issues))
        self._end(note)
