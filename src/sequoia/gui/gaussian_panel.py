from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sequoia.engines.cache import RunCache
from sequoia.engines.classical import simulate
from sequoia.engines.gaussian import GaussianResult, gaussian_window_groups, prepare_gaussian
from sequoia.gui.analysis_tabs import style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, car_text, eng, plain
from sequoia.gui.runguard import RunGuard

SWEEP_POINTS = 13
SCALE_TEXT = (
    "Pump x (and the per-source factor) scales the pair gain of a source and the classical "
    "leakage of the laser that pumps it. Fixed: detector dark counts, lasers that pump no "
    "simulated source, Fock-source flux."
)


@dataclass
class GaussianOutcome:
    result: GaussianResult | None
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    pairs: np.ndarray = field(default_factory=lambda: np.array([]))
    car: np.ndarray = field(default_factory=lambda: np.array([]))
    g2: np.ndarray = field(default_factory=lambda: np.array([]))
    groups: list[tuple[float, list[str]]] = field(default_factory=list)


class GaussianTab(RunGuard, QWidget):
    """Multimode Gaussian photon statistics, and CAR / heralded g2 against pump power."""

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
            "Multiplies the pump power of every pair source (times its own factor in the "
            "table): the pair gain and the leakage of its pump laser scale, dark counts and "
            "lasers that pump no simulated source stay fixed."
        )
        self.sweep = QCheckBox("sweep pump power")
        self.sweep.setChecked(True)
        self.groups: list[tuple[float, list[str]]] = []
        self.groups_label = QLabel("pair sources: not listed yet")
        self.group_box = QComboBox()
        self.group_box.setToolTip(
            "Sources with different time windows cannot be combined in one per-window model; "
            "the selected group is simulated and the other pair sources are left out."
        )
        self.group_box.setVisible(False)
        self.list_button = QPushButton("List sources")
        self.factor_table = _make_table(["Source", "Window", "Pump factor"])
        self.factor_table.setMaximumHeight(120)
        self.factors: dict[str, QDoubleSpinBox] = {}
        self.scope_label = QLabel(SCALE_TEXT)
        self.scope_label.setWordWrap(True)
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
        source_bar = QHBoxLayout()
        source_bar.setContentsMargins(6, 0, 6, 0)
        source_bar.addWidget(self.list_button)
        source_bar.addWidget(self.groups_label, 1)
        source_bar.addWidget(self.group_box)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.factor_table)
        ll.addWidget(self.scope_label)
        ll.addWidget(self.table, 1)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addLayout(source_bar)
        layout.addWidget(split, 1)
        self.list_button.clicked.connect(self.list_sources)
        self.group_box.currentIndexChanged.connect(self.refresh_stale)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        self.scale.valueChanged.connect(self.refresh_stale)
        self.sweep.toggled.connect(self.refresh_stale)

    def config_key(self):
        return (
            self.scale.value(),
            self.sweep.isChecked(),
            self.selected_sources(),
            tuple((k, w.value()) for k, w in self.factors.items() if w.value() != 1.0),
        )

    def selected_sources(self) -> tuple[str, ...] | None:
        ids = self.group_box.currentData() if len(self.groups) > 1 else None
        return tuple(ids) if ids else None

    def show_groups(self, groups: list[tuple[float, list[str]]]) -> None:
        previous = self.selected_sources()
        self.groups = groups
        self.groups_label.setText(
            "pair-source windows: "
            + ("; ".join(f"{eng(w, 's')}: {', '.join(ids)}" for w, ids in groups) or "none")
            + (" - differ, choose one group" if len(groups) > 1 else "")
        )
        self.group_box.blockSignals(True)
        self.group_box.clear()
        self.group_box.addItem("choose a source group", None)
        for w, ids in groups:
            self.group_box.addItem(f"{eng(w, 's')}: {', '.join(ids)}", list(ids))
            if previous == tuple(ids):
                self.group_box.setCurrentIndex(self.group_box.count() - 1)
        self.group_box.blockSignals(False)
        self.group_box.setVisible(len(groups) > 1)
        old = {k: w.value() for k, w in self.factors.items()}
        self.factors = {}
        rows = [(w, sid) for w, ids in groups for sid in ids]
        self.factor_table.setRowCount(len(rows))
        for r, (w, sid) in enumerate(rows):
            self.factor_table.setItem(r, 0, _NumItem(sid))
            self.factor_table.setItem(r, 1, _NumItem(eng(w, "s"), w))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1e6)
            spin.setDecimals(4)
            spin.setValue(old.get(sid, 1.0))
            spin.valueChanged.connect(self.refresh_stale)
            self.factor_table.setCellWidget(r, 2, spin)
            self.factors[sid] = spin
        self.refresh_stale()

    def list_sources(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return

        def job(should_cancel, progress):
            progress(0, 0)
            groups = gaussian_window_groups(setup, library, should_cancel=should_cancel)
            return GaussianOutcome(None, groups=groups)

        self.begin_run()
        self._start(job, "listing pair sources...")

    def set_outcome(self, outcome) -> None:
        if self.track_revision(outcome) and self.outcome is not None:
            self.status.setText(self.stale_text(self.note))

    def reset(self) -> None:
        self.invalidate()
        self.outcome = None
        self.note = ""
        self.show_groups([])
        self.groups_label.setText("pair sources: not listed yet")
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
        chosen = self.selected_sources()
        factors = {k: w.value() for k, w in self.factors.items()}

        def job(should_cancel, progress):
            progress(0, 0)
            cache = RunCache()
            classical = simulate(setup, library, should_cancel=should_cancel, cache=cache)
            groups = gaussian_window_groups(
                setup, library, classical=classical, cache=cache, should_cancel=should_cancel
            )
            selected = None
            if len(groups) > 1:
                if chosen is None or list(chosen) not in [ids for _, ids in groups]:
                    return GaussianOutcome(None, groups=groups)
                selected = chosen
            prepared = prepare_gaussian(
                setup,
                library,
                classical=classical,
                cache=cache,
                should_cancel=should_cancel,
                sources=selected,
            )
            gain = {k: scale * factors.get(k, 1.0) for k in prepared.model.pumps}
            result = prepared.evaluate(gain)
            outcome = GaussianOutcome(result, groups=groups)
            if not sweep or not result.sources:
                return outcome
            mu = next(iter(result.sources.values()))["pairs_per_window"]
            grid = np.logspace(-1, 3, SWEEP_POINTS)
            rows = []
            for i, t in enumerate(grid):
                progress(i, len(grid))
                if should_cancel():
                    break
                r = prepared.evaluate({k: v * float(t) for k, v in gain.items()})
                rows.append((t, r))
            names = [n for n, a in result.analyses.items() if a["type"] == "coincidences"]
            hnames = [n for n, a in result.analyses.items() if a["type"] == "heralded_g2"]
            outcome.scales = scale * np.array([t for t, _ in rows])
            outcome.pairs = np.array([t for t, _ in rows]) * mu
            if names:
                outcome.car = np.array([r.analyses[names[0]]["car"] for _, r in rows])
            if hnames:
                outcome.g2 = np.array([r.analyses[hnames[0]]["g2_heralded"] for _, r in rows])
            return outcome

        self.begin_run()
        self._start(job, "running...")

    def _start(self, job, text: str) -> None:
        self.run_button.setEnabled(False)
        self.list_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText(text)
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.list_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self.status.setText(text)

    def _done(self, outcome: GaussianOutcome) -> None:
        if not self.run_is_current():
            self._end("")
            return
        if outcome.groups or outcome.result is None:
            self.show_groups(outcome.groups)
        r = outcome.result
        if r is None:
            self._end(
                "not run: the pair sources have different time windows; choose one source group "
                "and run again"
                if len(outcome.groups) > 1
                else f"{len(outcome.groups)} pair-source window group(s) listed"
            )
            return
        if not r.channels:
            blocker = next((i for i in r.issues if i.code == "gaussian_no_detectors"), None)
            self.accept_run()
            self.outcome = None
            _fill(self.table, [])
            self.car_curve.setData([], [])
            self.g2_curve.setData([], [])
            if self.on_issues is not None:
                self.on_issues(r.issues)
            self._end(
                "not computed: "
                + (blocker.message if blocker else "no tagged detector in the setup")
            )
            return
        self.outcome = outcome
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
                rows.append([_NumItem(name), _NumItem("CAR"), _NumItem(car_text(a["car"]))])
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
        note += "; sources: " + (", ".join(r.sources) or "none")
        skipped = [i.subject for i in r.issues if i.code == "gaussian_source_skipped"]
        if skipped:
            note += " (left out: " + ", ".join(skipped) + ")"
        if len(set(r.source_scale.values())) > 1:
            note += "; pump x " + ", ".join(f"{k} {v:g}" for k, v in r.source_scale.items())
        warnings = [i for i in r.issues if i.severity in ("warning", "error")]
        if warnings:
            note += f"; {len(warnings)} note(s): " + "; ".join(str(i) for i in warnings[:2])
        self.note = note
        if self.on_issues is not None:
            self.on_issues(r.issues)
        self.status.setToolTip("\n".join(str(i) for i in r.issues))
        self._end(self.stale_text(note) if self.accept_run() else note)
