from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sequoia.analysis.correlation import analyse_tags, write_tags
from sequoia.engines.cache import RunCache
from sequoia.engines.montecarlo import (
    Comparison,
    TimeTags,
    compare_with_rates,
    simulate_timetags,
)
from sequoia.engines.rates import simulate_rates
from sequoia.gui.analysis_tabs import style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, eng, plain


@dataclass
class McOutcome:
    tags: TimeTags
    analyses: dict[str, dict[str, Any]]
    comparisons: list[Comparison] = field(default_factory=list)
    note: str = ""


class MonteCarloTab(QWidget):
    """Run the Monte Carlo time-tag engine on the open setup and compare with the rate engine."""

    def __init__(self, provider, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.provider = provider
        self.runner = JobRunner(self)
        self.outcome: McOutcome | None = None
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.001, 3600.0)
        self.duration.setDecimals(3)
        self.duration.setValue(5.0)
        self.duration.setSuffix(" s")
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2_000_000_000)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("random")
        self.run_button = QPushButton("Run Monte Carlo")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Save tags...")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setStyleSheet("color: palette(placeholder-text);")
        self.analysis = QComboBox()
        self.table = _make_table(["Item", "Quantity", "Monte Carlo", "± σ", "Rate engine", "z"])
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "delay [ns]", "events per bin")
        self.curve = self.plot.plot(
            pen=pg.mkPen("#1f77b4", width=1),
            stepMode="center",
            fillLevel=0,
            brush=(31, 119, 180, 80),
        )
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        for w in (
            QLabel("Duration"),
            self.duration,
            QLabel("Seed"),
            self.seed,
            self.run_button,
            self.cancel_button,
            self.export_button,
            self.progress,
        ):
            bar.addWidget(w)
        bar.addWidget(self.status, 1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.setContentsMargins(6, 0, 6, 0)
        top.addWidget(QLabel("Histogram"))
        top.addWidget(self.analysis)
        top.addStretch(1)
        rl.addLayout(top)
        rl.addWidget(self.plot, 1)
        split = QSplitter()
        split.addWidget(self.table)
        split.addWidget(right)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(split, 1)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.export_button.clicked.connect(self.save_tags)
        self.analysis.currentIndexChanged.connect(lambda _i: self._draw())
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(lambda m: self._end(f"failed: {m.splitlines()[0]}"))
        self.runner.cancelled.connect(lambda: self._end("cancelled"))

    def set_outcome(self, _outcome) -> None:
        pass

    def run(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return
        duration = self.duration.value()
        seed = None if self.seed.value() < 0 else self.seed.value()

        def job(should_cancel, progress):
            cache = RunCache()
            tags = simulate_timetags(
                setup,
                library,
                duration,
                seed=seed,
                cache=cache,
                should_cancel=should_cancel,
                progress=progress,
            )
            analyses = analyse_tags(
                tags.channels, tags.duration, setup.analyses, tags.repetition_rates
            )
            rates = simulate_rates(setup, library, cache=cache, should_cancel=should_cancel)
            return McOutcome(tags, analyses, compare_with_rates(tags, analyses, rates))

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

    def _done(self, outcome: McOutcome) -> None:
        self.outcome = outcome
        rows = []
        for c in outcome.comparisons:
            rows.append(
                [
                    _NumItem(c.name),
                    _NumItem(c.quantity),
                    _NumItem(plain(c.monte_carlo), c.monte_carlo),
                    _NumItem(plain(c.sigma, 3), c.sigma),
                    _NumItem(plain(c.expected), c.expected),
                    _NumItem(f"{c.z:+.2f}" + ("" if c.ok else "  !"), c.z),
                ]
            )
        for name, a in outcome.analyses.items():
            if a["type"] == "heralded_g2":
                rows.append(
                    [
                        _NumItem(name),
                        _NumItem("g2(0)"),
                        _NumItem(plain(a["g2_heralded"]), a["g2_heralded"]),
                        _NumItem(plain(a["g2_sigma"], 3), a["g2_sigma"]),
                        _NumItem(""),
                        _NumItem(""),
                    ]
                )
        _fill(self.table, rows)
        names = [n for n, a in outcome.analyses.items() if "tau_s" in a]
        self.analysis.blockSignals(True)
        self.analysis.clear()
        self.analysis.addItems(names)
        self.analysis.blockSignals(False)
        total = sum(outcome.tags.counts().values())
        text = f"{total} events in {eng(outcome.tags.duration, 's')}"
        detectors = outcome.tags.truth.get("detectors", {})
        if any(d.get("survival", 1.0) < 0.99 for d in detectors.values()):
            text += (
                "; dead-time loss above 1%: the rate engine treats detector dead times as "
                "independent, so differences of about 1% in coincidences are expected"
            )
        self._end(text)
        self._draw()

    def _draw(self) -> None:
        name = self.analysis.currentText()
        if self.outcome is None or not name:
            self.curve.setData([0, 1], [0])
            return
        a = self.outcome.analyses[name]
        tau = np.asarray(a["tau_s"]) * 1e9
        width = a["bin_width_s"] * 1e9
        edges = np.concatenate([tau - width / 2, [tau[-1] + width / 2]])
        self.curve.setData(edges, np.asarray(a["counts"]))

    def save_tags(self) -> None:
        if self.outcome is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save time tags", "tags.npz", "NumPy (*.npz)")
        if path:
            write_tags(path, self.outcome.tags.channels, self.outcome.tags.duration)
