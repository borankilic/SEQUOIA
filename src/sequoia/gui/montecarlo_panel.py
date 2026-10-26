from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sequoia.analysis.correlation import CorrelationError, confirm_centers, write_tags
from sequoia.analysis.planning import PlanningError, format_plan, plan_rate
from sequoia.engines.report import MC_QUANTITY
from sequoia.gui.analysis_tabs import style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, car_text, eng, plain
from sequoia.gui.runguard import RunGuard, RunRecords
from sequoia.measurement import effective_analyses
from sequoia.runs import McOutcome, new_run_id, run_montecarlo

__all__ = ["McOutcome", "MonteCarloTab"]
PLANNABLE = ("coincidences", "heralded_g2")


class MonteCarloTab(RunGuard, RunRecords, QWidget):
    """Run the Monte Carlo time-tag engine on the open setup and compare with the rate engine."""

    def __init__(self, provider, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard_init()
        self._records_init()
        self._measured_duration = None
        self._run_id = None
        self.provider = provider
        self.runner = JobRunner(self)
        self.outcome: McOutcome | None = None
        self.on_issues = None
        self.note = ""
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.001, 3600.0)
        self.duration.setDecimals(3)
        self.duration.setValue(5.0)
        self.duration.setSuffix(" s")
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2_000_000_000)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("random")
        self.schmidt = QDoubleSpinBox()
        self.schmidt.setRange(0.0, 1e4)
        self.schmidt.setDecimals(2)
        self.schmidt.setSpecialValueText("Poisson")
        self.schmidt.setToolTip(
            "Pair-number statistics of pulsed sources. Poisson: independent pairs. "
            "K >= 1: thermal statistics of K equally weighted modes (negative binomial). "
            "Values between 0 and 1 are refused."
        )
        self.freeze = QCheckBox("freeze centres from last search")
        self.freeze.setToolTip(
            "Measure with the coincidence centres found by the last run that searched for them, "
            "instead of searching again in the new data (confirmatory analysis)."
        )
        self.last_search: dict[str, dict[str, Any]] | None = None
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
        self.table = _make_table(
            [
                "Item",
                "Quantity [unit]",
                "Monte Carlo (recorded tags)",
                "± σ (Poisson counting)",
                "Rate engine (expected)",
                "z",
            ]
        )
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "delay t_a - t_b [ns]", "events per bin in the acquisition")
        self.plan_analysis = QComboBox()
        self.plan_target = QComboBox()
        self.plan_target.addItems(["relative precision", "expected events"])
        self.plan_value = QDoubleSpinBox()
        self.plan_value.setRange(1e-6, 1e12)
        self.plan_value.setDecimals(6)
        self.plan_value.setValue(0.01)
        self.plan_button = QPushButton("Plan acquisition")
        self.plan_label = QLabel("")
        self.plan_label.setWordWrap(True)
        self.plan_button.clicked.connect(self.plan)
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
            QLabel("Pairs per pulse, K"),
            self.schmidt,
            self.freeze,
            self.run_button,
            self.cancel_button,
            self.export_button,
            *self.run_buttons(),
            self.progress,
        ):
            bar.addWidget(w)
        bar.addWidget(self.status, 1)
        plan = QHBoxLayout()
        plan.setContentsMargins(6, 0, 6, 0)
        for w in (
            QLabel("Plan acquisition for"),
            self.plan_analysis,
            self.plan_target,
            self.plan_value,
            self.plan_button,
        ):
            plan.addWidget(w)
        plan.addWidget(self.plan_label, 1)
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
        layout.addWidget(self.model_card)
        layout.addWidget(split, 1)
        layout.addLayout(plan)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.export_button.clicked.connect(self.save_tags)
        self.analysis.currentIndexChanged.connect(lambda _i: self._draw())
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        self.duration.valueChanged.connect(self.refresh_stale)
        self.seed.valueChanged.connect(self.refresh_stale)
        self.schmidt.valueChanged.connect(self.refresh_stale)
        self.freeze.toggled.connect(self.refresh_stale)

    def config_key(self):
        return (
            self.duration.value(),
            self.seed.value(),
            self.schmidt.value(),
            self.freeze.isChecked(),
        )

    def set_outcome(self, outcome) -> None:
        setup, _library = self.provider()
        m = setup.measurement if setup is not None else None
        duration = m.acquisition_duration.si if m is not None and m.acquisition_duration else None
        if duration is not None and duration != self._measured_duration:
            self._measured_duration = duration
            self.duration.setValue(duration)
        if self.track_revision(outcome) and self.outcome is not None:
            self.status.setText(self.stale_text(self.note))

    def reset(self) -> None:
        self.invalidate()
        self.outcome = None
        self.last_search = None
        self._measured_duration = None
        self.note = ""
        self.clear_record()
        self.plan_analysis.clear()
        self.plan_label.setText("")
        if self.on_issues is not None:
            self.on_issues([])
        _fill(self.table, [])
        self.analysis.clear()
        self.curve.setData([0, 1], [0])
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
        duration = self.duration.value()
        seed = None if self.seed.value() < 0 else self.seed.value()
        k = self.schmidt.value()
        if 0.0 < k < 1.0:
            self.status.setText(
                f"K = {k:g} is not a mode number: use 0 for Poisson pair statistics or K >= 1 "
                "for thermal statistics of K modes"
            )
            return
        schmidt = k if k >= 1.0 else None
        specs = effective_analyses(setup)
        confirmatory = self.freeze.isChecked()
        if confirmatory:
            if self.last_search is None:
                self.status.setText(
                    "no searched centres to freeze: run once without 'freeze centres' first"
                )
                return
            try:
                specs = confirm_centers(self.last_search, specs)
            except CorrelationError as exc:
                self.status.setText(f"cannot freeze centres: {exc}")
                return
        self.begin_run()

        run_id = self._run_id = new_run_id("montecarlo")

        def job(should_cancel, progress):
            return run_montecarlo(
                setup,
                library,
                duration_s=duration,
                seed=seed,
                schmidt_number=schmidt,
                specs=specs,
                confirmatory=confirmatory,
                should_cancel=should_cancel,
                progress=progress,
                run_id=run_id,
            )

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
        if not self.run_is_current():
            self._end("")
            return
        self.outcome = outcome
        self.set_record(outcome.record, outcome.card)
        if any(a.get("center_mode") == "search" for a in outcome.analyses.values()):
            self.last_search = outcome.analyses
        rows = []
        for c in outcome.comparisons:
            a = outcome.analyses.get(c.name, {})
            where = f" in {eng(a['window_s'], 's')} full-width window" if "window_s" in a else ""
            rows.append(
                [
                    _NumItem(c.name),
                    _NumItem(MC_QUANTITY.get(c.quantity, c.quantity) + where),
                    _NumItem(plain(c.monte_carlo), c.monte_carlo),
                    _NumItem(plain(c.sigma, 3), c.sigma),
                    _NumItem(plain(c.expected), c.expected),
                    _NumItem(
                        f"{c.z:+.2f}"
                        + ("" if c.ok else "  !")
                        + ("  (model)" if c.basis == "model" else ""),
                        c.z,
                        "Compared with an analytic Poisson-multipair model, not a "
                        "like-for-like statistic: a large z can be a model difference."
                        if c.basis == "model"
                        else "",
                    ),
                ]
            )
        for name, a in outcome.analyses.items():
            if a["type"] == "heralded_g2":
                g2 = a["g2_heralded"]
                rows.append(
                    [
                        _NumItem(name),
                        _NumItem("heralded g2(0) (triples x herald / doubles)"),
                        _NumItem(
                            plain(g2) if np.isfinite(g2) else "undefined: no heralded coincidences",
                            g2,
                        ),
                        _NumItem(plain(a["g2_sigma"], 3), a["g2_sigma"]),
                        _NumItem(""),
                        _NumItem(""),
                    ]
                )
        for name, a in outcome.analyses.items():
            for key in ("center_s", "center_a_s", "center_b_s"):
                if key in a and "center_mode" in a:
                    label = "centre" + key[6:-2].replace("_", " ")
                    rows.append(
                        [
                            _NumItem(name),
                            _NumItem(f"{label} ({a['center_mode']})"),
                            _NumItem(eng(a[key], "s"), a[key]),
                            _NumItem(""),
                            _NumItem(""),
                            _NumItem(""),
                        ]
                    )
        for name, a in outcome.analyses.items():
            if a["type"] == "coincidences":
                rows.append(
                    [
                        _NumItem(name),
                        _NumItem("CAR (true / accidental)"),
                        _NumItem(car_text(a["car"]), a["car"]),
                        _NumItem(""),
                        _NumItem(""),
                        _NumItem(""),
                    ]
                )
        _fill(self.table, rows)
        planned = [n for n, a in outcome.analyses.items() if a["type"] in PLANNABLE]
        current = self.plan_analysis.currentText()
        self.plan_analysis.clear()
        self.plan_analysis.addItems(planned)
        if current in planned:
            self.plan_analysis.setCurrentText(current)
        names = [n for n, a in outcome.analyses.items() if "tau_s" in a]
        self.analysis.blockSignals(True)
        self.analysis.clear()
        self.analysis.addItems(names)
        self.analysis.blockSignals(False)
        total = sum(outcome.tags.counts().values())
        text = f"{total} events in {eng(outcome.tags.duration, 's')}"
        if outcome.confirmatory:
            text += "; fixed centres (confirmatory)"
        elif any(a.get("center_mode") == "search" for a in outcome.analyses.values()):
            text += "; centres searched in these data"
        detectors = outcome.tags.truth.get("detectors", {})
        if any(d.get("survival", 1.0) < 0.99 for d in detectors.values()):
            text += (
                "; dead-time loss above 1%: the rate engine treats detector dead times as "
                "independent, so differences of about 1% in coincidences are expected"
            )
        issues = [str(i) for i in outcome.tags.issues]
        if issues:
            text += f"; {len(issues)} note(s): " + "; ".join(issues[:2])
        self.note = text
        if self.on_issues is not None:
            self.on_issues(
                list(outcome.record.issues)
                if outcome.record is not None
                else [i.for_run("montecarlo", self._run_id) for i in outcome.tags.issues]
            )
        self.status.setToolTip("\n".join(issues))
        self._end(self.stale_text(text) if self.accept_run() else text)
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
        if not path:
            return
        tags = self.outcome.tags
        try:
            write_tags(
                path,
                tags.channels,
                tags.duration,
                tags.clock,
                tags.metadata(),
                tags.clock_ids,
                analyses=self.outcome.analyses,
            )
        except (OSError, ValueError, CorrelationError) as exc:
            QMessageBox.warning(self, "Cannot save time tags", f"{path}\n\n{exc}")
            return
        self.status.setText(f"saved {path}")

    def plan(self):
        """Acquisition time for the chosen analysis from this run's rates (Poisson counting)."""
        name = self.plan_analysis.currentText()
        if self.outcome is None or not name:
            self.plan_label.setText("run the Monte Carlo first")
            return None
        a = self.outcome.analyses[name]
        setup, _library = self.provider()
        m = setup.measurement if setup is not None else None
        repeats = m.repeats if m is not None else 1
        value = self.plan_value.value()
        target = (
            {"target_relative": value}
            if self.plan_target.currentIndex() == 0
            else {"target_events": value}
        )
        try:
            if a["type"] == "coincidences":
                result = plan_rate(
                    a["total_hz"], repeats=repeats, quantity="raw coincidences", **target
                )
            else:
                herald = a["herald_singles_hz"]
                success = a["triples_hz"] / herald if herald > 0 else 0.0
                result = plan_rate(
                    herald,
                    success_probability=success,
                    repeats=repeats,
                    quantity="heralded triples",
                    **target,
                )
        except PlanningError as exc:
            self.plan_label.setText(f"cannot plan: {exc}")
            return None
        self.plan_label.setText(format_plan(result))
        return result
