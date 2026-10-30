import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sequoia.analysis.optimize import METHODS, Variable, optimize
from sequoia.analysis.sweep import (
    Axis,
    Candidate,
    SweepError,
    candidate_targets,
    check_axis_unit,
    check_bounds,
    default_plan,
    metrics_of,
    with_values,
)
from sequoia.engines.cache import RunCache
from sequoia.errors import Issue, SequoiaError
from sequoia.gui.analysis_tabs import save_text, style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, plain
from sequoia.gui.runguard import RunGuard, RunRecords
from sequoia.runs import metric_label, new_run_id, run_sweep_record


def _spin(value: float = 0.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-1e12, 1e12)
    box.setDecimals(6)
    box.setValue(value)
    box.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    return box


def _structure(setup) -> tuple:
    return (
        tuple((i.id, i.part) for i in setup.instances),
        tuple((link.id, link.kind) for link in setup.links),
    )


class AxisRow(QWidget):
    changed = Signal()

    def __init__(self, candidates: list[Candidate], parent=None, *, use_bounds=False) -> None:
        super().__init__(parent)
        self.candidates = candidates
        self.use_bounds = use_bounds
        self.target = QComboBox()
        self.target.setMinimumWidth(220)
        self.target.addItems([c.target for c in candidates])
        self.unit = QLineEdit()
        self.unit.setMaximumWidth(60)
        self.start = _spin()
        self.stop = _spin(1.0)
        self.points = QSpinBox()
        self.points.setRange(2, 2000)
        self.points.setValue(21)
        self.log = QCheckBox("log", self)
        self.log.setToolTip("logarithmically spaced points (both limits must be positive)")
        self.bounds = QLabel("", self)
        self.bounds.setStyleSheet("color: palette(placeholder-text);")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for label, widget in (
            ("", self.target),
            ("from", self.start),
            ("to", self.stop),
            ("unit", self.unit),
            ("points", self.points),
            ("", self.log),
            ("", self.bounds),
        ):
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.target.currentIndexChanged.connect(self._reset_range)
        for signal in (
            self.target.currentIndexChanged,
            self.unit.textChanged,
            self.start.valueChanged,
            self.stop.valueChanged,
            self.points.valueChanged,
            self.log.toggled,
        ):
            signal.connect(lambda *_a: self.changed.emit())
        self._reset_range()

    def _reset_range(self) -> None:
        index = self.target.currentIndex()
        if index < 0:
            self.bounds.setText("")
            return
        cand = self.candidates[index]
        self.unit.setText(cand.unit)
        self.bounds.setText(f"physical: {cand.bounds_text()}")
        if self.use_bounds and cand.lower is not None and cand.upper is not None:
            self.set_range(cand.lower, cand.upper, self.points.value(), False)
        elif cand.default is not None:
            self.set_range(*cand.default)
        else:
            cur = float(cand.current) if cand.current is not None else 0.0
            span = abs(cur) * 0.2 or 1.0
            self.set_range(cur - span, cur + span, self.points.value(), False)

    def set_range(self, start: float, stop: float, points: int, log: bool) -> None:
        self.start.setValue(start)
        self.stop.setValue(stop)
        self.points.setValue(points)
        self.log.setChecked(log)

    def axis(self) -> Axis:
        start, stop, n = self.start.value(), self.stop.value(), self.points.value()
        if self.log.isChecked():
            if start <= 0 or stop <= 0:
                raise SweepError(f"{self.target.currentText()}: a log axis needs positive limits")
            values = np.geomspace(start, stop, n)
        else:
            values = np.linspace(start, stop, n)
        return Axis(self.target.currentText(), values, self.unit.text().strip())

    def set_candidates(self, candidates: list[Candidate]) -> None:
        previous = self.target.currentText()
        self.candidates = candidates
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItems([c.target for c in candidates])
        kept = previous in [c.target for c in candidates]
        if kept:
            self.target.setCurrentText(previous)
        self.target.blockSignals(False)
        if not kept:
            self._reset_range()


class SweepTab(RunGuard, RunRecords, QWidget):
    """1D/2D parameter sweeps of the open setup, plotted against a chosen result."""

    def __init__(self, provider, parent=None) -> None:
        super().__init__(parent)
        self._guard_init()
        self._records_init()
        self.on_issues = None
        self.provider = provider
        self.runner = JobRunner(self)
        self.result = None
        self.outputs: list[str] = []
        self.candidates: list[Candidate] = []
        self.rows = [AxisRow([]), AxisRow([])]
        self.two_d = QCheckBox("second axis (2D)")
        self.fock_check = QCheckBox("photon-number observables")
        self.fock_check.setToolTip(
            "Also compute the Photon counts quantities (fock:P(...) per detector state, "
            "fock:P_click per detector) at every point, with the Photon counts tab's detector "
            "mode, pair truncation and duration. Per detection window; no dead time."
        )
        self.fock_settings = None
        self.metric = QComboBox()
        self.metric.setMinimumWidth(240)
        self.run_button = QPushButton("Sweep")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.refresh_button = QPushButton("Reload parameters")
        self.default_button = QPushButton("Default sweep", self)
        self.default_button.setToolTip(
            "The recommended sweep for this kind of setup (HOM delay, Malus angle, Bell "
            "wave plate, pump power), or the default range of the selected parameter"
        )
        self.optimize_button = QPushButton("Optimize...")
        self.export_button = QPushButton("Export CSV")
        self.png_button = QPushButton("Export PNG")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setStyleSheet("color: palette(placeholder-text);")
        self.plot = pg.PlotWidget()
        style_plot(self.plot, "", "")
        self.image: pg.ImageItem | None = None
        self.colorbar = None

        form = QVBoxLayout()
        form.setContentsMargins(6, 4, 6, 4)
        form.addWidget(self.rows[0])
        second = QHBoxLayout()
        second.addWidget(self.two_d)
        second.addWidget(self.rows[1], 1)
        second.addWidget(self.fock_check)
        form.addLayout(second)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Output"))
        actions.addWidget(self.metric)
        for w in (
            self.run_button,
            self.cancel_button,
            self.refresh_button,
            self.default_button,
            self.optimize_button,
            self.export_button,
            self.png_button,
            *self.run_buttons(),
            self.progress,
        ):
            actions.addWidget(w)
        actions.addWidget(self.status, 1)
        form.addLayout(actions)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form)
        layout.addWidget(self.model_card)
        layout.addWidget(self.plot, 1)

        self.rows[1].setEnabled(False)
        self.two_d.toggled.connect(self.rows[1].setEnabled)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.refresh_button.clicked.connect(self.reload_candidates)
        self.default_button.clicked.connect(self.apply_default)
        self.optimize_button.clicked.connect(self.open_optimizer)
        self.export_button.clicked.connect(self.export_csv)
        self.png_button.clicked.connect(self.export_png)
        self.metric.currentIndexChanged.connect(lambda _i: self._draw())
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        self.optimizer = None
        self.status_text = ""
        self._structure = None
        self._planned = None
        self._outcome = None
        self.two_d.toggled.connect(self.refresh_stale)
        self.fock_check.toggled.connect(self.refresh_stale)
        for row in self.rows:
            row.changed.connect(self.refresh_stale)

    def config_key(self):
        axes = []
        for row in self.rows[: 2 if self.two_d.isChecked() else 1]:
            try:
                a = row.axis()
            except Exception:
                axes.append(None)
                continue
            axes.append((a.target, a.unit, tuple(np.asarray(a.values, float).tolist())))
        return (tuple(axes), self._fock_options())

    def _fock_options(self):
        if not self.fock_check.isChecked():
            return None
        return dict(self.fock_settings()) if self.fock_settings is not None else {}

    def _failed(self, message: str) -> None:
        if self.run_is_current():
            self._end(f"failed: {message.splitlines()[0]}")
        else:
            self._end("")

    def reset(self) -> None:
        self.invalidate()
        self.result = None
        self.status_text = ""
        self.clear_record()
        if self.optimizer is not None:
            self.optimizer.close()
            self.optimizer = None
        self.metric.blockSignals(True)
        self.metric.clear()
        self.metric.blockSignals(False)
        self.plot.clear()
        self.status.setText("")
        for row in self.rows:
            row.set_candidates([])
        self.candidates = []
        self._planned = None
        self._outcome = None

    def reload_candidates(self) -> None:
        setup, library = self.provider()
        if setup is None:
            return
        self._structure = _structure(setup)
        candidates = candidate_targets(setup, library, self._outcome)
        for row in self.rows:
            row.set_candidates(candidates)
        self.candidates = candidates

    def set_outcome(self, outcome) -> None:
        setup, _library = self.provider()
        if outcome is not None:
            self._outcome = outcome
        plan_now = setup is not None and outcome is not None and self._planned != _structure(setup)
        if setup is not None and (
            plan_now or not self.rows[0].candidates or _structure(setup) != self._structure
        ):
            self.reload_candidates()
        if self.track_revision(outcome) and self.result is not None:
            self.status.setText(self.stale_text(self.status_text))
        if outcome is not None and self.result is None:
            names = sorted(metrics_of(outcome)) if outcome.classical else []
            current = self.metric.currentText()
            self.metric.blockSignals(True)
            self.metric.clear()
            self.metric.addItems(names)
            if current in names:
                self.metric.setCurrentText(current)
            self.metric.blockSignals(False)
        if plan_now:
            self._planned = _structure(setup)
            if self.result is None:
                self.apply_default()

    def apply_default(self) -> None:
        setup, library = self.provider()
        if setup is None:
            return
        outcome = self._outcome
        if outcome is not None and outcome.classical is None:
            outcome = None
        try:
            plan = default_plan(setup, library, outcome) if outcome is not None else None
        except SequoiaError:
            plan = None
        row = self.rows[0]
        if plan is None or plan.target not in [c.target for c in row.candidates]:
            row._reset_range()
            index = row.target.currentIndex()
            if index >= 0:
                self.status.setText(f"default range of {row.target.currentText()}")
            return
        self.two_d.setChecked(False)
        row.target.setCurrentText(plan.target)
        row.unit.setText(plan.unit)
        row.set_range(plan.start, plan.stop, plan.points, plan.log)
        if self.metric.findText(plan.output) >= 0:
            self.metric.setCurrentText(plan.output)
        self.status.setText(
            f"default sweep ({plan.task}): {plan.target} vs {metric_label(plan.output)}"
        )

    def run(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return
        try:
            axes = [self.rows[0].axis()]
            if self.two_d.isChecked():
                if self.rows[1].target.currentText() == axes[0].target:
                    self.status.setText("the two axes must differ")
                    return
                axes.append(self.rows[1].axis())
        except SweepError as exc:
            self.status.setText(str(exc))
            return
        names = None
        if not axes[0].target:
            self.status.setText("no parameter to sweep")
            return
        for axis in axes:
            problem = self._axis_problem(axis, setup, library)
            if problem:
                self.status.setText(problem)
                return
        fock = self._fock_options()
        self.begin_run()
        run_id = new_run_id("sweep")

        def job(should_cancel, progress):
            return run_sweep_record(
                setup,
                library,
                axes,
                names,
                fock=fock,
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

    def _axis_problem(self, axis: Axis, setup, library) -> str:
        if not np.all(np.isfinite(axis.values)):
            return f"{axis.target}: the range is not finite"
        try:
            check_axis_unit(setup, axis.target, axis.unit)
            with_values(setup, [(axis.target, float(axis.values[0]), axis.unit)])
            check_bounds(setup, library, [axis])
        except Exception as exc:
            return f"{axis.target}: {str(exc).splitlines()[0]}"
        return ""

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self.status_text = text
        self.status.setText(text)

    def _done(self, run) -> None:
        if not self.run_is_current():
            self._end("")
            return
        result = run.result
        self.result = result
        failures = [
            Issue("warning", "sweep_point_failed", f"sweep point {index}", str(reason)).for_run(
                "sweep", run.record.run_id if run.record is not None else None
            )
            for index, reason in list(result.failures.items())[:20]
        ]
        card = list(run.card)
        if failures:
            card.append(("Approximations", f"first failure: {failures[0].message[:160]}"))
        self.set_record(run.record, card)
        if self.on_issues is not None:
            self.on_issues(failures)
        names = sorted(result.data)
        current = self.metric.currentText()
        self.metric.blockSignals(True)
        self.metric.clear()
        self.metric.addItems(names)
        if current in names:
            self.metric.setCurrentText(current)
        self.metric.blockSignals(False)
        failed = len(result.failures)
        text = f"{int(np.prod(result.shape))} points" + (f", {failed} failed" if failed else "")
        if failed:
            first = next(iter(result.failures), None)
            reason = result.failures[first] if isinstance(result.failures, dict) else first
            text += f" (first: {str(reason)[:90]})"
        self._end(text)
        if self.accept_run():
            self.status.setText(self.stale_text(text))
        self._draw()

    def _draw(self) -> None:
        result, name = self.result, self.metric.currentText()
        self.plot.clear()
        if self.colorbar is not None:
            self.plot.getPlotItem().layout.removeItem(self.colorbar)
            self.colorbar = None
        if result is None or name not in result.data:
            return
        data = result.data[name]
        ax = result.axes
        unit = lambda a: f"{a.target} [{a.unit}]" if a.unit else a.target  # noqa: E731
        if len(ax) == 1:
            style_plot(self.plot, unit(ax[0]), metric_label(name))
            self.plot.setTitle("")
            self.plot.plot(
                ax[0].values,
                data,
                pen=pg.mkPen("#1f77b4", width=2),
                symbol="o",
                symbolSize=5,
                symbolBrush="#1f77b4",
            )
            return
        style_plot(self.plot, unit(ax[0]), unit(ax[1]))
        self.plot.setTitle(metric_label(name))
        img = pg.ImageItem(np.nan_to_num(data, nan=0.0))
        x0, x1 = ax[0].values[0], ax[0].values[-1]
        y0, y1 = ax[1].values[0], ax[1].values[-1]
        nx, ny = data.shape
        dx = (x1 - x0) / max(nx - 1, 1) or 1.0
        dy = (y1 - y0) / max(ny - 1, 1) or 1.0
        img.setRect(x0 - dx / 2, y0 - dy / 2, dx * nx, dy * ny)
        finite = data[np.isfinite(data)]
        if finite.size:
            img.setLevels((float(finite.min()), float(finite.max())))
        img.setColorMap(pg.colormap.get("viridis"))
        self.plot.addItem(img)
        self.image = img

    def export_csv(self) -> None:
        if self.result is not None:
            save_text(self, "Export sweep", "sweep.csv", self.result.to_csv())

    def export_png(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export plot", "sweep.png")
        if path and not self.plot.grab().save(path):
            QMessageBox.warning(self, "Cannot save plot", f"{path} could not be written.")

    def open_optimizer(self) -> None:
        setup, _library = self.provider()
        if setup is None:
            return
        dialog = OptimizeDialog(self.provider, self.metric.currentText(), self)
        self.optimizer = dialog
        dialog.show()


class OptimizeDialog(QDialog):
    def __init__(self, provider, objective: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optimize")
        self.provider = provider
        self.runner = JobRunner(self)
        self.result = None
        setup, library = provider()
        self.candidates = candidate_targets(setup, library) if setup is not None else []
        self.objective = QComboBox()
        self.objective.setEditable(True)
        self.objective.setMinimumWidth(260)
        if setup is not None:
            from sequoia.analysis.sweep import evaluate

            try:
                names = sorted(metrics_of(evaluate(setup, library)))
            except Exception:
                names = []
            self.objective.addItems(names)
        if objective:
            self.objective.setCurrentText(objective)
        self.maximize = QCheckBox("maximize")
        self.maximize.setChecked(True)
        self.method = QComboBox()
        self.method.addItems(METHODS)
        self.evaluations = QSpinBox()
        self.evaluations.setRange(10, 5000)
        self.evaluations.setValue(300)
        self.vars = QGridLayout()
        self.rows: list[AxisRow] = []
        self.add_row()
        add = QPushButton("Add variable")
        add.clicked.connect(self.add_row)
        self.run_button = QPushButton("Optimize")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.apply_button = QPushButton("Apply to setup")
        self.apply_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.table = _make_table(["Parameter", "Start", "Best"])
        self.summary = QLabel("")
        top = QFormLayout()
        top.addRow("Objective", self.objective)
        top.addRow("", self.maximize)
        top.addRow("Method", self.method)
        top.addRow("Max evaluations", self.evaluations)
        buttons = QHBoxLayout()
        for w in (add, self.run_button, self.cancel_button, self.apply_button):
            buttons.addWidget(w)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(
            QLabel("Variables (bounds are the from / to values; they start at the physical range)")
        )
        layout.addLayout(self.vars)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.summary)
        self.resize(760, 520)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.apply_button.clicked.connect(self.apply)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(lambda m: self._end(f"failed: {m.splitlines()[0]}"))
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        self.runner.progress.connect(self._progress)
        self.on_apply = None

    def add_row(self) -> None:
        row = AxisRow(self.candidates, self, use_bounds=True)
        row.points.hide()
        row.log.hide()
        self.rows.append(row)
        self.vars.addWidget(row, len(self.rows) - 1, 0)

    def variables(self) -> list[Variable]:
        out = []
        seen = set()
        for row in self.rows:
            axis = row.axis()
            lo, hi = float(axis.values[0]), float(axis.values[-1])
            if not axis.target or axis.target in seen:
                continue
            seen.add(axis.target)
            out.append(Variable(axis.target, min(lo, hi), max(lo, hi), axis.unit))
        return out

    def run(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return
        variables = self.variables()
        try:
            check_bounds(setup, library, [Axis(v.target, [v.lo, v.hi], v.unit) for v in variables])
        except SweepError as exc:
            self.summary.setText(str(exc))
            return
        objective = self.objective.currentText().strip()
        maximize, method = self.maximize.isChecked(), self.method.currentText()
        evaluations = self.evaluations.value()

        def job(should_cancel, progress):
            return optimize(
                setup,
                library,
                variables,
                objective,
                maximize=maximize,
                method=method,
                max_evaluations=evaluations,
                cache=RunCache(),
                should_cancel=should_cancel,
                callback=lambda n, _v: progress(n, evaluations),
            )

        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.apply_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, evaluations)
        self.summary.setText("running...")
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self.summary.setText(text)

    def _done(self, result) -> None:
        self.result = result
        setup, _ = self.provider()
        from sequoia.analysis.sweep import get_value

        rows = []
        for v in self.variables():
            start = get_value(setup, v.target, v.unit)
            best = result.values.get(v.target)
            label = f"{v.target} [{v.unit}]" if v.unit else v.target
            rows.append(
                [
                    _NumItem(label),
                    _NumItem("-" if start is None else plain(start), start),
                    _NumItem("-" if best is None else plain(best), best),
                ]
            )
        _fill(self.table, rows)
        better = math.isfinite(result.objective)
        self.apply_button.setEnabled(better)
        self._end(
            f"objective {plain(result.baseline)} → {plain(result.objective)} "
            f"in {result.evaluations} evaluations ({result.message})"
        )

    def apply(self) -> None:
        if self.result is None:
            return
        if callable(self.on_apply):
            self.on_apply(self.result)
        self.accept()
