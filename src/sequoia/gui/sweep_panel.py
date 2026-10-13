import math

import numpy as np
import pyqtgraph as pg
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
    candidate_targets,
    check_axis_unit,
    metrics_of,
    run_sweep,
    with_values,
)
from sequoia.engines.cache import RunCache
from sequoia.gui.analysis_tabs import save_text, style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, plain
from sequoia.gui.runguard import RunGuard


def _spin(value: float = 0.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-1e12, 1e12)
    box.setDecimals(6)
    box.setValue(value)
    box.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    return box


class AxisRow(QWidget):
    def __init__(self, candidates: list[Candidate], parent=None) -> None:
        super().__init__(parent)
        self.candidates = candidates
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
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for label, widget in (
            ("", self.target),
            ("from", self.start),
            ("to", self.stop),
            ("unit", self.unit),
            ("points", self.points),
        ):
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.target.currentIndexChanged.connect(self._reset_range)
        self._reset_range()

    def _reset_range(self) -> None:
        index = self.target.currentIndex()
        if index < 0:
            return
        cand = self.candidates[index]
        self.unit.setText(cand.unit)
        cur = cand.current if cand.current is not None else 0.0
        if cand.unit and cand.current is not None:
            cur = float(cand.current)
        span = abs(cur) * 0.2 or 1.0
        self.start.setValue(cur - span)
        self.stop.setValue(cur + span)

    def axis(self) -> Axis:
        return Axis(
            self.target.currentText(),
            np.linspace(self.start.value(), self.stop.value(), self.points.value()),
            self.unit.text().strip(),
        )

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


class SweepTab(RunGuard, QWidget):
    """1D/2D parameter sweeps of the open setup, plotted against a chosen result."""

    def __init__(self, provider, parent=None) -> None:
        super().__init__(parent)
        self._guard_init()
        self.provider = provider
        self.runner = JobRunner(self)
        self.result = None
        self.outputs: list[str] = []
        self.candidates: list[Candidate] = []
        self.rows = [AxisRow([]), AxisRow([])]
        self.two_d = QCheckBox("second axis (2D)")
        self.metric = QComboBox()
        self.metric.setMinimumWidth(240)
        self.run_button = QPushButton("Sweep")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.refresh_button = QPushButton("Reload parameters")
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
        form.addLayout(second)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Output"))
        actions.addWidget(self.metric)
        for w in (
            self.run_button,
            self.cancel_button,
            self.refresh_button,
            self.optimize_button,
            self.export_button,
            self.png_button,
            self.progress,
        ):
            actions.addWidget(w)
        actions.addWidget(self.status, 1)
        form.addLayout(actions)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form)
        layout.addWidget(self.plot, 1)

        self.rows[1].setEnabled(False)
        self.two_d.toggled.connect(self.rows[1].setEnabled)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.refresh_button.clicked.connect(self.reload_candidates)
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

    def _failed(self, message: str) -> None:
        if self.run_is_current():
            self._end(f"failed: {message.splitlines()[0]}")
        else:
            self._end("")

    def reset(self) -> None:
        self.invalidate()
        self.result = None
        self.status_text = ""
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

    def reload_candidates(self) -> None:
        setup, library = self.provider()
        if setup is None:
            return
        candidates = candidate_targets(setup, library)
        for row in self.rows:
            row.set_candidates(candidates)
        self.candidates = candidates

    def set_outcome(self, outcome) -> None:
        if not self.rows[0].candidates:
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

    def run(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return
        axes = [self.rows[0].axis()]
        if self.two_d.isChecked():
            if self.rows[1].target.currentText() == axes[0].target:
                self.status.setText("the two axes must differ")
                return
            axes.append(self.rows[1].axis())
        names = None
        if not axes[0].target:
            self.status.setText("no parameter to sweep")
            return
        for axis in axes:
            problem = self._axis_problem(axis, setup, library)
            if problem:
                self.status.setText(problem)
                return
        self.begin_run()

        def job(should_cancel, progress):
            return run_sweep(
                setup,
                library,
                axes,
                names,
                cache=RunCache(),
                should_cancel=should_cancel,
                progress=progress,
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

    def _done(self, result) -> None:
        if not self.run_is_current():
            self._end("")
            return
        self.result = result
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
            style_plot(self.plot, unit(ax[0]), name)
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
        layout.addWidget(QLabel("Variables (bounds are the from / to values)"))
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
        row = AxisRow(self.candidates)
        row.points.hide()
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
