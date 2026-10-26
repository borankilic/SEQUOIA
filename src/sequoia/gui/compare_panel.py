import json
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sequoia.analysis.fitting import (
    MODELS,
    FitResult,
    fit,
    from_display,
    param_unit,
    record,
    table_csv,
)
from sequoia.analysis.measured import (
    Comparison,
    MeasuredData,
    compare_sweep,
    load_measured,
    poisson_sigma,
    subtract_background,
)
from sequoia.errors import SequoiaError
from sequoia.gui.analysis_tabs import style_plot
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, plain
from sequoia.gui.runguard import RunGuard
from sequoia.library.importers.table import Table, list_sheets, read_table
from sequoia.units import from_si

SWEEP_MODE = "Compare with a SEQUOIA sweep"
NONE = "(none)"
PREVIEW_ROWS = 12
CURVE_POINTS = 400
_ERROR = "color: #b00020;"


class CompareTab(RunGuard, QWidget):
    """Measured data: column mapping, fits with uncertainties and identifiability, and
    comparison with a sweep of the drawn setup."""

    def __init__(self, provider, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard_init()
        self.provider = provider
        self.runner = JobRunner(self)
        self.on_issues = None
        self.table: Table | None = None
        self.path: Path | None = None
        self.data: MeasuredData | None = None
        self.result: FitResult | None = None
        self.comparison: Comparison | None = None
        self.note = ""

        self.load_button = QPushButton("Load data...")
        self.load_button.setToolTip("CSV, TSV, TXT or XLSX; the file is never modified")
        self.file_label = QLabel("no data loaded")
        self.model = QComboBox()
        self.model.addItems([*MODELS, SWEEP_MODE])
        self.run_button = QPushButton("Fit")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Export...")
        self.export_button.setToolTip(
            "Writes the data, model and residuals (CSV) and a JSON "
            "record with the fit, file hash and mapping"
        )
        self.export_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.sheet = QComboBox()
        self.x_col, self.y_col = QComboBox(), QComboBox()
        self.sigma_col, self.bg_col, self.time_col = QComboBox(), QComboBox(), QComboBox()
        self.x_unit, self.y_unit = QLineEdit(), QLineEdit()
        self.x_unit.setPlaceholderText("e.g. deg, ns, mm")
        self.y_unit.setPlaceholderText("e.g. counts, Hz, mW, %")
        self.time_unit = QLineEdit("s")
        self.acq_time = QDoubleSpinBox()
        self.acq_time.setRange(0.0, 1e9)
        self.acq_time.setDecimals(6)
        self.acq_time.setSuffix(" s")
        self.acq_time.setSpecialValueText("none")
        self.acq_time.setToolTip(
            "One acquisition time for every row (when there is no time "
            "column); turns a rate into counts for Poisson weights"
        )
        self.y_kind = QComboBox()
        self.y_kind.addItems(["auto", "counts", "rate", "other"])
        self.subtract = QCheckBox("subtract background column")
        self.sidebands = QSpinBox()
        self.sidebands.setRange(1, 1000)
        self.sidebands.setPrefix("sidebands ")
        self.sidebands.setToolTip("The background column is the mean of this many equal windows")
        mapping = QFormLayout()
        mapping.addRow("Sheet", self.sheet)
        for label, combo, unit in (
            ("x", self.x_col, self.x_unit),
            ("y", self.y_col, self.y_unit),
        ):
            row = QHBoxLayout()
            row.addWidget(combo, 2)
            row.addWidget(QLabel("unit"))
            row.addWidget(unit, 1)
            mapping.addRow(label, row)
        mapping.addRow("y uncertainty", self.sigma_col)
        mapping.addRow("y kind", self.y_kind)
        trow = QHBoxLayout()
        trow.addWidget(self.time_col, 2)
        trow.addWidget(QLabel("unit"))
        trow.addWidget(self.time_unit, 1)
        mapping.addRow("time column", trow)
        mapping.addRow("acquisition time", self.acq_time)
        brow = QHBoxLayout()
        brow.addWidget(self.bg_col, 2)
        brow.addWidget(self.subtract)
        brow.addWidget(self.sidebands)
        mapping.addRow("background", brow)
        self.preview = _make_table([])
        self.mapping_box = QGroupBox("Column mapping")
        mv = QVBoxLayout(self.mapping_box)
        mv.addLayout(mapping)
        mv.addWidget(QLabel(f"Preview (first {PREVIEW_ROWS} rows)"))
        mv.addWidget(self.preview, 1)

        self.fixed = QLineEdit()
        self.fixed.setPlaceholderText("fix calibrated parameters, e.g. efficiency=0.2 (data units)")
        self.target = QComboBox()
        self.target.setEditable(True)
        self.target_unit = QLineEdit()
        self.target_unit.setPlaceholderText("unit of the swept parameter")
        self.metric_box = QComboBox()
        self.metric_box.setEditable(True)
        self.fit_scale = QCheckBox("fit one scale factor (loss budget)")
        self.sweep_box = QGroupBox("Sweep of the drawn setup at the data x values")
        sf = QFormLayout(self.sweep_box)
        sf.addRow("parameter", self.target)
        sf.addRow("unit", self.target_unit)
        sf.addRow("compared result", self.metric_box)
        sf.addRow(self.fit_scale)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.mapping_box, 1)
        lv.addWidget(self.fixed)
        lv.addWidget(self.sweep_box)

        self.plot = pg.PlotWidget()
        self.resid_plot = pg.PlotWidget()
        self.resid_plot.setXLink(self.plot)
        style_plot(self.plot, "x", "y")
        style_plot(self.resid_plot, "x", "residual / sigma")
        self.plot.addLegend()
        self.data_points = self.plot.plot(
            pen=None, symbol="o", symbolSize=5, symbolBrush="#1f77b4", name="data"
        )
        self.error_bars = pg.ErrorBarItem(x=np.array([]), y=np.array([]), pen="#1f77b4")
        self.plot.addItem(self.error_bars)
        self.model_curve = self.plot.plot(pen=pg.mkPen("#d62728", width=2), name="model")
        self.resid_points = self.resid_plot.plot(
            pen=None, symbol="o", symbolSize=4, symbolBrush="#555555"
        )
        self.resid_plot.addLine(y=0, pen=pg.mkPen("#888888"))

        self.params = _make_table(["Parameter", "Value", "Std. error", "Unit", "Status"])
        self.correlation = _make_table([])
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.details = QTabWidget()
        self.details.addTab(self.params, "Parameters")
        self.details.addTab(self.correlation, "Correlation")
        self.details.addTab(self.notes, "Notes && provenance")
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.plot)
        right.addWidget(self.resid_plot)
        right.addWidget(self.details)
        right.setSizes([260, 100, 180])

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([330, 600])
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        for w in (self.load_button, self.file_label, self.model, self.run_button):
            bar.addWidget(w)
        for w in (self.cancel_button, self.export_button, self.progress):
            bar.addWidget(w)
        bar.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)

        self.load_button.clicked.connect(lambda: self.load_file())
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.export_button.clicked.connect(lambda: self.export())
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        self.model.currentIndexChanged.connect(self._mode_changed)
        self.sheet.currentIndexChanged.connect(self._sheet_changed)
        for combo in (self.x_col, self.y_col, self.sigma_col, self.bg_col, self.time_col):
            combo.currentIndexChanged.connect(self._column_changed)
        self.y_kind.currentIndexChanged.connect(self.apply_mapping)
        self.target.currentIndexChanged.connect(self._target_changed)
        for edit in (self.x_unit, self.y_unit, self.time_unit):
            edit.editingFinished.connect(self.apply_mapping)
        self.acq_time.valueChanged.connect(self.apply_mapping)
        self.subtract.toggled.connect(self.apply_mapping)
        self.sidebands.valueChanged.connect(self.apply_mapping)
        for w in (self.fixed, self.target_unit):
            w.textChanged.connect(self.refresh_stale)
        for combo in (self.target, self.metric_box):
            combo.currentTextChanged.connect(self.refresh_stale)
        self.fit_scale.toggled.connect(self.refresh_stale)
        self._mode_changed()
        self._set_mapping_enabled(False)

    @property
    def sweep_mode(self) -> bool:
        return self.model.currentText() == SWEEP_MODE

    def document_key(self):
        return super().document_key() if self.sweep_mode else None

    def config_key(self):
        return (
            id(self.data),
            self.model.currentText(),
            self.fixed.text(),
            self.target.currentText(),
            self.target_unit.text(),
            self.metric_box.currentText(),
            self.fit_scale.isChecked(),
        )

    def _error(self, text: str) -> None:
        self.status.setStyleSheet(_ERROR)
        self.status.setText(text)

    def _info(self, text: str) -> None:
        self.status.setStyleSheet("color: palette(placeholder-text);")
        self.status.setText(text)

    def _set_mapping_enabled(self, enabled: bool) -> None:
        self.mapping_box.setEnabled(enabled)
        self.run_button.setEnabled(enabled and self.data is not None)

    def load_file(self, path: str | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load measured data", "", "Tables (*.csv *.tsv *.txt *.dat *.xlsx *.xlsm)"
            )
            if not path:
                return False
        path = Path(path)
        sheets: list[str] = []
        try:
            if path.suffix.lower() in (".xlsx", ".xlsm"):
                sheets = list_sheets(path)
            table = read_table(path, sheets[0] if sheets else None)
        except (SequoiaError, OSError, ValueError) as exc:
            self._error(f"cannot load {path.name}: {exc}")
            return False
        except Exception as exc:
            self._error(f"cannot load {path.name}: {type(exc).__name__}: {exc}")
            return False
        self.path = path
        self.sheet.blockSignals(True)
        self.sheet.clear()
        self.sheet.addItems(sheets)
        self.sheet.setEnabled(bool(sheets))
        self.sheet.blockSignals(False)
        self._set_table(table)
        return self.data is not None

    def _sheet_changed(self) -> None:
        if self.path is None or not self.sheet.currentText():
            return
        try:
            table = read_table(self.path, self.sheet.currentText())
        except SequoiaError as exc:
            self._error(str(exc))
            return
        self._set_table(table)

    def _set_table(self, table: Table) -> None:
        self.table = table
        self.file_label.setText(table.path.name)
        headers = [c.header for c in table.columns]
        guesses = {"x": 0, "y": 1 if len(headers) > 1 else 0}
        for i, h in enumerate(headers):
            low = h.lower()
            if any(k in low for k in ("sigma", "err", "uncert", "std")):
                guesses.setdefault("sigma", i)
            elif any(k in low for k in ("background", "accidental", "sideband")):
                guesses.setdefault("bg", i)
            elif any(k in low for k in ("acq", "integration", "duration")):
                guesses.setdefault("time", i)
        for key, combo, optional in (
            ("x", self.x_col, False),
            ("y", self.y_col, False),
            ("sigma", self.sigma_col, True),
            ("bg", self.bg_col, True),
            ("time", self.time_col, True),
        ):
            combo.blockSignals(True)
            combo.clear()
            if optional:
                combo.addItem(NONE)
            combo.addItems(headers)
            index = guesses.get(key)
            combo.setCurrentIndex(0 if index is None else index + (1 if optional else 0))
            combo.blockSignals(False)
        self.preview.setColumnCount(len(headers))
        self.preview.setHorizontalHeaderLabels(headers)
        n = min(PREVIEW_ROWS, len(table.columns[0].values)) if table.columns else 0
        rows = [
            [
                _NumItem(plain(float(c.values[r])) if np.isfinite(c.values[r]) else "missing")
                for c in table.columns
            ]
            for r in range(n)
        ]
        _fill(self.preview, rows)
        self._set_mapping_enabled(True)
        self._column_changed()

    def _column_changed(self) -> None:
        if self.table is None:
            return
        for combo, edit in ((self.x_col, self.x_unit), (self.y_col, self.y_unit)):
            column = self._column(combo)
            if column is not None:
                edit.setText(column.unit or "")
        self.apply_mapping()

    def _column(self, combo: QComboBox):
        if self.table is None or combo.currentText() in ("", NONE):
            return None
        return self.table.columns[combo.currentIndex() - (1 if combo.itemText(0) == NONE else 0)]

    def apply_mapping(self, *_args) -> bool:
        if self.table is None:
            return False
        refs = {}
        for key, combo in (
            ("x", self.x_col),
            ("y", self.y_col),
            ("sigma", self.sigma_col),
            ("background", self.bg_col),
            ("time", self.time_col),
        ):
            column = self._column(combo)
            if column is not None:
                refs[key] = self.table.columns.index(column)
        try:
            data = load_measured(
                self.table.path,
                refs["x"],
                refs["y"],
                sigma=refs.get("sigma"),
                background=refs.get("background"),
                time=refs.get("time"),
                x_unit=self.x_unit.text().strip() or None,
                y_unit=self.y_unit.text().strip() or None,
                time_unit=self.time_unit.text().strip() or "s",
                acquisition_time=None if "time" in refs else (self.acq_time.value() or None),
                y_kind=None if self.y_kind.currentText() == "auto" else self.y_kind.currentText(),
                table=self.table,
            )
            if self.subtract.isChecked():
                data = subtract_background(data, self.sidebands.value())
        except (SequoiaError, KeyError, ValueError) as exc:
            self.data = None
            self.run_button.setEnabled(False)
            self._error(f"mapping: {exc}")
            self._show_data()
            return False
        self.data = data
        self.run_button.setEnabled(not self.runner.busy)
        dropped = f", {len(data.dropped)} row(s) dropped" if data.dropped else ""
        self._info(
            f"{len(data)} points, x in {data.x_unit}, y in {data.y_unit} ({data.y_kind})"
            f"{dropped}" + ("; " + "; ".join(data.warnings) if data.warnings else "")
        )
        self._clear_results()
        self._show_data()
        return True

    def _mode_changed(self, *_args) -> None:
        sweep = self.sweep_mode
        self.sweep_box.setVisible(sweep)
        self.fixed.setVisible(not sweep)
        self.run_button.setText("Run sweep and compare" if sweep else "Fit")
        if sweep:
            self._fill_targets()
        self.refresh_stale()

    def _fill_targets(self) -> None:
        setup, library = self.provider()
        if setup is None:
            return
        from sequoia.analysis.sweep import candidate_targets

        try:
            candidates = candidate_targets(setup, library)
        except Exception:
            return
        text = self.target.currentText()
        self.target.blockSignals(True)
        self.target.clear()
        for c in candidates:
            self.target.addItem(c.target, c.unit)
        if text:
            self.target.setCurrentText(text)
        self.target.blockSignals(False)
        if not self.target_unit.text() and self.target.currentData() is not None:
            self.target_unit.setText(self.target.currentData())

    def _target_changed(self, *_args) -> None:
        unit = self.target.currentData()
        if unit is not None:
            self.target_unit.setText(unit)

    def set_outcome(self, outcome) -> None:
        if outcome is not None:
            from sequoia.analysis.sweep import metrics_of

            try:
                names = sorted(metrics_of(outcome))
            except Exception:
                names = []
            text = self.metric_box.currentText()
            self.metric_box.blockSignals(True)
            self.metric_box.clear()
            self.metric_box.addItems(names)
            self.metric_box.setCurrentText(text)
            self.metric_box.blockSignals(False)
            if self.sweep_mode:
                self._fill_targets()
        if self.track_revision(outcome) and (self.result or self.comparison) is not None:
            self._info(self.stale_text(self.note))

    def refresh_stale(self, *_args) -> None:
        if self.results_outdated() and (self.result or self.comparison) is not None:
            self._info(self.stale_text(self.note))

    def _clear_results(self) -> None:
        self.result = None
        self.comparison = None
        self.note = ""
        self.export_button.setEnabled(False)
        _fill(self.params, [])
        self.correlation.setRowCount(0)
        self.correlation.setColumnCount(0)
        self.notes.setPlainText("")
        self.model_curve.setData([], [])
        self.resid_points.setData([], [])

    def reset(self) -> None:
        self.invalidate()
        self._clear_results()
        self._end("")
        if self.on_issues is not None:
            self.on_issues([])

    def shutdown(self) -> None:
        self.runner.shutdown()

    def _sigma_display(self, data: MeasuredData):
        if data.sigma is not None:
            return np.asarray(data.sigma)
        return poisson_sigma(data)

    def _show_data(self) -> None:
        data = self.data
        if data is None:
            self.data_points.setData([], [])
            self.error_bars.setData(x=np.array([]), y=np.array([]), height=np.array([]))
            return
        self.data_points.setData(np.asarray(data.x), np.asarray(data.y))
        sigma = self._sigma_display(data)
        if sigma is not None:
            self.error_bars.setData(x=np.asarray(data.x), y=np.asarray(data.y), height=2 * sigma)
        else:
            self.error_bars.setData(x=np.array([]), y=np.array([]), height=np.array([]))
        x_label = f"{data.x_label} ({data.x_unit})"
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", f"{data.y_label} ({data.y_unit})")
        self.resid_plot.setLabel("bottom", x_label)

    def parse_fixed(self, model) -> tuple[dict[str, float], dict[str, str]]:
        text = self.fixed.text().strip()
        fixed: dict[str, float] = {}
        if not text:
            return fixed, {}
        for item in text.replace(";", ",").split(","):
            name, sep, value = item.partition("=")
            name = name.strip()
            if not sep or name not in model.params:
                raise SequoiaError(
                    f"fixed parameter {item.strip()!r}: use name=value with one of "
                    f"{', '.join(model.params)}"
                )
            kind = model.kinds[model.params.index(name)]
            unit = param_unit(kind, self.data.x_unit, self.data.y_unit)
            fixed[name] = from_display(float(value), unit)
        return fixed, {n: "independently calibrated (entered by the user)" for n in fixed}

    def run(self) -> None:
        data = self.data
        if data is None or self.runner.busy:
            return
        if self.sweep_mode:
            setup, library = self.provider()
            if setup is None:
                self._error("no setup to sweep: open or draw a setup first")
                return
            target = self.target.currentText().strip()
            unit = self.target_unit.text().strip()
            metric = self.metric_box.currentText().strip()
            scale = self.fit_scale.isChecked()
            if not target or not metric:
                self._error("choose the swept parameter and the compared result")
                return

            def job(should_cancel, progress):
                return compare_sweep(
                    data,
                    setup,
                    library,
                    target,
                    unit,
                    metric,
                    fit_scale=scale,
                    should_cancel=should_cancel,
                    progress=progress,
                )

        else:
            model = MODELS[self.model.currentText()]()
            try:
                fixed, sources = self.parse_fixed(model)
            except (SequoiaError, ValueError) as exc:
                self._error(str(exc))
                return

            def job(should_cancel, progress):
                return fit(data, model, fixed=fixed, fixed_sources=sources)

        self.begin_run()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._info("running...")
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(self.data is not None)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        if text:
            self._info(text)

    def _failed(self, message: str) -> None:
        if not self.run_is_current():
            self._end("")
            return
        self._end("")
        self._error(f"failed: {message.splitlines()[0]}")

    def _done(self, outcome) -> None:
        if not self.run_is_current():
            self._end("")
            return
        self._clear_results()
        data = self.data
        if isinstance(outcome, FitResult):
            self.result = outcome
            self._show_fit(outcome, data)
        else:
            self.comparison = outcome
            self._show_comparison(outcome)
        self.export_button.setEnabled(True)
        stale = self.accept_run()
        self._end(self.stale_text(self.note) if stale else self.note)
        if self.result is not None and self.result.flags:
            self.status.setStyleSheet(_ERROR)

    def _show_fit(self, r: FitResult, data: MeasuredData) -> None:
        lo, hi = float(np.min(data.x_si)), float(np.max(data.x_si))
        grid = np.linspace(lo, hi, CURVE_POINTS)
        gx = grid if data.x_unit in ("1", "counts") else from_si(grid, data.x_unit)
        self.model_curve.setData(np.asarray(gx), r.predict_display(grid))
        self.resid_points.setData(np.asarray(r.x), np.asarray(r.normalized))
        rows = []
        for name, value, se, unit, status in r.rows():
            rows.append(
                [
                    _NumItem(name),
                    _NumItem(plain(value), value),
                    _NumItem("unbounded" if math.isinf(se) else plain(se), se),
                    _NumItem(unit),
                    _NumItem(status),
                ]
            )
        _fill(self.params, rows)
        k = len(r.free)
        self.correlation.setColumnCount(k)
        self.correlation.setRowCount(k)
        self.correlation.setHorizontalHeaderLabels(r.free)
        self.correlation.setVerticalHeaderLabels(r.free)
        self.correlation.verticalHeader().setVisible(True)
        for i in range(k):
            for j in range(k):
                c = r.correlation[i, j]
                self.correlation.setItem(
                    i, j, QTableWidgetItem("-" if not math.isfinite(c) else f"{c:+.3f}")
                )
        self.note = (
            f"{r.model.name}: chi2/dof = {plain(r.reduced_chi2)} ({r.dof} dof), "
            f"{r.weighting}"
            + (
                "; identifiability: " + "; ".join(r.flags)
                if r.flags
                else "; all fitted parameters identifiable"
            )
        )
        self.notes.setPlainText(self._notes(r.flags, r.warnings, r.model.description))

    def _show_comparison(self, c: Comparison) -> None:
        order = np.argsort(np.asarray(c.x))
        self.model_curve.setData(np.asarray(c.x)[order], np.asarray(c.model)[order])
        self.resid_points.setData(np.asarray(c.x), np.asarray(c.normalized))
        rows = [
            [
                _NumItem("chi2"),
                _NumItem(plain(c.chi2), c.chi2),
                _NumItem(""),
                _NumItem(""),
                _NumItem(c.weighting),
            ],
            [
                _NumItem("chi2 / dof"),
                _NumItem(plain(c.reduced_chi2), c.reduced_chi2),
                _NumItem(""),
                _NumItem(""),
                _NumItem(f"{c.dof} dof"),
            ],
        ]
        if c.scale is not None:
            for name, value, se, unit, status in c.scale.rows():
                rows.append(
                    [
                        _NumItem(name),
                        _NumItem(plain(value), value),
                        _NumItem(plain(se), se),
                        _NumItem(unit),
                        _NumItem(status),
                    ]
                )
        _fill(self.params, rows)
        self.note = (
            f"{c.label} vs data: chi2/dof = {plain(c.reduced_chi2)} ({c.dof} dof), {c.weighting}"
        )
        self.notes.setPlainText(
            self._notes(c.scale.flags if c.scale else [], c.warnings, json.dumps(c.details))
        )

    def _notes(self, flags, warnings, description: str) -> str:
        data = self.data
        lines = [description, ""]
        if flags:
            lines += ["Identifiability:", *(f"- {f}" for f in flags), ""]
        if warnings:
            lines += ["Warnings:", *(f"- {w}" for w in warnings), ""]
        if data is not None:
            base = data
            while base.original is not None:
                base = base.original
            lines.append(
                f"Source: {base.source.get('file')} (sha256 {base.source.get('sha256', '')[:16]})"
            )
            lines.append(f"Columns: {base.source.get('columns')}")
            for row, why in base.dropped:
                lines.append(f"dropped data row {row}: {why}")
            for op in data.operations:
                lines.append(f"derived: {op}")
        return "\n".join(lines)

    def export(self, path: str | None = None) -> bool:
        data = self.data
        if data is None or (self.result is None and self.comparison is None):
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export comparison", "comparison.csv")
            if not path:
                return False
        base = Path(path).with_suffix("")
        if self.result is not None:
            r = self.result
            csv_text = table_csv(data, r.predict_display(data.x_si), r.residuals, r.normalized)
        else:
            c = self.comparison
            csv_text = table_csv(data, c.model, c.residuals, c.normalized)
        extra = {"gui": {"mode": self.model.currentText(), "fixed": self.fixed.text()}}
        if self.comparison is not None:
            setup, library = self.provider()
            if setup is not None and library is not None:
                try:
                    refs = {i.part for i in setup.instances} | {p.part for p in setup.probes}
                    extra["setup"] = {
                        "name": setup.name,
                        "library_fingerprint": library.fingerprint(refs),
                    }
                except Exception as exc:
                    extra["setup"] = {"name": setup.name, "fingerprint_error": str(exc)}
        rec = record(data, self.result, self.comparison, extra)
        try:
            base.with_suffix(".csv").write_text(csv_text, encoding="utf-8")
            base.with_suffix(".json").write_text(
                json.dumps(rec, indent=2, allow_nan=False), encoding="utf-8"
            )
        except (OSError, ValueError) as exc:
            self._error(f"cannot export: {exc}")
            return False
        self._info(f"exported {base.with_suffix('.csv').name} and {base.with_suffix('.json').name}")
        return True
