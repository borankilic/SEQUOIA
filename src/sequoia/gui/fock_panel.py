import copy
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sequoia.engines.cache import Cancelled
from sequoia.errors import Issue, SequoiaError
from sequoia.fock.analysis import condition, count_variance, format_state
from sequoia.fock.circuit import (
    BeamSplitter,
    Block,
    Circuit,
    FockError,
    Loss,
    Permutation,
    PhaseShifter,
)
from sequoia.fock.detectors import IdealPNR, PseudoPNR, Threshold
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import (
    DetectorSpec,
    Experiment,
    FockResult,
    load_experiment,
    run_experiment,
    save_experiment,
)
from sequoia.fock.from_setup import (
    ComponentRow,
    Extraction,
    InputDescription,
    SetupExperiment,
    SetupResult,
    build_experiment_from_setup,
    extraction_experiment,
    list_components,
    list_inputs,
    run_setup_experiment,
)
from sequoia.gui.jobs import JobRunner
from sequoia.gui.runguard import RunGuard

ELEMENT_TYPES = ("beam splitter", "phase shifter", "permutation", "loss", "block")
DETECTOR_CHOICES = ("ideal PNR", "threshold", "pseudo-PNR")
BAR_LIMIT = 40
_COND = re.compile(r"^\s*(\d+)\s*(==|=|!=|>=|<=|>|<)\s*(\d+)\s*$")
_NAMED = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(==|=|!=|>=|<=|>|<)\s*(\d+)\s*$")


class _SortItem(QTableWidgetItem):
    def __init__(self, text: str, key: float | None = None) -> None:
        super().__init__(text)
        self._key = key
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other) -> bool:
        if isinstance(other, _SortItem) and self._key is not None and other._key is not None:
            return self._key < other._key
        return self.text() < other.text()


def _read_only_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(22)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setSortingEnabled(True)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _parse_ints(text: str, what: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.replace(";", ",").split(",") if x.strip() != "")
    except ValueError:
        raise FockError(f"{what}: {text!r} is not a list of integers") from None


def _parse_int(text: str, what: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise FockError(f"{what}: {text!r} is not a whole number") from None


def _parse_float(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise FockError(f"{what}: {text!r} is not a number") from None


def parse_filter(text: str, names: list[str] | None = None) -> dict[int, tuple[str, int]]:
    out: dict[int, tuple[str, int]] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        m = _COND.match(part) or _NAMED.match(part)
        if m is None:
            raise FockError(
                f"cannot read condition {part.strip()!r}; use e.g. 0=1, 1>=1 or spad_1=1"
            )
        key = m.group(1)
        if key.isdigit():
            idx = int(key)
        elif names is not None and key in names:
            idx = names.index(key)
        else:
            raise FockError(f"unknown detector {key!r}")
        out[idx] = (m.group(2), int(m.group(3)))
    return out


@dataclass
class _Payload:
    kind: str
    components: list[ComponentRow] = field(default_factory=list)
    inputs: list[InputDescription] = field(default_factory=list)
    se: SetupExperiment | None = None
    result: SetupResult | None = None
    extraction: tuple[Experiment, Extraction] | None = None
    error: str = ""


def _num(p: float) -> str:
    return f"{p:.6g}"


class FockTab(RunGuard, QWidget):
    """Photon counts: the drawn circuit as a photon-number experiment, or a free circuit editor."""

    def __init__(self, provider=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard_init()
        self.provider = provider
        self.on_issues = None
        self.runner = JobRunner(self)
        self.result: FockResult | None = None
        self.setup_exp: SetupExperiment | None = None
        self.setup_result: SetupResult | None = None
        self.extraction: Extraction | None = None
        self.components: list[ComponentRow] = []
        self.inputs: list[InputDescription] = []
        self._mixture = None
        self._counts_keys: tuple = (None, None, None)
        self._elements: list = []
        self._dirty: set[int] = set()
        self._building = False
        self._stale_note = ""
        self._base: Experiment | None = None
        self._shown: dict = {}
        self._saved_key: str | None = None
        self._result_key: str | None = None
        self._run_key: str | None = None
        self.confirm_replace = None

        self.mode_box = QComboBox()
        self.mode_box.addItems(["From setup", "Circuit editor"])
        self.import_setup_button = QPushButton("Import from setup")
        self.import_setup_button.setToolTip(
            "Read the drawn circuit: list its components and the photon inputs found in it."
        )
        self.extract_button = QPushButton("Extract to circuit editor")
        self.extract_button.setToolTip(
            "Turn the drawn circuit into editable beam splitter, phase and loss elements "
            "and open it in the circuit editor."
        )
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.0, 1e9)
        self.duration.setDecimals(6)
        self.duration.setValue(1.0)
        self.duration.setPrefix("duration ")
        self.duration.setSuffix(" s")
        self.n_max_pairs = QSpinBox()
        self.n_max_pairs.setRange(0, 12)
        self.n_max_pairs.setSpecialValueText("pairs: auto")
        self.n_max_pairs.setPrefix("max pairs ")
        self.detector_mode = QComboBox()
        self.detector_mode.addItems(["setup detectors", "ideal PNR"])
        self.open_button = QPushButton("Open circuit...")
        self.save_button = QPushButton("Save circuit...")
        self.import_button = QPushButton("Import Perceval GUI circuit...")
        self.backend = QComboBox()
        self.backend.addItem("native")
        from sequoia.fock.perceval_bridge import perceval_version

        if perceval_version() is not None:
            self.backend.addItem("perceval")
        self.shots = QSpinBox()
        self.shots.setRange(0, 100_000_000)
        self.shots.setValue(0)
        self.shots.setPrefix("shots ")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(1)
        self.seed.setPrefix("seed ")
        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setStyleSheet("color: palette(placeholder-text);")

        self.n_modes = QSpinBox()
        self.n_modes.setRange(1, 64)
        self.n_modes.setValue(2)
        self.n_modes.setPrefix("modes ")
        self.brightness = QDoubleSpinBox()
        self.brightness.setRange(0.0, 1.0)
        self.brightness.setDecimals(4)
        self.brightness.setSingleStep(0.05)
        self.brightness.setValue(1.0)
        self.brightness.setPrefix("emission prob. ")
        self.indist = QDoubleSpinBox()
        self.indist.setRange(0.0, 1.0)
        self.indist.setDecimals(4)
        self.indist.setSingleStep(0.05)
        self.indist.setValue(1.0)
        self.indist.setPrefix("indistinguishability ")
        self.max_photons = QSpinBox()
        self.max_photons.setRange(1, 16)
        self.max_photons.setValue(Limits().max_photons)
        self.max_photons.setPrefix("max photons ")

        self.circuit_table = QTableWidget(0, 5)
        self.circuit_table.setHorizontalHeaderLabels(
            ["Element", "Modes", "Parameter 1", "Parameter 2", "Label"]
        )
        self.circuit_table.horizontalHeader().setStretchLastSection(True)
        self.circuit_table.verticalHeader().setVisible(False)
        self.circuit_table.setToolTip(
            "beam splitter: modes 'a,b'; theta (rad), phase on first input (rad)\n"
            "phase shifter: mode; phase (rad)\n"
            "permutation: modes 'a,b,c'; target positions '1,2,0'\n"
            "loss: mode; loss probability (photon-loss channel)\n"
            "Label: the component or link of the drawn circuit the element came from."
        )
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.up_button = QPushButton("Up")
        self.down_button = QPushButton("Down")
        self.new_type = QComboBox()
        self.new_type.addItems(ELEMENT_TYPES[:4])

        self.input_table = QTableWidget(0, 8)
        self.input_table.setHorizontalHeaderLabels(
            ["Mode", "Photons", "Tags", "Detector", "Efficiency", "Wires", "Group", "Dark / window"]
        )
        self.input_table.horizontalHeader().setStretchLastSection(True)
        self.input_table.verticalHeader().setVisible(False)
        self.input_table.setToolTip(
            "Tags: one label per photon, e.g. '0,1'. Equal tags interfere, different tags do not.\n"
            "Detector efficiency is applied once, in the detector response.\n"
            "Group: modes with the same group name are read by one detector (summed photons).\n"
            "Dark / window: probability of one dark click per detection window."
        )
        self.preserved_label = QLabel("")
        self.preserved_label.setWordWrap(True)
        self.preserved_label.setStyleSheet("color: palette(placeholder-text);")
        self.mixture_label = QLabel("")
        self.mixture_label.setWordWrap(True)
        self.mixture_label.setVisible(False)
        self.clear_mixture_button = QPushButton("Use the occupation table instead")
        self.clear_mixture_button.setVisible(False)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("condition on detected state, e.g. 0=1, 1>=1")
        self.filter_edit.setClearButtonEnabled(True)
        self.success_label = QLabel("")
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: palette(placeholder-text);")
        self.results_table = _read_only_table(
            ["State", "Probability", "Counts", "Conditional probability"]
        )
        self.plot = pg.PlotWidget()
        self._style_plot(self.plot)
        self.bars: pg.BarGraphItem | None = None
        self.normalized = QCheckBox("show conditional distribution")
        self.normalized.setChecked(False)

        self.components_table = _read_only_table(
            ["Id", "Type", "Category", "Part", "Settings and key parameters", "Ports (H, V modes)"]
        )
        self.inputs_table = _read_only_table(
            [
                "Source",
                "Kind",
                "Photons",
                "Mean pairs / window",
                "Schmidt K",
                "Window",
                "Omitted weight",
                "Description",
            ]
        )
        self.states_table = _read_only_table(
            [
                "State",
                "Detectors",
                "Probability / window",
                "Potential rate (1/s, no dead time)",
                "Expected counts in duration",
                "Sigma (binomial, fixed windows)",
                "Conditional probability",
            ]
        )
        self.state_filter = QLineEdit()
        self.state_filter.setPlaceholderText(
            "condition on detectors, e.g. spad_1=1, spad_2>=1 (index or detector id)"
        )
        self.state_filter.setClearButtonEnabled(True)
        self.state_summary = QLabel("")
        self.state_summary.setWordWrap(True)
        self.clicks_table = _read_only_table(
            [
                "Click pattern",
                "Probability / window",
                "Potential rate (1/s, no dead time)",
                "Expected counts in duration",
                "Sigma (binomial)",
            ]
        )
        self.detector_table = _read_only_table(
            [
                "Detector",
                "Model",
                "Efficiency",
                "Dark / window",
                "P(click)",
                "Mean counts / window",
                "Potential click rate (1/s, no dead time)",
                "Expected counts in duration",
                "Sigma (binomial)",
            ]
        )
        self.marginals_table = _read_only_table(
            ["Quantity", "Detector / mode", "n", "P(n photons arrive)", "P(n detected)"]
        )
        self.cond_table = QTableWidget(0, 3)
        self.cond_table.setHorizontalHeaderLabels(["Detector", "Condition", "Value"])
        self.cond_table.horizontalHeader().setStretchLastSection(True)
        self.cond_table.verticalHeader().setVisible(False)
        self.cond_result = _read_only_table(
            [
                "State",
                "Detectors",
                "Conditional probability",
                "Joint probability / window",
                "Accepted-event rate (1/s)",
            ]
        )
        self.cond_summary = QLabel("")
        self.cond_summary.setWordWrap(True)
        self.samples_table = _read_only_table(["State", "Detectors", "Shots", "Frequency", "P"])
        self.samples_note = QLabel("")
        self.model_card = QLabel("")
        self.model_card.setWordWrap(True)
        self.model_card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notes_view = QPlainTextEdit()
        self.notes_view.setReadOnly(True)
        self.setup_plot = pg.PlotWidget()
        self._style_plot(self.setup_plot)
        self.setup_bars: pg.BarGraphItem | None = None
        self.setup_normalized = QCheckBox("show conditional distribution")
        self.pair_plot = pg.PlotWidget()
        self._style_plot(self.pair_plot, "photons arriving at detectors", "probability")
        self.pair_bars: pg.BarGraphItem | None = None

        row_a = QHBoxLayout()
        row_a.setContentsMargins(6, 4, 6, 0)
        for w in (
            self.mode_box,
            self.import_setup_button,
            self.extract_button,
            self.duration,
            self.n_max_pairs,
            self.detector_mode,
            self.open_button,
            self.save_button,
            self.import_button,
        ):
            row_a.addWidget(w)
        row_a.addStretch(1)
        row_b = QHBoxLayout()
        row_b.setContentsMargins(6, 0, 6, 4)
        for w in (
            self.backend,
            self.shots,
            self.seed,
            self.run_button,
            self.cancel_button,
            self.progress,
        ):
            row_b.addWidget(w)
        row_b.addWidget(self.status, 1)

        settings = QHBoxLayout()
        for w in (self.n_modes, self.brightness, self.indist, self.max_photons):
            settings.addWidget(w)
        settings.addStretch(1)

        circuit_buttons = QHBoxLayout()
        for w in (self.new_type, self.add_button, self.remove_button, self.up_button):
            circuit_buttons.addWidget(w)
        circuit_buttons.addWidget(self.down_button)
        circuit_buttons.addStretch(1)
        circuit_page = QWidget()
        cl = QVBoxLayout(circuit_page)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addLayout(circuit_buttons)
        cl.addWidget(self.circuit_table, 1)
        input_page = QWidget()
        il = QVBoxLayout(input_page)
        il.setContentsMargins(0, 0, 0, 0)
        il.addLayout(settings)
        il.addWidget(self.mixture_label)
        il.addWidget(self.clear_mixture_button)
        il.addWidget(self.input_table, 1)
        il.addWidget(self.preserved_label)
        self.editor_tabs = QTabWidget()
        self.editor_tabs.addTab(circuit_page, "Circuit")
        self.editor_tabs.addTab(input_page, "Input and detectors")

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Condition"))
        filter_row.addWidget(self.filter_edit, 1)
        filter_row.addWidget(self.normalized)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(filter_row)
        rl.addWidget(self.success_label)
        rl.addWidget(self.message_label)
        splitter_r = QSplitter(Qt.Orientation.Vertical)
        splitter_r.addWidget(self.results_table)
        splitter_r.addWidget(self.plot)
        rl.addWidget(splitter_r, 1)
        editor_page = QSplitter()
        editor_page.addWidget(self.editor_tabs)
        editor_page.addWidget(right)
        editor_page.setStretchFactor(0, 1)
        editor_page.setStretchFactor(1, 1)

        states_page = QWidget()
        sl = QVBoxLayout(states_page)
        sl.setContentsMargins(0, 0, 0, 0)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Condition"))
        srow.addWidget(self.state_filter, 1)
        sl.addLayout(srow)
        sl.addWidget(self.state_summary)
        sl.addWidget(self.states_table, 1)
        clicks_page = QSplitter(Qt.Orientation.Vertical)
        clicks_page.addWidget(self.detector_table)
        clicks_page.addWidget(self.clicks_table)
        cond_page = QWidget()
        cl2 = QVBoxLayout(cond_page)
        cl2.setContentsMargins(0, 0, 0, 0)
        cl2.addWidget(self.cond_table)
        cl2.addWidget(self.cond_summary)
        cl2.addWidget(self.cond_result, 1)
        samples_page = QWidget()
        sm = QVBoxLayout(samples_page)
        sm.setContentsMargins(0, 0, 0, 0)
        sm.addWidget(self.samples_note)
        sm.addWidget(self.samples_table, 1)
        plot_page = QWidget()
        pl = QVBoxLayout(plot_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.setup_normalized)
        pl.addWidget(self.setup_plot, 2)
        pl.addWidget(self.pair_plot, 1)
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(states_page, "Output states")
        self.result_tabs.addTab(clicks_page, "Detector clicks")
        self.result_tabs.addTab(self.marginals_table, "Marginals")
        self.result_tabs.addTab(cond_page, "Conditional")
        self.result_tabs.addTab(samples_page, "Samples")
        self.result_tabs.addTab(plot_page, "Plot")
        self.result_tabs.addTab(self.notes_view, "Notes")
        self.info_tabs = QTabWidget()
        self.info_tabs.addTab(self.components_table, "Components")
        self.info_tabs.addTab(self.inputs_table, "Inputs")
        setup_page = QSplitter(Qt.Orientation.Vertical)
        setup_page.addWidget(self.info_tabs)
        results_box = QWidget()
        rb = QVBoxLayout(results_box)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.addWidget(self.model_card)
        rb.addWidget(self.result_tabs, 1)
        setup_page.addWidget(results_box)
        setup_page.setStretchFactor(0, 1)
        setup_page.setStretchFactor(1, 2)

        self.stack = QStackedWidget()
        self.stack.addWidget(setup_page)
        self.stack.addWidget(editor_page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row_a)
        layout.addLayout(row_b)
        layout.addWidget(self.stack, 1)

        self.n_modes.valueChanged.connect(self._resize_modes)
        self.add_button.clicked.connect(self.add_element)
        self.remove_button.clicked.connect(self.remove_element)
        self.up_button.clicked.connect(lambda: self.move_element(-1))
        self.down_button.clicked.connect(lambda: self.move_element(1))
        self.circuit_table.itemChanged.connect(self._item_changed)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.open_button.clicked.connect(self.open_dialog)
        self.save_button.clicked.connect(self.save_dialog)
        self.import_button.clicked.connect(self.import_dialog)
        self.import_setup_button.clicked.connect(self.import_from_setup)
        self.extract_button.clicked.connect(self.extract_to_editor)
        self.clear_mixture_button.clicked.connect(self.clear_mixture)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.filter_edit.textChanged.connect(self._refresh_results)
        self.normalized.toggled.connect(self._refresh_results)
        self.state_filter.textChanged.connect(self._show_states)
        self.setup_normalized.toggled.connect(self._show_states)
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        for w in (self.duration, self.n_max_pairs, self.shots, self.seed, self.max_photons):
            w.valueChanged.connect(self.refresh_stale)
        for w in (self.detector_mode, self.backend):
            w.currentIndexChanged.connect(self.refresh_stale)
        for w in (self.n_modes, self.shots, self.seed, self.max_photons):
            w.valueChanged.connect(self._editor_changed)
        for w in (self.brightness, self.indist):
            w.valueChanged.connect(self._editor_changed)
        self.backend.currentIndexChanged.connect(self._editor_changed)
        self.circuit_table.itemChanged.connect(self._editor_changed)
        self.input_table.itemChanged.connect(self._editor_changed)
        self._resize_modes(2)
        self.set_experiment(self._default_experiment())
        self._saved_key = self.editor_key()
        self.mode_box.setCurrentIndex(0 if provider is not None else 1)
        self._mode_changed()

    @property
    def setup_mode(self) -> bool:
        return self.mode_box.currentIndex() == 0

    def _mode_changed(self, *_args) -> None:
        setup = self.setup_mode
        self.stack.setCurrentIndex(0 if setup else 1)
        for w in (self.import_setup_button, self.extract_button, self.duration, self.n_max_pairs):
            w.setVisible(setup)
        self.detector_mode.setVisible(setup)
        for w in (self.open_button, self.save_button, self.import_button):
            w.setVisible(not setup)

    @staticmethod
    def _style_plot(plot, bottom: str = "output state (rank)", left: str = "probability") -> None:
        pal = QApplication.palette()
        plot.setBackground(pal.base().color())
        item = plot.getPlotItem()
        fg = pal.text().color()
        for axis in ("left", "bottom"):
            item.getAxis(axis).setPen(fg)
            item.getAxis(axis).setTextPen(fg)
        item.setLabel("bottom", bottom)
        item.setLabel("left", left)
        item.showGrid(x=False, y=True, alpha=0.25)

    @staticmethod
    def _default_experiment() -> Experiment:
        c = Circuit(2).beam_splitter(0, 1)
        return Experiment(c, [1, 1], name="HOM beam splitter")

    def config_key(self):
        return (
            self.duration.value(),
            self.n_max_pairs.value(),
            self.detector_mode.currentIndex(),
            self.shots.value(),
            self.seed.value(),
            self.backend.currentText(),
            self.max_photons.value(),
        )

    def set_outcome(self, outcome) -> None:
        if self.track_revision(outcome) and (self.setup_result is not None or self.setup_exp):
            text = self.stale_text(self._stale_note)
            if self.setup_mode:
                self.status.setText(text)
            if self.setup_result is not None:
                self.state_summary.setText(text)

    def reset(self) -> None:
        self.invalidate()
        self.result = None
        self._result_key = None
        self.results_table.setRowCount(0)
        if self.bars is not None:
            self.plot.removeItem(self.bars)
            self.bars = None
        self.setup_exp = None
        self.setup_result = None
        self.extraction = None
        self.components = []
        self.inputs = []
        self._stale_note = ""
        if self.on_issues is not None:
            self.on_issues([])
        for table in (
            self.components_table,
            self.inputs_table,
            self.states_table,
            self.clicks_table,
            self.detector_table,
            self.marginals_table,
            self.cond_result,
            self.samples_table,
        ):
            table.setRowCount(0)
        self.cond_table.setRowCount(0)
        self.notes_view.setPlainText("")
        self.model_card.setText("")
        self.state_summary.setText("")
        self.cond_summary.setText("")
        self.samples_note.setText("")
        if self.setup_bars is not None:
            self.setup_plot.removeItem(self.setup_bars)
            self.setup_bars = None
        if self.pair_bars is not None:
            self.pair_plot.removeItem(self.pair_bars)
            self.pair_bars = None
        self.status.setText("")
        self.success_label.setText("")
        self.message_label.setText("")

    def shutdown(self) -> None:
        self.runner.shutdown()

    def _resize_modes(self, n: int) -> None:
        self._building = True
        old = self.input_table.rowCount()
        self.input_table.setRowCount(n)
        for r in range(old, n):
            self._init_input_row(r)
        self._building = False

    def _init_input_row(self, r: int) -> None:
        mode = QTableWidgetItem(str(r))
        mode.setFlags(mode.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.input_table.setItem(r, 0, mode)
        self.input_table.setItem(r, 1, QTableWidgetItem("0"))
        self.input_table.setItem(r, 2, QTableWidgetItem(""))
        combo = QComboBox()
        combo.addItems(DETECTOR_CHOICES)
        combo.currentIndexChanged.connect(self._editor_changed)
        self.input_table.setCellWidget(r, 3, combo)
        self.input_table.setItem(r, 4, QTableWidgetItem("1"))
        self.input_table.setItem(r, 5, QTableWidgetItem("4"))
        self.input_table.setItem(r, 6, QTableWidgetItem(""))
        self.input_table.setItem(r, 7, QTableWidgetItem("0"))

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if not self._building:
            self._dirty.add(item.row())

    def _row_texts(self, e) -> tuple[str, str, str, str, str]:
        label = e.label or ""
        if isinstance(e, BeamSplitter):
            return (
                "beam splitter",
                ",".join(map(str, e.modes)),
                f"{e.theta:.10g}",
                f"{e.phi_tl:.10g}",
                label,
            )
        if isinstance(e, PhaseShifter):
            return "phase shifter", str(e.mode), f"{e.phase:.10g}", "", label
        if isinstance(e, Permutation):
            return (
                "permutation",
                ",".join(map(str, e.modes)),
                ",".join(map(str, e.perm)),
                "",
                label,
            )
        if isinstance(e, Loss):
            return "loss", str(e.mode), f"{e.loss:.10g}", e.kind, label
        return (
            "block",
            ",".join(map(str, e.modes)),
            f"{len(e.modes)}x{len(e.modes)} matrix",
            "",
            label,
        )

    def _fill_circuit_table(self) -> None:
        self._building = True
        self.circuit_table.setRowCount(len(self._elements))
        for r, e in enumerate(self._elements):
            for c, text in enumerate(self._row_texts(e)):
                item = QTableWidgetItem(text)
                if isinstance(e, Block) or c in (0, 4):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.circuit_table.setItem(r, c, item)
        self._dirty.clear()
        self._building = False

    def add_element(self) -> None:
        kind = self.new_type.currentText()
        n = self.n_modes.value()
        if kind == "beam splitter":
            e = BeamSplitter((0, min(1, n - 1)) if n > 1 else (0, 0))
        elif kind == "phase shifter":
            e = PhaseShifter(0, 0.0)
        elif kind == "permutation":
            e = Permutation((0, min(1, n - 1)), (1, 0)) if n > 1 else Permutation((0,), (0,))
        else:
            e = Loss(0, 0.0)
        self._elements = self._collect_elements(strict=False)
        self._elements.append(e)
        self._fill_circuit_table()
        self._editor_changed()

    def remove_element(self) -> None:
        row = self.circuit_table.currentRow()
        if row < 0:
            return
        self._elements = self._collect_elements(strict=False)
        del self._elements[row]
        self._fill_circuit_table()
        self._editor_changed()

    def move_element(self, step: int) -> None:
        row = self.circuit_table.currentRow()
        new = row + step
        if row < 0 or not 0 <= new < len(self._elements):
            return
        self._elements = self._collect_elements(strict=False)
        self._elements[row], self._elements[new] = self._elements[new], self._elements[row]
        self._fill_circuit_table()
        self.circuit_table.selectRow(new)
        self._editor_changed()

    def _text(self, table: QTableWidget, r: int, c: int) -> str:
        item = table.item(r, c)
        return item.text().strip() if item is not None else ""

    def _collect_elements(self, strict: bool = True) -> list:
        out = []
        for r, old in enumerate(self._elements[: self.circuit_table.rowCount()]):
            if r not in self._dirty or isinstance(old, Block):
                out.append(old)
                continue
            kind = self._text(self.circuit_table, r, 0)
            modes = self._text(self.circuit_table, r, 1)
            p1 = self._text(self.circuit_table, r, 2)
            p2 = self._text(self.circuit_table, r, 3)
            label = old.label
            try:
                if kind == "beam splitter":
                    m = _parse_ints(modes, f"row {r + 1} modes")
                    if len(m) != 2:
                        raise FockError(f"row {r + 1}: a beam splitter needs two modes")
                    base = old if isinstance(old, BeamSplitter) else BeamSplitter(m)
                    out.append(
                        BeamSplitter(
                            m,
                            _parse_float(p1 or "1.5707963267948966", f"row {r + 1} theta"),
                            _parse_float(p2 or "0", f"row {r + 1} phase"),
                            base.phi_bl,
                            base.phi_tr,
                            base.phi_br,
                            base.convention,
                            label,
                        )
                    )
                elif kind == "phase shifter":
                    (m,) = _parse_ints(modes, f"row {r + 1} mode")
                    out.append(
                        PhaseShifter(m, _parse_float(p1 or "0", f"row {r + 1} phase"), label)
                    )
                elif kind == "permutation":
                    out.append(
                        Permutation(
                            _parse_ints(modes, f"row {r + 1} modes"),
                            _parse_ints(p1, f"row {r + 1} targets"),
                            label,
                        )
                    )
                elif kind == "loss":
                    (m,) = _parse_ints(modes, f"row {r + 1} mode")
                    out.append(
                        Loss(m, _parse_float(p1 or "0", f"row {r + 1} loss"), old.kind, label)
                    )
                else:
                    out.append(old)
            except (FockError, ValueError) as exc:
                if strict:
                    raise FockError(str(exc)) from exc
                out.append(old)
        return out

    def _snapshot(self) -> dict:
        snap: dict = {
            "n_modes": self.n_modes.value(),
            "brightness": self.brightness.value(),
            "indist": self.indist.value(),
            "shots": self.shots.value(),
            "seed": self.seed.value(),
            "backend": self.backend.currentText(),
            "max_photons": self.max_photons.value(),
        }
        for r in range(self.input_table.rowCount()):
            combo = self.input_table.cellWidget(r, 3)
            snap[r] = (
                self._text(self.input_table, r, 1),
                self._text(self.input_table, r, 2),
                combo.currentText() if combo is not None else "",
                self._text(self.input_table, r, 4),
                self._text(self.input_table, r, 5),
                self._text(self.input_table, r, 6),
                self._text(self.input_table, r, 7),
            )
        return snap

    def experiment(self) -> Experiment:
        """The editor's experiment: the loaded document with only the edited fields replaced.
        Fields the editor does not show (conditions, description, count settings, custom wire
        parameters, the seed's None) are kept from the document."""
        base = self._base if self._base is not None else self._default_experiment()
        shown, now = self._shown, self._snapshot()
        n = now["n_modes"]
        base_n = base.circuit.n_modes

        def same(key, cols=None) -> bool:
            if key not in shown or key not in now:
                return False
            if cols is None:
                return shown[key] == now[key]
            return all(shown[key][c] == now[key][c] for c in cols)

        circuit = Circuit(n)
        for e in self._collect_elements(strict=True):
            circuit.add(e)
        if n == base_n and all(same(r, (0, 1)) for r in range(n)):
            occ, tags = list(base.occupations), copy.deepcopy(base.tags)
        else:
            occ, tags = [], []
            for r in range(n):
                if r < base_n and same(r, (0, 1)):
                    k = base.occupations[r]
                    t = list(base.tags[r]) if base.tags and r < len(base.tags) else [0] * k
                else:
                    k = _parse_int(self._text(self.input_table, r, 1) or "0", f"mode {r} photons")
                    if k < 0:
                        raise FockError(f"mode {r}: negative photon number")
                    raw = self._text(self.input_table, r, 2)
                    t = [x.strip() for x in raw.split(",") if x.strip()]
                    if t and len(t) != k:
                        raise FockError(f"mode {r}: {k} photons but {len(t)} tags")
                    t = [int(x) if str(x).lstrip("-").isdigit() else x for x in t] or [0] * k
                occ.append(k)
                tags.append(t)
        det_cols = (2, 3, 4, 5, 6)
        if n == base_n and all(same(r, det_cols) for r in range(n)):
            detectors = copy.deepcopy(base.detectors)
        else:
            detectors = self._detectors_from_table(base, n, same)
        limits = base.limits
        if not same("max_photons"):
            limits = replace(limits, max_photons=self.max_photons.value())
        labels = base.mode_labels
        if labels is not None and len(labels) != n:
            labels = (list(labels) + [""] * n)[:n]
        return Experiment(
            circuit,
            occ,
            tags,
            base.brightness if same("brightness") else self.brightness.value(),
            base.indistinguishability if same("indist") else self.indist.value(),
            detectors,
            base.shots if same("shots") else self.shots.value(),
            base.seed if same("seed") else self.seed.value(),
            base.backend if same("backend") else self.backend.currentText(),
            copy.deepcopy(base.conditions),
            base.name,
            base.description,
            labels,
            limits,
            self._mixture,
            base.rate_hz,
            base.window_s,
            base.duration_s,
            base.acquisition,
        )

    def _detectors_from_table(self, base: Experiment, n: int, same) -> list[DetectorSpec]:
        by_mode = {m: spec.model for spec in base.detectors for m in spec.modes}
        singles: list[DetectorSpec] = []
        groups: dict[str, list[int]] = {}
        models: dict[int, object] = {}
        for r in range(n):
            old = by_mode.get(r) if r < base.circuit.n_modes else None
            if old is not None and same(r, (2, 3, 4, 6)):
                model = old
            else:
                combo = self.input_table.cellWidget(r, 3)
                choice = combo.currentText() if combo is not None else DETECTOR_CHOICES[0]
                eff = _parse_float(
                    self._text(self.input_table, r, 4) or "1", f"mode {r} efficiency"
                )
                dark = _parse_float(self._text(self.input_table, r, 7) or "0", f"mode {r} dark")
                if choice == "threshold":
                    model = Threshold(efficiency=eff, dark=dark)
                elif choice == "pseudo-PNR":
                    wires = _parse_int(self._text(self.input_table, r, 5) or "4", "wires")
                    if isinstance(old, PseudoPNR) and old.wires == wires:
                        model = replace(old, efficiency=eff, dark=dark)
                    else:
                        model = PseudoPNR(wires=wires, efficiency=eff, dark=dark)
                else:
                    model = IdealPNR(efficiency=eff, dark=dark)
            models[r] = model
            group = self._text(self.input_table, r, 6)
            if group:
                groups.setdefault(group, []).append(r)
            else:
                singles.append(DetectorSpec((r,), model))
        return singles + [DetectorSpec(tuple(m), models[m[0]]) for m in groups.values()]

    def editor_key(self) -> str | None:
        try:
            return json.dumps(self.experiment().to_dict(), sort_keys=True, default=str)
        except (SequoiaError, ValueError, TypeError):
            return None

    @property
    def editor_dirty(self) -> bool:
        return self.editor_key() != self._saved_key

    def _editor_changed(self, *_args) -> None:
        if self._building:
            return
        self._update_preserved()
        if self.result is not None:
            self._mark_editor_stale()

    def _editor_result_outdated(self) -> bool:
        return self.result is not None and self.editor_key() != self._result_key

    def _mark_editor_stale(self) -> None:
        if self._editor_result_outdated():
            self.success_label.setText(
                "outdated - the circuit or its inputs were edited since this run; run again"
            )

    def _update_preserved(self) -> None:
        base = self._base
        if base is None:
            self.preserved_label.setText("")
            return
        parts = []
        if base.conditions:
            parts.append("named conditions " + ", ".join(base.conditions))
        if base.description:
            parts.append("description")
        counts = [
            f"{k} {v:g}"
            for k, v in (
                ("rate_hz", base.rate_hz),
                ("window_s", base.window_s),
                ("duration_s", base.duration_s),
            )
            if v is not None
        ]
        if counts:
            parts.append("counts " + ", ".join(counts))
        for spec in base.detectors:
            m = spec.model
            if isinstance(m, PseudoPNR) and (m.weights or m.wire_efficiencies):
                parts.append(f"custom wire weights/efficiencies on modes {spec.modes}")
        if base.seed is None:
            parts.append("random seed (none set)")
        if base.limits.max_work != Limits().max_work:
            parts.append(f"max work {base.limits.max_work:g}")
        if base.limits.max_photons > self.max_photons.maximum():
            parts.append(f"max photons {base.limits.max_photons}")
        self.preserved_label.setText(
            "Kept from the document and not editable here: " + "; ".join(parts) + "."
            if parts
            else ""
        )

    def set_experiment(self, exp: Experiment) -> None:
        self._name = exp.name
        self._mode_labels = list(exp.mode_labels) if exp.mode_labels else None
        self._mixture = exp.input_mixture
        self._counts_keys = (exp.rate_hz, exp.window_s, exp.duration_s)
        self._building = True
        self.n_modes.setValue(exp.circuit.n_modes)
        self._resize_modes(exp.circuit.n_modes)
        self._building = True
        self._elements = list(exp.circuit.elements)
        self._fill_circuit_table()
        self._building = True
        by_mode = {}
        group_of: dict[int, str] = {}
        for g, spec in enumerate(exp.detectors):
            for m in spec.modes:
                by_mode[m] = spec.model
                if len(spec.modes) > 1:
                    group_of[m] = f"g{g}"
        for r in range(exp.circuit.n_modes):
            label = exp.mode_labels[r] if exp.mode_labels and r < len(exp.mode_labels) else ""
            self.input_table.item(r, 0).setText(f"{r} {label}".strip())
            self.input_table.item(r, 1).setText(str(exp.occupations[r]))
            tags = exp.tags[r] if exp.tags and r < len(exp.tags) else []
            show = [str(t) for t in tags]
            self.input_table.item(r, 2).setText(
                "" if all(s == "0" for s in show) else ",".join(show)
            )
            model = by_mode.get(r, IdealPNR())
            combo = self.input_table.cellWidget(r, 3)
            combo.blockSignals(True)
            if isinstance(model, Threshold):
                combo.setCurrentIndex(1)
            elif isinstance(model, PseudoPNR):
                combo.setCurrentIndex(2)
                self.input_table.item(r, 5).setText(str(model.wires))
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
            self.input_table.item(r, 4).setText(f"{model.efficiency:.10g}")
            self.input_table.item(r, 6).setText(group_of.get(r, ""))
            self.input_table.item(r, 7).setText(f"{model.dark:.10g}")
        self.brightness.setValue(exp.brightness)
        self.indist.setValue(exp.indistinguishability)
        self.shots.setValue(exp.shots)
        self.seed.setValue(int(exp.seed) if exp.seed is not None else 0)
        idx = self.backend.findText(exp.backend)
        self.backend.setCurrentIndex(max(idx, 0))
        self.max_photons.setValue(exp.limits.max_photons)
        self._base = copy.deepcopy(exp)
        self._shown = self._snapshot()
        self._building = False
        self._show_mixture()
        self._update_preserved()
        if self.result is not None:
            self._mark_editor_stale()

    def _show_mixture(self) -> None:
        mix = self._mixture
        self.mixture_label.setVisible(mix is not None)
        self.clear_mixture_button.setVisible(mix is not None)
        if mix is not None:
            self.mixture_label.setText(
                f"Input: statistical mixture of {len(mix.branches)} photon configurations "
                f"(total weight {mix.total_weight:.6g}) taken from the drawn setup. It is used "
                "instead of the occupation table, which is ignored while it is set. The mixture "
                "can only be inspected here, not edited."
            )

    def clear_mixture(self) -> None:
        self._mixture = None
        self._show_mixture()
        self._editor_changed()

    def _start(self, text: str) -> None:
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.import_setup_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText(text)

    def _setup_job(self, kind: str):
        setup, library = self.provider() if self.provider is not None else (None, None)
        if setup is None:
            self.status.setText("no setup is open")
            return None
        duration = self.duration.value()
        n_max = self.n_max_pairs.value() or None
        mode = "pnr" if self.detector_mode.currentIndex() == 1 else "setup"
        limits = Limits(max_photons=self.max_photons.value())
        shots, seed, backend = self.shots.value(), self.seed.value(), self.backend.currentText()

        def job(should_cancel, progress):
            progress(0, 0)
            payload = _Payload(kind)
            try:
                payload.components = list_components(setup, library)
            except SequoiaError as exc:
                payload.error = str(exc)
                return payload
            try:
                se = build_experiment_from_setup(
                    setup,
                    library,
                    n_max=n_max,
                    duration=duration,
                    detectors_mode=mode,
                    should_cancel=should_cancel,
                    limits=limits,
                )
                payload.se = se
                payload.inputs = se.inputs
                if kind == "run":
                    payload.result = run_setup_experiment(
                        se,
                        shots=shots,
                        seed=seed,
                        backend=backend,
                        should_cancel=should_cancel,
                        progress=progress,
                    )
                elif kind == "extract":
                    payload.extraction = extraction_experiment(se, should_cancel=should_cancel)
            except Cancelled:
                raise
            except (SequoiaError, ValueError) as exc:
                payload.error = str(exc)
                try:
                    payload.inputs = list_inputs(
                        setup, library, n_max=n_max, should_cancel=should_cancel, limits=limits
                    )
                except (SequoiaError, ValueError):
                    pass
            return payload

        return job

    def import_from_setup(self) -> None:
        if self.runner.busy:
            return
        job = self._setup_job("import")
        if job is None:
            return
        self.begin_run()
        self._start("reading the drawn circuit...")
        self.runner.start(job)

    def extract_to_editor(self) -> None:
        if self.runner.busy or not self._may_replace():
            return
        job = self._setup_job("extract")
        if job is None:
            return
        self.begin_run()
        self._start("extracting the circuit...")
        self.runner.start(job)

    def run(self) -> None:
        if self.runner.busy:
            return
        if self.setup_mode:
            job = self._setup_job("run")
            if job is None:
                return
            self.begin_run()
            self._start("running...")
            self.runner.start(job)
            return
        try:
            exp = self.experiment()
            exp.photons()
        except (SequoiaError, ValueError) as exc:
            self._end(f"invalid input: {exc}")
            return
        self._run_key = self.editor_key()

        def job(should_cancel, progress):
            return run_experiment(exp, should_cancel, progress)

        self.begin_run()
        self._start("running...")
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.import_setup_button.setEnabled(True)
        self.extract_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText(text)

    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "failed"
        if not self.run_is_current():
            self._end("")
            return
        self.message_label.setText(first)
        self._end(f"failed: {first}")

    def _done(self, result) -> None:
        if not self.run_is_current():
            self._end("")
            return
        if isinstance(result, _Payload):
            self._payload_done(result)
            return
        self.result = result
        self._result_key = self._run_key
        r = result.resources
        self._end(
            f"{r['output_states']} states in {r['seconds']:.3g} s ({result.backend}, "
            f"{r['permanents']} permanents)"
        )
        self._refresh_results()

    def _issues(self, texts: list[str]) -> None:
        if self.on_issues is not None:
            self.on_issues([Issue("warning", "photon_counts", "photon counts", t) for t in texts])

    def _payload_done(self, p: _Payload) -> None:
        outdated = self.accept_run()
        self.components = p.components
        self.inputs = p.inputs
        self._fill_components()
        self._fill_inputs()
        if p.se is not None:
            self.setup_exp = p.se
        warnings: list[str] = []
        if p.error:
            self.setup_exp = None
            self.setup_result = None
            self._clear_results()
            self.notes_view.setPlainText(p.error)
            self.state_summary.setText(p.error)
            self.result_tabs.setCurrentIndex(self.result_tabs.count() - 1)
            self._issues([p.error])
            self._stale_note = ""
            self._end(f"cannot use the drawn circuit: {p.error.splitlines()[0]}")
            return
        se = p.se
        if se.omitted_weight > 1e-4:
            warnings.append(
                f"input weight beyond the pair-number truncation is not simulated: "
                f"{se.omitted_weight:.3g}"
            )
        if p.kind == "run" and p.result is not None:
            self.setup_result = p.result
            self._fill_results()
            for d, load in p.result.outside_dilute_regime.items():
                warnings.append(
                    f"'{d}': potential click rate x dead time = {load:.3g}; the shown rates are "
                    "potential clicks before dead time, not recorded counts"
                )
            r = p.result.result.resources
            note = (
                f"{len(p.result.detected)} detector states in {r['seconds']:.3g} s "
                f"({p.result.result.backend}, {r['permanents']} permanents)"
            )
            if p.result.omitted_weight > 1e-4:
                note += f"; omitted input weight {p.result.omitted_weight:.3g}"
            self._stale_note = note
            self._issues(warnings)
            self._end(self.stale_text(note) if outdated else note)
            if outdated:
                self.state_summary.setText(self.stale_text(note))
        elif p.kind == "extract" and p.extraction is not None:
            exp, ex = p.extraction
            self.extraction = ex
            self.set_experiment(exp)
            self._saved_key = None
            self.mode_box.setCurrentIndex(1)
            self.editor_tabs.setCurrentIndex(0)
            note = (
                f"extracted {len(ex.circuit.elements)} elements on {ex.circuit.n_modes} modes "
                f"({'verified against the exact compilation' if ex.verified else 'one block'}, "
                f"max deviation {ex.max_error:.1e})"
            )
            if ex.notes:
                note += "; " + "; ".join(ex.notes)
            note += (
                "; the editor holds a copy: later edits to the canvas do not reach it, and edits "
                "here do not change the canvas"
            )
            if outdated:
                note = "the setup changed while extracting; extract again. " + note
            self._issues(warnings + list(ex.notes))
            self._end(note)
        else:
            self.notes_view.setPlainText("\n".join(se.notes))
            self._stale_note = f"{len(p.components)} components, {len(p.inputs)} input source(s)"
            self._issues(warnings)
            self._end(self.stale_text(self._stale_note) if outdated else self._stale_note)

    def _clear_results(self) -> None:
        for table in (
            self.states_table,
            self.clicks_table,
            self.detector_table,
            self.marginals_table,
            self.cond_result,
            self.samples_table,
        ):
            table.setRowCount(0)
        self.cond_table.setRowCount(0)

    def _fill_components(self) -> None:
        t = self.components_table
        t.setSortingEnabled(False)
        t.setRowCount(len(self.components))
        for r, row in enumerate(self.components):
            for c, text in enumerate(
                (row.id, row.type, row.category, row.part, row.summary, row.ports)
            ):
                t.setItem(r, c, _SortItem(text))
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()

    def _fill_inputs(self) -> None:
        t = self.inputs_table
        t.setSortingEnabled(False)
        t.setRowCount(len(self.inputs))
        for r, d in enumerate(self.inputs):
            cells = (
                (d.source, None),
                (d.kind, None),
                (d.photons, None),
                ("" if d.mean_pairs is None else _num(d.mean_pairs), d.mean_pairs),
                ("" if d.schmidt is None else _num(d.schmidt), d.schmidt),
                (f"{d.window_s:.6g} s", d.window_s),
                (_num(d.omitted), d.omitted),
                (d.text, None),
            )
            for c, (text, key) in enumerate(cells):
                t.setItem(r, c, _SortItem(text, key))
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()

    def _fill_results(self) -> None:
        res = self.setup_result
        if res is None:
            return
        se = res.setup_experiment
        ids = res.detector_ids
        cnt = res.counts
        window = se.window_s
        models = {
            d: spec.model for d, spec in zip(se.detector_ids, se.compiled.detectors, strict=True)
        }
        dt = self.detector_table
        dt.setSortingEnabled(False)
        dt.setRowCount(len(ids))
        for k, d in enumerate(ids):
            p_click = sum(p for s, p in res.detected.items() if s[k] > 0)
            mean = sum(s[k] * p for s, p in res.detected.items())
            rate = p_click * cnt.windows_per_second
            total = rate * cnt.duration_s
            sig = math.sqrt(count_variance(p_click, cnt.n_windows))
            model = models[d]
            row = (
                (d, None),
                (f"{model.kind}", None),
                (f"{model.efficiency:.4g}", model.efficiency),
                (f"{model.dark:.3g}", model.dark),
                (_num(p_click), p_click),
                (_num(mean), mean),
                (f"{rate:.6g}", rate),
                (f"{total:.6g}", total),
                (f"{sig:.4g}", sig),
            )
            for c, (text, key) in enumerate(row):
                dt.setItem(k, c, _SortItem(text, key))
        dt.setSortingEnabled(True)
        dt.resizeColumnsToContents()
        patterns: dict[str, float] = {}
        for s, p in res.detected.items():
            label = " + ".join(
                f"{d}" if n == 1 else f"{d} x{n}" for d, n in zip(ids, s, strict=True) if n > 0
            )
            patterns[label or "no click"] = patterns.get(label or "no click", 0.0) + p
        ct = self.clicks_table
        ct.setSortingEnabled(False)
        ct.setRowCount(len(patterns))
        for r, (label, p) in enumerate(sorted(patterns.items(), key=lambda kv: -kv[1])):
            rate = p * cnt.windows_per_second
            total = rate * cnt.duration_s
            sig = math.sqrt(count_variance(p, cnt.n_windows))
            row = (
                (label, float(r)),
                (_num(p), p),
                (f"{rate:.6g}", rate),
                (f"{total:.6g}", total),
                (f"{sig:.4g}", sig),
            )
            for c, (text, key) in enumerate(row):
                ct.setItem(r, c, _SortItem(text, key))
        ct.setSortingEnabled(True)
        ct.resizeColumnsToContents()
        rows: list[tuple[str, str, int, float, float | None]] = []
        for k, d in enumerate(ids):
            arrive: dict[int, float] = {}
            for s, p in res.photon_numbers.items():
                arrive[s[k]] = arrive.get(s[k], 0.0) + p
            seen: dict[int, float] = {}
            for s, p in res.detected.items():
                seen[s[k]] = seen.get(s[k], 0.0) + p
            for n in sorted(set(arrive) | set(seen)):
                rows.append(("detector", d, n, arrive.get(n, 0.0), seen.get(n, 0.0)))
        labels = se.output_labels
        for m, lab in enumerate(labels):
            dist: dict[int, float] = {}
            for s, p in res.per_mode.items():
                dist[s[m]] = dist.get(s[m], 0.0) + p
            for n in sorted(dist):
                rows.append(("mode", lab, n, dist[n], None))
        mt = self.marginals_table
        mt.setSortingEnabled(False)
        mt.setRowCount(len(rows))
        for r, (kind, name, n, pa, pd) in enumerate(rows):
            for c, (text, key) in enumerate(
                (
                    (kind, None),
                    (name, None),
                    (str(n), float(n)),
                    (_num(pa), pa),
                    ("" if pd is None else _num(pd), -1.0 if pd is None else pd),
                )
            ):
                mt.setItem(r, c, _SortItem(text, key))
        mt.setSortingEnabled(True)
        mt.resizeColumnsToContents()
        self._fill_conditions(ids)
        self._show_states()
        self._fill_conditional()
        self._fill_samples()
        self._fill_pair_plot()
        notes = list(se.notes)
        notes += [m for m in res.messages if m not in notes]
        if res.omitted_weight > 1e-12:
            notes.append(f"omitted input weight (not simulated): {res.omitted_weight:.4g}")
        notes.append(
            f"detection window {window:.6g} s, {cnt.windows_per_second:.6g} windows per second, "
            f"duration {cnt.duration_s:g} s"
        )
        self.notes_view.setPlainText("\n".join(f"- {n}" for n in notes))
        self.window_note = window
        self.model_card.setText(self._model_text(res))

    def _model_text(self, res) -> str:
        se = res.setup_experiment
        cnt = res.counts
        models = sorted({spec.model.kind for spec in se.compiled.detectors})
        text = (
            f"Measurement: one detection window = {se.window_s:.4g} s (pulse period), "
            f"{cnt.windows_per_second:.4g} windows/s, duration {cnt.duration_s:g} s "
            f"({cnt.n_windows:.4g} windows); detectors {', '.join(se.detector_ids)} "
            f"({', '.join(models)}, efficiency and dark clicks per window included). "
            "Model: photon-number circuit; loss, interference, multiplexing included; dead time, "
            "jitter, afterpulsing and spectral/temporal structure not included, so rates are "
            "potential clicks before dead time. Uncertainty: binomial over a fixed number of "
            f"windows. Normalization: resolved probability {res.mass:.6g}, omitted input weight "
            f"{res.omitted_weight:.3g}."
        )
        high = res.outside_dilute_regime
        if high:
            text = (
                "Acquisition regime not modelled: potential click rate x dead time = "
                + ", ".join(f"{d} {v:.2g}" for d, v in high.items())
                + " (dilute regime < 0.1); recorded lab counts will be lower. "
                + text
            )
        return text

    def _fill_conditions(self, ids: list[str]) -> None:
        old = {}
        for r in range(self.cond_table.rowCount()):
            name = self.cond_table.item(r, 0).text()
            op = self.cond_table.cellWidget(r, 1)
            val = self.cond_table.cellWidget(r, 2)
            old[name] = (op.currentIndex(), val.value())
        self.cond_table.setRowCount(len(ids))
        for r, d in enumerate(ids):
            item = QTableWidgetItem(d)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.cond_table.setItem(r, 0, item)
            op = QComboBox()
            op.addItems(["(any)", "=", ">=", ">", "<=", "<", "!="])
            val = QSpinBox()
            val.setRange(0, 99)
            val.setValue(1)
            if d in old:
                op.setCurrentIndex(old[d][0])
                val.setValue(old[d][1])
            op.currentIndexChanged.connect(self._fill_conditional)
            val.valueChanged.connect(self._fill_conditional)
            self.cond_table.setCellWidget(r, 1, op)
            self.cond_table.setCellWidget(r, 2, val)

    def _cond_spec(self) -> dict[int, tuple[str, int]]:
        spec = {}
        for r in range(self.cond_table.rowCount()):
            op = self.cond_table.cellWidget(r, 1)
            val = self.cond_table.cellWidget(r, 2)
            if op is not None and op.currentIndex() > 0:
                spec[r] = (op.currentText(), val.value())
        return spec

    def _fill_conditional(self, *_args) -> None:
        res = self.setup_result
        if res is None:
            return
        spec = self._cond_spec()
        cnt = res.counts
        if not spec:
            self.cond_summary.setText("choose a condition on at least one detector")
            self.cond_result.setRowCount(0)
            return
        cond = condition(res.detected, spec)
        rate = cond.success_probability * cnt.windows_per_second
        self.cond_summary.setText(
            f"success probability {cond.success_probability:.6g} per window "
            f"({rate:.6g} per second, {rate * cnt.duration_s:.6g} in {cnt.duration_s:g} s); "
            f"{len(cond.joint)} detector states kept, distribution normalised"
        )
        t = self.cond_result
        t.setSortingEnabled(False)
        rows = sorted(cond.joint.items(), key=lambda kv: -kv[1])
        t.setRowCount(len(rows))
        for r, (s, p) in enumerate(rows):
            q = cond.probs.get(s, 0.0)
            vals = (
                (format_state(s), float(r)),
                (self._pattern(s, res.detector_ids), None),
                (_num(q), q),
                (_num(p), p),
                (f"{p * cnt.windows_per_second:.6g}", p * cnt.windows_per_second),
            )
            for c, (text, key) in enumerate(vals):
                t.setItem(r, c, _SortItem(text, key))
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()

    @staticmethod
    def _pattern(state, ids) -> str:
        return " ".join(f"{d}={n}" for d, n in zip(ids, state, strict=True))

    def _fill_samples(self) -> None:
        res = self.setup_result
        if res is None:
            return
        t = self.samples_table
        t.setSortingEnabled(False)
        if res.samples is None:
            t.setRowCount(0)
            self.samples_note.setText(
                "set shots above 0 to draw seeded samples of the detector states"
            )
            return
        sm = res.samples
        note = f"{sm.shots} shots, seed {sm.seed}"
        if sm.unresolved_probability:
            note += (
                f"; {sm.unresolved} shots fell in the unresolved input weight "
                f"({sm.unresolved_probability:.3g}, truncated pair numbers) and are not listed"
            )
        self.samples_note.setText(note)
        rows = sorted(sm.counts.items(), key=lambda kv: -kv[1])
        t.setRowCount(len(rows))
        for r, (s, c) in enumerate(rows):
            vals = (
                (format_state(s), float(r)),
                (self._pattern(s, res.detector_ids), None),
                (str(c), float(c)),
                (_num(c / sm.shots), c / sm.shots),
                (_num(res.detected.get(s, 0.0)), res.detected.get(s, 0.0)),
            )
            for k, (text, key) in enumerate(vals):
                t.setItem(r, k, _SortItem(text, key))
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()

    def _show_states(self, *_args) -> None:
        res = self.setup_result
        if res is None:
            return
        ids = res.detector_ids
        cnt = res.counts
        text = self.state_filter.text()
        cond = None
        try:
            spec = parse_filter(text, ids)
        except FockError as exc:
            spec = {}
            self.state_summary.setText(str(exc))
        else:
            if spec:
                cond = condition(res.detected, spec)
                self.state_summary.setText(
                    f"success probability {cond.success_probability:.6g} "
                    f"({len(cond.joint)} states kept, conditional distribution normalised)"
                )
            else:
                pn = ", ".join(
                    f"{n}: {v:.4g}"
                    for n, v in sorted(_by_photon_number(res.photon_numbers).items())
                )
                self.state_summary.setText(
                    f"total probability {res.mass:.12g}; photons arriving at the detectors: {pn}"
                )
        rows = sorted(res.detected.items(), key=lambda kv: -kv[1])
        t = self.states_table
        t.setSortingEnabled(False)
        t.setRowCount(len(rows))
        for r, (s, p) in enumerate(rows):
            rate = p * cnt.windows_per_second
            total = rate * cnt.duration_s
            cp = cond.probs.get(s) if cond is not None else None
            sig = math.sqrt(count_variance(p, cnt.n_windows))
            vals = (
                (format_state(s), float(r)),
                (self._pattern(s, ids), None),
                (_num(p), p),
                (f"{rate:.6g}", rate),
                (f"{total:.6g}", total),
                (f"{sig:.4g}", sig),
                ("" if cp is None else _num(cp), -1.0 if cp is None else cp),
            )
            for c, (text_, key) in enumerate(vals):
                t.setItem(r, c, _SortItem(text_, key))
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()
        if self.setup_normalized.isChecked() and cond is not None and cond.probs:
            shown = sorted(cond.probs.items(), key=lambda kv: -kv[1])
        else:
            shown = rows
        self.setup_bars = self._draw_bars(self.setup_plot, self.setup_bars, shown, ids)

    def _draw_bars(self, plot, bars, shown, ids):
        shown = shown[:BAR_LIMIT]
        if bars is not None:
            plot.removeItem(bars)
            bars = None
        if not shown:
            return None
        heights = np.array([p for _, p in shown])
        brush = QApplication.palette().highlight().color()
        bars = pg.BarGraphItem(x=np.arange(len(shown)), height=heights, width=0.8, brush=brush)
        plot.addItem(bars)
        axis = plot.getPlotItem().getAxis("bottom")
        step = max(1, math.ceil(len(shown) / 12))
        axis.setTicks([[(i, format_state(shown[i][0])) for i in range(0, len(shown), step)]])
        return bars

    def _fill_pair_plot(self) -> None:
        res = self.setup_result
        if res is None:
            return
        pn = sorted(_by_photon_number(res.photon_numbers).items())
        if self.pair_bars is not None:
            self.pair_plot.removeItem(self.pair_bars)
            self.pair_bars = None
        if not pn:
            return
        brush = QApplication.palette().highlight().color()
        self.pair_bars = pg.BarGraphItem(
            x=np.array([n for n, _ in pn], dtype=float),
            height=np.array([p for _, p in pn]),
            width=0.8,
            brush=brush,
        )
        self.pair_plot.addItem(self.pair_bars)

    def _refresh_results(self, *_args) -> None:
        res = self.result
        if res is None:
            return
        cond = None
        text = self.filter_edit.text()
        try:
            spec = parse_filter(text)
        except FockError as exc:
            spec = {}
            self.success_label.setText(str(exc))
        msgs = list(res.messages)
        if spec:
            cond = condition(res.probabilities, spec)
            self.success_label.setText(
                f"success probability {cond.success_probability:.6g} "
                f"({len(cond.joint)} states kept, conditional distribution normalized)"
            )
        elif not text.strip():
            self.success_label.setText(f"total probability {res.mass:.12g}")
        raw = res.raw
        pn = raw.by_photon_number()
        if len(pn) > 1:
            msgs.append("photon-number mass: " + ", ".join(f"{k}: {v:.4g}" for k, v in pn.items()))
        self.message_label.setText("; ".join(msgs))
        counts = res.samples.counts if res.samples else {}
        rows = sorted(res.probabilities.items(), key=lambda kv: -kv[1])
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(len(rows))
        for i, (s, p) in enumerate(rows):
            self.results_table.setItem(i, 0, _SortItem(format_state(s), float(i)))
            self.results_table.setItem(i, 1, _SortItem(f"{p:.6g}", p))
            c = counts.get(s)
            self.results_table.setItem(
                i, 2, _SortItem("" if res.samples is None else str(c or 0), float(c or 0))
            )
            cp = cond.probs.get(s) if cond is not None else None
            self.results_table.setItem(
                i, 3, _SortItem("" if cp is None else f"{cp:.6g}", -1.0 if cp is None else cp)
            )
        self.results_table.setSortingEnabled(True)
        self.results_table.resizeColumnsToContents()
        if self.normalized.isChecked() and cond is not None and cond.probs:
            shown = sorted(cond.probs.items(), key=lambda kv: -kv[1])
        else:
            shown = rows
        self.bars = self._draw_bars(self.plot, self.bars, shown, None)
        self._mark_editor_stale()

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open circuit experiment", "", "Circuit experiment (*.yaml *.yml *.json)"
        )
        if path:
            self.open_path(path)

    def save_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save circuit experiment", "", "Circuit experiment (*.yaml *.yml *.json)"
        )
        if path:
            self.save_path(path)

    def import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Perceval GUI circuit", "", "Perceval GUI circuit (*.json)"
        )
        if path:
            self.import_path(path)

    def _problem(self, title: str, message: str) -> None:
        self.status.setText(f"{title}: {message.splitlines()[0] if message else ''}")
        QMessageBox.warning(self, title, message)

    def _may_replace(self) -> bool:
        """Ask before an edited, unsaved circuit is replaced: save, discard or cancel."""
        if not self.editor_dirty:
            return True
        if self.confirm_replace is not None:
            answer = self.confirm_replace(self)
        else:
            box = QMessageBox.question(
                self,
                "Unsaved circuit",
                "The circuit in the editor has unsaved changes. Save them first?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            answer = {
                QMessageBox.StandardButton.Save: "save",
                QMessageBox.StandardButton.Discard: "discard",
            }.get(box, "cancel")
        if answer == "save":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save circuit experiment", "", "Circuit experiment (*.yaml *.yml *.json)"
            )
            return bool(path) and self.save_path(path)
        return answer == "discard"

    def open_path(self, path: str | Path) -> bool:
        if not self._may_replace():
            return False
        try:
            self.set_experiment(load_experiment(path))
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot open circuit", str(exc))
            return False
        self._saved_key = self.editor_key()
        self.status.setText(f"opened {Path(path).name}")
        return True

    def save_path(self, path: str | Path) -> bool:
        try:
            save_experiment(self.experiment(), path)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot save circuit", str(exc))
            return False
        self._saved_key = self.editor_key()
        self.status.setText(f"saved {Path(path).name}")
        return True

    def import_path(self, path: str | Path) -> bool:
        from sequoia.fock.import_perceval_gui import import_perceval_gui

        if not self._may_replace():
            return False
        try:
            imp = import_perceval_gui(path, strict=False)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot import circuit", str(exc))
            return False
        n = imp.circuit.n_modes
        imp.experiment.occupations = [1 if i < min(2, n) else 0 for i in range(n)]
        self.set_experiment(imp.experiment)
        self._saved_key = None
        note = f"imported {Path(path).name}: {n} modes, {imp.n_components} components"
        if imp.warnings:
            note += f"; {len(imp.warnings)} warning(s): " + "; ".join(imp.warnings)
        self.status.setText(note)
        return True


def _by_photon_number(dist: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for s, p in dist.items():
        out[sum(s)] = out.get(sum(s), 0.0) + p
    return out
