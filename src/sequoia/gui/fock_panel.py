import math
import re
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sequoia.errors import SequoiaError
from sequoia.fock.analysis import condition, format_state
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
from sequoia.gui.jobs import JobRunner

ELEMENT_TYPES = ("beam splitter", "phase shifter", "permutation", "loss", "block")
DETECTOR_CHOICES = ("ideal PNR", "threshold", "pseudo-PNR")
BAR_LIMIT = 40
_COND = re.compile(r"^\s*(\d+)\s*(==|=|!=|>=|<=|>|<)\s*(\d+)\s*$")


class _SortItem(QTableWidgetItem):
    def __init__(self, text: str, key: float | None = None) -> None:
        super().__init__(text)
        self._key = key
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other) -> bool:
        if isinstance(other, _SortItem) and self._key is not None and other._key is not None:
            return self._key < other._key
        return super().__lt__(other)


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


def _parse_float(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise FockError(f"{what}: {text!r} is not a number") from None


def parse_filter(text: str) -> dict[int, tuple[str, int]]:
    """'0=1, 1>=1' -> {0: ('=', 1), 1: ('>=', 1)} over the detected-state entries."""
    out: dict[int, tuple[str, int]] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        m = _COND.match(part)
        if m is None:
            raise FockError(f"cannot read condition {part.strip()!r}; use e.g. 0=1, 1>=1")
        out[int(m.group(1))] = (m.group(2), int(m.group(3)))
    return out


class FockTab(QWidget):
    """Photon-number (Fock) circuit experiment: circuit, input, detectors, distribution."""

    def __init__(self, provider=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.provider = provider
        self.runner = JobRunner(self)
        self.result: FockResult | None = None
        self._elements: list = []
        self._dirty: set[int] = set()
        self._building = False

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
        self.n_modes.setRange(1, 24)
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

        self.circuit_table = QTableWidget(0, 4)
        self.circuit_table.setHorizontalHeaderLabels(
            ["Element", "Modes", "Parameter 1", "Parameter 2"]
        )
        self.circuit_table.horizontalHeader().setStretchLastSection(True)
        self.circuit_table.verticalHeader().setVisible(False)
        self.circuit_table.setToolTip(
            "beam splitter: modes 'a,b'; theta (rad), phase on first input (rad)\n"
            "phase shifter: mode; phase (rad)\n"
            "permutation: modes 'a,b,c'; target positions '1,2,0'\n"
            "loss: mode; loss probability (photon-loss channel)"
        )
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.up_button = QPushButton("Up")
        self.down_button = QPushButton("Down")
        self.new_type = QComboBox()
        self.new_type.addItems(ELEMENT_TYPES[:4])

        self.input_table = QTableWidget(0, 6)
        self.input_table.setHorizontalHeaderLabels(
            ["Mode", "Photons", "Tags", "Detector", "Efficiency", "Wires"]
        )
        self.input_table.horizontalHeader().setStretchLastSection(True)
        self.input_table.verticalHeader().setVisible(False)
        self.input_table.setToolTip(
            "Tags: one label per photon, e.g. '0,1'. Equal tags interfere, different tags do not.\n"
            "Detector efficiency is applied once, in the detector response."
        )

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
        self._style_plot()
        self.bars: pg.BarGraphItem | None = None
        self.normalized = QCheckBox("show conditional distribution")
        self.normalized.setChecked(False)

        top = QHBoxLayout()
        top.setContentsMargins(6, 4, 6, 4)
        for w in (
            self.open_button,
            self.save_button,
            self.import_button,
            self.backend,
            self.shots,
            self.seed,
            self.run_button,
            self.cancel_button,
            self.progress,
        ):
            top.addWidget(w)
        top.addWidget(self.status, 1)

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
        il.addWidget(self.input_table, 1)
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

        split = QSplitter()
        split.addWidget(self.editor_tabs)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(split, 1)

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
        self.filter_edit.textChanged.connect(self._refresh_results)
        self.normalized.toggled.connect(self._refresh_results)
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        self._resize_modes(2)
        self.set_experiment(self._default_experiment())

    def _style_plot(self) -> None:
        pal = QApplication.palette()
        self.plot.setBackground(pal.base().color())
        item = self.plot.getPlotItem()
        fg = pal.text().color()
        for axis in ("left", "bottom"):
            item.getAxis(axis).setPen(fg)
            item.getAxis(axis).setTextPen(fg)
        item.setLabel("bottom", "output state (rank)")
        item.setLabel("left", "probability")
        item.showGrid(x=False, y=True, alpha=0.25)

    @staticmethod
    def _default_experiment() -> Experiment:
        c = Circuit(2).beam_splitter(0, 1)
        return Experiment(c, [1, 1], name="HOM beam splitter")

    def set_outcome(self, _outcome) -> None:
        pass

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
        self.input_table.setCellWidget(r, 3, combo)
        self.input_table.setItem(r, 4, QTableWidgetItem("1"))
        self.input_table.setItem(r, 5, QTableWidgetItem("4"))

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if not self._building:
            self._dirty.add(item.row())

    def _row_texts(self, e) -> tuple[str, str, str, str]:
        if isinstance(e, BeamSplitter):
            return (
                "beam splitter",
                ",".join(map(str, e.modes)),
                f"{e.theta:.10g}",
                f"{e.phi_tl:.10g}",
            )
        if isinstance(e, PhaseShifter):
            return "phase shifter", str(e.mode), f"{e.phase:.10g}", ""
        if isinstance(e, Permutation):
            return (
                "permutation",
                ",".join(map(str, e.modes)),
                ",".join(map(str, e.perm)),
                "",
            )
        if isinstance(e, Loss):
            return "loss", str(e.mode), f"{e.loss:.10g}", e.kind
        return "block", ",".join(map(str, e.modes)), f"{len(e.modes)}x{len(e.modes)} matrix", ""

    def _fill_circuit_table(self) -> None:
        self._building = True
        self.circuit_table.setRowCount(len(self._elements))
        for r, e in enumerate(self._elements):
            for c, text in enumerate(self._row_texts(e)):
                item = QTableWidgetItem(text)
                if isinstance(e, Block) or c == 0:
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

    def remove_element(self) -> None:
        row = self.circuit_table.currentRow()
        if row < 0:
            return
        self._elements = self._collect_elements(strict=False)
        del self._elements[row]
        self._fill_circuit_table()

    def move_element(self, step: int) -> None:
        row = self.circuit_table.currentRow()
        new = row + step
        if row < 0 or not 0 <= new < len(self._elements):
            return
        self._elements = self._collect_elements(strict=False)
        self._elements[row], self._elements[new] = self._elements[new], self._elements[row]
        self._fill_circuit_table()
        self.circuit_table.selectRow(new)

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
                        )
                    )
                elif kind == "phase shifter":
                    (m,) = _parse_ints(modes, f"row {r + 1} mode")
                    out.append(PhaseShifter(m, _parse_float(p1 or "0", f"row {r + 1} phase")))
                elif kind == "permutation":
                    out.append(
                        Permutation(
                            _parse_ints(modes, f"row {r + 1} modes"),
                            _parse_ints(p1, f"row {r + 1} targets"),
                        )
                    )
                elif kind == "loss":
                    (m,) = _parse_ints(modes, f"row {r + 1} mode")
                    out.append(Loss(m, _parse_float(p1 or "0", f"row {r + 1} loss"), old.kind))
                else:
                    out.append(old)
            except (FockError, ValueError) as exc:
                if strict:
                    raise FockError(str(exc)) from exc
                out.append(old)
        return out

    def experiment(self) -> Experiment:
        n = self.n_modes.value()
        circuit = Circuit(n)
        for e in self._collect_elements(strict=True):
            circuit.add(e)
        occ, tags, detectors = [], [], []
        for r in range(n):
            k = int(_parse_float(self._text(self.input_table, r, 1) or "0", f"mode {r} photons"))
            raw = self._text(self.input_table, r, 2)
            t: list = [x.strip() for x in raw.split(",") if x.strip()]
            if t and len(t) != k:
                raise FockError(f"mode {r}: {k} photons but {len(t)} tags")
            if not t:
                t = [0] * k
            occ.append(k)
            tags.append([int(x) if str(x).lstrip("-").isdigit() else x for x in t])
            combo = self.input_table.cellWidget(r, 3)
            choice = combo.currentText() if combo is not None else DETECTOR_CHOICES[0]
            eff = _parse_float(self._text(self.input_table, r, 4) or "1", f"mode {r} efficiency")
            if choice == "threshold":
                model = Threshold(efficiency=eff)
            elif choice == "pseudo-PNR":
                wires = int(_parse_float(self._text(self.input_table, r, 5) or "4", "wires"))
                model = PseudoPNR(wires=wires, efficiency=eff)
            else:
                model = IdealPNR(efficiency=eff)
            detectors.append(DetectorSpec((r,), model))
        return Experiment(
            circuit,
            occ,
            tags,
            self.brightness.value(),
            self.indist.value(),
            detectors,
            self.shots.value(),
            self.seed.value(),
            self.backend.currentText(),
            name=getattr(self, "_name", "photonic circuit experiment"),
            limits=Limits(max_photons=self.max_photons.value()),
        )

    def set_experiment(self, exp: Experiment) -> None:
        self._name = exp.name
        self._building = True
        self.n_modes.setValue(exp.circuit.n_modes)
        self._resize_modes(exp.circuit.n_modes)
        self._building = True
        self._elements = list(exp.circuit.elements)
        self._fill_circuit_table()
        by_mode = {}
        for spec in exp.detectors:
            if len(spec.modes) == 1:
                by_mode[spec.modes[0]] = spec.model
        for r in range(exp.circuit.n_modes):
            self.input_table.item(r, 1).setText(str(exp.occupations[r]))
            tags = exp.tags[r] if exp.tags and r < len(exp.tags) else []
            show = [str(t) for t in tags]
            self.input_table.item(r, 2).setText(
                "" if all(s == "0" for s in show) else ",".join(show)
            )
            model = by_mode.get(r, IdealPNR())
            combo = self.input_table.cellWidget(r, 3)
            if isinstance(model, Threshold):
                combo.setCurrentIndex(1)
            elif isinstance(model, PseudoPNR):
                combo.setCurrentIndex(2)
                self.input_table.item(r, 5).setText(str(model.wires))
            else:
                combo.setCurrentIndex(0)
            self.input_table.item(r, 4).setText(f"{model.efficiency:.10g}")
        self.brightness.setValue(exp.brightness)
        self.indist.setValue(exp.indistinguishability)
        self.shots.setValue(exp.shots)
        self.seed.setValue(int(exp.seed) if exp.seed is not None else 0)
        idx = self.backend.findText(exp.backend)
        self.backend.setCurrentIndex(max(idx, 0))
        self._building = False

    def run(self) -> None:
        if self.runner.busy:
            return
        try:
            exp = self.experiment()
            exp.photons()
        except (SequoiaError, ValueError) as exc:
            self._end(f"invalid input: {exc}")
            return

        def job(should_cancel, progress):
            return run_experiment(exp, should_cancel, progress)

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

    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "failed"
        self.message_label.setText(first)
        self._end(f"failed: {first}")

    def _done(self, result: FockResult) -> None:
        self.result = result
        r = result.resources
        self._end(
            f"{r['output_states']} states in {r['seconds']:.3g} s ({result.backend}, "
            f"{r['permanents']} permanents)"
        )
        self._refresh_results()

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
        shown = shown[:BAR_LIMIT]
        if self.bars is not None:
            self.plot.removeItem(self.bars)
            self.bars = None
        if shown:
            heights = np.array([p for _, p in shown])
            brush = QApplication.palette().highlight().color()
            self.bars = pg.BarGraphItem(
                x=np.arange(len(shown)), height=heights, width=0.8, brush=brush
            )
            self.plot.addItem(self.bars)
            axis = self.plot.getPlotItem().getAxis("bottom")
            step = max(1, math.ceil(len(shown) / 12))
            axis.setTicks([[(i, format_state(shown[i][0])) for i in range(0, len(shown), step)]])

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

    def open_path(self, path: str | Path) -> bool:
        try:
            self.set_experiment(load_experiment(path))
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot open circuit", str(exc))
            return False
        self.status.setText(f"opened {Path(path).name}")
        return True

    def save_path(self, path: str | Path) -> bool:
        try:
            save_experiment(self.experiment(), path)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot save circuit", str(exc))
            return False
        self.status.setText(f"saved {Path(path).name}")
        return True

    def import_path(self, path: str | Path) -> bool:
        from sequoia.fock.import_perceval_gui import import_perceval_gui

        try:
            imp = import_perceval_gui(path, strict=False)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot import circuit", str(exc))
            return False
        n = imp.circuit.n_modes
        imp.experiment.occupations = [1 if i < min(2, n) else 0 for i in range(n)]
        self.set_experiment(imp.experiment)
        note = f"imported {Path(path).name}: {n} modes, {imp.n_components} components"
        if imp.warnings:
            note += f"; {len(imp.warnings)} warning(s): " + "; ".join(imp.warnings)
        self.status.setText(note)
        return True
