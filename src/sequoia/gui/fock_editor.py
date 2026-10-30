import copy
import json
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sequoia.errors import SequoiaError
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
from sequoia.fock.experiment import DetectorSpec, Experiment
from sequoia.gui.fock_common import IntDelegate, parse_float, parse_int, parse_ints

ELEMENT_TYPES = ("beam splitter", "phase shifter", "permutation", "loss", "block")
DETECTOR_CHOICES = ("ideal PNR", "threshold", "pseudo-PNR")


def default_experiment() -> Experiment:
    c = Circuit(2).beam_splitter(0, 1)
    return Experiment(c, [1, 1], name="HOM beam splitter")


class CircuitEditor(QWidget):
    """Circuit editor: edits a retained Experiment document. ``experiment()`` returns the loaded
    document with only the fields edited here replaced. ``shots``, ``seed`` and ``backend`` are
    the run controls of the owning tab, which belong to the document too."""

    changed = Signal()

    def __init__(self, shots: QSpinBox, seed: QSpinBox, backend: QComboBox, parent=None) -> None:
        super().__init__(parent)
        self.shots, self.seed, self.backend = shots, seed, backend
        self._mixture = None
        self._counts_keys: tuple = (None, None, None)
        self._elements: list = []
        self._dirty: set[int] = set()
        self._building = False
        self._base: Experiment | None = None
        self._shown: dict = {}

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
            "Photons and Wires are whole numbers.\n"
            "Tags: one label per photon, e.g. '0,1'. Equal tags interfere, different tags do not.\n"
            "Detector efficiency is applied once, in the detector response.\n"
            "Group: modes with the same group name are read by one detector (summed photons).\n"
            "Dark / window: probability of one dark click per detection window."
        )
        self._photons_delegate = IntDelegate(0, 1000, self.input_table)
        self._wires_delegate = IntDelegate(1, 1000, self.input_table)
        self.input_table.setItemDelegateForColumn(1, self._photons_delegate)
        self.input_table.setItemDelegateForColumn(5, self._wires_delegate)
        self.preserved_label = QLabel("")
        self.preserved_label.setWordWrap(True)
        self.preserved_label.setStyleSheet("color: palette(placeholder-text);")
        self.mixture_label = QLabel("")
        self.mixture_label.setWordWrap(True)
        self.mixture_label.setVisible(False)
        self.clear_mixture_button = QPushButton("Use the occupation table instead")
        self.clear_mixture_button.setVisible(False)

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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor_tabs)

        self.n_modes.valueChanged.connect(self._resize_modes)
        self.add_button.clicked.connect(self.add_element)
        self.remove_button.clicked.connect(self.remove_element)
        self.up_button.clicked.connect(lambda: self.move_element(-1))
        self.down_button.clicked.connect(lambda: self.move_element(1))
        self.circuit_table.itemChanged.connect(self._item_changed)
        self.clear_mixture_button.clicked.connect(self.clear_mixture)
        for w in (self.n_modes, self.shots, self.seed, self.max_photons):
            w.valueChanged.connect(self._editor_changed)
        for w in (self.brightness, self.indist):
            w.valueChanged.connect(self._editor_changed)
        self.backend.currentIndexChanged.connect(self._editor_changed)
        self.circuit_table.itemChanged.connect(self._editor_changed)
        self.input_table.itemChanged.connect(self._editor_changed)
        self._resize_modes(2)

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
                    m = parse_ints(modes, f"row {r + 1} modes")
                    if len(m) != 2:
                        raise FockError(f"row {r + 1}: a beam splitter needs two modes")
                    base = old if isinstance(old, BeamSplitter) else BeamSplitter(m)
                    out.append(
                        BeamSplitter(
                            m,
                            parse_float(p1 or "1.5707963267948966", f"row {r + 1} theta"),
                            parse_float(p2 or "0", f"row {r + 1} phase"),
                            base.phi_bl,
                            base.phi_tr,
                            base.phi_br,
                            base.convention,
                            label,
                        )
                    )
                elif kind == "phase shifter":
                    (m,) = parse_ints(modes, f"row {r + 1} mode")
                    out.append(PhaseShifter(m, parse_float(p1 or "0", f"row {r + 1} phase"), label))
                elif kind == "permutation":
                    out.append(
                        Permutation(
                            parse_ints(modes, f"row {r + 1} modes"),
                            parse_ints(p1, f"row {r + 1} targets"),
                            label,
                        )
                    )
                elif kind == "loss":
                    (m,) = parse_ints(modes, f"row {r + 1} mode")
                    out.append(
                        Loss(m, parse_float(p1 or "0", f"row {r + 1} loss"), old.kind, label)
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
        base = self._base if self._base is not None else default_experiment()
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
                    k = parse_int(self._text(self.input_table, r, 1) or "0", f"mode {r} photons")
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
                eff = parse_float(self._text(self.input_table, r, 4) or "1", f"mode {r} efficiency")
                dark = parse_float(self._text(self.input_table, r, 7) or "0", f"mode {r} dark")
                if choice == "threshold":
                    model = Threshold(efficiency=eff, dark=dark)
                elif choice == "pseudo-PNR":
                    wires = parse_int(self._text(self.input_table, r, 5) or "4", "wires")
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

    def _editor_changed(self, *_args) -> None:
        if self._building:
            return
        self._update_preserved()
        self.changed.emit()

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
