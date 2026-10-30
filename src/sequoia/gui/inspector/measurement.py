import html
import re
from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sequoia.gui.document import SetupDocument
from sequoia.gui.inspector.fields import (
    MISSING,
    Editor,
    FieldRow,
    FieldSpec,
    analyze,
    error_message,
    make_editor,
)
from sequoia.measurement import HeraldCondition, Measurement, PulseClock, measurement_card

_FIELDS = (
    "purpose",
    "coincidence_window",
    "acquisition_duration",
    "repeats",
    "accidentals",
    "dead_time",
)
_LATER = ("channels", "centers")
LABELS = {
    "purpose": "Purpose",
    "coincidence_window": "Coincidence window (full width)",
    "acquisition_duration": "Acquisition duration per setting",
    "repeats": "Repeats",
    "accidentals": "Accidental estimator",
    "dead_time": "Dead-time model",
    "pulse_clock": "Pulse clock (tag analysis)",
    "channels": "Detector -> tagger channel",
    "centers": "Centres per analysis",
    "conditions": "Heralding conditions",
}
HINTS = {
    "purpose": "What this measurement is for; shown on the measurement card and in run records.",
    "coincidence_window": "Full width of the coincidence window, the default for every "
    "coincidence and heralded-g2 analysis that sets none.",
    "acquisition_duration": "Acquisition time per setting (Monte Carlo, CHSH and plans use it).",
    "repeats": "How often each setting is acquired.",
    "accidentals": "auto: sidebands one pulse period away with a common pulse clock, else the "
    "singles product. sideband: also off-peak windows for CW light. singles: S_a S_b x window.",
    "dead_time": "non-paralyzable: the engines' dead-time model; none: dead times set to zero.",
    "channels": "Detector id -> tagger channel (e.g. spad_s -> tagger.ch1); overrides the cable. "
    "Analyses and conditions may then name the detector.",
    "centers": "Fixed delay centre per analysis name (heralded g2: name.a, name.b), e.g. 0.5 ns; "
    "analyses with an explicit centre keep it.",
    "pulse_clock": "The pulse clock a time-tag analysis assumes: from the sources clocking each "
    "channel, continuous (CW), one pair source for every channel, or an explicit rate.",
    "conditions": "Herald channels must click, veto channels must not (channels or detector "
    "ids, separated by commas). Photon counts lists each as a conditional view; an analysis "
    "with 'condition: name' counts only events passing it.",
}
_CLOCKS = (
    ("from the sources", "sources"),
    ("continuous (CW)", "cw"),
    ("one source for every channel", "source"),
    ("explicit rate", "rate"),
)
_SPLIT = re.compile(r"[,\s]+")


def _refs(text: str) -> list[str]:
    return [r for r in _SPLIT.split(text.strip()) if r]


class PulseClockEditor(Editor):
    def __init__(self, sources: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = QComboBox(self)
        for text, kind in _CLOCKS:
            self.kind.addItem(text, kind)
        self.source = QComboBox(self)
        self.source.addItems(sources)
        self.rate = QLineEdit(self)
        self.rate.setPlaceholderText("e.g. 80 MHz")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.kind)
        layout.addWidget(self.source)
        layout.addWidget(self.rate)
        self._loading = False
        self._shown: dict = {"kind": "sources"}
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.source.activated.connect(lambda _i: self.commit())
        self.rate.editingFinished.connect(self.commit)
        self._update_visibility()

    def _update_visibility(self) -> None:
        kind = self.kind.currentData()
        self.source.setVisible(kind == "source")
        self.rate.setVisible(kind == "rate")

    def _kind_changed(self) -> None:
        self._update_visibility()
        if not self._loading:
            self.commit()

    def collect(self) -> tuple[dict | None, str]:
        kind = self.kind.currentData()
        out: dict[str, Any] = {"kind": kind}
        if kind == "source":
            if not self.source.currentText():
                return None, "the setup has no instance to take the clock from"
            out["source"] = self.source.currentText()
        elif kind == "rate":
            text = self.rate.text().strip()
            if not text:
                return None, "enter the pulse repetition rate, e.g. 80 MHz"
            out["rate"] = text
        try:
            PulseClock.model_validate(out)
        except ValidationError as exc:
            return None, error_message(exc)
        return out, ""

    def commit(self) -> None:
        if self._loading:
            return
        value, message = self.collect()
        self.problem.emit(message)
        if value is not None and value != self._shown:
            self.committed.emit(value)

    def set_raw(self, raw: Any) -> None:
        raw = raw if isinstance(raw, dict) else {"kind": "sources"}
        self._shown = dict(raw)
        self._loading = True
        self.kind.setCurrentIndex(max(self.kind.findData(raw.get("kind", "sources")), 0))
        if raw.get("source"):
            if self.source.findText(raw["source"]) < 0:
                self.source.addItem(raw["source"])
            self.source.setCurrentText(raw["source"])
        self.rate.setText(str(raw.get("rate") or ""))
        self._loading = False
        self._update_visibility()
        self.problem.emit("")


class ConditionsEditor(Editor):
    HEADERS = ("Name", "Herald (must click)", "Veto (must not click)", "Description")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(110)
        self.add_button = QPushButton("Add condition", self)
        self.remove_button = QPushButton("Remove selected", self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._shown: list = []
        self._loading = False
        self.table.itemChanged.connect(lambda _item: self.commit())
        self.add_button.clicked.connect(self.add_row)
        self.remove_button.clicked.connect(self.remove_selected)

    def _set_row(self, row: int, values: tuple[str, str, str, str]) -> None:
        for col, text in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(text))

    def add_row(self) -> None:
        names = {self.table.item(r, 0).text() for r in range(self.table.rowCount())}
        n = 1
        while f"C{n}" in names:
            n += 1
        self._loading = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_row(row, (f"C{n}", "", "", ""))
        self._loading = False
        self.table.setCurrentCell(row, 1)
        self.problem.emit("enter the herald channels (or detector ids) of the new condition")

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        self._loading = True
        for row in rows:
            self.table.removeRow(row)
        self._loading = False
        self.commit()

    def collect(self) -> tuple[list | None, str]:
        out = []
        for r in range(self.table.rowCount()):
            name, herald, veto, desc = (
                (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
                for c in range(4)
            )
            entry: dict[str, Any] = {"name": name, "herald": _refs(herald)}
            if _refs(veto):
                entry["veto"] = _refs(veto)
            if desc:
                entry["description"] = desc
            try:
                HeraldCondition.model_validate(entry)
            except ValidationError as exc:
                return None, f"condition '{name or r + 1}': {error_message(exc)}"
            if any(c["name"] == name for c in out):
                return None, f"condition names must be unique ('{name}' is used twice)"
            out.append(entry)
        return out, ""

    def commit(self) -> None:
        if self._loading:
            return
        value, message = self.collect()
        self.problem.emit(message)
        if value is not None and value != self._shown:
            self.committed.emit(value)

    def set_raw(self, raw: Any) -> None:
        conditions = list(raw) if isinstance(raw, list) else []
        self._shown = conditions
        self._loading = True
        self.table.setRowCount(len(conditions))
        for r, c in enumerate(conditions):
            self._set_row(
                r,
                (
                    str(c.get("name", "")),
                    ", ".join(c.get("herald", [])),
                    ", ".join(c.get("veto", [])),
                    str(c.get("description", "")),
                ),
            )
        self._loading = False
        self.problem.emit("")


class MeasurementEditor(QWidget):
    """The setup's measurement definition; every edit goes through
    ``SetupDocument.set_measurement`` and an invalid one is shown on its row, not applied."""

    def __init__(self, document: SetupDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document = document
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.state = QLabel("", self)
        self.state.setWordWrap(True)
        self.state.setProperty("role", "muted")
        layout.addWidget(self.state)
        self.rows: dict[str, FieldRow] = {}
        fields = Measurement.model_fields
        self.defaults = {n: f.get_default(call_default_factory=True) for n, f in fields.items()}

        def add(name: str, spec: FieldSpec, editor: Editor) -> None:
            row = FieldRow(
                name,
                spec,
                editor,
                self._set,
                description=HINTS[name],
                label=LABELS[name],
                parent=self,
            )
            layout.addWidget(row)
            self.rows[name] = row

        for name in _FIELDS:
            spec = analyze(fields[name].annotation, fields[name].metadata)
            add(name, spec, make_editor(spec))
        self.rows["coincidence_window"].editor.force_example("1 ns")
        self.rows["acquisition_duration"].editor.force_example("10 s")
        sources = [i.id for i in document.setup.instances]
        add("pulse_clock", FieldSpec("json"), PulseClockEditor(sources))
        for name in _LATER:
            spec = analyze(fields[name].annotation, fields[name].metadata)
            add(name, spec, make_editor(spec))
        add("conditions", FieldSpec("json"), ConditionsEditor())
        self.remove_button = QPushButton("Remove measurement", self)
        self.remove_button.setToolTip(
            "Delete the measurement definition; analyses and engines use their own settings"
        )
        self.remove_button.clicked.connect(self.remove)
        layout.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.error = QLabel("", self)
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.error.setVisible(False)
        layout.addWidget(self.error)
        title = QLabel("<b>Measurement card</b>", self)
        layout.addWidget(title)
        self.card = QLabel("", self)
        self.card.setWordWrap(True)
        self.card.setTextFormat(Qt.TextFormat.RichText)
        self.card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.card)
        self.refresh()

    def data(self) -> dict[str, Any]:
        m = self.document.setup.measurement
        return m.model_dump(mode="json", exclude_defaults=True) if m is not None else {}

    def _set(self, name: str, value: Any) -> None:
        data = self.data()
        if value is None or value is MISSING or value in ({}, []):
            data.pop(name, None)
        else:
            data[name] = value
        self.document.set_measurement(data)

    def remove(self) -> None:
        try:
            self.document.set_measurement(None)
        except ValueError as exc:
            self.error.setText(str(exc))
            self.error.setVisible(True)
            return
        self.error.setVisible(False)

    def refresh(self) -> None:
        data = self.data()
        exists = self.document.setup.measurement is not None
        self.state.setText(
            "Saved with the setup; edits are undoable and make earlier results outdated."
            if exists
            else "No measurement definition: the defaults below apply. Editing a field creates one."
        )
        self.remove_button.setEnabled(exists)
        for name, row in self.rows.items():
            default = self.defaults[name]
            if name == "pulse_clock":
                default = {"kind": "sources"}
            row.refresh(data.get(name, MISSING), MISSING if default is None else default)
        rows = measurement_card(self.document.setup)
        self.card.setText(
            "<table>"
            + "".join(
                f"<tr><td valign='top'>{html.escape(k.strip())}&nbsp;</td>"
                f"<td>{html.escape(v)}</td></tr>"
                for k, v in rows
            )
            + "</table>"
        )
