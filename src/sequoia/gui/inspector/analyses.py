import copy
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sequoia.components.base import PortDirection, PortKind
from sequoia.data import Duration
from sequoia.gui.document import ANALYSIS_TYPES, SetupDocument, check_analysis
from sequoia.gui.inspector.fields import ScalarEditor, analyze

_DESCRIBE = {
    "coincidences": "coincidences between two channels in a window",
    "heralded_g2": "heralded second-order correlation (herald, a, b)",
    "g2": "second-order correlation histogram of two channels",
    "histogram": "arrival-time difference histogram of two channels",
}
_ROLES = {
    "heralded_g2": ("herald", "a", "b"),
    "coincidences": ("channel 1", "channel 2"),
    "g2": ("channel 1", "channel 2"),
    "histogram": ("channel 1", "channel 2"),
}


def tagger_channels(document: SetupDocument) -> list[str]:
    out = []
    for inst in document.setup.instances:
        for port in document.ports(inst.id):
            if port.kind == PortKind.ELECTRICAL and port.direction == PortDirection.IN:
                out.append(f"{inst.id}.{port.name}")
    return out


def _summary(spec: dict[str, Any]) -> str:
    kind = spec.get("type", "?")
    if kind == "heralded_g2":
        chans = [spec.get("herald"), spec.get("a"), spec.get("b")]
    else:
        chans = list(spec.get("channels") or [])
    return f"{spec.get('name', '?')}  [{kind}]  " + " x ".join(str(c) for c in chans if c)


class AnalysisForm(QWidget):
    cancelled = Signal()

    def __init__(
        self,
        channels: list[str],
        spec: dict[str, Any] | None,
        default_name: str,
        submit: Callable[[dict[str, Any]], str | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._base = copy.deepcopy(spec) if spec else {}
        self._channels = list(channels)
        self.name = QLineEdit(self._base.get("name", default_name))
        self.kind = QComboBox()
        for kind in ANALYSIS_TYPES:
            self.kind.addItem(kind, kind)
        self.kind.setCurrentIndex(
            max(0, self.kind.findData(self._base.get("type", "coincidences")))
        )
        self.kind.setToolTip("\n".join(f"{k}: {v}" for k, v in _DESCRIBE.items()))
        self.channel_labels = [QLabel() for _ in range(3)]
        self.channel_combos = [QComboBox() for _ in range(3)]
        duration = analyze(Duration)
        self.window = ScalarEditor(duration)
        self.window.force_example("1 ns")
        self.range = ScalarEditor(duration)
        self.range.force_example("10 ns")
        self.center = QLineEdit()
        self.center.setPlaceholderText("auto, or a time such as 0 ns")
        self.bins = QSpinBox()
        self.bins.setRange(1, 1_000_000)
        self.bins.setValue(200)
        self.error = QLabel("")
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.error.setVisible(False)
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        self._rows: dict[str, tuple[QLabel, QWidget]] = {}

        def row(key: str, text: str, widget: QWidget) -> None:
            label = QLabel(text)
            layout.addWidget(label)
            layout.addWidget(widget)
            self._rows[key] = (label, widget)

        row("name", "Name", self.name)
        row("type", "Type", self.kind)
        for i in range(3):
            layout.addWidget(self.channel_labels[i])
            layout.addWidget(self.channel_combos[i])
            self._rows[f"ch{i}"] = (self.channel_labels[i], self.channel_combos[i])
        row("window", "Coincidence window (a duration, e.g. 1 ns)", self.window)
        row("range", "Histogram range (a duration, e.g. 10 ns)", self.range)
        row("bins", "Bins", self.bins)
        row("center", "Centre", self.center)
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._load(self._base)
        self.kind.currentIndexChanged.connect(lambda _i: self._layout_for_kind())
        self.apply_button.clicked.connect(self.apply)
        self.cancel_button.clicked.connect(self.cancelled)
        self._layout_for_kind()

    def _fill_combo(self, combo: QComboBox, current: str | None) -> None:
        combo.clear()
        for channel in self._channels:
            combo.addItem(channel, channel)
        if current and combo.findData(current) < 0:
            combo.addItem(f"{current} (not in this setup)", current)
        combo.setCurrentIndex(combo.findData(current) if current else (0 if combo.count() else -1))

    def _load(self, spec: dict[str, Any]) -> None:
        kind = spec.get("type", "coincidences")
        if kind == "heralded_g2":
            current = [spec.get("herald"), spec.get("a"), spec.get("b")]
        else:
            chans = list(spec.get("channels") or [])
            current = [*chans, None, None][:2] + [None]
        used: set[str] = set()
        for i, combo in enumerate(self.channel_combos):
            want = current[i] if i < len(current) else None
            if want is None:
                free = [c for c in self._channels if c not in used]
                want = free[0] if free and i < (3 if kind == "heralded_g2" else 2) else None
            self._fill_combo(combo, want)
            if want:
                used.add(want)
        if spec.get("window") is not None:
            self.window.set_raw(spec["window"])
        if spec.get("range") is not None:
            self.range.set_raw(spec["range"])
        if "bins" in spec:
            self.bins.setValue(int(spec["bins"]))
        center = spec.get("center", "auto")
        self.center.setText("" if str(center).strip().lower() == "auto" else str(center))

    def _layout_for_kind(self) -> None:
        kind = self.kind.currentData()
        roles = _ROLES.get(kind, ("channel 1", "channel 2"))
        for i in range(3):
            label, combo = self._rows[f"ch{i}"]
            visible = i < len(roles)
            label.setVisible(visible)
            combo.setVisible(visible)
            if visible:
                label.setText(f"Tagger input: {roles[i]}")
        timed = kind in ("coincidences", "heralded_g2")
        for key in ("window",):
            for widget in self._rows[key]:
                widget.setVisible(timed)
        for key in ("range", "bins"):
            for widget in self._rows[key]:
                widget.setVisible(not timed)
        if len(roles) == 3:
            taken = self._selected(2)
            combo = self.channel_combos[2]
            if combo.currentData() is None or combo.currentData() in taken:
                free = next((c for c in self._channels if c not in taken), None)
                if free is not None:
                    combo.setCurrentIndex(combo.findData(free))

    def _selected(self, count: int) -> list[str]:
        return [c.currentData() for c in self.channel_combos[:count] if c.currentData()]

    def collect(self) -> tuple[dict[str, Any] | None, str]:
        kind = self.kind.currentData()
        spec: dict[str, Any] = {
            k: v
            for k, v in self._base.items()
            if k not in ("window", "range", "bins", "channels", "herald", "a", "b", "center")
        }
        spec["name"] = self.name.text().strip()
        spec["type"] = kind
        roles = _ROLES[kind]
        picked = [self.channel_combos[i].currentData() for i in range(len(roles))]
        if not all(picked):
            return (
                None,
                "this setup has no free time-tagger input to analyse; add a tagger and cables",
            )
        if kind == "heralded_g2":
            spec["herald"], spec["a"], spec["b"] = picked
        else:
            spec["channels"] = picked
        if kind in ("coincidences", "heralded_g2"):
            raw, message = self.window.parse(self.window.text())
            if message:
                return None, f"window: {message}"
            spec["window"] = raw
        else:
            raw, message = self.range.parse(self.range.text())
            if message:
                return None, f"range: {message}"
            spec["range"] = raw
            spec["bins"] = int(self.bins.value())
        center = self.center.text().strip()
        if center and center.lower() != "auto":
            raw, message = self.window.parse(center)
            if message:
                return None, f"centre: {message}"
            spec["center"] = raw
        elif kind in ("coincidences", "heralded_g2"):
            spec["center"] = "auto"
        try:
            check_analysis(spec)
        except ValueError as exc:
            return None, str(exc)
        return spec, ""

    def show_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(bool(message))

    def apply(self) -> bool:
        spec, message = self.collect()
        if spec is not None:
            message = self._submit(spec) or ""
        self.show_error(message)
        return not message


class AnalysesEditor(QWidget):
    def __init__(self, document: SetupDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document = document
        self.form: AnalysisForm | None = None
        self._index: int | None = None
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.add_button = QPushButton("Add")
        self.edit_button = QPushButton("Edit")
        self.remove_button = QPushButton("Remove")
        self.error = QLabel("")
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.error.setVisible(False)
        self.hint = QLabel("")
        self.hint.setProperty("role", "muted")
        self.hint.setWordWrap(True)
        row = QHBoxLayout()
        for button in (self.add_button, self.edit_button, self.remove_button):
            row.addWidget(button)
        row.addStretch(1)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.list)
        self._layout.addLayout(row)
        self._layout.addWidget(self.hint)
        self._layout.addWidget(self.error)
        self.add_button.clicked.connect(self.start_add)
        self.edit_button.clicked.connect(self.start_edit)
        self.remove_button.clicked.connect(self.remove_current)
        self.list.itemDoubleClicked.connect(lambda _item: self.start_edit())
        self.list.currentRowChanged.connect(lambda _r: self._update_buttons())
        self.refresh()

    def _show_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(bool(message))

    def refresh(self) -> None:
        current = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for spec in self.document.setup.analyses:
            self.list.addItem(_summary(spec))
        if self.list.count():
            self.list.setCurrentRow(min(max(current, 0), self.list.count() - 1))
        self.list.blockSignals(False)
        channels = tagger_channels(self.document)
        self.hint.setText(
            "" if channels else "Add a time tagger and connect detectors to it to define analyses."
        )
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_row = self.list.currentRow() >= 0
        self.edit_button.setEnabled(has_row)
        self.remove_button.setEnabled(has_row)
        self.add_button.setEnabled(self.form is None)

    def _close_form(self) -> None:
        if self.form is not None:
            self._layout.removeWidget(self.form)
            self.form.hide()
            self.form.deleteLater()
            self.form = None
        self._index = None
        self._update_buttons()

    def _open_form(self, spec: dict[str, Any] | None, index: int | None) -> None:
        self._close_form()
        self._show_error("")
        taken = {a.get("name") for a in self.document.setup.analyses}
        n = 1
        while f"analysis_{n}" in taken:
            n += 1
        form = AnalysisForm(tagger_channels(self.document), spec, f"analysis_{n}", self._on_applied)
        form.cancelled.connect(self._close_form)
        self._layout.insertWidget(2, form)
        self.form, self._index = form, index
        self._update_buttons()

    def start_add(self) -> None:
        self._open_form(None, None)

    def start_edit(self) -> None:
        row = self.list.currentRow()
        analyses = self.document.setup.analyses
        if 0 <= row < len(analyses):
            self._open_form(analyses[row], row)

    def _on_applied(self, spec: dict[str, Any]) -> str | None:
        analyses = copy.deepcopy(list(self.document.setup.analyses))
        if self._index is None:
            analyses.append(spec)
        else:
            analyses[self._index] = spec
        try:
            self.document.set_analyses(analyses)
        except ValueError as exc:
            return str(exc)
        row = len(analyses) - 1 if self._index is None else self._index
        self._close_form()
        self.refresh()
        self.list.setCurrentRow(row)
        return None

    def remove_current(self) -> None:
        row = self.list.currentRow()
        analyses = copy.deepcopy(list(self.document.setup.analyses))
        if not 0 <= row < len(analyses):
            return
        del analyses[row]
        try:
            self.document.set_analyses(analyses)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._show_error("")
        self._close_form()
        self.refresh()
