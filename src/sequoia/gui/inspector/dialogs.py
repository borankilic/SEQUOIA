from typing import Any

import numpy as np
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sequoia.data import Curve, format_axis, parse_axis
from sequoia.library.editing import PROVENANCE_SOURCES


class ProvenanceDialog(QDialog):
    """Where a value came from: source, date, who, reference and a note."""

    def __init__(self, initial: dict[str, Any] | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Provenance")
        initial = initial or {}
        self.source = QComboBox()
        self.source.addItems(PROVENANCE_SOURCES)
        if initial.get("source") in PROVENANCE_SOURCES:
            self.source.setCurrentText(initial["source"])
        self.has_date = QCheckBox("date")
        self.date = QDateEdit()
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        date = initial.get("date")
        if date:
            parsed = QDate.fromString(str(date), "yyyy-MM-dd")
            if parsed.isValid():
                self.date.setDate(parsed)
                self.has_date.setChecked(True)
        if not self.has_date.isChecked():
            self.date.setDate(QDate.currentDate())
        self.date.setEnabled(self.has_date.isChecked())
        self.has_date.toggled.connect(self.date.setEnabled)
        self.by = QLineEdit(initial.get("by", ""))
        self.reference = QLineEdit(initial.get("reference", ""))
        self.reference.setPlaceholderText("datasheet page, file, DOI ...")
        self.note = QLineEdit(initial.get("note", ""))
        form = QFormLayout()
        form.addRow("source", self.source)
        row = QHBoxLayout()
        row.addWidget(self.has_date)
        row.addWidget(self.date, 1)
        form.addRow(row)
        form.addRow("by", self.by)
        form.addRow("reference", self.reference)
        form.addRow("note", self.note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.clear_button = buttons.addButton(
            "Remove provenance", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.cleared = False
        self.clear_button.clicked.connect(self._clear)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _clear(self) -> None:
        self.cleared = True
        self.accept()

    def provenance(self) -> dict[str, Any] | None:
        if self.cleared:
            return None
        out: dict[str, Any] = {"source": self.source.currentText()}
        if self.has_date.isChecked():
            out["date"] = self.date.date().toString("yyyy-MM-dd")
        for key, widget in (("by", self.by), ("reference", self.reference), ("note", self.note)):
            text = widget.text().strip()
            if text:
                out[key] = text
        return out


def ask_provenance(parent: QWidget | None, initial: dict[str, Any] | None):
    """Returns (accepted, provenance-or-None)."""
    dialog = ProvenanceDialog(initial, parent)
    if dialog.exec():
        return True, dialog.provenance()
    return False, None


class CurveTableDialog(QDialog):
    """Edit a curve as a table of (x, y) rows with its axis labels and units."""

    def __init__(self, curve: Curve | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit curve")
        self.resize(460, 520)
        self._source = curve
        self.x_axis = QLineEdit(format_axis(curve.x_label, curve.x_unit) if curve else "x (nm)")
        self.y_axis = QLineEdit(format_axis(curve.y_label, curve.y_unit) if curve else "y (1)")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["x", "y"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #b00020;")
        form = QFormLayout()
        form.addRow("x axis", self.x_axis)
        form.addRow("y axis", self.y_axis)
        add = QPushButton("Add row")
        remove = QPushButton("Remove row")
        paste = QPushButton("Paste from clipboard")
        add.clicked.connect(lambda: self._insert(self.table.currentRow() + 1))
        remove.clicked.connect(self._remove)
        paste.clicked.connect(self.paste)
        tools = QHBoxLayout()
        for w in (add, remove, paste):
            tools.addWidget(w)
        tools.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(tools)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.message)
        layout.addWidget(buttons)
        if curve is not None:
            for x, y in zip(curve.x, curve.y, strict=True):
                self._append(repr(float(x)), repr(float(y)))
        else:
            for _ in range(3):
                self._append("", "")
        self._result: Curve | None = None

    def _append(self, x: str, y: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(x))
        self.table.setItem(row, 1, QTableWidgetItem(y))

    def _insert(self, row: int) -> None:
        row = max(0, min(row, self.table.rowCount()))
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def paste(self, text: str | None = None) -> None:
        text = text if isinstance(text, str) else QApplication.clipboard().text()
        rows = []
        for line in text.splitlines():
            cells = [c for c in line.replace(",", "\t").replace(";", "\t").split() if c]
            if len(cells) >= 2:
                rows.append(cells[:2])
        if rows:
            self.table.setRowCount(0)
            for x, y in rows:
                self._append(x, y)

    def set_rows(self, rows: list[tuple[float, float]]) -> None:
        self.table.setRowCount(0)
        for x, y in rows:
            self._append(repr(float(x)), repr(float(y)))

    def _read(self) -> Curve:
        xs, ys = [], []
        for row in range(self.table.rowCount()):
            xi, yi = self.table.item(row, 0), self.table.item(row, 1)
            tx = xi.text().strip() if xi else ""
            ty = yi.text().strip() if yi else ""
            if not tx and not ty:
                continue
            try:
                xs.append(float(tx))
                ys.append(float(ty))
            except ValueError:
                raise ValueError(f"row {row + 1}: '{tx}', '{ty}' are not numbers") from None
        if len(xs) < 2:
            raise ValueError("a curve needs at least two points")
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        x_label, x_unit = parse_axis(self.x_axis.text())
        y_label, y_unit = parse_axis(self.y_axis.text())
        base = self._source
        return Curve(
            x=xs,
            y=ys,
            x_label=x_label,
            x_unit=x_unit,
            y_label=y_label,
            y_unit=y_unit,
            provenance=base.provenance if base else None,
            extrapolation=base.extrapolation if base else "warn_clamp",
            file=base.file if base else None,
        )

    def _accept(self) -> None:
        try:
            self._result = self._read()
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self.accept()

    def curve(self) -> Curve | None:
        return self._result
