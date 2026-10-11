from pathlib import Path
from typing import get_args

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sequoia.data import Curve
from sequoia.errors import SequoiaError
from sequoia.library.importers.table import (
    Table,
    YKind,
    guess_y_kind,
    import_curve,
    list_sheets,
    read_table,
)

TABLE_FILTER = "Tables (*.csv *.tsv *.txt *.dat *.asc *.xlsx *.xlsm);;All files (*)"


class ImportCurveDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import curve")
        self.resize(520, 560)
        self._table: Table | None = None
        self._curve: Curve | None = None
        self._loading = False

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)

        self.sheet = QComboBox()
        self.x_column = QComboBox()
        self.y_column = QComboBox()
        self.x_unit = QLineEdit()
        self.x_unit.setPlaceholderText("unit of the x column, e.g. nm")
        self.y_kind = QComboBox()
        for kind in get_args(YKind):
            self.y_kind.addItem(kind, kind)
        self.y_unit = QLineEdit()
        self.y_unit.setPlaceholderText("only for 'raw'")
        self.target = QComboBox()
        self.target.addItems(["fraction", "responsivity"])

        form = QFormLayout()
        form.addRow("File", path_row)
        form.addRow("Sheet", self.sheet)
        form.addRow("x column", self.x_column)
        form.addRow("x unit", self.x_unit)
        form.addRow("y column", self.y_column)
        form.addRow("y kind", self.y_kind)
        form.addRow("y unit", self.y_unit)
        form.addRow("convert to", self.target)

        import pyqtgraph as pg

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(180)
        self.plot.setMenuEnabled(False)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.message)
        layout.addWidget(self.buttons)

        self.sheet.currentIndexChanged.connect(self._on_sheet)
        for widget in (self.x_column, self.y_column, self.y_kind, self.target):
            widget.currentIndexChanged.connect(self._on_column_change)
        self.x_unit.editingFinished.connect(self.reimport)
        self.y_unit.editingFinished.connect(self.reimport)
        self.sheet.setVisible(False)
        form.labelForField(self.sheet).setVisible(False)
        self._sheet_label = form.labelForField(self.sheet)

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import table", "", TABLE_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path: str | Path, sheet: str | None = None) -> None:
        path = Path(path)
        self.path_edit.setText(str(path))
        excel = path.suffix.lower() in (".xlsx", ".xlsm")
        self._loading = True
        try:
            self.sheet.clear()
            if excel:
                try:
                    self.sheet.addItems(list_sheets(path))
                except Exception as exc:
                    self._fail(f"cannot open workbook: {exc}")
                    return
                if sheet is not None:
                    self.sheet.setCurrentText(sheet)
            self.sheet.setVisible(excel)
            self._sheet_label.setVisible(excel)
        finally:
            self._loading = False
        self._read(path, self.sheet.currentText() if excel else None)

    def _on_sheet(self) -> None:
        if not self._loading and self.path_edit.text():
            self._read(Path(self.path_edit.text()), self.sheet.currentText())

    def _read(self, path: Path, sheet: str | None) -> None:
        try:
            table = read_table(path, sheet)
        except SequoiaError as exc:
            self._table = None
            self._fail(str(exc))
            return
        self._table = table
        self._loading = True
        try:
            self.x_column.clear()
            self.y_column.clear()
            for i, col in enumerate(table.columns):
                text = col.header + (f"  [{col.unit}]" if col.unit else "")
                self.x_column.addItem(text, i)
                self.y_column.addItem(text, i)
            if len(table.columns) > 1:
                self.y_column.setCurrentIndex(1)
            self._sync_defaults()
        finally:
            self._loading = False
        self.reimport()

    def _sync_defaults(self) -> None:
        if self._table is None or self.x_column.currentIndex() < 0:
            return
        x = self._table.columns[self.x_column.currentIndex()]
        y = self._table.columns[self.y_column.currentIndex()]
        self.x_unit.setText(x.unit or "")
        index = self.y_kind.findData(guess_y_kind(y.name, y.unit))
        self.y_kind.setCurrentIndex(index)
        self.y_unit.setText(y.unit or "")

    def _on_column_change(self) -> None:
        if self._loading:
            return
        sender = self.sender()
        if sender in (self.x_column, self.y_column):
            self._loading = True
            self._sync_defaults()
            self._loading = False
        self.reimport()

    def _fail(self, text: str) -> None:
        self._curve = None
        self.plot.clear()
        self.message.setText(text)
        self.message.setStyleSheet("color: #c0392b;")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def reimport(self) -> None:
        if self._table is None or self.x_column.currentIndex() < 0:
            return
        kind = self.y_kind.currentData()
        try:
            result = import_curve(
                self._table,
                self.x_column.currentData(),
                self.y_column.currentData(),
                x_unit=self.x_unit.text().strip() or None,
                y_unit=self.y_unit.text().strip() or None,
                y_kind=kind,
                target=self.target.currentText(),
            )
        except SequoiaError as exc:
            self._fail(str(exc))
            return
        self._curve = result.curve
        curve = result.curve
        self.plot.clear()
        self.plot.plot(curve.x, curve.y, pen="#2a7ab0")
        self.plot.setLabel("bottom", f"{curve.x_label} ({curve.x_unit})")
        self.plot.setLabel("left", f"{curve.y_label} ({curve.y_unit})")
        warnings = "\n".join(result.warnings)
        self.message.setStyleSheet("")
        self.message.setText(f"{len(curve.x)} points" + (f"\n{warnings}" if warnings else ""))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def curve(self) -> Curve | None:
        return self._curve
