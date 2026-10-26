import math
import re
from collections.abc import Callable, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLineEdit,
    QMenu,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
)

from sequoia.fock.analysis import format_state
from sequoia.fock.circuit import FockError

BAR_LIMIT = 40
ROWS_SHOWN = 500
MAX_COLUMN = 240
_COND = re.compile(r"^\s*(\d+)\s*(==|=|!=|>=|<=|>|<)\s*(\d+)\s*$")
_NAMED = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(==|=|!=|>=|<=|>|<)\s*(\d+)\s*$")


def parse_ints(text: str, what: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.replace(";", ",").split(",") if x.strip() != "")
    except ValueError:
        raise FockError(f"{what}: {text!r} is not a list of integers") from None


def parse_int(text: str, what: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise FockError(f"{what}: {text!r} is not a whole number") from None


def parse_float(text: str, what: str) -> float:
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


def num(p: float) -> str:
    return f"{p:.6g}"


class SortItem(QTableWidgetItem):
    def __init__(self, text: str, key: float | None = None) -> None:
        super().__init__(text)
        self._key = None if key is None else float(key)
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other) -> bool:
        if isinstance(other, HiddenItem):
            return not other.order[0]
        if isinstance(other, SortItem) and self._key is not None and other._key is not None:
            return bool(self._key < other._key)
        return bool(self.text() < other.text())


class HiddenItem(SortItem):
    """Cell of the 'more rows' line, which stays last whatever the sort order. ``order[0]`` is
    True while the table sorts descending; it is shared with the table, which updates it before
    Qt sorts."""

    def __init__(self, order: list[bool]) -> None:
        super().__init__("")
        self.order = order

    def __lt__(self, other) -> bool:
        return bool(self.order[0])


Cells = Sequence[tuple[str, float | None]]


class ResultTable(QTableWidget):
    """Read-only result table: copy of the selected cells (Ctrl/Cmd+C, context menu), full header
    text in tooltips, an empty-state message, and long tables shown as the first ``limit`` rows
    plus one line with the number and probability of the rest ('show more')."""

    def __init__(self, headers: list[str], empty: str = "", tips: Sequence[str] = ()) -> None:
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(headers)
        for c, h in enumerate(headers):
            tip = tips[c] if c < len(tips) and tips[c] else h.replace("\n", " ")
            self.horizontalHeaderItem(c).setToolTip(tip)
        self.empty_text = empty
        self.limit = ROWS_SHOWN
        self.hidden_rows = 0
        self.hidden_mass = 0.0
        self._items: Sequence = ()
        self._cells: Callable[[int, object], Cells] | None = None
        self._mass: Callable[[object], float] | None = None
        self._mass_column = 1
        self._descending = [False]
        self.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.horizontalHeader().sortIndicatorChanged.connect(self._sort_order)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setSortingEnabled(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.cellDoubleClicked.connect(self._double_clicked)
        self.fit()

    def _sort_order(self, _column: int, order) -> None:
        self._descending[0] = order == Qt.SortOrder.DescendingOrder

    def sortItems(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        self._descending[0] = order == Qt.SortOrder.DescendingOrder
        super().sortItems(column, order)

    def fit(self) -> None:
        self.resizeColumnsToContents()
        for c in range(self.columnCount()):
            if self.columnWidth(c) > MAX_COLUMN:
                self.setColumnWidth(c, MAX_COLUMN)

    def selected_text(self, headers: bool = False) -> str:
        cells = sorted((i.row(), i.column()) for i in self.selectedIndexes())
        if not cells:
            return ""
        cols = sorted({c for _, c in cells})
        rows: dict[int, dict[int, str]] = {}
        for r, c in cells:
            item = self.item(r, c)
            rows.setdefault(r, {})[c] = item.text() if item is not None else ""
        lines = []
        if headers:
            lines.append(
                "\t".join(self.horizontalHeaderItem(c).text().replace("\n", " ") for c in cols)
            )
        for r in sorted(rows):
            lines.append("\t".join(rows[r].get(c, "") for c in cols))
        return "\n".join(lines)

    def copy_selection(self, headers: bool = False) -> None:
        QApplication.clipboard().setText(self.selected_text(headers))

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            return
        super().keyPressEvent(event)

    def context_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addAction("Copy", self.copy_selection)
        menu.addAction("Copy with headers", lambda: self.copy_selection(True))
        menu.addAction("Select all", self.selectAll)
        if self.hidden_rows:
            menu.addAction(f"Show {min(ROWS_SHOWN, self.hidden_rows)} more rows", self.show_more)
            menu.addAction(f"Show all {len(self._items)} rows", self.show_all)
        return menu

    def _menu(self, pos) -> None:
        self.context_menu().exec(self.viewport().mapToGlobal(pos))

    def _double_clicked(self, row: int, _col: int) -> None:
        if isinstance(self.item(row, 0), HiddenItem):
            self.show_more()

    def show_more(self) -> None:
        self.limit += ROWS_SHOWN
        self._render()

    def show_all(self) -> None:
        self.limit = max(len(self._items), ROWS_SHOWN)
        self._render()

    def set_rows(
        self,
        items: Sequence,
        cells: Callable[[int, object], Cells],
        mass: Callable[[object], float] | None = None,
        mass_column: int = 1,
        keep_limit: bool = False,
    ) -> None:
        self._items, self._cells, self._mass, self._mass_column = items, cells, mass, mass_column
        if not keep_limit:
            self.limit = ROWS_SHOWN
        self._render()

    def clear_rows(self, empty: str | None = None) -> None:
        self._items = ()
        self.hidden_rows, self.hidden_mass = 0, 0.0
        self.setRowCount(0)
        if empty is not None:
            self.empty_text = empty
        self.viewport().update()

    def _render(self) -> None:
        items, shown = self._items, self._items[: self.limit]
        self.hidden_rows = len(items) - len(shown)
        self.hidden_mass = (
            float(sum(self._mass(x) for x in items[self.limit :]))
            if self.hidden_rows and self._mass is not None
            else 0.0
        )
        self.setSortingEnabled(False)
        self.setRowCount(len(shown) + (1 if self.hidden_rows else 0))
        for r, x in enumerate(shown):
            for c, (text, key) in enumerate(self._cells(r, x)):
                self.setItem(r, c, SortItem(text, key))
        if self.hidden_rows:
            r = len(shown)
            text = f"... {self.hidden_rows} more rows (double-click to show more)"
            for c in range(self.columnCount()):
                self.setItem(r, c, HiddenItem(self._descending))
            self.item(r, 0).setText(text)
            if self._mass is not None:
                self.item(r, self._mass_column).setText(f"{num(self.hidden_mass)} (hidden)")
            for c in range(self.columnCount()):
                self.item(r, c).setToolTip(
                    f"{self.hidden_rows} less probable rows are not listed; their total "
                    f"probability is {self.hidden_mass:.6g}. Right-click to show more or all."
                )
        self.setSortingEnabled(True)
        self.fit()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.rowCount() == 0 and self.empty_text:
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(
                self.viewport().rect().adjusted(12, 12, -12, -12),
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextWordWrap,
                self.empty_text,
            )
            painter.end()


class IntDelegate(QStyledItemDelegate):
    """Whole-number cell editor: fractional or negative input cannot be typed."""

    def __init__(self, minimum: int = 0, maximum: int = 1_000_000, parent=None) -> None:
        super().__init__(parent)
        self.minimum, self.maximum = minimum, maximum

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QIntValidator(self.minimum, self.maximum, editor))
        return editor


def style_plot(plot, bottom: str = "output state (rank)", left: str = "probability") -> None:
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


def draw_bars(plot, bars, shown):
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
