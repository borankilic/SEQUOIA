from PySide6.QtCore import QByteArray, QMimeData, QSize, Qt, Signal
from PySide6.QtWidgets import QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from sequoia.components.base import CATEGORIES
from sequoia.library.store import Library, PartInfo

PART_MIME = "application/x-sequoia-part"
PROBLEMS = "Problems"
_REF_ROLE = Qt.ItemDataRole.UserRole
_SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1


class _PartTree(QTreeWidget):
    def mimeData(self, items) -> QMimeData:
        mime = QMimeData()
        refs = [i.data(0, _REF_ROLE) for i in items if i.data(0, _REF_ROLE)]
        if refs:
            mime.setData(PART_MIME, QByteArray(refs[0].encode("utf-8")))
        return mime

    def mimeTypes(self) -> list[str]:
        return [PART_MIME]


def part_label(info: PartInfo) -> str:
    pieces = [p for p in (info.manufacturer, info.model) if p]
    return " ".join(pieces) if pieces else info.ref.rsplit("/", 1)[-1]


class Palette(QWidget):
    part_activated = Signal(str)
    part_edit_requested = Signal(str)

    def __init__(self, library: Library, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.library = library
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search parts")
        self.search.setClearButtonEnabled(True)
        self.tree = _PartTree()
        self.tree.setHeaderHidden(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree.setIconSize(QSize(22, 22))
        self.tree.setIndentation(14)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.search)
        layout.addWidget(self.tree)
        self.search.textChanged.connect(self._apply_filter)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.refresh()

    def refresh(self) -> None:
        expanded = {
            self._path(item)
            for item in self._all_items()
            if item.isExpanded() and item.childCount()
        }
        first = not expanded and self.tree.topLevelItemCount() == 0
        self.tree.clear()
        infos = self.library.index()
        by_category: dict[str, dict[str, list[PartInfo]]] = {}
        problems: list[PartInfo] = []
        for info in infos:
            if info.error is not None or info.category is None or info.type is None:
                problems.append(info)
            else:
                by_category.setdefault(info.category, {}).setdefault(info.type, []).append(info)
        for category in CATEGORIES:
            types = by_category.get(category)
            if not types:
                continue
            top = self._group(category.capitalize(), category)
            for type_name in sorted(types):
                node = self._group(type_name, f"{category}/{type_name}", top)
                for info in sorted(types[type_name], key=lambda i: i.ref):
                    self._part(info, node)
        if problems:
            top = self._group(PROBLEMS, PROBLEMS)
            for info in sorted(problems, key=lambda i: i.ref):
                part = self._part(info, top)
                if info.error is not None:
                    reason = info.error
                else:
                    reason = f"unknown component type {info.type!r}"
                part.setText(0, info.ref)
                part.setToolTip(0, reason)
        for item in self._all_items():
            if item.childCount() and (first or self._path(item) in expanded):
                item.setExpanded(True)
        self._apply_filter(self.search.text())

    def _group(self, text, key, parent=None) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent) if parent is not None else QTreeWidgetItem(self.tree)
        item.setText(0, text)
        item.setData(0, Qt.ItemDataRole.UserRole + 2, key)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setData(0, _SEARCH_ROLE, text.lower())
        return item

    def _part(self, info: PartInfo, parent: QTreeWidgetItem) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent)
        item.setText(0, part_label(info))
        item.setData(0, _REF_ROLE, info.ref)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled
        )
        tip = [info.ref]
        if info.type:
            tip.append(f"type: {info.type}")
        if info.serial:
            tip.append(f"serial: {info.serial}")
        item.setToolTip(0, "\n".join(tip))
        if info.type:
            try:
                from PySide6.QtGui import QIcon

                from sequoia.gui.canvas.icons import component_pixmap

                item.setIcon(0, QIcon(component_pixmap(info.type, 28)))
            except ImportError:
                pass
        haystack = " ".join(
            str(x) for x in (info.ref, info.type, info.manufacturer, info.model, info.serial) if x
        )
        item.setData(0, _SEARCH_ROLE, haystack.lower())
        return item

    def _path(self, item: QTreeWidgetItem) -> str:
        return str(item.data(0, Qt.ItemDataRole.UserRole + 2))

    def _all_items(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _apply_filter(self, text: str) -> None:
        words = text.lower().split()

        def visit(item: QTreeWidgetItem, inherited: bool) -> bool:
            haystack = item.data(0, _SEARCH_ROLE) or ""
            own = all(w in haystack for w in words)
            is_part = bool(item.data(0, _REF_ROLE))
            if is_part:
                visible = not words or own or inherited
            else:
                group_match = bool(words) and own
                visible = False
                for i in range(item.childCount()):
                    if visit(item.child(i), inherited or group_match):
                        visible = True
                if not words:
                    visible = True
            item.setHidden(not visible)
            if words and visible and not is_part:
                item.setExpanded(True)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i), False)

    def visible_refs(self) -> list[str]:
        return [
            item.data(0, _REF_ROLE)
            for item in self._all_items()
            if item.data(0, _REF_ROLE) and not item.isHidden()
        ]

    def item_for_ref(self, ref: str) -> QTreeWidgetItem | None:
        for item in self._all_items():
            if item.data(0, _REF_ROLE) == ref:
                return item
        return None

    def _context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        ref = item.data(0, _REF_ROLE) if item is not None else None
        if not ref:
            return
        menu = QMenu(self)
        add = menu.addAction("Add to canvas")
        edit = menu.addAction("Edit in Library Manager...")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is add:
            self.part_activated.emit(ref)
        elif chosen is edit:
            self.part_edit_requested.emit(ref)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        ref = item.data(0, _REF_ROLE)
        if ref:
            self.part_activated.emit(ref)
