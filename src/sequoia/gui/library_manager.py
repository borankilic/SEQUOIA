from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sequoia.components.base import get_component, registered_types
from sequoia.errors import PartError, SequoiaError
from sequoia.gui.inspector.fields import ModelForm
from sequoia.gui.palette import part_label
from sequoia.library.editing import (
    HEADER_FIELDS,
    build_part,
    draft_from_part,
    keep_provenance,
    save_edited,
    suggest_ref,
)
from sequoia.library.store import Library, Part

CATEGORY_TITLES = {
    "sources": "Sources",
    "nonlinear": "Nonlinear",
    "freespace": "Free-space optics",
    "polarization": "Polarization",
    "filters": "Filters",
    "fiber": "Fiber",
    "detectors": "Detectors",
    "electronics": "Electronics",
    "probes": "Probes",
}


class LibraryManager(QDialog):
    """Browse, create, duplicate and edit library parts with per-value provenance."""

    library_changed = Signal()

    def __init__(self, library: Library, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Library Manager")
        self.resize(980, 700)
        self.library = library
        self.ref: str | None = None
        self.type_name: str | None = None
        self.original: Part | None = None
        self.draft: dict[str, Any] = {}
        self.attachments: list[str] = []
        self.is_new = False
        self.dirty = False
        self.form: ModelForm | None = None
        self.ask_text: Callable[[str, str, str], str | None] = self._ask_text
        self.ask_type: Callable[[list[str]], str | None] = self._ask_type
        self.confirm: Callable[[str], bool] = self._confirm
        self.ask_file: Callable[[], str | None] = self._ask_file

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search parts")
        self.search.setClearButtonEnabled(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.new_button = QPushButton("New...")
        self.duplicate_button = QPushButton("Duplicate...")
        self.rename_button = QPushButton("Rename...")
        self.delete_button = QPushButton("Delete")
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.search)
        lv.addWidget(self.tree, 1)
        for row in (
            (self.new_button, self.duplicate_button),
            (self.rename_button, self.delete_button),
        ):
            h = QHBoxLayout()
            for b in row:
                h.addWidget(b)
            lv.addLayout(h)

        self.title = QLabel("Select a part or create a new one")
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.type_label = QLabel("")
        self.manufacturer = QLineEdit()
        self.model = QLineEdit()
        self.serial = QLineEdit()
        self.description = QPlainTextEdit()
        self.description.setMaximumHeight(60)
        header = QFormLayout()
        header.addRow("Type", self.type_label)
        header.addRow("Manufacturer", self.manufacturer)
        header.addRow("Model", self.model)
        header.addRow("Serial", self.serial)
        header.addRow("Description", self.description)
        self.attach_list = QListWidget()
        self.attach_list.setMaximumHeight(70)
        self.attach_add = QPushButton("Attach file...")
        self.attach_remove = QPushButton("Remove")
        attach_row = QHBoxLayout()
        attach_row.addWidget(self.attach_add)
        attach_row.addWidget(self.attach_remove)
        attach_row.addStretch(1)
        self.params_holder = QWidget()
        self.params_layout = QVBoxLayout(self.params_holder)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.save_button = QPushButton("Save")
        self.revert_button = QPushButton("Revert")
        self.save_button.setDefault(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.revert_button)
        buttons.addWidget(self.save_button)

        content = QWidget()
        cv = QVBoxLayout(content)
        cv.addWidget(self.title)
        cv.addLayout(header)
        cv.addWidget(QLabel("Parameters"))
        cv.addWidget(self.params_holder)
        cv.addWidget(QLabel("Attachments (datasheets, raw files)"))
        cv.addWidget(self.attach_list)
        cv.addLayout(attach_row)
        cv.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(content)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.scroll, 1)
        rv.addLayout(buttons)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 650])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.search.textChanged.connect(self._filter)
        self.tree.currentItemChanged.connect(self._on_item)
        self.new_button.clicked.connect(self.new_part)
        self.duplicate_button.clicked.connect(self.duplicate_part)
        self.rename_button.clicked.connect(self.rename_part)
        self.delete_button.clicked.connect(self.delete_part)
        self.save_button.clicked.connect(self.save)
        self.revert_button.clicked.connect(self.revert)
        self.attach_add.clicked.connect(self.add_attachment)
        self.attach_remove.clicked.connect(self.remove_attachment)
        for edit in (self.manufacturer, self.model, self.serial):
            edit.textEdited.connect(self._mark_dirty)
        self.description.textChanged.connect(self._description_changed)
        self.refresh()
        self._set_enabled()

    # ---- dialogs (replaceable in tests)
    def _ask_text(self, title: str, label: str, default: str) -> str | None:
        text, ok = QInputDialog.getText(self, title, label, text=default)
        return text.strip() if ok and text.strip() else None

    def _ask_type(self, types: list[str]) -> str | None:
        text, ok = QInputDialog.getItem(self, "New part", "Component type", types, 0, False)
        return text if ok else None

    def _confirm(self, message: str) -> bool:
        return (
            QMessageBox.question(self, "Library Manager", message) == QMessageBox.StandardButton.Yes
        )

    def _ask_file(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(self, "Attach file")
        return path or None

    # ---- tree
    def set_library(self, library: Library) -> None:
        self.library = library
        self._clear_editor()
        self.refresh()

    def refresh(self, select: str | None = None) -> None:
        keep = select or self.ref
        self.tree.blockSignals(True)
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for info in self.library.index():
            key = info.category or "other"
            group = groups.get(key)
            if group is None:
                group = QTreeWidgetItem([CATEGORY_TITLES.get(key, key.title())])
                group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.tree.addTopLevelItem(group)
                groups[key] = group
            label = f"{part_label(info)}  [{info.type or 'invalid'}]"
            item = QTreeWidgetItem(group, [label])
            item.setData(0, Qt.ItemDataRole.UserRole, info.ref)
            item.setToolTip(0, info.error or info.ref)
        self.tree.expandAll()
        self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.tree.blockSignals(False)
        if keep:
            self.select_ref(keep, load=False)
        self._filter(self.search.text())

    def select_ref(self, ref: str, load: bool = True) -> bool:
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            for i in range(group.childCount()):
                item = group.child(i)
                if item.data(0, Qt.ItemDataRole.UserRole) == ref:
                    self.tree.blockSignals(not load)
                    self.tree.setCurrentItem(item)
                    self.tree.blockSignals(False)
                    return True
        return False

    def visible_refs(self) -> list[str]:
        out = []
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            for i in range(group.childCount()):
                if not group.child(i).isHidden():
                    out.append(group.child(i).data(0, Qt.ItemDataRole.UserRole))
        return out

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            shown = 0
            for i in range(group.childCount()):
                item = group.child(i)
                ref = item.data(0, Qt.ItemDataRole.UserRole)
                match = not needle or needle in (item.text(0) + " " + ref).lower()
                item.setHidden(not match)
                shown += match
            group.setHidden(shown == 0)

    def _on_item(self, current, _previous) -> None:
        if current is None:
            return
        ref = current.data(0, Qt.ItemDataRole.UserRole)
        if ref is None or ref == self.ref and not self.is_new:
            return
        if not self._discard_ok():
            self.select_ref(self.ref or "", load=False)
            return
        self.open_ref(ref)

    def _discard_ok(self) -> bool:
        return not self.dirty or self.confirm("Discard unsaved changes?")

    # ---- editor
    def _clear_editor(self) -> None:
        self.ref = None
        self.type_name = None
        self.original = None
        self.draft = {}
        self.attachments = []
        self.is_new = False
        self.dirty = False
        self._rebuild_form(None)
        for w in (self.manufacturer, self.model, self.serial):
            w.setText("")
        self.description.setPlainText("")
        self.attach_list.clear()
        self.title.setText("Select a part or create a new one")
        self.type_label.setText("")
        self.status.setText("")
        self._set_enabled()

    def _rebuild_form(self, type_name: str | None) -> None:
        if self.form is not None:
            self.params_layout.removeWidget(self.form)
            self.form.deleteLater()
            self.form = None
        if type_name is None:
            return
        cls = get_component(type_name)
        required = tuple(n for n, i in cls.Params.model_fields.items() if i.is_required())
        self.form = ModelForm(
            cls.Params,
            self._commit_param,
            required=required,
            provenance_editable=True,
        )
        self.params_layout.addWidget(self.form)
        self.form.refresh(self.draft)

    def _commit_param(self, name: str, value: Any) -> None:
        if value is None:
            self.draft.pop(name, None)
        else:
            self.draft[name] = keep_provenance(self.draft.get(name), value)
        self._mark_dirty()

    def _mark_dirty(self, *_args) -> None:
        self.dirty = True
        self.status.setText("unsaved changes")
        self.status.setStyleSheet("")
        self._set_enabled()

    def _description_changed(self) -> None:
        if self.ref is not None and self.description.hasFocus():
            self._mark_dirty()

    def _set_enabled(self) -> None:
        has = self.ref is not None
        saved = has and not self.is_new
        self.save_button.setEnabled(has and (self.dirty or self.is_new))
        self.revert_button.setEnabled(has and self.dirty)
        self.duplicate_button.setEnabled(saved)
        self.rename_button.setEnabled(saved)
        self.delete_button.setEnabled(saved)
        for w in (
            self.manufacturer,
            self.model,
            self.serial,
            self.description,
            self.attach_add,
            self.attach_remove,
        ):
            w.setEnabled(has)

    def open_ref(self, ref: str) -> bool:
        try:
            part = self.library.load_part(ref)
        except SequoiaError as exc:
            self._clear_editor()
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #b00020;")
            return False
        self.ref = ref
        self.type_name = part.type
        self.original = part
        self.draft = draft_from_part(part)
        self.attachments = list(part.attachments)
        self.is_new = False
        self.dirty = False
        self._show_header(part)
        self._rebuild_form(part.type)
        self.title.setText(ref)
        self.type_label.setText(part.type)
        self.status.setText("")
        self._set_enabled()
        return True

    def _show_header(self, part: Part | None) -> None:
        for name, widget in (
            ("manufacturer", self.manufacturer),
            ("model", self.model),
            ("serial", self.serial),
        ):
            widget.setText((getattr(part, name) if part else "") or "")
        self.description.blockSignals(True)
        self.description.setPlainText((part.description if part else "") or "")
        self.description.blockSignals(False)
        self._fill_attachments()

    def _fill_attachments(self) -> None:
        self.attach_list.clear()
        self.attach_list.addItems(self.attachments)

    def header(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "manufacturer": self.manufacturer.text().strip(),
            "model": self.model.text().strip(),
            "serial": self.serial.text().strip(),
            "description": self.description.toPlainText().strip(),
        }
        out["attachments"] = list(self.attachments)
        assert set(HEADER_FIELDS) <= set(out)
        return out

    def revert(self) -> None:
        if self.is_new:
            self._clear_editor()
        elif self.ref is not None:
            self.open_ref(self.ref)

    # ---- actions
    def new_part(self, type_name: str | None = None, ref: str | None = None) -> bool:
        if not self._discard_ok():
            return False
        types = sorted(registered_types())
        type_name = type_name if isinstance(type_name, str) else self.ask_type(types)
        if not type_name:
            return False
        ref = ref or self.ask_text(
            "New part", "Library path", suggest_ref(self.library, type_name, None)
        )
        if not ref:
            return False
        try:
            if self.library.exists(ref):
                raise PartError(f"part {ref!r} already exists")
            self.library.part_dir(ref)
        except PartError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #b00020;")
            return False
        self.ref = ref
        self.type_name = type_name
        self.original = None
        self.draft = {}
        self.attachments = []
        self.is_new = True
        self.dirty = True
        self._show_header(None)
        self._rebuild_form(type_name)
        self.title.setText(f"{ref} (new)")
        self.type_label.setText(type_name)
        self.status.setText("fill in the required parameters, then Save")
        self.status.setStyleSheet("")
        self._set_enabled()
        return True

    def save(self) -> bool:
        if self.ref is None or self.type_name is None:
            return False
        try:
            part = build_part(self.ref, self.type_name, self.header(), self.draft)
            save_edited(self.library, part, self.original)
        except SequoiaError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #b00020;")
            return False
        ref = self.ref
        self.dirty = False
        self.is_new = False
        self.refresh(select=ref)
        self.open_ref(ref)
        self.status.setText(f"saved {ref}")
        self.library_changed.emit()
        return True

    def duplicate_part(self, new_ref: str | None = None) -> bool:
        if self.ref is None or self.is_new or not self._discard_ok():
            return False
        new_ref = (
            new_ref
            if isinstance(new_ref, str)
            else self.ask_text("Duplicate part", "New library path", f"{self.ref}_copy")
        )
        if not new_ref:
            return False
        try:
            self.library.duplicate_part(self.ref, new_ref)
        except PartError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #b00020;")
            return False
        self.refresh(select=new_ref)
        self.open_ref(new_ref)
        self.library_changed.emit()
        return True

    def rename_part(self, new_ref: str | None = None) -> bool:
        if self.ref is None or self.is_new or not self._discard_ok():
            return False
        new_ref = (
            new_ref
            if isinstance(new_ref, str)
            else self.ask_text("Rename part", "New library path", self.ref)
        )
        if not new_ref or new_ref == self.ref:
            return False
        try:
            self.library.rename_part(self.ref, new_ref)
        except PartError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #b00020;")
            return False
        self.ref = new_ref
        self.refresh(select=new_ref)
        self.open_ref(new_ref)
        self.library_changed.emit()
        return True

    def delete_part(self) -> bool:
        if self.ref is None or self.is_new:
            return False
        if not self.confirm(f"Delete part {self.ref} and its data files? This cannot be undone."):
            return False
        try:
            self.library.delete_part(self.ref)
        except PartError as exc:
            self.status.setText(str(exc))
            return False
        self._clear_editor()
        self.refresh()
        self.library_changed.emit()
        return True

    def add_attachment(self, path: str | None = None) -> bool:
        if self.ref is None:
            return False
        path = path if isinstance(path, str) else self.ask_file()
        if not path:
            return False
        try:
            name = self.library.attach_file(self.ref, Path(path))
        except PartError as exc:
            self.status.setText(str(exc))
            return False
        if name not in self.attachments:
            self.attachments.append(name)
        self._fill_attachments()
        self._mark_dirty()
        return True

    def remove_attachment(self) -> None:
        row = self.attach_list.currentRow()
        if row >= 0:
            del self.attachments[row]
            self._fill_attachments()
            self._mark_dirty()
