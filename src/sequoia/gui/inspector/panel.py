from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sequoia.components.base import get_component
from sequoia.data import Length
from sequoia.errors import SequoiaError
from sequoia.gui.document import MAX_ABS_CELLS, SetupDocument
from sequoia.gui.inspector.analyses import AnalysesEditor
from sequoia.gui.inspector.fields import (
    MISSING,
    STYLE,
    CollapsibleGroup,
    Editor,
    FieldRow,
    FieldSpec,
    LiteralEditor,
    ModelForm,
    NumberEditor,
    ScalarEditor,
    StrEditor,
    analyze,
    error_message,
    make_editor,
)
from sequoia.setup.model import CableLink, FiberLink, FiberPolarization, FreeSpaceLink
from sequoia.setup.validate import FIBER_TYPE

LINK_MODELS = {"free_space": FreeSpaceLink, "fiber": FiberLink, "cable": CableLink}
_LINK_SKIP = ("id", "kind", "src", "dst")
MEDIA = ("air", "vacuum")
LINK_LABELS = {
    "length": "Length",
    "lock_to_canvas": "Lock length to canvas distance",
    "medium": "Medium",
    "fiber": "Fiber type",
    "connector_loss_db": "Loss (dB)",
    "polarization": "Polarization transform",
    "delay": "Cable delay",
}
LINK_HINTS = {
    "length": "Physical path length of the link, e.g. 100 mm or 0.5 m.",
    "lock_to_canvas": "Use the distance between the two ports on the canvas "
    "(one grid cell is the setup's grid cell size) instead of a typed length.",
    "medium": "Propagation medium of the free-space path.",
    "fiber": "Library fiber part this link is made of.",
    "connector_loss_db": "Extra loss of this fiber link in dB (connectors, splices, bends); "
    "0 by default. Fiber attenuation per length comes from the fiber part.",
    "polarization": "Fixed polarization rotation of the fiber.",
    "delay": "Signal delay of the cable, e.g. 5 ns.",
}
LINK_EXAMPLES = {"length": "100 mm", "delay": "5 ns"}


def _label(text: str, role: str = "", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if role:
        label.setProperty("role", role)
    return label


class PolarizationEditor(Editor):
    def __init__(self, spec: FieldSpec | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QComboBox, QLineEdit

        self.kind = QComboBox()
        self.kind.addItems(["identity", "su2", "random"])
        self.vector = [QLineEdit() for _ in range(3)]
        for i, edit in enumerate(self.vector):
            edit.setPlaceholderText("xyz"[i] + " (rad)")
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("seed (integer)")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.kind)
        row = QHBoxLayout()
        for edit in self.vector:
            row.addWidget(edit)
        self.vector_box = QWidget()
        self.vector_box.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vector_box)
        layout.addWidget(self.seed)
        self._loading = False
        self.kind.currentIndexChanged.connect(self._kind_changed)
        for edit in (*self.vector, self.seed):
            edit.editingFinished.connect(self.commit)
        self._update_visibility()

    def _update_visibility(self) -> None:
        kind = self.kind.currentText()
        self.vector_box.setVisible(kind == "su2")
        self.seed.setVisible(kind == "random")

    def _kind_changed(self) -> None:
        self._update_visibility()
        if not self._loading:
            self.commit()

    def collect(self) -> tuple[dict | None, str]:
        kind = self.kind.currentText()
        out: dict[str, Any] = {"kind": kind}
        try:
            if kind == "su2":
                texts = [e.text().strip() for e in self.vector]
                if not all(texts):
                    return None, "enter the three rotation vector components (radians)"
                out["rotation_vector"] = [float(t) for t in texts]
            elif kind == "random":
                text = self.seed.text().strip()
                if not text:
                    return None, "enter an integer seed so results are reproducible"
                out["seed"] = int(text)
        except ValueError:
            return None, "rotation components must be numbers and the seed an integer"
        try:
            FiberPolarization.model_validate(out)
        except ValidationError as exc:
            return None, error_message(exc)
        return out, ""

    def commit(self) -> None:
        if self._loading:
            return
        value, message = self.collect()
        self.problem.emit(message)
        if value is not None:
            self.committed.emit(value)

    def set_raw(self, raw: Any) -> None:
        raw = raw if isinstance(raw, dict) else {}
        self._loading = True
        self.kind.setCurrentText(raw.get("kind", "identity"))
        vec = raw.get("rotation_vector") or ["", "", ""]
        for edit, v in zip(self.vector, vec, strict=False):
            edit.setText(str(v))
        self.seed.setText("" if raw.get("seed") is None else str(raw["seed"]))
        self._loading = False
        self._update_visibility()
        self.problem.emit("")


class _TextEdit(QPlainTextEdit):
    editingFinished = Signal()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.editingFinished.emit()


class MultilineEditor(Editor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = _TextEdit()
        self.text.setMinimumHeight(110)
        self.text.setMaximumHeight(240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text)
        self._shown = ""
        self.text.editingFinished.connect(self.commit)

    def commit(self) -> None:
        text = self.text.toPlainText()
        if text != self._shown:
            self.committed.emit(text)

    def set_raw(self, raw: Any) -> None:
        text = "" if raw in (None, MISSING) else str(raw)
        self._shown = text
        if text != self.text.toPlainText():
            self.text.setPlainText(text)


def _placed_label(placed) -> str:
    return placed.part


class PartHeader(QWidget):
    def __init__(self, document: SetupDocument, item_id: str, part_ref: str, kind: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.part = None
        self.error: str | None = None
        try:
            self.part = document.library.load_part(part_ref)
        except SequoiaError as exc:
            self.error = str(exc)
        if self.part is not None:
            title = " ".join(p for p in (self.part.manufacturer, self.part.model) if p)
            title = title or part_ref.rsplit("/", 1)[-1]
            head = _label(f"<b>{title}</b>")
            layout.addWidget(head)
            layout.addWidget(_label(f"{self.part.type} ({kind}) - {part_ref}", "muted"))
            if self.part.serial:
                layout.addWidget(_label(f"serial {self.part.serial}", "muted"))
            if self.part.description:
                layout.addWidget(_label(self.part.description))
        else:
            layout.addWidget(_label(f"<b>{part_ref}</b>"))
            layout.addWidget(_label(self.error or "", "error"))
        layout.addWidget(_label(f"id: {item_id}", "muted"))
        self.issue = _label("", "error")
        self.issue.setVisible(False)
        layout.addWidget(self.issue)

    def set_issue(self, message: str | None) -> None:
        self.issue.setText(message or "")
        self.issue.setVisible(bool(message))


class InstanceForm(QWidget):
    def __init__(self, document: SetupDocument, item_id: str, tag: str) -> None:
        super().__init__()
        self.document = document
        self.item_id = item_id
        self.tag = tag
        placed = self._placed()
        self.header = PartHeader(document, item_id, placed.part, tag)
        self.settings_form: ModelForm | None = None
        self.overrides_form: ModelForm | None = None
        self.overrides_group: CollapsibleGroup | None = None
        self.placement: dict[str, Any] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.header)
        spec = FieldSpec("str")
        self.id_row = FieldRow(
            "id",
            spec,
            StrEditor(spec),
            self._rename,
            description="The identifier used in links, probes and analyses; renaming updates them.",
            track_default=False,
        )
        layout.addWidget(self.id_row)
        if tag == "i":
            layout.addWidget(self._build_placement())
        part = self.header.part
        if part is not None:
            cls = get_component(part.type)
            settings_cls = cls.Settings
            if settings_cls.model_fields:
                layout.addWidget(_label("<b>Settings</b>"))
                self.settings_form = ModelForm(settings_cls, self._set_setting)
                layout.addWidget(self.settings_form)
            else:
                layout.addWidget(_label("This component has no settings.", "muted"))
            params_cls = cls.Params
            if params_cls.model_fields:
                self.overrides_form = ModelForm(params_cls, self._set_override, override_mode=True)
                self.overrides_group = CollapsibleGroup("Overrides", self.overrides_form)
                layout.addWidget(self.overrides_group)
        layout.addStretch(1)
        self.refresh()

    def _placed(self):
        s = self.document.setup
        for item in (*s.instances, *s.probes):
            if item.id == self.item_id:
                return item
        raise KeyError(self.item_id)

    def _rename(self, _name: str, value: Any) -> None:
        new_id = str(value).strip()
        if new_id and new_id != self.item_id:
            self.document.rename_instance(self.item_id, new_id)

    def _set_setting(self, name: str, value: Any) -> None:
        self.document.set_setting(self.item_id, name, value)

    def _set_override(self, name: str, value: Any) -> None:
        self.document.set_override(self.item_id, name, value)

    def _build_placement(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        doc, iid = self.document, self.item_id

        def float_spec(minimum=None, maximum=None) -> FieldSpec:
            return FieldSpec("float", minimum=minimum, maximum=maximum)

        def commit_pos(axis: int):
            def commit(_name, value):
                inst = self._placed()
                pos = list(inst.position)
                pos[axis] = float(value)
                doc.move_instance(iid, pos)

            return commit

        def commit_rot(_name, value):
            doc.rotate_instance(iid, float(value))

        def commit_flip(_name, value):
            if bool(value) != self._placed().flipped:
                doc.flip_instance(iid)

        rows = {
            "x": FieldRow(
                "x (cells)",
                float_spec(-MAX_ABS_CELLS, MAX_ABS_CELLS),
                NumberEditor(float_spec(-MAX_ABS_CELLS, MAX_ABS_CELLS)),
                commit_pos(0),
                track_default=False,
            ),
            "y": FieldRow(
                "y (cells)",
                float_spec(-MAX_ABS_CELLS, MAX_ABS_CELLS),
                NumberEditor(float_spec(-MAX_ABS_CELLS, MAX_ABS_CELLS)),
                commit_pos(1),
                track_default=False,
            ),
            "rotation": FieldRow(
                "rotation (deg)",
                float_spec(0.0, 360.0),
                NumberEditor(float_spec(0.0, 360.0)),
                commit_rot,
                track_default=False,
            ),
            "flipped": FieldRow(
                "flipped",
                FieldSpec("bool"),
                make_editor(FieldSpec("bool")),
                commit_flip,
                track_default=False,
            ),
        }
        rows["rotation"].editor.spin.setSingleStep(15.0)
        self.placement = rows
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, 23)
        self.rotation_slider.setPageStep(6)
        self.rotation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rotation_slider.setTickInterval(6)
        self.rotation_slider.setToolTip("Rotation in 15 degree steps")
        self._slider_loading = False

        def slide(value: int) -> None:
            if self._slider_loading:
                return
            try:
                doc.rotate_instance(iid, float(value * 15))
            except ValueError:
                pass

        self.rotation_slider.valueChanged.connect(slide)
        layout.addWidget(_label("<b>Placement</b>"))
        for name, row in rows.items():
            layout.addWidget(row)
            if name == "rotation":
                layout.addWidget(self.rotation_slider)
        return box

    def refresh(self) -> None:
        placed = self._placed()
        self.id_row.refresh(placed.id)
        if self.tag == "i":
            x, y = placed.position
            self.placement["x"].refresh(x)
            self.placement["y"].refresh(y)
            self.placement["rotation"].refresh(placed.rotation)
            self.placement["flipped"].refresh(placed.flipped)
            self._slider_loading = True
            self.rotation_slider.setValue(int(round(placed.rotation / 15.0)) % 24)
            self._slider_loading = False
        error = self.document.component_error(self.item_id)
        self.header.set_issue(f"Invalid: {error}" if error else None)
        part = self.header.part
        if part is None:
            return
        if self.settings_form is not None:
            self.settings_form.refresh(dict(placed.settings))
        if self.overrides_form is not None:
            params = {n: getattr(part.params, n) for n in type(part.params).model_fields}
            self.overrides_form.refresh(dict(placed.overrides), params)
            count = len(placed.overrides)
            self.overrides_group.set_title(f"Overrides ({count})" if count else "Overrides")


class LinkForm(QWidget):
    def __init__(self, document: SetupDocument, link_id: str) -> None:
        super().__init__()
        self.document = document
        self.link_id = link_id
        link = self._link()
        self.link_kind = link.kind
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_label(f"<b>Link {link.id}</b>"))
        layout.addWidget(_label(f"{link.kind}: {link.src} -> {link.dst}", "muted"))
        id_spec = FieldSpec("str")
        self.id_row = FieldRow(
            "id",
            id_spec,
            StrEditor(id_spec),
            self._rename,
            description="The identifier used by probes and diagnostics; renaming updates them.",
            track_default=False,
        )
        layout.addWidget(self.id_row)
        self.medium_combo = None
        if link.kind in ("free_space", "fiber"):
            self.medium_combo = QComboBox()
            self.medium_combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self.medium_combo.setMinimumContentsLength(12)
            self.medium_combo.addItem("Air (free space)", "free_space")
            self.medium_combo.addItem("Fiber (inserts launch couplers)", "fiber")
            self.medium_combo.setCurrentIndex(0 if link.kind == "free_space" else 1)
            self.medium_combo.setToolTip(
                "Carry the beam through air or through an optical fiber. Switching an air link "
                "to fiber inserts a launch and a receive fiber coupler; switching back removes "
                "them again."
            )
            self.medium_combo.activated.connect(self._medium_chosen)
            row = QHBoxLayout()
            row.addWidget(QLabel("Link type"))
            row.addWidget(self.medium_combo, 1)
            layout.addLayout(row)
        editors: dict = {}
        specs: dict = {}
        self.fiber_editor: LiteralEditor | None = None
        if link.kind == "fiber":
            specs["fiber"] = FieldSpec("literal", literals=())
            specs["polarization"] = FieldSpec("json")
            editors["fiber"] = self._fiber_editor
            editors["polarization"] = lambda spec: PolarizationEditor(spec)
        if link.kind == "free_space":
            specs["medium"] = FieldSpec("literal", literals=MEDIA)
        self.form = ModelForm(
            LINK_MODELS[link.kind],
            self._commit,
            track_defaults=False,
            skip=_LINK_SKIP,
            editors=editors,
            specs=specs,
            labels=LINK_LABELS,
            descriptions=LINK_HINTS,
            required=("length", "delay", "fiber"),
        )
        for name, text in LINK_EXAMPLES.items():
            row = self.form.rows.get(name)
            if row is not None and isinstance(row.editor, ScalarEditor):
                row.editor.force_example(text)
        layout.addWidget(self.form)
        layout.addStretch(1)
        self.refresh()

    def _medium_chosen(self, index: int) -> None:
        wanted = self.medium_combo.itemData(index)
        if wanted == self._link().kind:
            return
        try:
            if wanted == "fiber":
                new_id = self.document.convert_link_to_fiber(self.link_id)
            else:
                new_id = self.document.convert_link_to_air(self.link_id)
            self.document.selection_requested.emit(new_id)
        except ValueError as exc:
            self.medium_combo.setCurrentIndex(0 if self._link().kind == "free_space" else 1)
            QMessageBox.warning(self, "Link type", str(exc))

    _FIBER_KINDS = {"smf": "SM", "pm": "PM", "mmf": "MM"}

    def _fiber_editor(self, spec: FieldSpec) -> LiteralEditor:
        editor = LiteralEditor(spec)
        self.fiber_editor = editor
        items = []
        for info in self.document.library.index():
            if info.type == FIBER_TYPE:
                kind = self.document._fiber_kind(info.ref)
                tag = self._FIBER_KINDS.get(kind, "?")
                name = " ".join(p for p in (info.manufacturer, info.model) if p)
                items.append((f"[{tag}] {name} ({info.ref})" if name else info.ref, info.ref))
        items.sort(key=lambda x: x[0])
        editor.set_choices(items)
        return editor

    def _link(self):
        for link in self.document.setup.links:
            if link.id == self.link_id:
                return link
        raise KeyError(self.link_id)

    def _commit(self, name: str, value: Any) -> None:
        self.document.set_link_field(self.link_id, name, value)

    def _rename(self, _name: str, value: Any) -> None:
        new_id = str(value).strip()
        if new_id and new_id != self.link_id:
            self.document.rename_link(self.link_id, new_id)

    def refresh(self) -> None:
        self.id_row.refresh(self.link_id)
        raw = self._link().model_dump(by_alias=True)
        explicit = {k: v for k, v in raw.items() if k not in _LINK_SKIP}
        if "polarization" in explicit and isinstance(explicit["polarization"], dict):
            explicit["polarization"] = {
                k: v for k, v in explicit["polarization"].items() if v is not None
            }
        if self.fiber_editor is not None:
            ref = explicit.get("fiber")
            if self.fiber_editor.combo.findData(ref) < 0 and ref:
                items = [
                    (self.fiber_editor.combo.itemText(i), self.fiber_editor.combo.itemData(i))
                    for i in range(self.fiber_editor.combo.count())
                ]
                self.fiber_editor.set_choices([*items, (f"{ref} (not in library)", ref)])
        self.form.refresh(
            {k: (MISSING if v is None else v) for k, v in explicit.items()},
            {k: MISSING for k in self.form.rows},
        )


class SetupForm(QWidget):
    def __init__(self, document: SetupDocument) -> None:
        super().__init__()
        self.document = document
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_label("<b>Setup</b>"))
        spec = FieldSpec("str")
        self.rows: dict[str, FieldRow] = {
            "name": FieldRow("name", spec, make_editor(spec), self._name, track_default=False),
            "description": FieldRow(
                "description", spec, MultilineEditor(), self._description, track_default=False
            ),
            "grid_cell": FieldRow(
                "grid cell",
                analyze(Length),
                make_editor(analyze(Length)),
                self._grid,
                description="physical size of one canvas grid cell (used by lock-to-canvas links)",
                track_default=False,
            ),
        }
        for row in self.rows.values():
            layout.addWidget(row)
        layout.addWidget(_label("<b>Analyses</b>"))
        self.analyses = AnalysesEditor(document)
        layout.addWidget(self.analyses)
        self.summary = _label("", "muted")
        layout.addWidget(self.summary)
        layout.addStretch(1)
        self.refresh()

    def _name(self, _n: str, value: Any) -> None:
        self.document.set_name(str(value))

    def _description(self, _n: str, value: Any) -> None:
        self.document.set_description(str(value))

    def _grid(self, _n: str, value: Any) -> None:
        self.document.set_grid_cell(value)

    def refresh(self) -> None:
        s = self.document.setup
        self.rows["name"].refresh(s.name)
        self.rows["description"].refresh(s.description)
        self.rows["grid_cell"].refresh(s.grid_cell.model_dump())
        if self.analyses.form is None:
            self.analyses.refresh()
        self.summary.setText(
            f"{len(s.instances)} instances, {len(s.links)} links, {len(s.probes)} probes"
        )


class InspectorPanel(QWidget):
    def __init__(self, document: SetupDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document: SetupDocument | None = None
        self._kind: str | None = None
        self._id: str | None = None
        self._signature: tuple | None = None
        self.form: QWidget | None = None
        self.setStyleSheet(STYLE)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll)
        self.set_document(document)

    def set_document(self, document: SetupDocument) -> None:
        if self.document is not None:
            self.document.changed.disconnect(self._on_changed)
        self.document = document
        document.changed.connect(self._on_changed)
        self._kind = self._id = None
        self._rebuild()

    def selection(self) -> tuple[str | None, str | None]:
        return self._kind, self._id

    def show_selection(self, kind: str | None, item_id: str | None = None) -> None:
        if kind in ("component",):
            kind = "instance"
        self._kind, self._id = (kind, item_id) if kind and item_id else (None, None)
        self._rebuild()

    def reload(self) -> None:
        self._rebuild()

    def _current_signature(self) -> tuple | None:
        s = self.document.setup
        if self._kind in ("instance", "probe"):
            for item in (*s.instances, *s.probes):
                if item.id == self._id:
                    return (self._kind, item.id, item.part)
            return None
        if self._kind == "link":
            for link in s.links:
                if link.id == self._id:
                    return ("link", link.id, link.kind)
            return None
        return ("setup",)

    def _rebuild(self) -> None:
        sig = self._current_signature()
        if sig is None:
            self._kind = self._id = None
            sig = ("setup",)
        self._signature = sig
        if sig[0] == "setup":
            form: QWidget = SetupForm(self.document)
        elif sig[0] == "link":
            form = LinkForm(self.document, sig[1])
        else:
            tag = "i" if sig[0] == "instance" else "p"
            if tag == "i" and not any(i.id == sig[1] for i in self.document.setup.instances):
                tag = "p"
            form = InstanceForm(self.document, sig[1], tag)
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(8, 8, 8, 8)
        box.addWidget(form)
        self.form = form
        previous = self.scroll.takeWidget()
        self.scroll.setWidget(holder)
        if previous is not None:
            previous.hide()
            previous.deleteLater()

    def _on_changed(self) -> None:
        sig = self._current_signature()
        if sig is None:
            self._kind = self._id = None
            sig = ("setup",)
        if sig != self._signature:
            self._rebuild()
        elif self.form is not None:
            self.form.refresh()
