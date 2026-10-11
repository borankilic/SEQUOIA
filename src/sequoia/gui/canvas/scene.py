from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QKeyEvent, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QLineEdit,
)

from sequoia.gui.canvas.items import CELL, SNAP_CELLS, ComponentItem, ProbeItem, snap_value
from sequoia.gui.canvas.links import LinkItem, edit_field, label_text
from sequoia.gui.document import SetupDocument

_PROBE_T = (0.25, 0.75, 0.15, 0.85, 0.35, 0.65)


class _Clipboard:
    data: dict | None = None
    pastes = 0


class _Editor(QLineEdit):
    cancelled = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)


class CanvasScene(QGraphicsScene):
    selection_changed = Signal(list)
    item_activated = Signal(str)
    status_message = Signal(str)

    def __init__(self, document: SetupDocument, parent=None) -> None:
        super().__init__(parent)
        self.document = document
        self.snap = True
        self.highlights: dict[str, str] = {}
        self.component_items: dict[str, ComponentItem] = {}
        self.link_items: dict[str, LinkItem] = {}
        self.probe_items: dict[str, ProbeItem] = {}
        self._syncing = False
        self._mouse_down = False
        self._last_selection: list[str] = []
        self._drag: dict | None = None
        self._rubber: QGraphicsLineItem | None = None
        self.link_editor: QLineEdit | None = None
        self._editor_proxy: QGraphicsProxyWidget | None = None
        self._editor_link: str | None = None
        self.selectionChanged.connect(self._on_selection)
        document.structure_changed.connect(self.rebuild)
        document.changed.connect(self._on_changed)
        self.rebuild()

    def set_snap(self, enabled: bool) -> None:
        self.snap = bool(enabled)

    def snap_cells(self, value: float) -> float:
        return snap_value(value, SNAP_CELLS) if self.snap else value

    def _instances(self):
        return {i.id: i for i in self.document.setup.instances}

    def rebuild(self) -> None:
        self.cancel_edit()
        selected = set(self.selected_ids())
        self._syncing = True
        self.blockSignals(True)
        try:
            for item in (
                *self.component_items.values(),
                *self.link_items.values(),
                *self.probe_items.values(),
            ):
                self.removeItem(item)
            self.component_items, self.link_items, self.probe_items = {}, {}, {}
            doc = self.document
            for inst in doc.setup.instances:
                item = ComponentItem(
                    self, inst, doc.ports(inst.id), doc.component(inst.id) is not None
                )
                self.component_items[inst.id] = item
                self.addItem(item)
            for link in doc.setup.links:
                item = LinkItem(self, link)
                self.link_items[link.id] = item
                self.addItem(item)
            for probe in doc.setup.probes:
                item = ProbeItem(self, probe.id, probe.part, probe.link)
                self.probe_items[probe.id] = item
                self.addItem(item)
            self._place_probes()
            for item_id, item in self._all_items().items():
                if item_id in selected:
                    item.setSelected(True)
        finally:
            self.blockSignals(False)
            self._syncing = False
        self._update_scene_rect()
        self._on_selection()

    def _all_items(self) -> dict:
        return {**self.component_items, **self.link_items, **self.probe_items}

    def _ids_match(self) -> bool:
        s = self.document.setup
        return (
            [i.id for i in s.instances] == list(self.component_items)
            and [x.id for x in s.links] == list(self.link_items)
            and [p.id for p in s.probes] == list(self.probe_items)
            and all(self.link_items[x.id].link.kind == x.kind for x in s.links)
        )

    def _on_changed(self) -> None:
        if not self._ids_match():
            return
        self._syncing = True
        try:
            doc = self.document
            for inst in doc.setup.instances:
                self.component_items[inst.id].update_from(
                    inst, doc.ports(inst.id), doc.component(inst.id) is not None
                )
            for link in doc.setup.links:
                self.link_items[link.id].update_link(link)
            self._place_probes()
        finally:
            self._syncing = False
        self._update_scene_rect()

    def _place_probes(self) -> None:
        counts: dict[str, int] = {}
        for probe in self.document.setup.probes:
            item = self.probe_items.get(probe.id)
            link = self.link_items.get(probe.link)
            if item is None or link is None:
                continue
            n = counts.get(probe.link, 0)
            counts[probe.link] = n + 1
            item.setPos(link.point_at(_PROBE_T[n % len(_PROBE_T)]))

    def _update_scene_rect(self) -> None:
        rect = self.itemsBoundingRect()
        if rect.isNull():
            rect = QRectF(-400, -300, 800, 600)
        self.setSceneRect(rect.adjusted(-3000, -2000, 3000, 2000))

    def port_anchor(self, instance_id: str, port_name: str):
        item = self.component_items.get(instance_id)
        if item is None:
            return None
        pos = item.port_scene_position(port_name)
        if pos is None:
            return None
        return pos, -item.port_inward(port_name)

    def port_scene_position(self, instance_id: str, port_name: str) -> QPointF | None:
        item = self.component_items.get(instance_id)
        return None if item is None else item.port_scene_position(port_name)

    def item_moved(self, item: ComponentItem) -> None:
        if self._syncing:
            return
        for link in self.link_items.values():
            if item.instance_id in (link.link.src.split(".")[0], link.link.dst.split(".")[0]):
                link.update_geometry()
        self._place_probes()
        if not self._mouse_down:
            self._commit_moves()

    def _commit_moves(self) -> None:
        moved = []
        pos = {i.id: i.position for i in self.document.setup.instances}
        for item_id, item in self.component_items.items():
            cells = [item.pos().x() / CELL, item.pos().y() / CELL]
            old = pos.get(item_id)
            if old is not None and (abs(old[0] - cells[0]) > 1e-9 or abs(old[1] - cells[1]) > 1e-9):
                moved.append((item_id, cells))
        if not moved:
            return
        if len(moved) == 1:
            self.document.move_instance(*moved[0])
        else:
            with self.document.macro("Move items"):
                for item_id, cells in moved:
                    self.document.move_instance(item_id, cells)
        self._update_scene_rect()

    def selected_ids(self) -> list[str]:
        return [i for i, item in self._all_items().items() if item.isSelected()]

    def _on_selection(self) -> None:
        if self._syncing:
            return
        ids = self.selected_ids()
        if ids != self._last_selection:
            self._last_selection = ids
            self.selection_changed.emit(ids)

    def select(self, item_id: str | None) -> None:
        self.select_many([] if item_id is None else [item_id])

    def select_many(self, ids) -> None:
        wanted = set(ids)
        self.blockSignals(True)
        for item_id, item in self._all_items().items():
            item.setSelected(item_id in wanted)
        self.blockSignals(False)
        self.update()
        self._on_selection()

    def set_highlight(self, ids, severity: str | None) -> None:
        for item_id in ids:
            if severity:
                self.highlights[item_id] = severity
            else:
                self.highlights.pop(item_id, None)
        self.update()

    def clear_highlights(self) -> None:
        self.highlights.clear()
        self.update()

    def add_part(self, part_ref: str, cell_position) -> str | None:
        position = [self.snap_cells(cell_position[0]), self.snap_cells(cell_position[1])]
        try:
            new_id = self.document.add_instance(part_ref, position)
        except ValueError as exc:
            self.status_message.emit(str(exc))
            return None
        self.select(new_id)
        return new_id

    def _selected_instances(self) -> list[str]:
        return [i for i in self.selected_ids() if i in self.component_items]

    def rotate_selected(self, delta: float) -> None:
        ids = self._selected_instances()
        if not ids:
            return
        instances = self._instances()
        with self.document.macro("Rotate"):
            for item_id in ids:
                self.document.rotate_instance(item_id, instances[item_id].rotation + delta)

    def flip_selected(self) -> None:
        ids = self._selected_instances()
        with self.document.macro("Flip"):
            for item_id in ids:
                self.document.flip_instance(item_id)

    def delete_selected(self) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        doc = self.document
        with doc.macro("Delete"):
            for item_id in ids:
                if item_id in self.probe_items and any(p.id == item_id for p in doc.setup.probes):
                    doc.remove_probe(item_id)
            for item_id in ids:
                if item_id in self.link_items and any(x.id == item_id for x in doc.setup.links):
                    doc.remove_link(item_id)
            for item_id in ids:
                if item_id in self.component_items and any(
                    i.id == item_id for i in doc.setup.instances
                ):
                    doc.remove_instance(item_id)

    def copy_selection(self) -> bool:
        names = set(self._selected_instances())
        if not names:
            return False
        s = self.document.setup
        links = [
            x for x in s.links if x.src.split(".")[0] in names and x.dst.split(".")[0] in names
        ]
        link_ids = {x.id for x in links}
        _Clipboard.data = {
            "instances": [i.model_dump(by_alias=True) for i in s.instances if i.id in names],
            "links": [x.model_dump(by_alias=True) for x in links],
            "probes": [p.model_dump(by_alias=True) for p in s.probes if p.link in link_ids],
        }
        _Clipboard.pastes = 0
        return True

    def paste(self) -> list[str]:
        data = _Clipboard.data
        if not data:
            return []
        _Clipboard.pastes += 1
        offset = float(_Clipboard.pastes)
        doc = self.document
        mapping: dict[str, str] = {}
        new_ids: list[str] = []
        try:
            with doc.macro("Paste"):
                for raw in data["instances"]:
                    new_id = doc.new_id(raw["id"])
                    mapping[raw["id"]] = new_id
                    pos = [raw["position"][0] + offset, raw["position"][1] + offset]
                    doc.add_instance(raw["part"], pos, raw["rotation"], instance_id=new_id)
                    if raw["flipped"]:
                        doc.flip_instance(new_id)
                    for name, value in raw["settings"].items():
                        doc.set_setting(new_id, name, value)
                    for name, value in raw["overrides"].items():
                        doc.set_override(new_id, name, value)
                    new_ids.append(new_id)
                for raw in data["links"]:
                    fields = {
                        k: v
                        for k, v in raw.items()
                        if k not in ("id", "kind", "from", "to") and v is not None
                    }
                    src = self._remap(raw["from"], mapping)
                    dst = self._remap(raw["to"], mapping)
                    new_id = doc.add_link(
                        raw["kind"], src, dst, link_id=doc.new_id(raw["id"]), **fields
                    )
                    mapping[raw["id"]] = new_id
                    new_ids.append(new_id)
                for raw in data["probes"]:
                    new_ids.append(
                        doc.add_probe(raw["part"], mapping[raw["link"]], doc.new_id(raw["id"]))
                    )
        except ValueError as exc:
            self.status_message.emit(str(exc))
            return []
        self.select_many(new_ids)
        return new_ids

    @staticmethod
    def _remap(endpoint: str, mapping: dict[str, str]) -> str:
        inst, port = endpoint.split(".")
        return f"{mapping[inst]}.{port}"

    def apply_layout(self, positions: dict[str, list[float]]) -> None:
        with self.document.macro("Auto layout"):
            for item_id, pos in positions.items():
                self.document.move_instance(item_id, pos)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.focusItem() is not None or self.link_editor is not None:
            super().keyPressEvent(event)
            return
        key, mods = event.key(), event.modifiers()
        ctrl = bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
        if key == Qt.Key_R and not ctrl:
            self.rotate_selected(-90.0 if mods & Qt.ShiftModifier else 90.0)
        elif key == Qt.Key_F and not ctrl:
            self.flip_selected()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif key == Qt.Key_C and ctrl:
            self.copy_selection()
        elif key == Qt.Key_V and ctrl:
            self.paste()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _port_hit(self, scene_pos: QPointF):
        for item in reversed(list(self.component_items.values())):
            name = item.port_at(scene_pos)
            if name is not None:
                return item, name
        return None

    def start_link(self, item: ComponentItem, port: str, scene_pos: QPointF) -> bool:
        endpoint = f"{item.instance_id}.{port}"
        for link in self.document.setup.links:
            if endpoint in (link.src, link.dst):
                self.status_message.emit(
                    f"port '{endpoint}' is already connected by link '{link.id}'"
                )
                return False
        start = item.port_scene_position(port)
        line = QGraphicsLineItem(start.x(), start.y(), scene_pos.x(), scene_pos.y())
        line.setPen(QPen(Qt.gray, 1.5, Qt.DashLine))
        line.setZValue(10)
        self.addItem(line)
        self._rubber = line
        self._drag = {"endpoint": endpoint, "start": start, "target": None, "kind": None}
        return True

    def update_link_drag(self, scene_pos: QPointF) -> None:
        drag = self._drag
        if drag is None:
            return
        self._rubber.setLine(drag["start"].x(), drag["start"].y(), scene_pos.x(), scene_pos.y())
        self._clear_port_states()
        drag["target"], drag["kind"] = None, None
        hit = self._port_hit(scene_pos)
        if hit is None:
            self.status_message.emit("")
            return
        item, name = hit
        target = f"{item.instance_id}.{name}"
        if target == drag["endpoint"]:
            return
        kind, reason = self.document.can_connect(drag["endpoint"], target)
        if kind is None:
            item.set_port_state(name, "invalid")
            self.status_message.emit(reason)
        else:
            item.set_port_state(name, "valid")
            drag["target"], drag["kind"] = target, kind
            self.status_message.emit(f"connect {drag['endpoint']} to {target} ({kind})")

    def _clear_port_states(self) -> None:
        for item in self.component_items.values():
            item.set_port_state(None, None)

    def finish_link(self, scene_pos: QPointF) -> str | None:
        drag = self._drag
        if drag is None:
            return None
        self.update_link_drag(scene_pos)
        target, kind = drag["target"], drag["kind"]
        self._clear_port_states()
        self.removeItem(self._rubber)
        self._rubber, self._drag = None, None
        if target is None:
            return None
        src, dst = self.document.normalize_connection(drag["endpoint"], target)
        try:
            return self.document.add_link(kind, src, dst)
        except ValueError as exc:
            self.status_message.emit(str(exc))
            return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.link_editor is None:
            hit = self._port_hit(event.scenePos())
            if hit is not None and self.start_link(hit[0], hit[1], event.scenePos()):
                event.accept()
                return
        self._mouse_down = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            self.update_link_drag(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None:
            self.finish_link(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._mouse_down = False
        self._commit_moves()

    def begin_edit_link(self, link_id: str) -> None:
        item = self.link_items.get(link_id)
        if item is None:
            return
        self.cancel_edit()
        editor = _Editor(label_text(self.document, item.link)[0])
        editor.setFixedWidth(100)
        editor.editingFinished.connect(self._commit_edit)
        editor.cancelled.connect(self.cancel_edit)
        proxy = self.addWidget(editor)
        proxy.setZValue(20)
        proxy.setPos(item.label.pos() - QPointF(50, editor.sizeHint().height() / 2))
        self.link_editor, self._editor_proxy, self._editor_link = editor, proxy, link_id
        editor.setFocus()
        editor.selectAll()

    def _commit_edit(self) -> None:
        editor, link_id = self.link_editor, self._editor_link
        if editor is None:
            return
        text = editor.text().strip()
        self.cancel_edit()
        link = self.link_items.get(link_id)
        if link is None or not text:
            return
        doc = self.document
        try:
            with doc.macro(f"Edit {link_id}"):
                field = edit_field(link.link)
                doc.set_link_field(link_id, field, text)
                if link.link.kind == "free_space" and link.link.lock_to_canvas:
                    doc.set_link_field(link_id, "lock_to_canvas", False)
        except ValueError as exc:
            self.status_message.emit(str(exc))

    def cancel_edit(self) -> None:
        proxy = self._editor_proxy
        self.link_editor, self._editor_proxy, self._editor_link = None, None, None
        if proxy is not None:
            proxy.widget().hide()
            if proxy.scene() is self:
                self.removeItem(proxy)
            proxy.deleteLater()
