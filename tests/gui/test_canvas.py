from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QKeyEvent
from PySide6.QtTest import QTest

from sequoia.gui.canvas import PART_MIME, CanvasScene, CanvasView, auto_layout
from sequoia.gui.canvas.items import CELL
from sequoia.gui.document import SetupDocument

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-borankilic-Desktop-Research-Quantum-Technologies-SEQUOIA/"
    "5d0a0d99-6543-4fe7-aef3-c7336f569617/scratchpad"
)


@pytest.fixture
def scene(doc, qtbot):
    s = CanvasScene(doc)
    yield s
    doc.undo_stack.cleanChanged.disconnect()


@pytest.fixture
def view(scene, qtbot):
    v = CanvasView(scene)
    qtbot.addWidget(v)
    v.resize(800, 600)
    v.show()
    return v


def chain(doc):
    doc.add_instance("sources/laser", [0, 0], instance_id="laser")
    doc.add_instance("freespace/mirror", [4, 0], instance_id="mirror")
    doc.add_instance("fiber/coupler", [8, 0], instance_id="coupler")
    doc.add_instance("detectors/spad", [12, 0], instance_id="det")


def key(scene, k, mods=Qt.NoModifier):
    ev = QKeyEvent(QEvent.KeyPress, k, mods)
    scene.keyPressEvent(ev)


def test_items_follow_document(doc, scene):
    chain(doc)
    doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    assert set(scene.component_items) == {"laser", "mirror", "coupler", "det"}
    assert set(scene.link_items) == {"link"}
    item = scene.component_items["mirror"]
    assert item.pos() == QPointF(4 * CELL, 0)


def test_port_positions_rotation_flip(doc, scene):
    doc.add_instance("freespace/mirror", [2, 3], instance_id="m")
    item = scene.component_items["m"]
    w = item._w
    p_in = scene.port_scene_position("m", "in")
    p_out = scene.port_scene_position("m", "out")
    assert p_in.x() == pytest.approx(2 * CELL - w / 2)
    assert p_out.x() == pytest.approx(2 * CELL + w / 2)
    assert p_in.y() == pytest.approx(3 * CELL)
    doc.rotate_instance("m", 90)
    assert scene.port_scene_position("m", "in").x() == pytest.approx(2 * CELL)
    assert scene.port_scene_position("m", "in").y() == pytest.approx(3 * CELL + w / 2)
    assert scene.port_scene_position("m", "out").y() == pytest.approx(3 * CELL - w / 2)
    doc.flip_instance("m")
    assert scene.port_scene_position("m", "in").y() == pytest.approx(3 * CELL - w / 2)
    doc.rotate_instance("m", 0)
    assert scene.port_scene_position("m", "in").x() == pytest.approx(2 * CELL + w / 2)


def test_ports_spacing_for_multiport(doc, scene):
    doc.add_instance("freespace/bs", [0, 0], instance_id="bs")
    ys = [scene.port_scene_position("bs", n).y() for n in ("in_a", "in_b")]
    assert ys[1] - ys[0] == pytest.approx(scene.component_items["bs"]._h / 3)
    assert scene.port_scene_position("bs", "out_t").y() == pytest.approx(ys[0])


def test_link_tool_creates_and_rejects(doc, scene):
    chain(doc)
    messages = []
    scene.status_message.connect(messages.append)
    start = scene.port_scene_position("laser", "out")
    item = scene.component_items["laser"]
    assert scene.start_link(item, "out", start)
    scene.update_link_drag(scene.port_scene_position("mirror", "in"))
    assert messages[-1].startswith("connect")
    assert scene.component_items["mirror"]._port_states["in"] == "valid"
    link_id = scene.finish_link(scene.port_scene_position("mirror", "in"))
    assert doc.setup.links[0].id == link_id
    assert doc.setup.links[0].src == "laser.out"

    assert scene.start_link(scene.component_items["mirror"], "out", QPointF())
    scene.update_link_drag(scene.port_scene_position("det", "in"))
    assert scene.component_items["det"]._port_states["in"] == "invalid"
    assert "free_space" in messages[-1]
    assert scene.finish_link(scene.port_scene_position("det", "in")) is None
    assert scene.start_link(scene.component_items["mirror"], "out", QPointF())
    assert scene.finish_link(QPointF(500, 500)) is None
    assert len(doc.setup.links) == 1


def test_link_tool_reverse_direction_normalised(doc, scene):
    chain(doc)
    assert scene.start_link(scene.component_items["mirror"], "in", QPointF())
    scene.finish_link(scene.port_scene_position("laser", "out"))
    link = doc.setup.links[0]
    assert (link.src, link.dst) == ("laser.out", "mirror.in")


def test_link_tool_via_mouse_events(doc, scene, view):
    chain(doc)
    a = view.mapFromScene(scene.port_scene_position("laser", "out"))
    b = view.mapFromScene(scene.port_scene_position("mirror", "in"))
    vp = view.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, a)
    QTest.mouseMove(vp, b)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, b)
    assert len(doc.setup.links) == 1


def test_undo_restores_scene(doc, scene):
    chain(doc)
    doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    doc.undo_stack.undo()
    assert not scene.link_items
    doc.undo_stack.undo()
    assert "det" not in scene.component_items
    doc.undo_stack.redo()
    assert "det" in scene.component_items


def test_snapping_and_move_writeback(doc, scene):
    doc.add_instance("freespace/mirror", [0, 0], instance_id="m")
    item = scene.component_items["m"]
    item.setPos(QPointF(3.3 * CELL, 1.1 * CELL))
    assert item.pos() == QPointF(3.5 * CELL, 1.0 * CELL)
    assert doc.setup.instances[0].position == [3.5, 1.0]
    scene.set_snap(False)
    item.setPos(QPointF(3.3 * CELL, 1.1 * CELL))
    assert doc.setup.instances[0].position == pytest.approx([3.3, 1.1])


def test_moves_merge_in_undo_stack(doc, scene):
    doc.add_instance("freespace/mirror", [0, 0], instance_id="m")
    n = doc.undo_stack.count()
    item = scene.component_items["m"]
    item.setPos(QPointF(40, 0))
    item.setPos(QPointF(80, 0))
    assert doc.undo_stack.count() == n + 1
    doc.undo_stack.undo()
    assert item.pos() == QPointF(0, 0)
    assert doc.setup.instances[0].position == [0.0, 0.0]


def test_drag_with_mouse_moves_once(doc, scene, view):
    doc.add_instance("freespace/mirror", [0, 0], instance_id="m")
    n = doc.undo_stack.count()
    vp = view.viewport()
    start = view.mapFromScene(QPointF(0, 0))
    end = view.mapFromScene(QPointF(3 * CELL, 2 * CELL))
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(vp, QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
    QTest.mouseMove(vp, end)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end)
    assert doc.setup.instances[0].position == pytest.approx([3.0, 2.0], abs=0.5)
    assert doc.undo_stack.count() == n + 1


def test_keyboard_rotate_flip_delete(doc, scene):
    chain(doc)
    doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    scene.select("mirror")
    key(scene, Qt.Key_R)
    assert doc.setup.instances[1].rotation == 90.0
    key(scene, Qt.Key_R, Qt.ShiftModifier)
    key(scene, Qt.Key_R, Qt.ShiftModifier)
    assert doc.setup.instances[1].rotation == 270.0
    key(scene, Qt.Key_F)
    assert doc.setup.instances[1].flipped
    key(scene, Qt.Key_Delete)
    assert [i.id for i in doc.setup.instances] == ["laser", "coupler", "det"]
    assert not doc.setup.links
    doc.undo_stack.undo()
    assert "mirror" in scene.component_items and scene.link_items


def test_delete_link_and_selection_signal(doc, scene):
    chain(doc)
    link = doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    seen = []
    scene.selection_changed.connect(seen.append)
    scene.select(link)
    assert seen[-1] == [link]
    key(scene, Qt.Key_Delete)
    assert not doc.setup.links
    assert seen[-1] == []


def test_copy_paste(doc, scene):
    chain(doc)
    doc.add_link("free_space", "laser.out", "mirror.in", length="150 mm")
    doc.set_setting("mirror", "angle", "45 deg")
    scene.select_many(["laser", "mirror"])
    key(scene, Qt.Key_C, Qt.ControlModifier)
    n = doc.undo_stack.count()
    key(scene, Qt.Key_V, Qt.ControlModifier)
    assert doc.undo_stack.count() == n + 1
    ids = [i.id for i in doc.setup.instances]
    assert ids == ["laser", "mirror", "coupler", "det", "laser_2", "mirror_2"]
    new = doc.setup.instances[-1]
    assert new.position == [5.0, 1.0] and new.settings["angle"] == "45 deg"
    assert len(doc.setup.links) == 2
    assert doc.setup.links[1].src == "laser_2.out"
    assert scene.selected_ids() == ["laser_2", "mirror_2", "link_2"]
    doc.undo_stack.undo()
    assert len(doc.setup.instances) == 4


def test_drop_part(doc, scene, view):
    mime = QMimeData()
    mime.setData(PART_MIME, b"freespace/mirror")
    pos = view.mapFromScene(QPointF(5.2 * CELL, 2.9 * CELL))
    ev = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    view.dropEvent(ev)
    assert doc.setup.instances[0].part == "freespace/mirror"
    assert doc.setup.instances[0].position == pytest.approx([5.0, 3.0])
    assert scene.selected_ids() == ["mirror"]
    bad = QMimeData()
    bad.setData(PART_MIME, b"nope/none")
    messages = []
    scene.status_message.connect(messages.append)
    view.dropEvent(QDropEvent(QPointF(pos), Qt.CopyAction, bad, Qt.LeftButton, Qt.NoModifier))
    assert len(doc.setup.instances) == 1 and messages


def test_labels_and_inline_edit(doc, scene):
    chain(doc)
    a = doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    f = doc.add_link("fiber", "coupler.out", "det.in", length="2 m")
    assert scene.link_items[a].label.text == "100 mm"
    assert scene.link_items[f].label.text == "2 m"
    scene.begin_edit_link(a)
    scene.link_editor.setText("150 mm")
    scene.link_editor.editingFinished.emit()
    assert doc.setup.links[0].length.value == 150.0
    assert scene.link_items[a].label.text == "150 mm"
    scene.begin_edit_link(a)
    scene.link_editor.setText("garbage")
    messages = []
    scene.status_message.connect(messages.append)
    scene.link_editor.editingFinished.emit()
    assert messages and scene.link_items[a].label.text == "150 mm"


def test_lock_to_canvas_label_and_cable(doc, scene):
    doc.add_instance("sources/laser", [0, 0], instance_id="laser")
    doc.add_instance("freespace/mirror", [4, 0], instance_id="mirror")
    link = doc.add_link("free_space", "laser.out", "mirror.in", lock_to_canvas=True)
    label = scene.link_items[link].label
    assert label.text == "100 mm" and label.italic
    doc.move_instance("mirror", [8, 0])
    assert label.text == "200 mm"
    scene.begin_edit_link(link)
    scene.link_editor.setText("30 mm")
    scene.link_editor.editingFinished.emit()
    assert not label.italic and label.text == "30 mm"


def test_cable_label(doc, scene):
    doc.add_instance("detectors/spad", [0, 0], instance_id="det")
    doc.add_instance("electronics/tagger", [4, 0], instance_id="tag")
    link = doc.add_link("cable", "det.out", "tag.ch1", delay="5 ns")
    assert scene.link_items[link].label.text == "5 ns"


def test_probe_marker_and_highlight(doc, scene):
    chain(doc)
    link = doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    probe = doc.add_probe("probes/powermeter", link)
    assert probe in scene.probe_items
    scene.select(probe)
    assert scene.selected_ids() == [probe]
    scene.set_highlight([link, "laser"], "warning")
    assert scene.highlights == {link: "warning", "laser": "warning"}
    scene.clear_highlights()
    assert not scene.highlights


def test_view_zoom_fit_and_snap(doc, scene, view):
    chain(doc)
    z = view.zoom_level
    view.zoom_by(2.0)
    assert view.zoom_level == pytest.approx(2 * z)
    view.zoom_by(1e6)
    assert view.zoom_level <= 4.0
    view.fit_to_content()
    r = view.mapToScene(view.viewport().rect()).boundingRect()
    assert r.contains(scene.itemsBoundingRect())
    view.set_snap(False)
    assert not scene.snap


def test_auto_layout_layers(doc):
    chain(doc)
    doc.add_link("free_space", "laser.out", "mirror.in", length="100 mm")
    doc.add_link("free_space", "mirror.out", "coupler.in", length="100 mm")
    pos = auto_layout(doc)
    xs = [pos[i][0] for i in ("laser", "mirror", "coupler")]
    assert xs == sorted(xs) and len(set(xs)) == 3
    assert pos["det"][0] == 0.0


def test_render_hbt_example(qtbot):
    import sequoia.components.builtin  # noqa: F401
    from sequoia.library.store import Library

    root = Path(__file__).resolve().parents[2]
    doc = SetupDocument.open(root / "setups" / "hbt_heralded.yaml", Library(root / "library"))
    scene = CanvasScene(doc)
    assert len(scene.component_items) == 14
    assert len(scene.link_items) == 15
    view = CanvasView(scene)
    qtbot.addWidget(view)
    view.resize(1600, 700)
    view.show()
    view.fit_to_content()
    if SCRATCH.is_dir():
        view.grab().save(str(SCRATCH / "hbt_canvas.png"))
        doc2 = SetupDocument.open(root / "setups" / "hbt_heralded.yaml", Library(root / "library"))
        scene2 = CanvasScene(doc2)
        scene2.apply_layout(auto_layout(doc2))
        view2 = CanvasView(scene2)
        qtbot.addWidget(view2)
        view2.resize(1800, 700)
        view2.show()
        view2.fit_to_content()
        view2.grab().save(str(SCRATCH / "hbt_canvas_auto.png"))
        doc2.undo_stack.cleanChanged.disconnect()
    doc.undo_stack.cleanChanged.disconnect()


def test_scene_rect_is_fixed_so_edits_never_move_the_view(doc, scene):
    before = scene.sceneRect()
    chain(doc)
    doc.add_instance("freespace/bs", [200, 150], instance_id="far")
    assert scene.sceneRect() == before


def test_pressing_a_port_selects_the_component(doc, scene):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent

    chain(doc)
    item = scene.component_items["mirror"]
    scene.clearSelection()
    event = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
    pos = scene.port_scene_position("mirror", "in")
    event.setScenePos(pos)
    event.setButton(Qt.LeftButton)
    event.setButtons(Qt.LeftButton)
    scene.mousePressEvent(event)
    assert scene.selected_ids() == ["mirror"]
    assert item.isSelected()
