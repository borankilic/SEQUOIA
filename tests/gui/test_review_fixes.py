import math
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest

from sequoia.gui.canvas import CanvasScene
from sequoia.gui.canvas.items import CELL
from sequoia.gui.canvas.view import MIN_ZOOM
from sequoia.gui.document import SetupDocument
from sequoia.gui.inspector.fields import MISSING, FieldRow, FieldSpec, ScalarEditor, analyze
from sequoia.gui.main_window import MainWindow
from sequoia.gui.results import ResultsDock
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "setups" / "hbt_heralded.yaml"


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(Library(ROOT / "library"), settings=settings)
    win.confirm_discard = lambda w: "discard"
    qtbot.addWidget(win)
    win.show()
    return win


def messages_of(window):
    out = []
    window.statusBar().messageChanged.connect(out.append)
    return out


def exercise_actions(window, qtbot):
    window.scene.status_message.emit("probe-message")
    assert window.statusBar().currentMessage() == "probe-message"
    window.scene.status_message.emit("")

    ids = [i.id for i in window.document.setup.instances]
    victim = ids[-1]
    n = len(ids)

    window.scene.select(victim)
    flipped = window.document.setup.instances[-1].flipped
    window.act_flip.trigger()
    assert window.document.setup.instances[-1].flipped != flipped

    window.scene.select(victim)
    window.act_copy.trigger()
    window.act_paste.trigger()
    assert len(window.document.setup.instances) == n + 1

    window.scene.select(victim)
    window.act_delete.trigger()
    assert len(window.document.setup.instances) == n
    assert victim not in [i.id for i in window.document.setup.instances]

    window.scene.select(ids[0])
    before = window.document.setup.instances[0].rotation
    window.act_rotate.trigger()
    assert window.document.setup.instances[0].rotation == (before + 90) % 360

    refusal = []
    window.scene.status_message.connect(refusal.append)
    link = window.document.setup.links[0]
    item = window.scene.component_items[link.src.split(".")[0]]
    assert not window.scene.start_link(item, link.src.split(".")[1], QPointF(0, 0))
    assert refusal and "already connected" in refusal[-1]
    assert window.statusBar().currentMessage() == refusal[-1]


def test_actions_and_status_after_open(window, qtbot):
    assert window.open_path(EXAMPLE)
    exercise_actions(window, qtbot)


def test_actions_and_status_after_new(window, qtbot):
    assert window.open_path(EXAMPLE)
    assert window.new_document()
    window.palette.part_activated.emit("freespace/lens_f200mm")
    window.palette.part_activated.emit("freespace/lens_f100mm")
    window.scene.status_message.emit("after-new")
    assert window.statusBar().currentMessage() == "after-new"
    n = len(window.document.setup.instances)
    window.scene.select_many([i.id for i in window.document.setup.instances])
    window.act_copy.trigger()
    window.act_paste.trigger()
    assert len(window.document.setup.instances) == 2 * n
    window.scene.select(window.document.setup.instances[0].id)
    window.act_delete.trigger()
    assert len(window.document.setup.instances) == 2 * n - 1
    window.scene.select(window.document.setup.instances[0].id)
    window.act_flip.trigger()
    assert window.document.setup.instances[0].flipped


def test_actions_after_open_library_and_second_open(window, qtbot, tmp_path):
    assert window.open_path(EXAMPLE)
    window.ask_library_path = lambda: str(ROOT / "library")
    assert window.open_library_dialog()
    exercise_actions(window, qtbot)
    assert window.open_path(EXAMPLE)
    exercise_actions(window, qtbot)


def test_view_follows_new_scene(window):
    first = window.scene
    window.open_path(EXAMPLE)
    assert window.scene is not first
    assert window.view.scene() is window.scene
    got = []
    window.view.status_message.connect(got.append)
    window.scene.status_message.emit("x")
    assert got == ["x"]
    first_alive = []
    try:
        first.status_message.emit("stale")
        first_alive.append(True)
    except RuntimeError:
        pass
    assert got == ["x"]


def test_draw_background_is_bounded(window, qtbot):
    image = QImage(200, 200, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    started = time.perf_counter()
    window.view.drawBackground(painter, QRectF(-1e12, -1e12, 2e12, 2e12))
    window.view.drawBackground(painter, QRectF(-4e7, -4e7, 8e7, 8e7))
    window.view.drawBackground(painter, QRectF(0, 0, math.inf, 10))
    painter.end()
    assert time.perf_counter() - started < 2.0


def test_fit_to_content_respects_min_zoom(doc, qtbot):
    from sequoia.gui.canvas import CanvasView

    doc.add_instance("sources/laser", [0, 0], instance_id="a")
    doc.add_instance("fiber/coupler", [99999, 99999], instance_id="b")
    scene = CanvasScene(doc)
    view = CanvasView(scene)
    qtbot.addWidget(view)
    view.resize(800, 600)
    view.show()
    view.fit_to_content()
    assert view.zoom_level >= MIN_ZOOM - 1e-9
    view.zoom_by(1.0)
    assert view.zoom_level >= MIN_ZOOM - 1e-9
    doc.undo_stack.cleanChanged.disconnect()


@pytest.mark.parametrize(
    "position",
    [[math.nan, 0], [0, math.inf], [1e6, 0], [0, -1e12]],
)
def test_document_rejects_absurd_positions(doc, position):
    doc.add_instance("sources/laser", [0, 0], instance_id="a")
    with pytest.raises(ValueError, match="position"):
        doc.move_instance("a", position)
    with pytest.raises(ValueError, match="position"):
        doc.add_instance("freespace/mirror", position)
    assert doc.setup.instances[0].position == [0.0, 0.0]


@pytest.mark.parametrize("rotation", [math.nan, math.inf, -math.inf])
def test_document_rejects_nonfinite_rotation(doc, rotation):
    doc.add_instance("sources/laser", [0, 0], instance_id="a")
    with pytest.raises(ValueError, match="rotation"):
        doc.rotate_instance("a", rotation)
    assert doc.setup.instances[0].rotation == 0.0
    doc.rotate_instance("a", 1e9)
    assert math.isfinite(doc.setup.instances[0].rotation)


def test_document_rejects_nonfinite_lengths(doc):
    doc.add_instance("sources/laser", [0, 0], instance_id="a")
    doc.add_instance("freespace/mirror", [4, 0], instance_id="m")
    link = doc.add_link("free_space", "a.out", "m.in", length="100 mm")
    for bad in ("1e400 m", "inf m", "1e999 mm"):
        with pytest.raises(ValueError, match="finite"):
            doc.set_link_field(link, "length", bad)
    with pytest.raises(ValueError, match="finite"):
        doc.set_grid_cell("inf m")
    with pytest.raises(ValueError, match="finite"):
        doc.set_setting("m", "angle", "1e999 deg")
    assert doc.setup.links[0].length.value == 100.0


def test_inf_allowed_where_it_is_the_default(tmp_path):
    import sequoia.components.builtin  # noqa: F401
    from sequoia.components.base import get_component
    from sequoia.library.store import Part

    library = Library(tmp_path / "lib")
    library.save_part(Part("freespace/iris_test", "Iris", get_component("Iris").Params()))
    doc = SetupDocument(library)
    doc.add_instance("freespace/iris_test", [0, 0], instance_id="iris")
    doc.set_setting("iris", "diameter", "5 mm")
    doc.set_setting("iris", "diameter", "inf m")
    assert doc.setup.instances[0].settings["diameter"] == "inf m"


def write(path, text):
    path.write_text(text)
    return path


def test_open_rejects_nan_position_with_dialog(window, tmp_path):
    bad = write(
        tmp_path / "nan.yaml",
        "instances:\n  - {id: a, part: sources/laser_405_cw, position: [.nan, .inf]}\n",
    )
    assert window.open_path(EXAMPLE)
    errors = []
    window.show_error = lambda title, message: errors.append((title, message))
    scene, doc = window.scene, window.document
    assert not window.open_path(bad)
    assert len(errors) == 1 and "position" in errors[0][1]
    assert window.scene is scene and window.document is doc
    assert window.view.scene() is scene
    assert len(window.document.setup.instances) == 14
    exercise_actions(window, None)


def test_scene_failure_leaves_window_intact(window, monkeypatch):
    assert window.open_path(EXAMPLE)
    scene, doc = window.scene, window.document
    errors = []
    window.show_error = lambda title, message: errors.append((title, message))

    class Boom(Exception):
        pass

    def broken(document, parent=None):
        raise Boom("scene construction failed")

    monkeypatch.setattr("sequoia.gui.main_window.CanvasScene", broken)
    assert not window.open_path(ROOT / "setups" / "mzi.yaml")
    assert errors and "scene construction failed" in errors[0][1]
    monkeypatch.undo()
    assert window.scene is scene and window.document is doc
    assert window.view.scene() is scene
    assert window.inspector.document is doc
    assert len(window.document.setup.instances) == 14
    exercise_actions(window, None)


def test_open_rejects_duplicate_ids(window, tmp_path):
    bad = write(
        tmp_path / "dup.yaml",
        "instances:\n"
        "  - {id: a, part: sources/laser_405_cw, position: [0, 0]}\n"
        "  - {id: a, part: freespace/mirror_ideal, position: [4, 0]}\n",
    )
    errors = []
    window.show_error = lambda title, message: errors.append((title, message))
    assert not window.open_path(bad)
    assert errors and "'a'" in errors[0][1] and "more than once" in errors[0][1]
    with pytest.raises(ValueError, match="more than once"):
        SetupDocument.open(bad, Library(ROOT / "library"))


def test_results_tabs_fail_independently_and_visibly(qtbot, window, monkeypatch):
    assert window.open_path(EXAMPLE)
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=60000)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=60000)
    dock = window.results
    outcome = dock.outcome
    assert dock.readouts.topLevelItemCount() >= 0
    assert dock.analyses.topLevelItemCount() > 0

    def broken(_outcome):
        raise AttributeError("'NoneType' object has no attribute 'w_x'")

    monkeypatch.setattr(dock, "_fill_links", broken)
    dock.analyses.clear()
    dock.set_outcome(outcome)
    assert dock.analyses.topLevelItemCount() > 0
    assert "Links" in dock.tab_errors
    assert "Links" in dock.status.text() and "w_x" in dock.status.text()
    assert "(!)" in dock.tabs.tabText(1)


def test_links_tab_survives_missing_mode(qtbot):
    dock = ResultsDock()
    qtbot.addWidget(dock)
    light = SimpleNamespace(power=1e-3, mode=None, dop=0.5)
    state = SimpleNamespace(id="l", kind="free_space", src="a.out", dst="b.in", end=light, info={})
    classical = SimpleNamespace(links={"l": state}, readouts={}, probes={}, issues=[])
    outcome = SimpleNamespace(
        issues=[], classical=classical, rates=None, elapsed=0.1, revision=1, ok=True
    )
    dock.set_outcome(outcome)
    assert not dock.tab_errors
    assert dock.links.rowCount() == 1
    assert dock.links.item(0, 4).text() == "n/a"


def test_undo_limit_and_memory(parts_library):
    doc = SetupDocument(parts_library)
    assert doc.undo_stack.undoLimit() == 200
    ids = []
    with doc.macro("build"):
        for i in range(120):
            ids.append(doc.add_instance("freespace/mirror", [(i % 20) * 4, (i // 20) * 3]))
        for i in range(119):
            if i % 20 != 19:
                doc.add_link("free_space", f"{ids[i]}.out", f"{ids[i + 1]}.in", length="10 mm")
    link = doc.setup.links[0].id
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for k in range(300):
        doc.set_link_field(link, "length", f"{20 + k} mm")
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = sum(s.size_diff for s in after.compare_to(before, "filename")) / 1e6
    assert doc.undo_stack.count() == 200
    assert growth < 25.0, growth
    for _ in range(200):
        doc.undo_stack.undo()
    assert not doc.undo_stack.canUndo()
    assert doc.setup.links[0].length.value == 20.0 + 99
    for _ in range(200):
        doc.undo_stack.redo()
    assert doc.setup.links[0].length.value == 20.0 + 299


def test_structural_sharing_keeps_unchanged_items(doc):
    doc.add_instance("sources/laser", [0, 0], instance_id="a")
    doc.add_instance("freespace/mirror", [4, 0], instance_id="m")
    before = doc.setup
    doc.move_instance("m", [5, 0])
    assert doc.setup.instances[0] is before.instances[0]
    assert doc.setup.instances[1] is not before.instances[1]
    doc.undo_stack.undo()
    assert doc.setup.instances[1].position == [4.0, 0.0]


def test_rejected_unset_resyncs_widget():
    spec = analyze(__import__("sequoia.data", fromlist=["Length"]).Length | None)
    assert spec.optional
    calls = []

    def commit(name, value):
        calls.append(value)
        if value is None:
            raise ValueError("needs a length")

    row = FieldRow("length", spec, ScalarEditor(spec), commit, track_default=False)
    row.refresh("100 mm", MISSING)
    assert row.optional_box.isChecked() and row.editor.isEnabled()
    row.optional_box.click()
    assert calls == [None]
    assert row.optional_box.isChecked()
    assert row.editor.isEnabled()
    assert row.editor.text() == "100 mm"
    assert "needs a length" in row.error.text() and not row.error.isHidden()


def test_cleared_text_restores_document_value(window, qtbot):
    assert window.open_path(EXAMPLE)
    window.scene.select("s_pump")
    form = window.inspector.form
    row = form.form.rows["length"]
    before = window.document.setup
    row.editor.edit.setText("")
    row.editor.edit.editingFinished.emit()
    assert window.document.setup is before
    assert row.editor.text() == "100 mm"
    assert row.error.text() and not row.error.isHidden()


def test_rejected_document_value_resyncs(window):
    assert window.open_path(EXAMPLE)
    window.scene.select("s_pump")
    row = window.inspector.form.form.rows["length"]
    row.editor.edit.setText("1e400 m")
    row.editor.edit.editingFinished.emit()
    assert window.document.setup.links[0].length.value == 100.0
    assert row.editor.text() == "100 mm"
    assert "finite" in row.error.text()


def test_separate_drags_do_not_merge(window):
    window.document.add_instance("freespace/mirror_ideal", [0, 0], instance_id="m")
    scene = window.scene
    view = window.view
    n = window.document.undo_stack.count()
    vp = view.viewport()
    for target in ((3, 1), (6, 2)):
        item = scene.component_items["m"]
        start = view.mapFromScene(item.pos())
        end = view.mapFromScene(QPointF(target[0] * CELL, target[1] * CELL))
        QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
        QTest.mouseMove(vp, end)
        QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end)
    assert window.document.undo_stack.count() == n + 2
    window.undo()
    assert window.document.setup.instances[0].position == pytest.approx([3.0, 1.0], abs=0.5)
    window.undo()
    assert window.document.setup.instances[0].position == [0.0, 0.0]


def test_curve_preview_axis_label_not_clipped(qtbot):
    from sequoia.data import Curve
    from sequoia.gui.inspector.fields import CurveEditor

    editor = CurveEditor(FieldSpec("curve"))
    qtbot.addWidget(editor)
    curve = Curve(
        x=[700.0, 800.0, 900.0],
        y=[0.1, 0.6, 0.3],
        x_label="wavelength",
        x_unit="nm",
        y_label="efficiency",
        y_unit="1",
    )
    editor.set_raw(curve)
    axis = editor.holder.plot.getAxis("left")
    assert axis.autoSIPrefix is False
    assert "(1)" not in axis.labelText
    assert axis.labelText == "efficiency"


def test_inspector_position_spin_is_bounded(window):
    assert window.open_path(EXAMPLE)
    window.scene.select("crystal")
    spin = window.inspector.form.placement["x"].editor.spin
    assert spin.maximum() == 1e5 and spin.minimum() == -1e5
    spin.setValue(1e9)
    assert window.document.setup.instances[2].position[0] == 1e5
