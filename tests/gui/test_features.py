from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, QSettings, Qt
from PySide6.QtGui import QKeyEvent

from sequoia.gui.canvas import CanvasScene
from sequoia.gui.canvas.items import CELL, font, label_visible
from sequoia.gui.canvas.layout import plan_layout
from sequoia.gui.inspector.panel import InspectorPanel
from sequoia.gui.main_window import MainWindow
from sequoia.gui.palette import part_label
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "setups" / "hbt_heralded.yaml"


@pytest.fixture
def scene(doc, qtbot):
    s = CanvasScene(doc)
    yield s
    doc.undo_stack.cleanChanged.disconnect()


@pytest.fixture
def panel(qtbot, doc):
    p = InspectorPanel(doc)
    qtbot.addWidget(p)
    p.resize(360, 800)
    return p


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(Library(ROOT / "library"), settings=settings)
    win.confirm_discard = lambda w: "discard"
    qtbot.addWidget(win)
    win.show()
    return win


def find_action(menu, *path):
    current = menu
    for text in path:
        match = None
        for action in current.actions():
            if action.text() == text:
                match = action
                break
        assert match is not None, f"{text!r} not in {[a.text() for a in current.actions()]}"
        if text == path[-1]:
            return match
        current = match.menu()
    raise AssertionError


def chain(doc):
    doc.add_instance("sources/laser", [0, 0], instance_id="laser")
    doc.add_instance("freespace/mirror", [4, 0], instance_id="mirror")
    doc.add_instance("fiber/coupler", [8, 0], instance_id="coupler")
    doc.add_instance("detectors/spad", [12, 0], instance_id="det")
    doc.add_instance("electronics/tagger", [16, 0], instance_id="tag")
    doc.add_link("free_space", "laser.out", "mirror.in", link_id="l1", length="100 mm")
    doc.add_link("free_space", "mirror.out", "coupler.in", link_id="l2", length="100 mm")
    doc.add_link("fiber", "coupler.out", "det.in", link_id="f1", fiber="fiber/smf", length="1 m")
    doc.add_link("cable", "det.out", "tag.ch1", link_id="c1", delay="0 ns")


def test_context_menu_adds_and_removes_probe(doc, scene):
    chain(doc)
    point = scene.link_items["l1"].point_at(0.5)
    menu = scene.build_context_menu(point)
    assert "Add probe" in [a.text() for a in menu.actions()]
    probe_menu = find_action(menu, "Add probe").menu()
    labels = [a.text() for a in probe_menu.actions()]
    infos = [i for i in doc.library.index() if i.category == "probes"]
    assert labels == [part_label(i) for i in infos] and labels
    find_action(menu, "Add probe", labels[0]).trigger()
    assert [(p.part, p.link) for p in doc.setup.probes] == [("probes/powermeter", "l1")]
    probe_id = doc.setup.probes[0].id
    assert scene.selected_ids() == [probe_id]

    probe_item = scene.probe_items[probe_id]
    menu = scene.build_context_menu(probe_item.pos() + QPointF(0, -4))
    find_action(menu, "Remove probe").trigger()
    assert not doc.setup.probes
    doc.undo_stack.undo()
    assert len(doc.setup.probes) == 1


def test_context_menu_cable_and_fiber_links_take_probes(doc, scene):
    chain(doc)
    for link in ("f1", "c1"):
        menu = scene.build_context_menu(scene.link_items[link].point_at(0.4))
        find_action(menu, "Add probe", part_label(scene.probe_parts()[0])).trigger()
    assert {p.link for p in doc.setup.probes} == {"f1", "c1"}


def test_context_menu_component_actions(doc, scene):
    chain(doc)
    pos = scene.component_items["mirror"].pos()
    menu = scene.build_context_menu(pos)
    find_action(menu, "Rotate 90°").trigger()
    assert doc.setup.instances[1].rotation == 90.0
    menu = scene.build_context_menu(pos)
    find_action(menu, "Flip").trigger()
    assert doc.setup.instances[1].flipped
    menu = scene.build_context_menu(pos)
    find_action(menu, "Delete").trigger()
    assert "mirror" not in [i.id for i in doc.setup.instances]


def test_context_menu_on_empty_canvas_adds_instance_here(doc, scene):
    menu = scene.build_context_menu(QPointF(5.2 * CELL, 3.1 * CELL))
    sub = find_action(menu, "Add instance here").menu()
    categories = [a.text() for a in sub.actions()]
    assert "Freespace" in categories and "Probes" not in categories
    part = find_action(
        sub,
        "Freespace",
        part_label(next(i for i in doc.library.index() if i.ref == "freespace/mirror")),
    )
    part.trigger()
    inst = doc.setup.instances[0]
    assert inst.part == "freespace/mirror"
    assert inst.position == [5.0, 3.0]


def test_edit_menu_probe_entries(qtbot, window):
    assert window.open_path(EXAMPLE)
    edit = next(a.menu() for a in window.menuBar().actions() if a.text().replace("&", "") == "Edit")
    names = [a.text() for a in edit.actions()]
    assert "Add probe" in names and "Remove probe" in names and "Add instance" in names
    assert not window.menu_add_probe.isEnabled()
    window.scene.select("s_signal")
    assert window.menu_add_probe.isEnabled()
    window.menu_add_probe.aboutToShow.emit()
    entry = window.menu_add_probe.actions()[0]
    entry.trigger()
    assert [p.link for p in window.document.setup.probes] == ["s_signal"]
    assert window.scene.selected_ids() == [window.document.setup.probes[0].id]
    assert window.act_remove_probe.isEnabled()
    window.act_remove_probe.trigger()
    assert not window.document.setup.probes
    window.menu_add_instance.aboutToShow.emit()
    assert window.menu_add_instance.actions()


def test_probe_through_link_is_simulated(qtbot, window):
    assert window.open_path(EXAMPLE)
    window.scene.select("s_signal")
    window.menu_add_probe.aboutToShow.emit()
    window.menu_add_probe.actions()[0].trigger()
    qtbot.waitUntil(
        lambda: (
            window.results.outcome is not None
            and window.results.outcome.revision == window.controller.revision
            and window.results.outcome.classical is not None
            and bool(window.results.outcome.classical.probes)
        ),
        timeout=60000,
    )
    assert window.results.readouts.topLevelItemCount() > 0


def analyses_editor(panel, doc):
    panel.show_selection(None)
    return panel.form.analyses


def test_analyses_editor_adds_each_type(doc, panel):
    chain(doc)
    editor = analyses_editor(panel, doc)
    assert editor.list.count() == 0
    editor.add_button.click()
    form = editor.form
    items = [form.channel_combos[0].itemData(i) for i in range(form.channel_combos[0].count())]
    assert items == ["tag.ch1", "tag.ch2", "tag.ch3", "tag.ch4"]
    form.name.setText("coinc")
    form.window.edit.setText("1 ns")
    form.channel_combos[1].setCurrentIndex(0)
    assert not form.apply()
    assert "differ" in form.error.text()
    assert doc.setup.analyses == []
    form.cancel_button.click()
    assert editor.form is None


@pytest.fixture
def multichannel(doc):
    chain(doc)
    doc.add_instance("detectors/spad", [12, 4], instance_id="det2")
    doc.add_instance("detectors/spad", [12, 8], instance_id="det3")
    doc.add_instance("fiber/coupler", [8, 4], instance_id="cpl2")
    doc.add_instance("fiber/coupler", [8, 8], instance_id="cpl3")
    doc.add_link("cable", "det2.out", "tag.ch2", link_id="c2", delay="0 ns")
    doc.add_link("cable", "det3.out", "tag.ch3", link_id="c3", delay="0 ns")
    return doc


def test_analyses_editor_roundtrip(multichannel, panel):
    doc = multichannel
    editor = analyses_editor(panel, doc)
    editor.add_button.click()
    form = editor.form
    assert form.channel_combos[0].count() == 4
    form.name.setText("herald_a")
    form.window.edit.setText("1 ns")
    assert form.apply()
    assert doc.setup.analyses == [
        {
            "name": "herald_a",
            "type": "coincidences",
            "channels": ["tag.ch1", "tag.ch2"],
            "window": "1 ns",
            "center": "auto",
        }
    ]
    assert editor.list.count() == 1 and "herald_a" in editor.list.item(0).text()

    editor.add_button.click()
    form = editor.form
    form.kind.setCurrentIndex(form.kind.findData("heralded_g2"))
    form.name.setText("hbt")
    for combo, channel in zip(form.channel_combos, ("tag.ch1", "tag.ch2", "tag.ch3"), strict=True):
        combo.setCurrentIndex(combo.findData(channel))
    form.window.edit.setText("2 ns")
    form.center.setText("0.5 ns")
    assert form.apply()
    assert doc.setup.analyses[1] == {
        "name": "hbt",
        "type": "heralded_g2",
        "herald": "tag.ch1",
        "a": "tag.ch2",
        "b": "tag.ch3",
        "window": "2 ns",
        "center": "0.5 ns",
    }

    editor.add_button.click()
    form = editor.form
    form.kind.setCurrentIndex(form.kind.findData("g2"))
    form.name.setText("g2_ab")
    form.channel_combos[0].setCurrentIndex(form.channel_combos[0].findData("tag.ch2"))
    form.channel_combos[1].setCurrentIndex(form.channel_combos[1].findData("tag.ch3"))
    form.range.edit.setText("10 ns")
    form.bins.setValue(100)
    assert form.apply()
    assert doc.setup.analyses[2] == {
        "name": "g2_ab",
        "type": "g2",
        "channels": ["tag.ch2", "tag.ch3"],
        "range": "10 ns",
        "bins": 100,
    }

    editor.add_button.click()
    form = editor.form
    form.kind.setCurrentIndex(form.kind.findData("histogram"))
    form.name.setText("hist")
    form.range.edit.setText("5 ns")
    assert form.apply()
    assert doc.setup.analyses[3]["type"] == "histogram"
    assert doc.setup.analyses[3]["bins"] == 200
    assert editor.list.count() == 4

    editor.list.setCurrentRow(0)
    editor.edit_button.click()
    form = editor.form
    assert form.name.text() == "herald_a" and form.window.text() == "1 ns"
    form.window.edit.setText("3 ns")
    assert form.apply()
    assert doc.setup.analyses[0]["window"] == "3 ns"
    assert doc.setup.analyses[0]["name"] == "herald_a"

    editor.list.setCurrentRow(1)
    editor.remove_button.click()
    assert [a["name"] for a in doc.setup.analyses] == ["herald_a", "g2_ab", "hist"]
    doc.undo_stack.undo()
    assert [a["name"] for a in doc.setup.analyses][1] == "hbt"


def test_analyses_editor_rejects_invalid(multichannel, panel):
    doc = multichannel
    editor = analyses_editor(panel, doc)
    editor.add_button.click()
    form = editor.form
    form.name.setText("a")
    form.window.edit.setText("1 bananas")
    assert not form.apply()
    assert form.error.text() and not form.error.isHidden()
    form.window.edit.setText("-1 ns")
    assert not form.apply()
    assert "positive" in form.error.text()
    form.window.edit.setText("1")
    assert not form.apply()
    assert "unit" in form.error.text()
    form.window.edit.setText("1 ns")
    form.name.setText("  ")
    assert not form.apply()
    assert "name" in form.error.text()
    form.name.setText("ok")
    form.center.setText("soon")
    assert not form.apply()
    form.center.setText("auto")
    assert form.apply()
    editor.add_button.click()
    form = editor.form
    form.name.setText("ok")
    form.window.edit.setText("1 ns")
    assert not form.apply()
    assert "used twice" in form.error.text()
    assert len(doc.setup.analyses) == 1


def test_document_set_analyses_validates(multichannel):
    doc = multichannel
    good = {
        "name": "a",
        "type": "coincidences",
        "channels": ["tag.ch1", "tag.ch2"],
        "window": "1 ns",
    }
    doc.set_analyses([good])
    for bad in (
        {**good, "type": "nonsense"},
        {**good, "channels": ["tag.ch1"]},
        {**good, "channels": ["tag.ch1", "tag.ch1"]},
        {**good, "window": "0 ns"},
        {**good, "window": "1 m"},
        {**good, "name": ""},
        {"name": "g", "type": "g2", "channels": ["tag.ch1", "tag.ch2"], "range": "1 ns", "bins": 0},
    ):
        with pytest.raises(ValueError):
            doc.set_analyses([bad])
    assert doc.setup.analyses == [good]


def test_analyses_editor_empty_setup_hint(doc, panel):
    editor = analyses_editor(panel, doc)
    assert "time tagger" in editor.hint.text()
    editor.add_button.click()
    form = editor.form
    form.name.setText("x")
    form.window.edit.setText("1 ns")
    assert not form.apply()
    assert "tagger" in form.error.text()


def test_rename_instance_cascades_and_undoes(multichannel):
    doc = multichannel
    doc.add_probe("probes/powermeter", "l1", probe_id="pm")
    doc.set_analyses(
        [
            {
                "name": "a",
                "type": "coincidences",
                "channels": ["tag.ch1", "tag.ch2"],
                "window": "1 ns",
            }
        ]
    )
    doc.rename_instance("tag", "tagger")
    s = doc.setup
    assert {i.id for i in s.instances} >= {"tagger"} and "tag" not in {i.id for i in s.instances}
    ends = {(link.src, link.dst) for link in s.links}
    assert ("det.out", "tagger.ch1") in ends and ("det2.out", "tagger.ch2") in ends
    assert s.analyses[0]["channels"] == ["tagger.ch1", "tagger.ch2"]
    doc.undo_stack.undo()
    assert "tag" in {i.id for i in doc.setup.instances}
    assert doc.setup.analyses[0]["channels"] == ["tag.ch1", "tag.ch2"]
    doc.undo_stack.redo()
    with pytest.raises(ValueError, match="already used"):
        doc.rename_instance("laser", "mirror")
    with pytest.raises(ValueError):
        doc.rename_instance("laser", "1bad")
    with pytest.raises(ValueError, match="no instance"):
        doc.rename_instance("ghost", "x")
    doc.rename_instance("pm", "meter")
    assert doc.setup.probes[0].id == "meter"


def test_rename_link_cascades_to_probes(multichannel):
    doc = multichannel
    doc.add_probe("probes/powermeter", "l1", probe_id="pm")
    doc.rename_link("l1", "pump_to_mirror")
    assert doc.setup.probes[0].link == "pump_to_mirror"
    assert "pump_to_mirror" in [link.id for link in doc.setup.links]
    with pytest.raises(ValueError, match="already used"):
        doc.rename_link("l2", "laser")
    doc.undo_stack.undo()
    assert doc.setup.probes[0].link == "l1"


def test_inspector_rename_keeps_selection(qtbot, window):
    assert window.open_path(EXAMPLE)
    window.scene.select("crystal")
    form = window.inspector.form
    row = form.id_row
    assert row.editor.text() == "crystal"
    row.editor.edit.setText("ppktp")
    row.editor.edit.editingFinished.emit()
    assert "ppktp" in [i.id for i in window.document.setup.instances]
    assert window.scene.selected_ids() == ["ppktp"]
    assert window.inspector.selection() == ("instance", "ppktp")
    assert window.inspector.form.id_row.editor.text() == "ppktp"
    ends = [x.dst for x in window.document.setup.links if x.id == "s_focus"]
    assert ends == ["ppktp.in"]
    window.scene.select("s_focus")
    row = window.inspector.form.id_row
    row.editor.edit.setText("focus_link")
    row.editor.edit.editingFinished.emit()
    assert window.scene.selected_ids() == ["focus_link"]
    assert window.inspector.selection() == ("link", "focus_link")
    row = window.inspector.form.id_row
    row.editor.edit.setText("pump")
    row.editor.edit.editingFinished.emit()
    assert window.inspector.form.id_row.editor.text() == "focus_link"
    assert "already used" in window.inspector.form.id_row.error.text()


def test_link_form_polish(window):
    assert window.open_path(EXAMPLE)
    window.scene.select("s_pump")
    form = window.inspector.form
    rows = form.form.rows
    assert rows["length"].label.text() == "Length"
    assert rows["lock_to_canvas"].label.text() == "Lock length to canvas distance"
    assert rows["medium"].label.text() == "Medium"
    assert not rows["length"].optional_box.isVisibleTo(form)
    assert not rows["length"].optional_box.isVisible()
    medium = rows["medium"].editor.combo
    assert [medium.itemText(i) for i in range(medium.count())] == ["air", "vacuum"]
    assert "100 mm" in rows["length"].editor.edit.placeholderText()
    medium.setCurrentIndex(medium.findData("vacuum"))
    assert window.document.setup.links[0].medium == "vacuum"
    assert rows["length"].label.toolTip()
    window.scene.select("f_signal")
    fiber_rows = window.inspector.form.form.rows
    assert fiber_rows["connector_loss_db"].label.text() == "Loss (dB)"
    assert fiber_rows["fiber"].label.text() == "Fiber type"
    window.scene.select("c_h")
    assert window.inspector.form.form.rows["delay"].label.text() == "Cable delay"


def key(scene, k, mods=Qt.NoModifier):
    scene.keyPressEvent(QKeyEvent(QEvent.KeyPress, k, mods))


def test_arrow_keys_nudge_by_snap_step_and_merge(doc, scene):
    chain(doc)
    scene.select("mirror")
    n = doc.undo_stack.count()
    key(scene, Qt.Key_Right)
    assert doc.setup.instances[1].position == [4.5, 0.0]
    key(scene, Qt.Key_Right)
    key(scene, Qt.Key_Down)
    assert doc.setup.instances[1].position == [5.0, 0.5]
    key(scene, Qt.Key_Left)
    key(scene, Qt.Key_Up)
    assert doc.setup.instances[1].position == [4.5, 0.0]
    assert doc.undo_stack.count() == n + 1
    doc.undo_stack.undo()
    assert doc.setup.instances[1].position == [4.0, 0.0]


def test_nudge_without_selection_does_nothing(doc, scene):
    chain(doc)
    n = doc.undo_stack.count()
    key(scene, Qt.Key_Left)
    assert doc.undo_stack.count() == n


def test_rotation_slider_in_inspector(doc, panel):
    chain(doc)
    panel.show_selection("instance", "mirror")
    form = panel.form
    slider = form.rotation_slider
    assert slider.minimum() == 0 and slider.maximum() == 23
    slider.setValue(3)
    assert doc.setup.instances[1].rotation == 45.0
    assert form.placement["rotation"].editor.spin.value() == 45.0
    form.placement["rotation"].editor.spin.setValue(90.0)
    assert slider.value() == 6
    doc.undo_stack.undo()
    assert slider.value() == 3
    doc.rotate_instance("mirror", 37.0)
    assert slider.value() == 2


def test_auto_layout_fits_readably(qtbot, window):
    window.resize(1500, 900)
    for name in ("hbt_heralded", "hom", "bell_test"):
        assert window.open_path(ROOT / "setups" / f"{name}.yaml")
        window.auto_layout()
        assert not window._overlapping(), name
        assert window.view.zoom_level >= 0.6, (name, window.view.zoom_level)
        assert label_visible("id", window.view.zoom_level)
        window.undo()


def test_text_scales_with_zoom_and_hides_when_tiny():
    assert font("id", 0.1).pixelSize() == font("id", 2.0).pixelSize()
    assert label_visible("id", 1.0)
    assert not label_visible("id", 0.2)
    assert not label_visible("port", 0.3)


def test_layout_has_no_overlap_and_keeps_all_instances(window):
    window.open_path(EXAMPLE)
    plan = plan_layout(window.document, (848.0, 602.0))
    assert set(plan.positions) == {i.id for i in window.document.setup.instances}
    window.auto_layout()
    items = list(window.scene.component_items.values())
    for i, a in enumerate(items):
        ra = a.mapRectToScene(a._body_path().boundingRect())
        for b in items[i + 1 :]:
            rb = b.mapRectToScene(b._body_path().boundingRect())
            box = ra.intersected(rb)
            assert box.width() <= 0.5 or box.height() <= 0.5, (a.instance_id, b.instance_id)


def test_overlap_offer_uses_real_body_overlap(qtbot, window):
    ids = []
    for x in (0, 6, 12):
        ids.append(window.document.add_instance("freespace/lens_f200mm", [x, 0]))
    assert not window._overlapping()
    w = window.scene.component_items[ids[0]]._w / CELL
    window.document.move_instance(ids[1], [w - 1.0, 0])
    assert window._overlapping()
    window.document.move_instance(ids[1], [w, 0])
    assert not window._overlapping()
    window.document.move_instance(ids[1], [w + 0.5, 0])
    assert not window._overlapping()
