from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from sequoia.gui import app as gui_app
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "setups" / "hbt_heralded.yaml"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-borankilic-Desktop-Research-Quantum-Technologies-SEQUOIA/"
    "5d0a0d99-6543-4fe7-aef3-c7336f569617/scratchpad"
)
WAIT = 60000


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def real_library():
    return Library(ROOT / "library")


@pytest.fixture
def make_window(qtbot, settings, real_library):
    def make(setup_path=None, **kwargs):
        window = MainWindow(real_library, setup_path, settings=settings, **kwargs)
        window.confirm_discard = lambda w: "discard"
        qtbot.addWidget(window)
        window.show()
        return window

    return make


def menu_actions(window, title):
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == title:
            return {a.text().replace("&", ""): a for a in action.menu().actions() if a.text()}
    raise AssertionError(title)


def test_menus_and_shortcuts(make_window):
    window = make_window()
    assert [a.text().replace("&", "") for a in window.menuBar().actions()] == [
        "File",
        "Edit",
        "View",
        "Run",
    ]
    file_menu = menu_actions(window, "File")
    for name in ("New", "Open...", "Open Recent", "Save", "Save As...", "Quit"):
        assert name in file_menu
    edit = menu_actions(window, "Edit")
    for name in ("Delete", "Copy", "Paste", "Rotate", "Flip", "Auto layout", "Snap to grid"):
        assert name in edit
    assert edit["Snap to grid"].isCheckable() and edit["Snap to grid"].isChecked()
    run = menu_actions(window, "Run")
    assert run["Auto-run"].isChecked()
    assert window.act_undo.shortcut().toString() != ""
    assert not window.act_undo.isEnabled()


def test_empty_hint_and_title(make_window):
    window = make_window()
    assert window.hint.isVisible()
    assert window.windowTitle().startswith("untitled")
    window.palette.part_activated.emit("freespace/lens_f200mm")
    assert not window.hint.isVisible()
    assert window.isWindowModified()
    assert len(window.document.setup.instances) == 1


def test_palette_double_click_adds_at_centre(make_window):
    window = make_window()
    ref = "freespace/lens_f200mm"
    item = window.palette.item_for_ref(ref)
    window.palette.tree.itemDoubleClicked.emit(item, 0)
    window.palette.tree.itemDoubleClicked.emit(item, 0)
    first, second = window.document.setup.instances
    cx, cy = window.view.view_center_cells()
    assert abs(first.position[0] - cx) <= 1 and abs(first.position[1] - cy) <= 1
    assert first.position != second.position


def test_selection_drives_inspector(make_window):
    window = make_window(EXAMPLE)
    window.scene.select("crystal")
    assert window.inspector.selection() == ("instance", "crystal")
    window.scene.select("s_pump")
    assert window.inspector.selection() == ("link", "s_pump")
    window.scene.select_many(["crystal", "pump"])
    assert window.inspector.selection() == (None, None)


def test_overlap_offer_and_auto_layout(make_window):
    window = make_window(EXAMPLE)
    assert window.overlap_bar.isVisible()
    assert not window.document.dirty
    before = [list(i.position) for i in window.document.setup.instances]
    window.overlap_button.click()
    assert not window.overlap_bar.isVisible()
    assert window.document.dirty
    after = [list(i.position) for i in window.document.setup.instances]
    assert before != after
    assert not window._overlapping()
    window.undo()
    assert [list(i.position) for i in window.document.setup.instances] == before


def test_unsaved_changes_prompt_on_close(make_window, tmp_path):
    window = make_window()
    window.document.set_name("changed")
    asked = []
    window.confirm_discard = lambda w: asked.append(1) or "cancel"
    assert not window.close()
    assert asked == [1] and window.isVisible()
    window.confirm_discard = lambda w: "discard"
    assert window.close()
    assert not window.isVisible()


def test_prompt_save_choice_writes_file(make_window, tmp_path):
    window = make_window()
    window.document.set_name("to_save")
    target = tmp_path / "out.yaml"
    window.confirm_discard = lambda w: "save"
    window.ask_save_path = lambda: str(target)
    assert window.close()
    assert target.is_file()


def test_prompt_on_new_and_open(make_window):
    window = make_window()
    window.document.set_name("changed")
    answers = iter(["cancel", "discard", "discard"])
    window.confirm_discard = lambda w: next(answers)
    assert not window.new_document()
    assert window.document.setup.name == "changed"
    assert window.new_document()
    assert window.document.setup.name == "untitled"
    window.document.set_name("again")
    window.ask_open_path = lambda: str(EXAMPLE)
    assert window.open_dialog()
    assert window.document.path == EXAMPLE
    assert not window.document.dirty


def test_open_error_is_reported(make_window, tmp_path):
    window = make_window()
    errors = []
    window.show_error = lambda title, message: errors.append((title, message))
    bad = tmp_path / "bad.yaml"
    bad.write_text("instances: [1, 2")
    assert not window.open_path(bad)
    assert len(errors) == 1
    assert window.document.path is None


def test_save_as_and_recent_files(make_window, settings, tmp_path):
    window = make_window(EXAMPLE)
    assert window.recent_files() == [str(EXAMPLE.resolve())]
    window.document.set_name("renamed")
    target = tmp_path / "copy.yaml"
    window.ask_save_path = lambda: str(target)
    assert window.save_as()
    assert not window.isWindowModified()
    assert window.windowTitle().startswith("copy.yaml")
    assert window.recent_files()[0] == str(target.resolve())
    window.close()
    other = MainWindow(window.library, settings=settings)
    other.confirm_discard = lambda w: "discard"
    try:
        assert other.recent_files()[0] == str(target.resolve())
        assert other.recent_menu.isEnabled()
        assert any(a.text() == str(target.resolve()) for a in other.recent_menu.actions())
    finally:
        other.close()


def test_geometry_persisted(make_window, settings, qtbot, real_library):
    window = make_window()
    window.resize(1111, 777)
    window.results_dock.hide()
    window.close()
    other = MainWindow(real_library, settings=settings)
    other.confirm_discard = lambda w: "discard"
    qtbot.addWidget(other)
    other.show()
    assert other.results_dock.isHidden()
    other.reset_layout()
    assert not other.results_dock.isHidden()
    other.close()


def test_auto_run_toggle_and_run_now(make_window, qtbot):
    window = make_window(EXAMPLE)
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    window.act_autorun.setChecked(False)
    assert not window.controller.auto_run
    assert "Auto-run off" in window.run_label.text()
    count = window.results.outcome.revision
    window.document.set_link_field("s_pump", "length", "101 mm")
    qtbot.wait(500)
    assert window.results.outcome.revision == count
    window.act_run.trigger()
    qtbot.waitUntil(lambda: window.results.outcome.revision != count, timeout=WAIT)


def test_status_bar_counts_and_highlights(make_window, qtbot):
    window = make_window(EXAMPLE)
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    text = window.run_label.text()
    assert "Done" in text and "error" in text and "warning" in text
    window.document.set_link_field("s_pump", "length", "101 mm")
    window.scene.clear_highlights()
    window.controller.run_now()
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    outcome = window.results.outcome
    from sequoia.gui.results import ids_for_issue

    expected = {
        i
        for issue in outcome.issues
        for i in ids_for_issue(issue.subject)
        if i in window.scene._all_items()
    }
    assert set(window.scene.highlights) == expected


def test_issue_activation_selects(make_window, qtbot):
    window = make_window(EXAMPLE)
    window.results.issue_activated.emit("link 's_lp'")
    assert window.scene.selected_ids() == ["s_lp"]
    assert window.inspector.selection() == ("link", "s_lp")
    window.results.issue_activated.emit("data")
    assert window.scene.selected_ids() == ["s_lp"]


def test_undo_redo_actions(make_window):
    window = make_window()
    window.palette.part_activated.emit("freespace/lens_f200mm")
    assert window.act_undo.isEnabled()
    window.act_undo.trigger()
    assert not window.document.setup.instances
    assert window.act_redo.isEnabled()
    window.act_redo.trigger()
    assert len(window.document.setup.instances) == 1


def test_canvas_actions_work_through_view(make_window):
    window = make_window()
    window.palette.part_activated.emit("freespace/lens_f200mm")
    assert window.scene.selected_ids()
    window.act_rotate.trigger()
    assert window.document.setup.instances[0].rotation == 90.0
    window.act_delete.trigger()
    assert not window.document.setup.instances


def test_snap_toggle(make_window):
    window = make_window()
    window.act_snap.setChecked(False)
    assert not window.scene.snap
    window.act_snap.setChecked(True)
    assert window.scene.snap


def test_new_document_resets_everything(make_window, qtbot):
    window = make_window(EXAMPLE)
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    assert window.new_document()
    assert window.results.outcome is None
    assert window.document.path is None
    assert not window.scene.component_items
    assert window.hint.isVisible()
    assert not window.overlap_bar.isVisible()


def test_window_renders(make_window, qtbot):
    window = make_window(EXAMPLE)
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    window.auto_layout()
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    image = window.grab()
    assert not image.isNull() and image.width() > 500
    if SCRATCH.is_dir():
        image.save(str(SCRATCH / "main_window_hbt.png"))


def test_find_library_order(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "library").mkdir(parents=True)
    (project / "sub").mkdir()
    setup = project / "sub" / "a.yaml"
    setup.write_text("")
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "library").mkdir(parents=True)
    explicit = tmp_path / "explicit_lib"
    explicit.mkdir()
    monkeypatch.chdir(elsewhere)
    assert gui_app.find_library(str(explicit), str(setup)) == explicit
    assert gui_app.find_library(None, str(setup)) == project / "library"
    assert gui_app.find_library(None, None) == Path("library")
    assert gui_app.find_library("missing", str(setup)) is None
    monkeypatch.chdir(tmp_path)
    assert gui_app.find_library(None, None) is None


def test_create_window_from_argv(qtbot, settings):
    window = gui_app.create_window([str(EXAMPLE), "--library", str(ROOT / "library")], settings)
    qtbot.addWidget(window)
    window.confirm_discard = lambda w: "discard"
    assert window.document.path == EXAMPLE
    assert len(window.document.setup.instances) == 14
    window.close()


def test_open_library_replaces_palette(make_window, tmp_path):
    window = make_window()
    count = len(window.palette.visible_refs())
    assert count > 0
    empty = tmp_path / "empty_library"
    empty.mkdir()
    window.ask_library_path = lambda: str(empty)
    assert window.open_library_dialog()
    assert window.palette.visible_refs() == []
    assert window.library.root == empty
