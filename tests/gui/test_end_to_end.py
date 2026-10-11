from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, QSettings, Qt
from PySide6.QtGui import QDropEvent

from sequoia.engines.cache import RunCache
from sequoia.engines.classical import simulate
from sequoia.engines.rates import has_rate_devices, simulate_rates
from sequoia.engines.report import format_rates_report, format_report
from sequoia.gui.canvas import PART_MIME
from sequoia.gui.canvas.items import CELL
from sequoia.gui.main_window import MainWindow
from sequoia.gui.palette import part_label
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, save_setup
from sequoia.setup.model import Setup

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "setups" / "hbt_heralded.yaml"
WAIT = 60000


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(Library(ROOT / "library"), settings=settings)
    qtbot.addWidget(win)
    win.show()
    win.confirm_discard = lambda w: "discard"
    return win


def drag_link(scene, src, dst):
    item = scene.component_items[src.split(".")[0]]
    start = scene.port_scene_position(*src.split("."))
    end = scene.port_scene_position(*dst.split("."))
    assert scene.start_link(item, src.split(".")[1], start)
    return scene.finish_link(end)


def commit_text(editor, text):
    editor = getattr(editor, "value_editor", editor)
    editor.edit.setText(text)
    editor.commit()


def drop_part(window, ref, position):
    mime = QMimeData()
    mime.setData(PART_MIME, ref.encode("utf-8"))
    point = window.view.mapFromScene(QPointF(position[0] * CELL, position[1] * CELL))
    event = QDropEvent(QPointF(point), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    window.view.dropEvent(event)
    selected = window.scene.selected_ids()
    assert len(selected) == 1
    return selected[0]


def augmented_example(tmp_path):
    raw = load_setup(EXAMPLE).model_dump(by_alias=True)
    for inst in raw["instances"]:
        if inst["id"] == "cpl_signal":
            inst["settings"] = {"defocus": "0.5 mm"}
        if inst["id"] == "spad_a":
            inst["overrides"] = {"dark_count_rate": "250 Hz"}
    raw["probes"] = [{"id": "pm_signal", "part": "probes/power_meter", "link": "s_signal"}]
    setup = Setup.model_validate(raw)
    path = tmp_path / "hbt_augmented.yaml"
    save_setup(setup, path)
    return path


def build_with_gui(window, target_path):
    doc, scene = window.document, window.scene
    example = load_setup(target_path)
    assert not doc.setup.instances
    scene.choose_fiber = lambda refs: "fiber/smf_810"

    for inst in example.instances:
        new_id = drop_part(window, inst.part, inst.position)
        form = window.inspector.form
        commit_text(form.id_row.editor, inst.id)
        assert window.scene.selected_ids() == [inst.id], (new_id, inst.id)
        form = window.inspector.form
        if inst.rotation:
            form.placement["rotation"].editor.spin.setValue(inst.rotation)
        for name, value in inst.settings.items():
            commit_text(form.settings_form.rows[name].editor, value)
        for name, value in inst.overrides.items():
            form.overrides_group.set_expanded(True)
            commit_text(form.overrides_form.rows[name].editor, value)
    assert not window.hint.isVisible()

    for link in example.links:
        link_id = drag_link(window.scene, link.src, link.dst)
        assert link_id is not None, (link.src, link.dst)
        window.scene.select(link_id)
        form = window.inspector.form
        assert form.link_kind == link.kind
        field = "delay" if link.kind == "cable" else "length"
        target = getattr(link, field)
        if link.kind == "fiber":
            assert doc.setup.links[-1].fiber == link.fiber
        if link.id == "s_focus":
            window.scene.begin_edit_link(link_id)
            window.scene.link_editor.setText(f"{target.value:g} {target.unit}")
            window.scene.link_editor.editingFinished.emit()
        else:
            commit_text(form.form.rows[field].editor, f"{target.value:g} {target.unit}")
        window.scene.select(link_id)
        commit_text(window.inspector.form.id_row.editor, link.id)
        assert window.scene.selected_ids() == [link.id]

    for probe in example.probes:
        point = window.scene.link_items[probe.link].point_at(0.5)
        menu = window.scene.build_context_menu(point)
        info = next(i for i in doc.library.index() if i.ref == probe.part)
        add = next(a for a in menu.actions() if a.text() == "Add probe")
        next(a for a in add.menu().actions() if a.text() == part_label(info)).trigger()
        (new_id,) = window.scene.selected_ids()
        assert new_id in window.scene.probe_items
        commit_text(window.inspector.form.id_row.editor, probe.id)

    window.scene.select_many([])
    form = window.inspector.form
    commit_text(form.rows["name"].editor, example.name)
    form.rows["description"].editor.text.setPlainText(example.description)
    form.rows["description"].editor.commit()
    editor = window.inspector.form.analyses
    for spec in example.analyses:
        editor.add_button.click()
        analysis = editor.form
        analysis.kind.setCurrentIndex(analysis.kind.findData(spec["type"]))
        analysis.name.setText(spec["name"])
        roles = (
            [spec["herald"], spec["a"], spec["b"]]
            if spec["type"] == "heralded_g2"
            else spec["channels"]
        )
        for combo, channel in zip(analysis.channel_combos, roles, strict=False):
            combo.setCurrentIndex(combo.findData(channel))
        if "window" in spec:
            analysis.window.edit.setText(spec["window"])
        if "range" in spec:
            analysis.range.edit.setText(spec["range"])
            analysis.bins.setValue(spec["bins"])
        assert analysis.apply(), analysis.error.text()
    return example


def report(path):
    library = Library(ROOT / "library")
    setup = load_setup(path)
    cache = RunCache()
    result = simulate(setup, library, cache=cache)
    text = format_report(result)
    if has_rate_devices(setup, library, cache, result.network):
        rates = simulate_rates(setup, library, classical=result, cache=cache)
        text += "\n" + format_rates_report(rates)
    return text


@pytest.fixture
def augmented(tmp_path):
    return augmented_example(tmp_path)


def test_build_hbt_from_scratch(window, tmp_path, qtbot, augmented):
    example = build_with_gui(window, augmented)
    assert window.isWindowModified()
    saved = tmp_path / "hbt_built.yaml"
    window.ask_save_path = lambda: str(saved)
    assert window.save_as()
    assert not window.isWindowModified()

    reloaded = load_setup(saved)
    assert len(reloaded.instances) == 14 and len(reloaded.links) == 15
    assert len(reloaded.probes) == 1 and len(reloaded.analyses) == 4
    built = reloaded.model_dump(by_alias=True)
    wanted = example.model_dump(by_alias=True)
    assert any(i["settings"] for i in wanted["instances"])
    assert any(i["overrides"] for i in wanted["instances"])
    assert built["probes"] == wanted["probes"] != []
    assert built["analyses"] == wanted["analyses"]
    assert built["instances"] == wanted["instances"]
    assert built["links"] == wanted["links"]
    assert built == wanted
    assert report(saved) == report(augmented)

    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    outcome = window.results.outcome
    assert outcome.ok and outcome.rates is not None
    assert outcome.revision == window.controller.revision
    assert outcome.classical.probes and "pm_signal" in outcome.classical.probes


def test_undo_everything_gives_empty_setup(window, augmented):
    build_with_gui(window, augmented)
    stack = window.document.undo_stack
    assert stack.count() > 10
    while stack.canUndo():
        stack.undo()
    setup = window.document.setup
    assert not setup.instances and not setup.links and not setup.probes and not setup.analyses
    assert setup.name == "untitled" and setup.description == ""
    assert not window.document.dirty
    assert not window.scene.component_items and not window.scene.link_items
    assert window.hint.isVisible()
    while stack.canRedo():
        stack.redo()
    assert len(window.document.setup.instances) == 14
    assert len(window.document.setup.probes) == 1


def test_example_open_shows_overlap_offer_and_results(window, qtbot):
    assert window.open_path(EXAMPLE)
    assert window.overlap_bar.isVisible()
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    outcome = window.results.outcome
    assert outcome.classical is not None and outcome.rates is not None
    assert window.results.links.rowCount() == 15
    assert "Done" in window.results.status.text()
    assert "Done" in window.run_label.text()


def test_edit_cancels_run_in_flight(window, qtbot):
    window.open_path(EXAMPLE)
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    window.controller.clear_cache()
    cancelled = []
    window.controller.cancelled.connect(lambda: cancelled.append(1))
    window.controller.run_now()
    assert window.controller.busy
    window.document.set_link_field("s_pump", "length", "123 mm")
    qtbot.waitUntil(lambda: bool(cancelled), timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    outcome = window.results.outcome
    assert outcome.revision == window.controller.revision


def test_render_png(window, qtbot):
    scratch = Path(
        "/private/tmp/claude-501/-Users-borankilic-Desktop-Research-Quantum-Technologies-SEQUOIA/"
        "5d0a0d99-6543-4fe7-aef3-c7336f569617/scratchpad"
    )
    window.resize(1500, 950)
    window.open_path(EXAMPLE)
    window.auto_layout()
    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    window.view.fit_to_content()
    image = window.grab()
    assert not image.isNull()
    if scratch.is_dir():
        image.save(str(scratch / "main_window_hbt_auto.png"))
