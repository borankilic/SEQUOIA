import re
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from sequoia.engines.cache import RunCache
from sequoia.engines.classical import simulate
from sequoia.engines.rates import has_rate_devices, simulate_rates
from sequoia.engines.report import format_rates_report, format_report
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

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


def build_hbt_from_scratch(window, target_example):
    doc, scene = window.document, window.scene
    example = load_setup(target_example)
    assert not doc.setup.instances

    with doc.macro("Place parts"):
        for inst in example.instances:
            if inst.id == "pump":
                continue
            doc.add_instance(inst.part, inst.position, instance_id=inst.id)
    new_id = scene.add_part("sources/laser_405_cw", example.instances[0].position)
    assert new_id == "laser_405_cw"
    assert not window.hint.isVisible()
    doc.remove_instance(new_id)
    doc.add_instance("sources/laser_405_cw", example.instances[0].position, instance_id="pump")

    for inst in example.instances:
        if inst.rotation:
            doc.rotate_instance(inst.id, inst.rotation)

    for link in example.links:
        if link.kind == "free_space":
            link_id = drag_link(scene, link.src, link.dst)
            assert link_id is not None
            doc.set_link_field(link_id, "length", link.length.model_dump(by_alias=True))
        else:
            kind, reason = doc.can_connect(link.src, link.dst)
            assert kind == link.kind, reason
            fields = {}
            if link.kind == "fiber":
                fields = {"fiber": link.fiber, "length": link.length.model_dump(by_alias=True)}
            else:
                fields = {"delay": link.delay.model_dump(by_alias=True)}
            doc.add_link(kind, link.src, link.dst, **fields)

    raw = example.model_dump(by_alias=True)
    doc.set_name(raw["name"])
    doc.set_description(raw["description"])
    doc.set_analyses(raw["analyses"])
    return example


def normalized(setup):
    raw = setup.model_dump(by_alias=True)
    instances = {i["id"]: {k: v for k, v in i.items() if k != "position"} for i in raw["instances"]}
    links = {}
    for link in raw["links"]:
        link = dict(link)
        link.pop("id")
        links[(link.get("from"), link.get("to"))] = link
    probes = {p["id"]: p for p in raw["probes"]}
    return {
        "name": raw["name"],
        "description": raw["description"],
        "grid_cell": raw["grid_cell"],
        "instances": instances,
        "links": links,
        "probes": probes,
        "analyses": raw["analyses"],
    }


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


def test_build_hbt_from_scratch(window, tmp_path, qtbot):
    example = build_hbt_from_scratch(window, EXAMPLE)
    assert window.isWindowModified()
    saved = tmp_path / "hbt_built.yaml"
    window.ask_save_path = lambda: str(saved)
    assert window.save_as()
    assert not window.isWindowModified()

    reloaded = load_setup(saved)
    assert len(reloaded.instances) == 14 and len(reloaded.links) == 15
    assert normalized(reloaded) == normalized(example)
    assert normalized(reloaded) == normalized(load_setup(EXAMPLE))
    for link in reloaded.links:
        assert link.id
    assert {i.id for i in reloaded.instances} == {i.id for i in example.instances}

    by_ends = {(link.src, link.dst): link.id for link in load_setup(EXAMPLE).links}
    rename = {link.id: by_ends[(link.src, link.dst)] for link in reloaded.links}
    pattern = re.compile(r"\b(" + "|".join(sorted(rename, key=len, reverse=True)) + r")\b")

    def squeeze(text):
        return [" ".join(line.split()) for line in text.splitlines()]

    def rename_line(line):
        head, _, tail = line.partition(" ")
        bare = head.rstrip(":")
        if bare in rename and tail.split()[:1] != ["kind"]:
            head = rename[bare] + head[len(bare) :]
        tail = pattern.sub(
            lambda m: m.group(1) if m.group(1) == "link" else rename[m.group(1)], tail
        )
        return f"{head} {tail}".strip()

    renamed = [rename_line(line) for line in squeeze(report(saved))]
    assert renamed == squeeze(report(EXAMPLE))

    qtbot.waitUntil(lambda: window.results.outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    outcome = window.results.outcome
    assert outcome.ok and outcome.rates is not None
    assert outcome.revision == window.controller.revision


def test_undo_everything_gives_empty_setup(window):
    build_hbt_from_scratch(window, EXAMPLE)
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
