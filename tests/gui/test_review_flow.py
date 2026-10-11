import random
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest

from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_to_dict

ROOT = Path(__file__).resolve().parents[2]
WAIT = 90000
TEMPLATES = sorted((ROOT / "templates").glob("*.yaml"))


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(Library(ROOT / "library"), settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.ask_save_path = lambda: str(tmp_path / "saved.yaml")
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


def settle(qtbot, w):
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.stem)
def test_every_template_opens_and_every_tab_fills(window, qtbot, path):
    assert window.open_path(path)
    qtbot.waitUntil(lambda: window.controller.last_outcome is not None, timeout=WAIT)
    settle(qtbot, window)
    outcome = window.controller.last_outcome
    assert outcome.ok, [i for i in outcome.issues if i.severity == "error"]
    window.auto_layout()
    results = window.results
    for index in range(results.tabs.count()):
        results.tabs.setCurrentIndex(index)
        qtbot.wait(5)
    assert not results.tab_errors, results.tab_errors
    for link_id in window.scene.link_items:
        results.select_link(link_id)
    assert window._overlapping() is False


def test_save_reopen_and_back_to_start(window, qtbot, tmp_path):
    started = []
    window.launcher = lambda: started.append(1)
    work = tmp_path / "work.yaml"
    work.write_text((ROOT / "templates" / "malus_law.yaml").read_text())
    window.open_path(work)
    settle(qtbot, window)
    window.document.set_setting("hwp", "angle", "30 deg")
    assert window.document.dirty
    assert window.save()
    saved = work
    assert saved.is_file()
    assert setup_to_dict(load_setup(saved)) == setup_to_dict(window.document.setup)
    assert not window.document.dirty
    window.new_window()
    assert started == [1]
    assert window.back_to_start()
    assert started == [1, 1]
    assert not window.isVisible()


def test_dirty_experiment_prompts_before_going_back(window, qtbot):
    window.open_path(ROOT / "templates" / "malus_law.yaml")
    settle(qtbot, window)
    window.document.set_setting("hwp", "angle", "5 deg")
    window.confirm_discard = lambda _w: "cancel"
    assert not window.back_to_start()
    assert window.isVisible()
    window.confirm_discard = lambda _w: "discard"
    assert window.back_to_start()


@pytest.mark.parametrize("seed", range(3))
def test_canvas_gesture_fuzz(window, qtbot, seed):
    rng = random.Random(seed)
    window.open_path(ROOT / "templates" / "hbt_heralded.yaml")
    settle(qtbot, window)
    window.auto_layout()
    view = window.view
    vp = view.viewport()
    ids = list(window.scene.component_items)
    for _ in range(60):
        action = rng.choice(["click", "drag", "pan", "zoom", "key", "link", "undo", "fit"])
        if action == "click":
            item = window.scene.component_items[rng.choice(ids)]
            QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, view.mapFromScene(item.pos()))
        elif action == "drag":
            item = window.scene.component_items[rng.choice(ids)]
            start = view.mapFromScene(item.pos())
            QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
            for k in range(1, 5):
                QTest.mouseMove(
                    vp, start + QPoint(rng.randint(-60, 60) * k // 4, rng.randint(-60, 60) * k // 4)
                )
            QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, start + QPoint(20, 10))
        elif action == "pan":
            start = QPoint(rng.randint(5, vp.width() - 5), rng.randint(5, vp.height() - 5))
            if view.itemAt(start) is None:
                QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
                QTest.mouseMove(vp, start + QPoint(rng.randint(-80, 80), rng.randint(-80, 80)))
                QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, start)
        elif action == "zoom":
            view.zoom_by(rng.choice([0.5, 0.8, 1.25, 2.0]))
        elif action == "key":
            window.scene.select(rng.choice(ids))
            QTest.keyClick(
                vp,
                rng.choice([Qt.Key_R, Qt.Key_F, Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]),
            )
        elif action == "link":
            a, b = rng.sample(ids, 2)
            ports_a = [p.name for p in window.document.ports(a)]
            ports_b = [p.name for p in window.document.ports(b)]
            if ports_a and ports_b:
                pa = window.scene.port_scene_position(a, rng.choice(ports_a))
                pb = window.scene.port_scene_position(b, rng.choice(ports_b))
                item = window.scene.component_items[a]
                name = rng.choice(ports_a)
                if window.scene.start_link(item, name, pa):
                    window.scene.update_link_drag(pb)
                    window.scene.finish_link(pb)
        elif action == "undo":
            if window.document.undo_stack.canUndo():
                window.undo()
            elif window.document.undo_stack.canRedo():
                window.redo()
        elif action == "fit":
            view.fit_to_content()
        qtbot.wait(2)
    settle(qtbot, window)
    assert set(window.scene.component_items) == {i.id for i in window.document.setup.instances}
    assert set(window.scene.link_items) == {x.id for x in window.document.setup.links}
    _ = QPointF


def test_link_type_switch_through_the_inspector(window, qtbot):
    window.open_path(ROOT / "templates" / "hbt_heralded.yaml")
    settle(qtbot, window)
    window.scene.select("s_bp")
    form = window.inspector.form
    assert form.medium_combo.currentData() == "free_space"
    form.medium_combo.setCurrentIndex(1)
    form._medium_chosen(1)
    settle(qtbot, window)
    fibers = [
        x
        for x in window.document.setup.links
        if x.kind == "fiber" and x.id not in ("f_signal", "f_idler", "f_a", "f_b")
    ]
    assert fibers
    window.scene.select(fibers[0].id)
    assert window.inspector.form.medium_combo.currentData() == "fiber"


@pytest.mark.parametrize("template", ["laser_power_meter.yaml", None])
def test_analysis_tabs_survive_setups_without_photon_detectors(window, qtbot, template):
    if template is not None:
        window.open_path(ROOT / "templates" / template)
    settle(qtbot, window)
    results = window.results
    for tab in (results.sweeps, results.montecarlo, results.gaussian):
        tab.run()
        qtbot.waitUntil(lambda t=tab: not t.runner.busy, timeout=WAIT)
    results.budget.run_ideal()
    qtbot.waitUntil(lambda: not results.budget.runner.busy, timeout=WAIT)
    window.show_optimizer()
    dialog = results.sweeps.optimizer
    assert dialog is not None
    dialog.run()
    qtbot.waitUntil(lambda: not dialog.runner.busy, timeout=WAIT)
    dialog.close()
    assert window.isVisible()
