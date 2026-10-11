from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest

from sequoia.gui import app as gui_app
from sequoia.gui.start_screen import (
    StartScreen,
    Template,
    box_size,
    discover_templates,
    render_thumbnail,
)
from sequoia.library.store import Library
from sequoia.setup.io import load_setup
from sequoia.setup.validate import check_setup

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
EXPECTED = {
    "laser_power_meter",
    "malus_law",
    "beam_expander",
    "mach_zehnder",
    "laser_fiber_coupling",
    "spdc_car",
    "hbt_heralded",
    "hom",
    "bell_test",
}


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def screen(qtbot, settings):
    widget = StartScreen(TEMPLATES, settings)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_templates_discovered():
    templates = discover_templates(TEMPLATES)
    assert all(isinstance(t, Template) for t in templates)
    assert {t.path.stem for t in templates} == EXPECTED
    assert all(t.name and t.description and t.tags for t in templates)
    assert discover_templates(TEMPLATES / "missing") == []


@pytest.mark.parametrize("template", discover_templates(TEMPLATES), ids=lambda t: t.path.stem)
def test_template_valid(template):
    setup = load_setup(template.path)
    issues = check_setup(setup, Library(ROOT / "library"))
    assert [i for i in issues if i.severity == "error"] == []


@pytest.mark.parametrize("template", discover_templates(TEMPLATES), ids=lambda t: t.path.stem)
def test_template_no_overlap(template):
    setup = load_setup(template.path)
    boxes = []
    for inst in setup.instances:
        w, h = box_size(inst.rotation)
        x, y = inst.position
        boxes.append((inst.id, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            overlap = min(a[3], b[3]) - max(a[1], b[1]) > 1e-9 and (
                min(a[4], b[4]) - max(a[2], b[2]) > 1e-9
            )
            assert not overlap, (a[0], b[0])


def test_thumbnails(screen):
    assert len(screen.cards) == len(EXPECTED)
    for card in screen.cards:
        pixmap = card.thumbnail.pixmap()
        assert not pixmap.isNull()
        image = pixmap.toImage()
        colors = {
            image.pixel(x, y)
            for x in range(0, image.width(), 3)
            for y in range(0, image.height(), 3)
        }
        assert len(colors) > 3
    assert not render_thumbnail(load_setup(TEMPLATES / "bell_test.yaml")).isNull()


def test_buttons(screen, qtbot):
    with qtbot.waitSignal(screen.new_requested):
        QTest.mouseClick(screen.new_button, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(screen.open_requested):
        QTest.mouseClick(screen.open_button, Qt.MouseButton.LeftButton)


def test_template_click(screen, qtbot):
    card = screen.cards[0]
    with qtbot.waitSignal(screen.template_requested) as blocker:
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert blocker.args == [str(card.template.path)]


def test_recent(qtbot, settings):
    good = str(ROOT / "setups" / "mzi.yaml")
    settings.setValue("recent_files", [str(ROOT / "nope.yaml"), good])
    widget = StartScreen(TEMPLATES, settings)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.recent_list.count() == 1
    item = widget.recent_list.item(0)
    with qtbot.waitSignal(widget.path_requested) as blocker:
        widget.recent_list.itemDoubleClicked.emit(item)
    assert blocker.args == [good]


def test_no_recent(screen):
    assert screen.recent_list.count() == 0
    assert not screen.recent_empty.isHidden()


def test_launch_template_untitled(qtbot, settings, tmp_path, monkeypatch):
    args = ["--library", str(ROOT / "library")]
    screen = gui_app.show_start_screen(args, settings, TEMPLATES)
    qtbot.addWidget(screen)
    screen.show()
    template = str(TEMPLATES / "mach_zehnder.yaml")
    screen.template_requested.emit(template)
    window = screen.windows[0]
    qtbot.addWidget(window)
    window.confirm_discard = lambda w: "discard"
    assert window.document.path is None
    assert window.document.setup.name == "Mach-Zehnder interferometer"
    assert template not in window.recent_files()
    assert not screen.isVisible()


def test_launch_new_and_path(qtbot, settings):
    args = ["--library", str(ROOT / "library")]
    screen = gui_app.show_start_screen(args, settings, TEMPLATES)
    qtbot.addWidget(screen)
    screen.show()
    screen.new_requested.emit()
    assert screen.windows[0].document.path is None
    qtbot.addWidget(screen.windows[0])
    example = ROOT / "setups" / "mzi.yaml"
    screen.path_requested.emit(str(example))
    window = screen.windows[1]
    qtbot.addWidget(window)
    assert window.document.path == example
