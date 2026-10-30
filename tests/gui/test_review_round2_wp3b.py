import html
import math
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from sequoia.errors import Issue
from sequoia.gui import app as gui_app
from sequoia.gui.main_window import MainWindow
from sequoia.gui.recipes import ALL_TOOLS, RECIPES, START_CHOICES, TASKS
from sequoia.gui.results import ResultsDock
from sequoia.gui.start_screen import StartScreen
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
WAIT = 120000
ARGS = ["--library", str(ROOT / "library")]


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qtbot, settings):
    w = MainWindow(Library(ROOT / "library"), settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


def open_template(qtbot, w, name):
    assert w.open_path(TEMPLATES / f"{name}.yaml")
    qtbot.waitUntil(lambda: w.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not w.controller.busy, timeout=WAIT)
    qtbot.wait(30)


class WindowSpy(QObject):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.shown = []

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.Type.Show
            and hasattr(obj, "isWindow")
            and obj.isWindow()
            and obj is not self.main
            and not (int(obj.windowFlags()) & int(Qt.WindowType.Popup) == int(Qt.WindowType.Popup))
        ):
            self.shown.append(type(obj).__name__)
        return False


def titles(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_tasks_move_the_same_widgets_and_all_tools_restores_the_order(qtbot):
    dock = ResultsDock()
    qtbot.addWidget(dock)
    order = list(dock.tools)
    assert dock.task == ALL_TOOLS and titles(dock.tabs) == order
    assert "Compare" in order and dock.tools["Compare"] is dock.compare
    for name, recipe in RECIPES.items():
        dock.show_task(name)
        assert dock.pages.currentIndex() == 0
        assert titles(dock.task_tabs) == list(recipe.tools)
        for i, title in enumerate(recipe.tools):
            assert dock.task_tabs.widget(i) is dock.tools[title]
        assert dock.tabs.count() + dock.task_tabs.count() == len(order)
        assert html.escape(recipe.name) in dock.recipe.text()
        assert "Default engine" in dock.recipe.text()
        for i in range(dock.task_tabs.count()):
            dock.task_tabs.setCurrentIndex(i)
            assert "Model:" in dock.help.text(), dock.task_tabs.tabText(i)
    dock.show_task(ALL_TOOLS)
    assert dock.pages.currentIndex() == 1 and dock.task_tabs.count() == 0
    assert titles(dock.tabs) == order
    for i in range(dock.tabs.count()):
        dock.tabs.setCurrentIndex(i)
        assert dock.help.text(), dock.tabs.tabText(i)


def test_task_is_chosen_from_list_menu_and_persisted(window, qtbot, settings):
    r = window.results
    r.task_list.setCurrentRow(TASKS.index("Beam & alignment"))
    assert r.task == "Beam & alignment" and r.current_tool() is r.caustic
    assert settings.value("results/task") == "Beam & alignment"
    window.task_actions["Compare & export"].trigger()
    assert r.task == "Compare & export" and r.current_tool() is r.compare
    assert window.task_actions["Compare & export"].isChecked()
    window.show_fock()
    assert r.task == "Two-photon interference" and r.current_tool() is r.fock
    window.close()
    again = MainWindow(Library(ROOT / "library"), settings=settings)
    qtbot.addWidget(again)
    assert again.results.task == "Two-photon interference"
    assert again.results.task_tabs.indexOf(again.results.fock) >= 0


def test_switching_tasks_opens_no_stray_windows(window, qtbot):
    open_template(qtbot, window, "hbt_heralded")
    spy = WindowSpy(window)
    QApplication.instance().installEventFilter(spy)
    try:
        for name in (*TASKS, *TASKS):
            window.show_task(name)
            qtbot.wait(5)
    finally:
        QApplication.instance().removeEventFilter(spy)
    assert spy.shown == []


def test_compare_tab_is_registered_with_the_run_guard(window):
    r = window.results
    assert r.plot_tabs["Compare"] is r.compare
    assert r.compare.revision_source == window.controller.document_key
    assert r.compare.on_issues is not None


def test_diagnostics_show_engine_and_run_sorted(window, qtbot):
    open_template(qtbot, window, "hbt_heralded")
    r = window.results
    headers = [r.diagnostics.horizontalHeaderItem(c).text() for c in range(6)]
    assert headers == ["", "Engine", "Run", "Code", "Subject", "Message"]
    r.set_engine_issues(
        "Gaussian",
        [
            Issue("warning", "gaussian_truncated", "'crystal'", "dropped 1% of pairs").for_run(
                "gaussian", "run-7"
            ),
            Issue("error", "gaussian_failed", "'crystal'", "no detectors"),
        ],
    )
    rows = [
        [r.diagnostics.item(i, c).text() for c in range(1, 6)]
        for i in range(r.diagnostics.rowCount())
    ]
    tagged = next(row for row in rows if row[2] == "gaussian_truncated")
    assert tagged[:2] == ["gaussian", "run-7"] and tagged[4] == "dropped 1% of pairs"
    untagged = next(row for row in rows if row[2] == "gaussian_failed")
    assert untagged[:2] == ["Gaussian", "last run"]
    assert all(row[:2] == ["setup", "current"] for row in rows if not row[2].startswith("gauss"))
    severities = [
        r.diagnostics.item(i, 0).data(Qt.ItemDataRole.UserRole + 1) for i in range(len(rows))
    ]
    assert severities == sorted(severities)
    assert rows[0][2] == "gaussian_failed"
    window.show_tool(r.gaussian)
    assert "dropped 1% of pairs" in r.help.text()
    activated = []
    r.issue_activated.connect(activated.append)
    r.diagnostics.selectRow(rows.index(tagged))
    assert activated == ["'crystal'"]
    r.set_engine_issues("Gaussian", [])
    assert "dropped" not in r.help.text()


def test_rate_counts_name_rates_and_denominators(window, qtbot):
    open_template(qtbot, window, "hbt_heralded")
    r = window.results
    headers = [r.channels.horizontalHeaderItem(c).text() for c in range(r.channels.columnCount())]
    assert "Recorded singles [Hz]" in headers and "Potential clicks [Hz]" in headers
    assert "dead time" in r.channels.horizontalHeaderItem(2).toolTip()
    col = headers.index("Dead-time survival")
    assert all(r.channels.item(i, col).text().endswith("%") for i in range(r.channels.rowCount()))
    metrics = {r.metrics.item(i, 0).text() for i in range(r.metrics.rowCount())}
    assert {"CAR", "heralding efficiency", "heralded g2(0)"} <= metrics
    window.show_tool(r.counts)
    assert "non-paralyzable dead time" in r.help.text()
    link_headers = [r.links.horizontalHeaderItem(c).text() for c in range(r.links.columnCount())]
    assert "P end [mW]" in link_headers and "Pair photons [1/s]" in link_headers


def test_metric_text_gives_reasons():
    from sequoia.gui.results import metric_text

    assert metric_text("CAR", math.nan) == "undefined: no accidental estimate"
    assert metric_text("heralded g2(0)", math.nan).startswith("undefined: ")
    assert metric_text("heralding efficiency", 0.25) == "25 %"


def test_start_screen_offers_tasks(qtbot, settings):
    screen = StartScreen(TEMPLATES, settings)
    qtbot.addWidget(screen)
    screen.show()
    assert set(screen.choices) == {c.label for c in START_CHOICES}
    with qtbot.waitSignal(screen.recipe_requested) as blocker:
        QTest.mouseClick(screen.choices["Characterise a heralded source"], Qt.LeftButton)
    assert blocker.args == [str(TEMPLATES / "hbt_heralded.yaml"), "Source characterisation"]


def test_new_user_runs_and_exports_a_bell_measurement_plan(qtbot, settings, tmp_path, monkeypatch):
    screen = gui_app.show_start_screen(ARGS, settings, TEMPLATES)
    qtbot.addWidget(screen)
    screen.show()
    QTest.mouseClick(screen.choices["Measure polarization correlations"], Qt.LeftButton)
    window = screen.windows[0]
    qtbot.addWidget(window)
    window.confirm_discard = lambda _w: "discard"
    qtbot.waitUntil(lambda: window.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    r = window.results
    assert window.document.path is None
    assert r.task == "Polarization correlations / CHSH"
    assert titles(r.task_tabs) == ["CHSH", "Polarization"]
    assert r.current_tool() is r.chsh and r.chsh.isVisible()
    assert r.polarization.views.currentIndex() == 1
    assert "four setting pairs" in r.recipe.text()
    chsh = r.chsh
    assert [o.currentText() for o in chsh.outcomes] == ["tt", "tr", "rt", "rr"]
    QTest.mouseClick(chsh.run_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: chsh.result is not None and not chsh.runner.busy, timeout=WAIT)
    pairs = [chsh.table.item(i, 0).text() for i in range(chsh.table.rowCount())]
    assert pairs == ["a b", "a b'", "a' b", "a' b'"]
    assert abs(chsh.result.s) > 2
    target = tmp_path / "plan.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    QTest.mouseClick(chsh.export_button, Qt.LeftButton)
    text = target.read_text()
    assert "acquisition plan" in text and "raw coincidences" in text
    assert "per setting pair" in text
