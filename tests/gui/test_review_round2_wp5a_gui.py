import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtGui import QValidator
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStyleOptionViewItem,
    QTableWidgetSelectionRange,
)

from sequoia.fock import Circuit
from sequoia.fock.detectors import IdealPNR, PseudoPNR, Threshold
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import DetectorSpec, Experiment, run_experiment, save_experiment
from sequoia.fock.inputs import bell_pair
from sequoia.fock.run import SetupRequest, run_setup_request
from sequoia.gui.fock_common import ROWS_SHOWN, ResultTable
from sequoia.gui.fock_panel import FockTab
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[2]
LIB = Library(ROOT / "library")
WAIT = 120000


@pytest.fixture
def tab(qtbot):
    t = FockTab()
    t.confirm_replace = lambda _t: "discard"
    qtbot.addWidget(t)
    yield t
    t.shutdown()


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(Library(ROOT / "library"), settings=settings)
    w.confirm_discard = lambda _w: "discard"
    w.show_error = lambda title, message: pytest.fail(f"{title}: {message}")
    qtbot.addWidget(w)
    w.show()
    return w


def settle(qtbot, t):
    qtbot.waitUntil(lambda: not t.runner.busy, timeout=WAIT)
    qtbot.wait(30)


def result_tables(t: FockTab) -> list[ResultTable]:
    return [w for w in vars(t).values() if isinstance(w, ResultTable)]


def inside(widget, window) -> bool:
    top_left = widget.mapTo(window, QPoint(0, 0))
    return widget.isVisible() and window.rect().contains(QRect(top_left, widget.size()))


# U10: laptop-sized window


def test_run_controls_and_model_card_fit_a_1280x800_window(qtbot):
    win = QMainWindow()
    t = FockTab(lambda: (None, None))
    win.setCentralWidget(t)
    qtbot.addWidget(win)
    win.resize(1280, 800)
    win.show()
    qtbot.waitExposed(win)
    assert win.width() == 1280 and t.minimumSizeHint().width() <= 1280
    for w in (t.run_button, t.cancel_button, t.model_card, t.estimate_label, t.result_tabs):
        assert inside(w, win), w
    assert "run" in t.model_card.text().lower()
    tables = result_tables(t)
    assert len(tables) >= 9
    for table in tables:
        header = table.horizontalHeader()
        fm = header.fontMetrics()
        for c in range(table.columnCount()):
            item = table.horizontalHeaderItem(c)
            widest = max(fm.horizontalAdvance(line) for line in item.text().split("\n"))
            assert widest + 8 <= header.sectionSize(c) or item.toolTip(), item.text()
            assert item.toolTip()
    t.mode_box.setCurrentIndex(1)
    for w in (t.run_button, t.results_table, t.estimate_label):
        assert inside(w, win), w
    t.shutdown()


def test_empty_states_before_running(tab):
    for table in result_tables(tab):
        assert table.rowCount() == 0 and table.empty_text
    assert "Run" in tab.states_table.empty_text
    assert "Run" in tab.results_table.empty_text
    assert "Import" in tab.components_table.empty_text
    assert "Resources" in tab.estimate_label.text()


# U10: copy selected cells


def test_copy_selected_cells_with_keyboard_and_menu(tab, qtbot):
    tab.run()
    settle(qtbot, tab)
    t = tab.results_table
    t.clearSelection()
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)
    QApplication.clipboard().clear()
    QTest.keyClick(t, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    lines = QApplication.clipboard().text().split("\n")
    assert len(lines) == 2
    assert lines[0] == f"{t.item(0, 0).text()}\t{t.item(0, 1).text()}"
    with_headers = t.selected_text(headers=True).split("\n")
    assert with_headers[0] == "State\tProbability"
    seen = [a.text() for a in t.context_menu().actions()]
    assert "Copy" in seen and "Copy with headers" in seen


# U11: estimate shown before Run, large distributions as top-k with hidden mass


def test_estimate_updates_with_edits(tab):
    assert "modes" in tab.estimate_label.text() and "work" in tab.estimate_label.text()
    tab.max_photons.setValue(1)
    assert "will not run" in tab.estimate_label.text()
    tab.max_photons.setValue(10)
    assert "will not run" not in tab.estimate_label.text()
    tab.input_table.item(0, 1).setText("x")
    assert "not valid" in tab.estimate_label.text()


def test_large_distribution_shows_top_rows_and_hidden_mass(tab, qtbot):
    c = Circuit(10)
    for k in range(9):
        c.beam_splitter(k, k + 1, 1.1)
    for k in range(0, 9, 2):
        c.beam_splitter(k, k + 1, 0.6)
    tab.set_experiment(Experiment(c, [1, 1, 1, 1] + [0] * 6))
    tab.run()
    settle(qtbot, tab)
    n = len(tab.result.probabilities)
    assert n > ROWS_SHOWN
    t = tab.results_table
    assert t.rowCount() == ROWS_SHOWN + 1
    assert t.hidden_rows == n - ROWS_SHOWN
    shown = sum(float(t.item(r, 1).text()) for r in range(ROWS_SHOWN))
    assert t.hidden_mass == pytest.approx(1.0 - shown, abs=1e-4)
    assert "more rows" in t.item(ROWS_SHOWN, 0).text()
    assert "hidden" in t.item(ROWS_SHOWN, 1).text()
    first = float(t.item(0, 1).text())
    assert first == max(float(t.item(r, 1).text()) for r in range(ROWS_SHOWN))
    t.sortItems(1, Qt.SortOrder.AscendingOrder)
    assert "more rows" in t.item(t.rowCount() - 1, 0).text()
    assert float(t.item(0, 1).text()) <= float(t.item(1, 1).text())
    t.sortItems(1, Qt.SortOrder.DescendingOrder)
    assert "more rows" in t.item(t.rowCount() - 1, 0).text()
    assert float(t.item(0, 1).text()) == first
    t.show_more()
    assert t.rowCount() == min(n, 2 * ROWS_SHOWN) + (1 if n > 2 * ROWS_SHOWN else 0)
    t.show_all()
    assert t.rowCount() == n and t.hidden_rows == 0
    assert "most probable" in tab.plot.getPlotItem().titleLabel.text


def test_preflight_refusal_in_setup_mode(window, qtbot):
    assert window.open_path(ROOT / "templates" / "fock_hom.yaml")
    qtbot.waitUntil(lambda: window.controller.last_outcome is not None, timeout=WAIT)
    qtbot.waitUntil(lambda: not window.controller.busy, timeout=WAIT)
    tab = window.results.fock
    tab.import_from_setup()
    settle(qtbot, tab)
    assert "Resources of the drawn setup" in tab.estimate_label.text()
    tab.max_photons.setValue(1)
    tab.run()
    settle(qtbot, tab)
    assert tab.status.text().startswith("not run")
    assert tab.setup_result is None and tab.states_table.rowCount() == 0
    assert "Not run" in tab.states_table.empty_text
    assert "Estimate:" in tab.notes_view.toPlainText()
    assert tab.components_table.rowCount() > 0
    tab.max_photons.setValue(10)
    tab.run()
    settle(qtbot, tab)
    assert tab.setup_result is not None and tab.states_table.rowCount() > 0


def test_partial_setup_result_is_labelled(window, qtbot):
    tab = window.results.fock
    flag = threading.Event()
    setup = load_setup(ROOT / "templates" / "fock_hom.yaml")
    out = run_setup_request(
        SetupRequest("run", setup, LIB, shots=50),
        flag.is_set,
        lambda d, _t: flag.set() if d >= 4 else None,
    )
    tab.begin_run()
    tab._done(out)
    assert "partial" in tab.status.text()
    assert "PARTIAL" in tab.model_card.text()
    assert "cancelled before sampling" in tab.samples_note.text()
    assert tab.states_table.rowCount() > 0


# U10: validated integer editors


def test_photon_and_wire_editors_accept_whole_numbers_only(tab):
    for col in (1, 5):
        index = tab.input_table.model().index(0, col)
        delegate = tab.input_table.itemDelegateForColumn(col)
        editor = delegate.createEditor(tab.input_table.viewport(), QStyleOptionViewItem(), index)
        v = editor.validator()
        assert v.validate("2", 0)[0] == QValidator.State.Acceptable
        assert v.validate("1.5", 0)[0] != QValidator.State.Acceptable
        assert v.validate("-1", 0)[0] != QValidator.State.Acceptable
    tab.input_table.item(0, 1).setText("1.5")
    with pytest.raises(Exception, match="whole number"):
        tab.experiment()


# U10: open and save without edits preserves the experiment


def _rich() -> Experiment:
    c = Circuit(4).beam_splitter(0, 1).beam_splitter(2, 3, 1.0).phase_shifter(1, 0.3, "arm")
    return Experiment(
        c,
        [1, 1, 1, 0],
        tags=[[0], [0], ["x"], []],
        brightness=0.123456789,
        indistinguishability=0.987654321,
        detectors=[
            DetectorSpec(
                (0,),
                PseudoPNR(wires=2, weights=(0.9, 0.1), wire_efficiencies=(0.5, 0.8), dark=0.2),
            ),
            DetectorSpec((1,), Threshold(efficiency=0.7, dark=0.05)),
            DetectorSpec((2, 3), IdealPNR(efficiency=0.9, dark=0.01)),
        ],
        shots=17,
        seed=None,
        conditions={"herald": {0: 1}, "pair": {1: (">=", 1), 2: 1}},
        name="rich",
        description="preserve this",
        mode_labels=["a", "b", "c", "d"],
        limits=Limits(max_photons=12),
        rate_hz=1e6,
        duration_s=3.0,
    )


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
@pytest.mark.parametrize(
    "exp",
    [
        _rich(),
        Experiment(Circuit(4).beam_splitter(0, 2), [0] * 4, input_mixture=bell_pair(0, 1, 2, 3)),
    ],
)
def test_open_then_save_without_edits_is_byte_identical(tab, tmp_path, exp, suffix):
    first = tmp_path / f"a{suffix}"
    second = tmp_path / f"b{suffix}"
    save_experiment(exp, first)
    assert tab.open_path(first)
    assert not tab.editor_dirty
    assert tab.save_path(second)
    assert second.read_bytes() == first.read_bytes()
    a, b = run_experiment(exp), run_experiment(tab.experiment())
    assert a.probabilities == pytest.approx(b.probabilities)
    for k, cond in a.conditionals.items():
        assert b.conditionals[k].success_probability == pytest.approx(cond.success_probability)
