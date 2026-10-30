from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication

import sequoia.components.builtin  # noqa: F401
from sequoia.gui.document import SetupDocument
from sequoia.gui.inspector.panel import InspectorPanel
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[2]
LIB = Library(ROOT / "library")


class Spy(QObject):
    def __init__(self):
        super().__init__()
        self.shown = []
        self.main = None

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


@pytest.fixture
def spy():
    s = Spy()
    QApplication.instance().installEventFilter(s)
    yield s
    QApplication.instance().removeEventFilter(s)


@pytest.fixture
def doc():
    return SetupDocument.open(ROOT / "templates" / "spdc_car.yaml", LIB)


@pytest.fixture
def editor(qtbot, doc, spy):
    panel = InspectorPanel(doc)
    qtbot.addWidget(panel)
    spy.main = panel
    panel.show()
    panel.show_selection(None)
    panel.form.measurement_group.set_expanded(True)
    yield panel.form.measurement
    qtbot.wait(20)
    assert spy.shown == []


def type_in(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def rows(editor):
    return editor.rows


def test_every_field_edits_the_measurement(editor, doc):
    r = rows(editor)
    assert doc.setup.measurement is None
    assert not editor.remove_button.isEnabled()
    type_in(r["purpose"].editor.edit, "heralding efficiency")
    m = doc.setup.measurement
    assert m is not None and m.purpose == "heralding efficiency"
    assert editor.remove_button.isEnabled()
    r["coincidence_window"].optional_box.click()
    type_in(r["coincidence_window"].editor.edit, "2 ns")
    assert doc.setup.measurement.coincidence_window.si == pytest.approx(2e-9)
    r["acquisition_duration"].optional_box.click()
    type_in(r["acquisition_duration"].editor.edit, "250 ms")
    assert doc.setup.measurement.acquisition_duration.si == pytest.approx(0.25)
    r["repeats"].editor.spin.setValue(3)
    r["accidentals"].editor.combo.setCurrentIndex(
        r["accidentals"].editor.combo.findData("sideband")
    )
    r["dead_time"].editor.combo.setCurrentIndex(r["dead_time"].editor.combo.findData("none"))
    m = doc.setup.measurement
    assert (m.repeats, m.accidentals, m.dead_time) == (3, "sideband", "none")
    clock = r["pulse_clock"].editor
    clock.kind.setCurrentIndex(clock.kind.findData("rate"))
    assert doc.setup.measurement.pulse_clock.kind == "sources"
    assert "repetition rate" in r["pulse_clock"].error.text()
    type_in(clock.rate, "80 MHz")
    assert doc.setup.measurement.pulse_clock.rate.si == pytest.approx(80e6)
    clock.kind.setCurrentIndex(clock.kind.findData("source"))
    clock.source.setCurrentText("pump")
    clock.commit()
    assert doc.setup.measurement.pulse_clock.source in {i.id for i in doc.setup.instances}
    clock.kind.setCurrentIndex(clock.kind.findData("cw"))
    assert doc.setup.measurement.pulse_clock.kind == "cw"
    channels = r["channels"].editor
    channels.add_button.click()
    _, k, v = channels.rows[-1]
    k.setText("spad_s")
    type_in(v, "tagger.ch3")
    assert doc.setup.measurement.channels == {"spad_s": "tagger.ch3"}
    centers = r["centers"].editor
    centers.add_button.click()
    _, k, v = centers.rows[-1]
    k.setText("pairs")
    type_in(v, "0.5 ns")
    assert doc.setup.measurement.centers["pairs"].si == pytest.approx(0.5e-9)
    cond = r["conditions"].editor
    cond.add_button.click()
    assert doc.setup.measurement.conditions == []
    cond.table.item(0, 0).setText("H")
    cond.table.item(0, 1).setText("tagger.ch1, spad_i")
    cond.table.item(0, 2).setText("tagger.ch4")
    cond.table.item(0, 3).setText("idler heralds, no third click")
    c = doc.setup.measurement.conditions[0]
    assert (c.name, c.herald, c.veto, c.description) == (
        "H",
        ["tagger.ch1", "spad_i"],
        ["tagger.ch4"],
        "idler heralds, no third click",
    )
    card = editor.card.text()
    assert "heralding efficiency" in card and "H: click on tagger.ch1 and spad_i" in card
    assert "2 ns full width" in card and "spad_s -&gt; tagger.ch3" in card


def test_undo_redo_and_reopen(editor, doc, tmp_path):
    r = rows(editor)
    r["coincidence_window"].optional_box.click()
    type_in(r["coincidence_window"].editor.edit, "3 ns")
    r["repeats"].editor.spin.setValue(4)
    saved = doc.setup.measurement
    assert saved.repeats == 4
    doc.undo_stack.undo()
    assert doc.setup.measurement.repeats == 1
    assert r["repeats"].editor.spin.value() == 1
    doc.undo_stack.redo()
    assert doc.setup.measurement == saved
    assert r["repeats"].editor.spin.value() == 4
    doc.save(tmp_path / "m.yaml")
    assert load_setup(tmp_path / "m.yaml").measurement == saved
    reopened = SetupDocument.open(tmp_path / "m.yaml", LIB)
    panel = InspectorPanel(reopened)
    editor2 = panel.form.measurement
    assert editor2.rows["coincidence_window"].editor.edit.text() == "3 ns"
    assert editor2.rows["repeats"].editor.spin.value() == 4
    assert panel.form.measurement_group.is_expanded()
    panel.deleteLater()


def test_invalid_values_are_shown_and_not_applied(editor, doc):
    r = rows(editor)
    r["coincidence_window"].optional_box.click()
    type_in(r["coincidence_window"].editor.edit, "1 ns")
    before = doc.setup.measurement
    type_in(r["coincidence_window"].editor.edit, "-1 ns")
    assert doc.setup.measurement == before
    assert "positive" in r["coincidence_window"].error.text()
    assert not r["coincidence_window"].error.isHidden()
    type_in(r["coincidence_window"].editor.edit, "3 kg")
    assert doc.setup.measurement == before and r["coincidence_window"].error.text()
    cond = r["conditions"].editor
    cond.add_button.click()
    cond.table.item(0, 1).setText("not a ref!")
    assert doc.setup.measurement.conditions == []
    assert "condition 'C1'" in r["conditions"].error.text()
    cond.add_button.click()
    cond.table.item(0, 1).setText("tagger.ch1")
    cond.table.item(1, 0).setText("C1")
    cond.table.item(1, 1).setText("tagger.ch2")
    assert doc.setup.measurement.conditions == []
    assert "unique" in r["conditions"].error.text()
    assert cond.table.rowCount() == 2
    cond.table.item(1, 0).setText("C2")
    assert [c.name for c in doc.setup.measurement.conditions] == ["C1", "C2"]
    channels = r["channels"].editor
    channels.add_button.click()
    _, k, v = channels.rows[-1]
    k.setText("spad_s")
    type_in(v, "no_dot")
    assert doc.setup.measurement.channels == {}
    assert r["channels"].error.text()


def test_remove_measurement_and_undo(editor, doc):
    type_in(rows(editor)["purpose"].editor.edit, "x")
    assert doc.setup.measurement is not None
    editor.remove_button.click()
    assert doc.setup.measurement is None
    assert not editor.remove_button.isEnabled()
    assert "No measurement definition" in editor.state.text()
    doc.undo_stack.undo()
    assert doc.setup.measurement.purpose == "x"
    assert editor.rows["purpose"].editor.edit.text() == "x"


def test_remove_refused_when_analyses_need_the_window(editor, doc):
    rows(editor)["coincidence_window"].optional_box.click()
    type_in(rows(editor)["coincidence_window"].editor.edit, "1 ns")
    spec = {k: v for k, v in doc.setup.analyses[0].items() if k != "window"}
    doc.set_analyses([spec])
    editor.remove_button.click()
    assert doc.setup.measurement is not None
    assert "window" in editor.error.text() and not editor.error.isHidden()


def test_photon_counts_conditional_tab_lists_named_conditions(qtbot, spy):
    from sequoia.fock.run import SetupRequest, run_setup_request
    from sequoia.gui.fock_results import SetupResultsView
    from sequoia.setup.io import setup_from_dict, setup_to_dict

    raw = setup_to_dict(load_setup(ROOT / "setups" / "hbt_heralded.yaml"))
    raw["measurement"] = {
        "conditions": [{"name": "H", "herald": ["tagger.ch1"], "veto": ["spad_b"]}]
    }
    out = run_setup_request(SetupRequest("run", setup_from_dict(raw), LIB))
    view = SetupResultsView()
    qtbot.addWidget(view)
    spy.main = view
    view.show()
    view.show_result(out.result, out.views)
    nc = out.views.conditions[0]
    assert "H: click on tagger.ch1, none on spad_b" in view.named_summary.text()
    view.named_condition.setCurrentIndex(view.named_condition.findText("H"))
    text = view.cond_summary.text()
    assert "'H'" in text and f"{nc.conditional.success_probability:.6g}" in text
    assert not view.cond_table.isEnabled()
    assert view.cond_result.rowCount() == len(nc.conditional.joint)
    view.named_condition.setCurrentIndex(0)
    assert view.cond_table.isEnabled()
    qtbot.wait(20)
    assert spy.shown == []
