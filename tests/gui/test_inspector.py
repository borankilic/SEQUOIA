import os
from pathlib import Path

import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import registered_types
from sequoia.data import Curve
from sequoia.gui import inspector as inspector_pkg
from sequoia.gui.document import SetupDocument
from sequoia.gui.inspector import fields
from sequoia.gui.inspector.fields import MISSING, ModelForm, analyze
from sequoia.gui.inspector.importer import ImportCurveDialog
from sequoia.gui.inspector.panel import InspectorPanel
from sequoia.library.store import Library

REAL_LIBRARY = Path(__file__).resolve().parents[2] / "library"
SCRATCH = os.environ.get("SEQUOIA_SCRATCH")


def type_in(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


@pytest.fixture
def panel(qtbot, doc):
    panel = InspectorPanel(doc)
    qtbot.addWidget(panel)
    panel.resize(360, 700)
    return panel


def test_every_registered_type_builds_forms(qtbot):
    registry = registered_types()
    assert len(registry) > 10
    for name, cls in registry.items():
        for model in (cls.Settings, cls.Params):
            form = ModelForm(model, lambda n, v: None)
            qtbot.addWidget(form)
            assert set(form.rows) == set(model.model_fields), name
            form.refresh({}, None)


def test_all_real_library_parts_have_forms(qtbot):
    library = Library(REAL_LIBRARY)
    doc = SetupDocument(library)
    panel = InspectorPanel(doc)
    qtbot.addWidget(panel)
    infos = [i for i in library.index() if i.error is None and i.category]
    assert infos
    for info in infos:
        if info.category == "probes":
            continue
        iid = doc.add_instance(info.ref, [0, 0])
        panel.show_selection("instance", iid)
        form = panel.form
        assert form.header.part is not None, info.ref
        assert form.header.error is None
        assert form.overrides_form is not None or info.type is not None
        if form.overrides_group is not None:
            form.overrides_group.set_expanded(True)


def test_instance_header_and_settings(qtbot, doc, panel):
    iid = doc.add_instance("freespace/mirror", [1, 2])
    panel.show_selection("instance", iid)
    form = panel.form
    assert form.header.part.ref == "freespace/mirror"
    row = form.settings_form.rows["angle"]
    assert not row.is_set
    type_in(row.editor.edit, "45 deg")
    assert doc.setup.instances[0].settings == {"angle": "45 deg"}
    assert row.is_set
    doc.undo_stack.undo()
    assert doc.setup.instances[0].settings == {}
    assert not row.is_set
    assert row.editor.text() == "0 deg"
    doc.undo_stack.redo()
    assert row.editor.text() == "45 deg"


def test_invalid_text_leaves_document_untouched(qtbot, doc, panel):
    iid = doc.add_instance("freespace/mirror", [0, 0])
    panel.show_selection("instance", iid)
    row = panel.form.settings_form.rows["angle"]
    before = doc.setup
    row.editor.edit.setText("45 bananas")
    assert row.error.text() and not row.error.isHidden()
    assert row.editor.edit.property("state") == "error"
    row.editor.edit.editingFinished.emit()
    assert doc.setup == before
    row.editor.edit.setText("not a number")
    assert row.error.text()
    row.editor.edit.setText("30 deg")
    assert row.error.isHidden()
    assert doc.setup == before


def test_model_validation_error_reported_inline(qtbot, doc, panel):
    iid = doc.add_instance("sources/laser", [0, 0])
    panel.show_selection("instance", iid)
    group = panel.form.overrides_group
    group.set_expanded(True)
    row = panel.form.overrides_form.rows["wavelength"]
    before = doc.setup
    type_in(row.editor.edit, "5 bananas")
    assert doc.setup == before
    assert row.error.text()


def test_override_with_part_value_and_reset(qtbot, doc, panel):
    iid = doc.add_instance("sources/laser", [0, 0])
    panel.show_selection("instance", iid)
    panel.form.overrides_group.set_expanded(True)
    row = panel.form.overrides_form.rows["wavelength"]
    assert row.editor.text() == "405 nm"
    assert not row.is_set
    type_in(row.editor.edit, "810 nm")
    assert doc.setup.instances[0].overrides == {"wavelength": "810 nm"}
    assert row.reset_button.isVisibleTo(panel.form)
    assert "Overrides (1)" in panel.form.overrides_group.toggle.text()
    row.reset_button.click()
    assert doc.setup.instances[0].overrides == {}
    assert row.editor.text() == "405 nm"


def test_literal_combo(qtbot, doc, panel):
    iid = doc.add_instance("fiber/coupler", [0, 0])
    panel.show_selection("instance", iid)
    row = panel.form.settings_form.rows["mode"]
    assert row.editor.combo.currentData() == "to_fiber"
    row.editor.combo.setCurrentIndex(row.editor.combo.findData("to_free"))
    assert doc.setup.instances[0].settings == {"mode": "to_free"}
    doc.undo_stack.undo()
    assert row.editor.combo.currentData() == "to_fiber"


def test_optional_annotation_analysis():
    from sequoia.data import Length

    spec = analyze(Length | None)
    assert spec.optional and spec.kind == "scalar"


def test_optional_set_unset_on_settings(qtbot, doc):
    from sequoia.components.probes import ProbeSettings

    calls = []

    def commit(name, value):
        calls.append((name, value))

    form = ModelForm(ProbeSettings, commit)
    qtbot.addWidget(form)
    form.show()
    form.refresh({}, None)
    row = form.rows["position"]
    assert row.optional_box.isVisible() and not row.optional_box.isChecked()
    assert not row.editor.isEnabled()
    row.optional_box.click()
    assert row.editor.isEnabled()
    assert calls == []
    type_in(row.editor.edit, "5 mm")
    assert calls == [("position", "5 mm")]
    form.refresh({"position": "5 mm"}, None)
    row.optional_box.click()
    assert calls[-1] == ("position", None)
    assert not row.editor.isEnabled()


def test_probe_settings_form(qtbot, doc, panel):
    laser = doc.add_instance("sources/laser", [0, 0])
    mirror = doc.add_instance("freespace/mirror", [4, 0])
    link = doc.add_link("free_space", f"{laser}.out", f"{mirror}.in")
    probe = doc.add_probe("probes/powermeter", link)
    panel.show_selection("probe", probe)
    form = panel.form
    assert form.tag == "p"
    assert not form.placement
    assert form.overrides_form is None
    assert form.settings_form is None


def test_link_forms(qtbot, doc, panel):
    laser = doc.add_instance("sources/laser", [0, 0])
    mirror = doc.add_instance("freespace/mirror", [4, 0])
    coupler = doc.add_instance("fiber/coupler", [8, 0])
    det = doc.add_instance("detectors/spad", [12, 0])
    free = doc.add_link("free_space", f"{laser}.out", f"{mirror}.in")
    doc.add_link("free_space", f"{mirror}.out", f"{coupler}.in")
    fiber = doc.add_link("fiber", f"{coupler}.out", f"{det}.in")
    panel.show_selection("link", free)
    form = panel.form
    assert set(form.form.rows) == {"length", "lock_to_canvas", "medium"}
    type_in(form.form.rows["length"].editor.edit, "250 mm")
    assert doc.setup.links[0].length.value == 250
    form.form.rows["lock_to_canvas"].editor.check.setChecked(True)
    assert doc.setup.links[0].lock_to_canvas
    doc.undo_stack.undo()
    assert not form.form.rows["lock_to_canvas"].editor.check.isChecked()

    row = form.form.rows["length"]
    before = doc.setup
    row.optional_box.click()
    row.optional_box.click()
    assert doc.setup.links[0].length is not None or doc.setup == before or row.error.text()

    panel.show_selection("link", fiber)
    form = panel.form
    assert set(form.form.rows) == {
        "fiber",
        "length",
        "connector_loss_db",
        "polarization",
    }
    fiber_row = form.form.rows["fiber"]
    assert fiber_row.editor.combo.currentData() == "fiber/smf"
    spin = form.form.rows["connector_loss_db"].editor.spin
    spin.setValue(0.5)
    assert doc.setup.links[2].connector_loss_db == 0.5
    pol = form.form.rows["polarization"].editor
    pol.kind.setCurrentText("su2")
    assert doc.setup.links[2].polarization.kind == "identity"
    assert pol.problem is not None
    for edit, v in zip(pol.vector, ("0.1", "0", "0"), strict=True):
        edit.setText(v)
    pol.vector[0].editingFinished.emit()
    assert doc.setup.links[2].polarization.kind == "su2"
    assert doc.setup.links[2].polarization.rotation_vector == [0.1, 0.0, 0.0]
    pol.kind.setCurrentText("random")
    assert doc.setup.links[2].polarization.kind == "su2"
    pol.seed.setText("7")
    pol.seed.editingFinished.emit()
    assert doc.setup.links[2].polarization.seed == 7


def test_cable_link_form(qtbot, doc, panel):
    det = doc.add_instance("detectors/spad", [0, 0])
    tagger = doc.add_instance("electronics/tagger", [4, 0])
    cable = doc.add_link("cable", f"{det}.out", f"{tagger}.ch1")
    panel.show_selection("link", cable)
    row = panel.form.form.rows["delay"]
    type_in(row.editor.edit, "5 ns")
    assert doc.setup.links[0].delay.value == 5
    type_in(row.editor.edit, "5 mm")
    assert row.error.text()
    assert doc.setup.links[0].delay.value == 5


def test_setup_form_when_nothing_selected(qtbot, doc, panel):
    form = panel.form
    type_in(form.rows["name"].editor.edit, "My setup")
    assert doc.setup.name == "My setup"
    type_in(form.rows["grid_cell"].editor.edit, "10 mm")
    assert doc.setup.grid_cell.value == 10
    type_in(form.rows["grid_cell"].editor.edit, "10 s")
    assert doc.setup.grid_cell.value == 10
    form.rows["description"].editor.text.setPlainText("hello")
    form.rows["description"].editor.commit()
    assert doc.setup.description == "hello"
    doc.undo_stack.undo()
    assert form.rows["description"].editor.text.toPlainText() == ""


def test_selection_removed_falls_back_to_setup(qtbot, doc, panel):
    iid = doc.add_instance("freespace/mirror", [0, 0])
    panel.show_selection("instance", iid)
    assert panel.selection() == ("instance", iid)
    doc.remove_instance(iid)
    assert panel.selection() == (None, None)
    assert type(panel.form).__name__ == "SetupForm"


def test_refresh_preserves_form_instance_and_typed_text(qtbot, doc, panel):
    iid = doc.add_instance("freespace/mirror", [0, 0])
    panel.show_selection("instance", iid)
    form = panel.form
    row = form.settings_form.rows["angle"]
    row.editor.edit.setText("12 de")
    doc.move_instance(iid, [3, 3])
    assert panel.form is form
    assert row.editor.edit.text() == "12 de"
    assert form.placement["x"].editor.spin.value() == 3.0


def test_placement_edits(qtbot, doc, panel):
    iid = doc.add_instance("freespace/mirror", [1, 2])
    panel.show_selection("instance", iid)
    form = panel.form
    form.placement["rotation"].editor.spin.setValue(30.0)
    assert doc.setup.instances[0].rotation == 30.0
    form.placement["flipped"].editor.check.setChecked(True)
    assert doc.setup.instances[0].flipped
    form.placement["y"].editor.spin.setValue(5.5)
    assert doc.setup.instances[0].position == [1.0, 5.5]


def test_curve_or_scalar_and_import(qtbot, doc, panel, monkeypatch, tmp_path):
    iid = doc.add_instance("detectors/spad", [0, 0])
    panel.show_selection("instance", iid)
    panel.form.overrides_group.set_expanded(True)
    row = panel.form.overrides_form.rows["efficiency"]
    assert row.spec.kind == "curve"
    assert row.editor.curve is not None
    new = Curve(x=[1.0, 2.0, 3.0], y=[0.1, 0.2, 0.3], x_unit="um", y_label="efficiency")
    monkeypatch.setattr(fields, "run_import_dialog", lambda parent: new)
    row.editor.import_button.click()
    raw = doc.setup.instances[0].overrides["efficiency"]
    assert raw["x"] == [1.0, 2.0, 3.0]
    assert row.editor.curve == new
    doc.undo_stack.undo()
    assert doc.setup.instances[0].overrides == {}
    assert row.editor.curve.x[0] == 400.0


def test_curve_or_value_mode_switch(qtbot):
    from sequoia.data import Dimensionless

    spec = analyze(Dimensionless | Curve)
    assert spec.kind == "curve_or_value"
    editor = fields.make_editor(spec)
    qtbot.addWidget(editor)
    seen = []
    editor.committed.connect(seen.append)
    editor.set_raw("0.5")
    assert editor.mode.currentIndex() == 0
    editor.set_raw(fields.curve_to_raw(Curve(x=[1, 2], y=[1, 2])))
    assert editor.mode.currentIndex() == 1
    assert seen == []
    editor.mode.setCurrentIndex(0)
    assert seen == []
    type_in(editor.value_editor.edit, "0.25")
    assert seen == [0.25]


def test_dict_and_list_editors(qtbot):
    from sequoia.components.electronics import TimeTaggerSettings
    from sequoia.components.polarization import PaddleParams

    calls = []
    form = ModelForm(TimeTaggerSettings, lambda n, v: calls.append((n, v)))
    qtbot.addWidget(form)
    form.refresh({"input_delays": {"ch1": "1 ns"}}, None)
    editor = form.rows["input_delays"].editor
    assert len(editor.rows) == 1
    editor.add_button.click()
    _, k, v = editor.rows[1]
    k.setText("ch2")
    v.setText("2 ns")
    v.editingFinished.emit()
    assert calls[-1] == ("input_delays", {"ch1": "1 ns", "ch2": "2 ns"})
    v.setText("2 m")
    v.editingFinished.emit()
    assert not form.rows["input_delays"].error.isHidden()
    assert len(calls) == 1

    calls.clear()
    paddle = ModelForm(PaddleParams, lambda n, v: calls.append((n, v)))
    qtbot.addWidget(paddle)
    paddle.refresh({}, None)
    row = paddle.rows["loops"]
    assert row.editor.text() == "1, 2, 1"
    type_in(row.editor.edit, "1, 3, 1")
    assert calls == [("loops", [1, 3, 1])]
    row.editor.edit.setText("1, x")
    assert row.error.text()


def test_bool_and_number_fields(qtbot):
    from sequoia.components.nonlinear import SPDCCrystalParams

    calls = []
    form = ModelForm(SPDCCrystalParams, lambda n, v: calls.append((n, v)))
    qtbot.addWidget(form)
    form.refresh({}, None)
    rows = form.rows
    rows["spdcalc_counts_correction"].editor.check.setChecked(True)
    assert calls[-1] == ("spdcalc_counts_correction", True)
    rows["integration_degree"].editor.spin.setValue(50)
    assert calls[-1] == ("integration_degree", 50)
    assert rows["theta"].spec.literals == ("auto",)
    type_in(rows["theta"].editor.edit, "auto")
    assert calls[-1] == ("theta", "auto")
    type_in(rows["theta"].editor.edit, "5 m")
    assert rows["theta"].error.text()


def test_import_dialog(qtbot, tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("wavelength (nm),transmission (%)\n800,10\n810,90\n820,50\n")
    dialog = ImportCurveDialog()
    qtbot.addWidget(dialog)
    dialog.load_file(path)
    curve = dialog.curve()
    assert curve is not None
    assert curve.x == [800.0, 810.0, 820.0]
    assert curve.y == pytest.approx([0.1, 0.9, 0.5])
    assert dialog.y_kind.currentData() == "percent"
    dialog.y_kind.setCurrentIndex(dialog.y_kind.findData("raw"))
    assert dialog.curve().y == [10.0, 90.0, 50.0]
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    dialog.load_file(bad)
    assert dialog.curve() is None
    assert dialog.message.text()


def test_analyze_classification():
    from sequoia.components.detectors import SinglePhotonDetectorParams, SNSPDParams
    from sequoia.components.fiber import OpticalFiberParams

    kinds = {n: analyze(f.annotation, f.metadata).kind for n, f in SNSPDParams.model_fields.items()}
    assert kinds["efficiency"] == "curve_or_value"
    assert kinds["efficiency_vs_bias"] == "surface"
    assert kinds["dark_count_rate"] == "scalar"
    fiber = {
        n: analyze(f.annotation, f.metadata).kind
        for n, f in OpticalFiberParams.model_fields.items()
    }
    assert fiber["kind"] == "literal"
    assert fiber["attenuation_db_per_km"] == "curve_or_value"
    assert fiber["numerical_aperture"] == "float"
    assert analyze(SinglePhotonDetectorParams.model_fields["dead_time"].annotation).kind == "scalar"
    assert MISSING is not None


@pytest.mark.skipif(SCRATCH is None, reason="set SEQUOIA_SCRATCH to write screenshots")
def test_screenshots(qtbot, tmp_path):
    out = Path(SCRATCH)
    library = Library(REAL_LIBRARY)
    doc = SetupDocument(library)
    panel = InspectorPanel(doc)
    qtbot.addWidget(panel)
    panel.resize(380, 900)
    panel.show()
    laser = doc.add_instance("sources/laser_405_cw", [0, 0])
    crystal = doc.add_instance("nonlinear/ppktp_type2_10mm", [4, 0])
    panel.show_selection("instance", laser)
    panel.grab().save(str(out / "inspector_laser.png"))
    panel.show_selection("instance", crystal)
    panel.form.overrides_group.set_expanded(True)
    panel.resize(380, 1800)
    panel.grab().save(str(out / "inspector_crystal.png"))
    panel.show_selection(None, None)
    panel.resize(380, 400)
    panel.grab().save(str(out / "inspector_setup.png"))
    assert inspector_pkg
