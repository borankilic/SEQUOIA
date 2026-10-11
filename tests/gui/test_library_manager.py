import shutil
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook
from PySide6.QtCore import QSettings

import sequoia.gui.inspector.fields as fields
from sequoia.analysis.sweep import evaluate, metrics_of
from sequoia.gui.inspector.dialogs import CurveTableDialog
from sequoia.gui.inspector.importer import ImportCurveDialog
from sequoia.gui.library_manager import LibraryManager
from sequoia.gui.main_window import MainWindow
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[2]
WAIT = 60000


@pytest.fixture
def lib_copy(tmp_path):
    target = tmp_path / "library"
    shutil.copytree(ROOT / "library", target)
    return Library(target)


@pytest.fixture
def xlsx(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Wavelength (nm)", "Responsivity (A/W)"])
    for wl, r in ((400, 0.10), (600, 0.35), (800, 0.55), (900, 0.62), (1000, 0.30)):
        ws.append([wl, r])
    path = tmp_path / "FDS100_responsivity.xlsx"
    wb.save(path)
    return path


def test_create_photodiode_from_xlsx(qtbot, lib_copy, xlsx, monkeypatch):
    manager = LibraryManager(lib_copy)
    qtbot.addWidget(manager)
    changed = []
    manager.library_changed.connect(lambda: changed.append(1))

    def fake_import(parent):
        dialog = ImportCurveDialog(parent)
        dialog.load_file(xlsx)
        return dialog.curve()

    monkeypatch.setattr(fields, "run_import_dialog", fake_import)
    assert manager.new_part("Photodiode", "detectors/thorlabs_fds100")
    manager.manufacturer.setText("Thorlabs")
    manager.model.setText("FDS100")
    manager._mark_dirty()
    row = manager.form.rows["responsivity"]
    row.editor.mode.setCurrentIndex(1)
    row.editor.curve_editor.import_button.click()
    assert "responsivity" in manager.draft
    row_d = manager.form.rows["active_diameter"]
    row_d.editor.edit.setText("3.6 mm")
    row_d.editor.commit()
    row_c = manager.form.rows["dark_current"]
    row_c.editor.edit.setText("2.5 nA")
    row_c.editor.commit()
    assert manager.save(), manager.status.text()
    assert changed

    part = lib_copy.load_part("detectors/thorlabs_fds100")
    assert part.manufacturer == "Thorlabs"
    assert part.params.active_diameter.value == 3.6
    curve = part.params.responsivity
    assert len(curve.x) == 5 and curve.y_unit == "A/W"
    assert np.interp(800e-9, curve.x_si, curve.y_si) == pytest.approx(0.55)
    assert (lib_copy.part_dir("detectors/thorlabs_fds100") / "responsivity.csv").is_file()

    info = {i.ref: i for i in lib_copy.index()}
    assert info["detectors/thorlabs_fds100"].type == "Photodiode"


def test_new_part_with_missing_required_is_refused(qtbot, lib_copy):
    manager = LibraryManager(lib_copy)
    qtbot.addWidget(manager)
    assert manager.new_part("Photodiode", "detectors/empty")
    assert not manager.save()
    assert "responsivity" in manager.status.text()
    assert not lib_copy.exists("detectors/empty")


def test_provenance_button_edits_value(qtbot, lib_copy, monkeypatch):
    import sequoia.gui.inspector.dialogs as dialogs

    manager = LibraryManager(lib_copy)
    qtbot.addWidget(manager)
    assert manager.select_ref("detectors/spad_si_generic")
    manager.open_ref("detectors/spad_si_generic")
    row = manager.form.rows["dark_count_rate"]
    monkeypatch.setattr(
        dialogs,
        "ask_provenance",
        lambda parent, initial: (True, {"source": "measured", "by": "Boran", "date": "2026-03-01"}),
    )
    row._edit_provenance()
    assert manager.draft["dark_count_rate"]["provenance"]["source"] == "measured"
    assert manager.dirty
    assert manager.save()
    part = lib_copy.load_part("detectors/spad_si_generic")
    prov = part.params.dark_count_rate.provenance
    assert prov.source == "measured" and prov.by == "Boran"
    assert str(prov.date) == "2026-03-01"
    manager.form.rows["dark_count_rate"].editor.edit.setText("250 Hz")
    manager.form.rows["dark_count_rate"].editor.commit()
    assert manager.draft["dark_count_rate"]["provenance"]["source"] == "measured"


def test_duplicate_rename_delete(qtbot, lib_copy):
    manager = LibraryManager(lib_copy)
    qtbot.addWidget(manager)
    manager.open_ref("freespace/lens_f100mm")
    assert manager.duplicate_part("freespace/lens_f100mm_copy")
    assert lib_copy.exists("freespace/lens_f100mm_copy")
    assert manager.rename_part("freespace/lens_f100_b")
    assert lib_copy.exists("freespace/lens_f100_b")
    assert not lib_copy.exists("freespace/lens_f100mm_copy")
    manager.confirm = lambda message: False
    assert not manager.delete_part()
    manager.confirm = lambda message: True
    assert manager.delete_part()
    assert not lib_copy.exists("freespace/lens_f100_b")
    assert "freespace/lens_f100_b" not in manager.visible_refs()
    assert not manager.duplicate_part("freespace/lens_f50mm")


def test_search_filters_parts(qtbot, lib_copy):
    manager = LibraryManager(lib_copy)
    qtbot.addWidget(manager)
    manager.search.setText("snspd")
    assert manager.visible_refs() == ["detectors/snspd_generic", "detectors/snspd_telecom"]
    manager.search.setText("")
    assert len(manager.visible_refs()) == len(lib_copy.index())


def test_curve_table_dialog(qtbot):
    dialog = CurveTableDialog(None)
    qtbot.addWidget(dialog)
    dialog.x_axis.setText("wavelength (nm)")
    dialog.y_axis.setText("transmission (1)")
    dialog.paste("700\t0.1\n800,0.9\n750 0.5\n")
    dialog._accept()
    curve = dialog.curve()
    assert curve is not None
    assert list(curve.x) == [700.0, 750.0, 800.0]
    assert list(curve.y) == [0.1, 0.5, 0.9]
    bad = CurveTableDialog(None)
    qtbot.addWidget(bad)
    bad.set_rows([(1.0, 1.0)])
    bad._accept()
    assert bad.curve() is None and "two points" in bad.message.text()


def test_added_part_shows_in_palette_and_simulates(qtbot, lib_copy, xlsx, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    window = MainWindow(lib_copy, ROOT / "setups" / "laser_fiber_coupling.yaml", settings=settings)
    window.confirm_discard = lambda _w: "discard"
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.controller.last_outcome is not None, timeout=WAIT)
    manager = window.show_library_manager()

    def fake_import(parent):
        dialog = ImportCurveDialog(parent)
        dialog.load_file(xlsx)
        return dialog.curve()

    monkeypatch.setattr(fields, "run_import_dialog", fake_import)
    assert manager.new_part("Photodiode", "detectors/thorlabs_fds100")
    row = manager.form.rows["responsivity"]
    row.editor.mode.setCurrentIndex(1)
    row.editor.curve_editor.import_button.click()
    assert manager.save()
    assert window.palette.item_for_ref("detectors/thorlabs_fds100") is not None

    setup = load_setup(ROOT / "setups" / "laser_fiber_coupling.yaml")
    raw = setup.model_dump(by_alias=True, exclude_defaults=True)
    for inst in raw["instances"]:
        if inst["id"] == "pd":
            inst["part"] = "detectors/thorlabs_fds100"
    from sequoia.setup.io import setup_from_dict

    swapped = setup_from_dict(raw)
    outcome = evaluate(swapped, lib_copy)
    current = metrics_of(outcome)["readout:pd.photocurrent_A"]
    power = metrics_of(outcome)["readout:pd.incident_power_W"]
    expected = float(np.interp(780e-9, [600e-9, 800e-9], [0.35, 0.55]))
    assert current == pytest.approx(power * expected, rel=0.02)
