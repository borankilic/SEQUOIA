import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import sequoia.components.builtin  # noqa: F401
from sequoia.errors import PartError
from sequoia.library.editing import (
    accepts_provenance,
    build_part,
    draft_from_part,
    get_provenance,
    keep_provenance,
    save_edited,
    suggest_ref,
    with_provenance,
)
from sequoia.library.store import Library
from sequoia.yamlio import load_yaml

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


def read_csv(path):
    with path.open() as f:
        rows = list(csv.reader(f))
    return rows[0], np.array(rows[1:], dtype=float)


def test_every_part_loads():
    infos = LIB.index()
    assert len(infos) >= 35
    types = {i.type for i in infos}
    import sequoia.components.base as base

    assert set(base.registered_types()) <= types
    for info in infos:
        LIB.load_part(info.ref)


@pytest.mark.parametrize("ref", [i.ref for i in Library(ROOT / "library").index()])
def test_gui_style_round_trip_matches_hand_written(ref, tmp_path):
    part = LIB.load_part(ref)
    draft = draft_from_part(part)
    header = {
        "manufacturer": part.manufacturer,
        "model": part.model,
        "serial": part.serial,
        "description": part.description,
        "attachments": part.attachments,
    }
    rebuilt = build_part(ref, part.type, header, draft)
    out = Library(tmp_path)
    out.save_part(rebuilt)
    original = load_yaml(LIB.part_dir(ref) / "part.yaml")
    written = load_yaml(out.part_dir(ref) / "part.yaml")
    assert set(written) == set(original)
    assert set(written.get("params") or {}) == set(original.get("params") or {})
    reloaded = out.load_part(ref)
    assert reloaded.params == part.params
    assert (reloaded.manufacturer, reloaded.model, reloaded.description) == (
        part.manufacturer,
        part.model,
        part.description,
    )
    for name in {p.name for p in LIB.part_dir(ref).glob("*.csv")}:
        h0, d0 = read_csv(LIB.part_dir(ref) / name)
        h1, d1 = read_csv(out.part_dir(ref) / name)
        assert h0 == h1
        assert np.allclose(d0, d1, rtol=0, atol=0)


def test_build_part_reports_missing_required():
    with pytest.raises(PartError, match="invalid parameters"):
        build_part("detectors/x", "Photodiode", {}, {})
    part = build_part("detectors/x", "Photodiode", {"model": "pd"}, {"responsivity": "0.5 A/W"})
    assert part.model == "pd" and part.manufacturer is None


def test_provenance_helpers():
    raw = with_provenance("10 mm", {"source": "datasheet", "reference": "p. 3"})
    assert raw == {
        "value": 10.0,
        "unit": "mm",
        "provenance": {"source": "datasheet", "reference": "p. 3"},
    }
    assert get_provenance(raw)["source"] == "datasheet"
    assert keep_provenance(raw, "12 mm")["provenance"]["reference"] == "p. 3"
    assert keep_provenance("10 mm", "12 mm") == "12 mm"
    assert "provenance" not in with_provenance(raw, None)
    assert accepts_provenance("5 mW") and not accepts_provenance("to_free")
    with pytest.raises(ValueError):
        with_provenance(["a"], {"source": "model"})
    with pytest.raises(ValidationError):
        with_provenance("10 mm", {"source": "guess"})


def test_store_duplicate_rename_delete_attach(tmp_path):
    lib = Library(tmp_path)
    part = build_part("filters/f1", "NDFilter", {"model": "nd"}, {"optical_density": "1.0"})
    lib.save_part(part)
    lib.duplicate_part("filters/f1", "filters/f2")
    assert lib.exists("filters/f2")
    with pytest.raises(PartError):
        lib.duplicate_part("filters/f1", "filters/f2")
    lib.rename_part("filters/f2", "filters/sub/f3")
    assert lib.exists("filters/sub/f3") and not lib.exists("filters/f2")
    source = tmp_path / "datasheet one.pdf"
    source.write_bytes(b"%PDF")
    name = lib.attach_file("filters/f1", source)
    assert name == "datasheet_one.pdf" and (lib.part_dir("filters/f1") / name).is_file()
    lib.delete_part("filters/sub/f3")
    assert not lib.part_dir("filters/sub").exists()
    with pytest.raises(PartError):
        lib.delete_part("filters/sub/f3")
    with pytest.raises(PartError):
        lib.part_dir("../escape")


def test_save_edited_drops_stale_curve_file(tmp_path):
    lib = Library(tmp_path)
    curve = {
        "x": [400.0, 800.0],
        "y": [0.1, 0.9],
        "x_label": "wavelength",
        "x_unit": "nm",
        "y_label": "transmission",
        "y_unit": "1",
    }
    first = build_part("filters/nd", "NDFilter", {}, {"optical_density": curve})
    save_edited(lib, first)
    assert (lib.part_dir("filters/nd") / "optical_density.csv").is_file()
    second = build_part("filters/nd", "NDFilter", {}, {"optical_density": "2.0"})
    save_edited(lib, second, previous=lib.load_part("filters/nd"))
    assert not (lib.part_dir("filters/nd") / "optical_density.csv").exists()
    assert lib.load_part("filters/nd").params.optical_density.value == 2.0


def test_suggest_ref_is_unique(tmp_path):
    lib = Library(tmp_path)
    assert suggest_ref(lib, "SPAD", "My Detector 1") == "detectors/my_detector_1"
    lib.save_part(build_part("detectors/my_detector_1", "SPAD", {}, {"efficiency": 0.5}))
    assert suggest_ref(lib, "SPAD", "My Detector 1") == "detectors/my_detector_1_2"
