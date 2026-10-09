import datetime
import textwrap

import pytest
from dummies import (
    CrystalParams,
    DetectorParams,
    FilterParams,
    make_curve,
    make_parts,
)

from sequoia.data import FileRef, Provenance, Scalar, Surface
from sequoia.errors import PartError
from sequoia.library.store import Part
from sequoia.yamlio import load_yaml


def write(library, ref, text, **files):
    directory = library.part_dir(ref)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "part.yaml").write_text(textwrap.dedent(text))
    for name, content in files.items():
        (directory / name.replace("__", ".")).write_text(textwrap.dedent(content))


def test_roundtrip_with_curve_surface_scalar_and_file(library):
    provenance = Provenance(source="datasheet", date=datetime.date(2026, 3, 1), reference="p.4")
    surface = Surface(
        x=[0.0, 5.0, 10.0],
        y=[20.0, 30.0],
        z=[[1.0, 2.0], [3.0, 4.5], [6.0, 7.25]],
        x_label="pump_power",
        x_unit="mW",
        y_label="temperature",
        y_unit="degC",
        z_label="brightness",
        z_unit="kcps",
        provenance=provenance,
        extrapolation="clamp",
    )
    params = CrystalParams(
        length=Scalar(value=10.0, unit="mm", provenance=provenance),
        brightness=surface,
        datasheet=FileRef(file="datasheet.pdf"),
    )
    (library.part_dir("nonlinear/ppktp")).mkdir(parents=True)
    (library.part_dir("nonlinear/ppktp") / "datasheet.pdf").write_bytes(b"%PDF")
    part = Part(
        "nonlinear/ppktp",
        "DummyCrystal",
        params,
        manufacturer="Lab",
        model="PPKTP-1",
        serial="0042",
        description="test crystal",
        attachments=["datasheet.pdf"],
    )
    library.save_part(part)
    loaded = library.load_part("nonlinear/ppktp")
    assert loaded.params == params
    assert loaded.params.brightness.file == "brightness.csv"
    assert (loaded.type, loaded.manufacturer, loaded.model, loaded.serial) == (
        "DummyCrystal",
        "Lab",
        "PPKTP-1",
        "0042",
    )
    assert loaded.attachments == ["datasheet.pdf"]
    assert loaded.description == "test crystal"


def test_curve_roundtrip_is_exact_and_files_are_readable(library):
    curve = make_curve(x=[400.0 + 0.1 * k for k in range(5)], y=[0.1 * 3.0**-k for k in range(5)])
    part = Part("filters/odd", "DummyFilter", FilterParams(transmission=curve))
    library.save_part(part)
    assert library.load_part("filters/odd").params.transmission == curve
    csv_text = (library.part_dir("filters/odd") / "transmission.csv").read_text()
    assert csv_text.splitlines()[0] == "wavelength (nm),transmission (1)"
    raw = load_yaml(library.part_dir("filters/odd") / "part.yaml")
    assert raw["params"]["transmission"] == {
        "curve": "transmission.csv",
        "x": "wavelength (nm)",
        "y": "transmission (1)",
    }


def test_saving_twice_overwrites(library):
    part = Part("filters/f", "DummyFilter", FilterParams(transmission=make_curve()))
    library.save_part(part)
    part.params = FilterParams(transmission=make_curve(y=[0.2, 0.3, 0.4]))
    library.save_part(part)
    assert library.load_part("filters/f").params.transmission.y == [0.2, 0.3, 0.4]


def test_hand_written_part_with_shorthand(library):
    write(
        library,
        "detectors/snspd/eos",
        """\
        type: DummyDetector
        manufacturer: Single Quantum
        model: Eos
        serial: 12345
        params:
          efficiency: {curve: sde.csv, x: wavelength (nm), y: efficiency (1),
                       provenance: {source: measured, date: 2026-03-01, by: Boran, note: Ib=12 uA}}
          dark_count_rate: 50 Hz
          dead_time: {value: 20, unit: ns}
        attachments: [datasheet.pdf]
        """,
        sde__csv="wavelength (nm),efficiency (1)\n780,0.8\n810,0.9\n850,0.85\n",
    )
    part = library.load_part("detectors/snspd/eos")
    assert part.serial == "12345"
    assert part.params.dark_count_rate.si == 50.0
    assert part.params.dead_time.si == pytest.approx(20e-9)
    assert part.params.efficiency(810e-9) == pytest.approx(0.9)
    assert part.params.efficiency.provenance.by == "Boran"
    assert part.params.efficiency.provenance.date == datetime.date(2026, 3, 1)


def test_headerless_csv_keeps_first_row(library):
    write(
        library,
        "detectors/d",
        """\
        type: DummyDetector
        params:
          efficiency: {curve: e.csv, x: wavelength (nm), y: efficiency (1)}
        """,
        e__csv="780,0.8\n810,0.9\n",
    )
    assert library.load_part("detectors/d").params.efficiency.x == [780.0, 810.0]


def test_index_lists_parts_with_categories_and_errors(library):
    make_parts(library)
    write(library, "broken/part", "type: [unterminated\n")
    infos = {i.ref: i for i in library.index()}
    assert infos["detectors/spad"].category == "detectors"
    assert infos["fiber/smf"].type == "OpticalFiber"
    assert infos["broken/part"].error and infos["broken/part"].type is None
    assert len(infos) == 10


def test_index_of_missing_library_is_empty(library):
    assert library.index() == []


@pytest.mark.parametrize("ref", ["", "../x", "/abs", "a/../b", "a//b", "a\\b", "./a"])
def test_bad_refs_are_rejected(library, ref):
    with pytest.raises(PartError, match="invalid part reference"):
        library.part_dir(ref)


@pytest.mark.parametrize(
    ("yaml_text", "files", "message"),
    [
        ("type: Nope\nparams: {}\n", {}, "unknown component type 'Nope'"),
        ("params: {}\n", {}, "no 'type'"),
        ("type: DummyLaser\nbogus: 1\n", {}, "unknown key"),
        ("- a\n- b\n", {}, "must contain a mapping"),
        ("type: DummyLaser\nparams: {wavelength: 405 nm, colour: red}\n", {}, "colour"),
        ("type: DummyLaser\nparams: {wavelength: 405 florps}\n", {}, "unknown unit"),
        ("type: DummyLaser\nparams: {}\n", {}, "wavelength"),
        (
            "type: DummyFilter\nparams:\n"
            "  transmission: {curve: t.csv, x: 'wavelength nm', y: t (1)}\n",
            {"t__csv": "400,0.1\n500,0.2\n"},
            "name \\(unit\\)",
        ),
        (
            "type: DummyFilter\nparams:\n"
            "  transmission: {curve: t.csv, x: w (nm), y: t (1), z: 1}\n",
            {"t__csv": "400,0.1\n500,0.2\n"},
            "unknown key",
        ),
        (
            "type: DummyFilter\nparams:\n"
            "  transmission: {curve: missing.csv, x: w (nm), y: t (1)}\n",
            {},
            "file not found",
        ),
        (
            "type: DummyFilter\nparams:\n  transmission: {curve: ../x.csv, x: w (nm), y: t (1)}\n",
            {},
            "inside the part folder",
        ),
        (
            "type: DummyFilter\nparams:\n  transmission: {curve: t.csv, x: w (nm), y: t (1)}\n",
            {"t__csv": "500,0.1\n400,0.2\n"},
            "strictly increasing",
        ),
        (
            "type: DummyFilter\nparams:\n  transmission: {curve: t.csv, x: w (nm), y: t (1)}\n",
            {"t__csv": "w (nm)\n400\n500\n"},
            "no numeric data|needs 2 numeric columns",
        ),
        (
            "type: DummyCrystal\nparams:\n  length: 10 mm\n"
            "  brightness: {surface: s.csv, x: p (mW), y: t (degC), z: b (kcps)}\n",
            {"s__csv": "0,20,1\n0,30,2\n5,20,3\n"},
            "complete rectangular",
        ),
        (
            "type: DummyCrystal\nparams:\n  length: 10 mm\n  brightness: 3\n"
            "  datasheet: {file: nope.pdf}\n",
            {},
            "nope.pdf.*not found",
        ),
    ],
)
def test_load_errors_are_clear(library, yaml_text, files, message):
    write(library, "p/bad", yaml_text, **files)
    with pytest.raises(PartError, match=message) as info:
        library.load_part("p/bad")
    assert "p/bad" in str(info.value)


def test_missing_part(library):
    with pytest.raises(PartError, match="not found in library"):
        library.load_part("nothing/here")
    assert not library.exists("nothing/here")


def test_detector_part_defaults_come_from_params(library):
    part = Part("detectors/d", "DummyDetector", DetectorParams(efficiency=make_curve()))
    library.save_part(part)
    loaded = library.load_part("detectors/d")
    assert loaded.params.dead_time == Scalar(value=50.0, unit="ns")
