import math
import shutil
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.data import Curve, Scalar
from sequoia.errors import PartError
from sequoia.library.calibration import (
    CALIBRATION_FILE,
    CalibrationRecord,
    Uncertain,
    UncertaintyError,
    from_calibration,
    placeholder_report,
    propagate_uncertainty,
    save_calibration,
)
from sequoia.library.store import Library
from sequoia.setup.io import load_setup

ROOT = Path(__file__).resolve().parents[1]
PARTS = (
    "sources/laser_780_cw",
    "sources/laser_633_hene",
    "filters/nd_filter_od1",
    "detectors/photodiode_generic",
    "probes/power_meter",
    "polarization/hwp_810",
    "polarization/polarizer_generic",
    "probes/polarimeter",
)


@pytest.fixture
def lib(tmp_path):
    for ref in PARTS:
        shutil.copytree(ROOT / "library" / ref, tmp_path / "lib" / ref)
    return Library(tmp_path / "lib")


def od_record(value=1.05, std=0.02, **kw):
    return CalibrationRecord(
        id=kw.pop("id", "od_cal"),
        parameter="optical_density",
        value=Scalar.model_validate(value),
        std=Scalar.model_validate(std),
        **kw,
    )


def test_fingerprint_is_stable_and_changes_with_a_calibration(lib):
    ref = "filters/nd_filter_od1"
    part_fp, lib_fp = lib.part_fingerprint(ref), lib.fingerprint()
    assert lib.part_fingerprint(ref) == part_fp and lib.fingerprint() == lib_fp
    assert lib.load_part(ref).fingerprint == part_fp
    other = lib.part_fingerprint("probes/power_meter")
    save_calibration(lib.part_dir(ref), [od_record(date="2026-10-01", measured_by="BK")])
    assert lib.part_fingerprint(ref) != part_fp
    assert lib.fingerprint() != lib_fp
    assert lib.part_fingerprint("probes/power_meter") == other
    changed = lib.part_fingerprint(ref)
    save_calibration(lib.part_dir(ref), [od_record(value=1.06)])
    assert lib.part_fingerprint(ref) != changed
    assert lib.fingerprint([ref]) != lib.fingerprint(["probes/power_meter"])


def test_measured_record_replaces_nominal_and_keeps_it_traceable(lib):
    ref = "filters/nd_filter_od1"
    older = od_record(value=1.02, id="old", date="2025-01-01")
    newer = od_record(value=1.05, id="new", date="2026-10-01", method="power ratio")
    model = od_record(value=0.9, id="model", source="model", date="2027-01-01")
    save_calibration(lib.part_dir(ref), [older, newer, model])
    part = lib.load_part(ref)
    assert part.params.optical_density.value == 1.05
    assert part.calibrated == {"optical_density": "new"}
    provenance = part.params.optical_density.provenance
    assert provenance.source == "measured" and "new" in provenance.note
    assert provenance.reference == CALIBRATION_FILE
    nominal = lib.load_part(ref, calibrated=False)
    assert nominal.params.optical_density.value == 1.0 and nominal.calibrated == {}
    assert [r.id for r in nominal.calibration] == ["old", "new", "model"]
    with pytest.raises(PartError, match="calibrated=False"):
        lib.save_part(part)
    lib.save_part(nominal)
    assert lib.load_part(ref).params.optical_density.value == 1.05


def test_ranking_against_the_nominal_provenance(lib):
    laser = "sources/laser_633_hene"
    nominal = CalibrationRecord(
        id="vendor",
        parameter="power",
        source="vendor nominal",
        value=Scalar.model_validate("4.5 mW"),
        std=Scalar.model_validate("0.2 mW"),
    )
    save_calibration(lib.part_dir(laser), [nominal])
    assert lib.load_part(laser).params.power.value == 4.5
    save_calibration(
        lib.part_dir("filters/nd_filter_od1"), [od_record(id="m", source="model", value=2.0)]
    )
    assert lib.load_part("filters/nd_filter_od1").params.optical_density.value == 1.0


def test_curve_records_with_covariance_round_trip(lib, tmp_path):
    curve = Curve(x=[600, 700, 800], y=[0.5, 0.6, 0.7], x_label="wavelength", x_unit="nm")
    cov = [[1e-4, 2e-5, 0.0], [2e-5, 1e-4, 0.0], [0.0, 0.0, 4e-4]]
    record = CalibrationRecord(
        id="r_curve",
        parameter="responsivity",
        curve=curve.model_copy(update={"y_label": "responsivity", "y_unit": "A/W"}),
        covariance=cov,
        conditions="22 degC",
        local_id="PD-3",
    )
    directory = lib.part_dir("detectors/photodiode_generic")
    save_calibration(directory, [record])
    part = lib.load_part("detectors/photodiode_generic")
    loaded = part.calibration[0]
    assert loaded.covariance == cov and loaded.local_id == "PD-3"
    assert part.params.responsivity.y == [0.5, 0.6, 0.7]
    assert part.params.responsivity.provenance.source == "measured"
    (directory / CALIBRATION_FILE).write_text(
        "records:\n- {id: a, parameter: nonsense, value: 1}\n", encoding="utf-8"
    )
    with pytest.raises(PartError, match="unknown parameter 'nonsense'"):
        lib.load_part("detectors/photodiode_generic")
    (directory / CALIBRATION_FILE).write_text(
        "records:\n- {id: a, parameter: dark_current, value: 1 nA, std: 1 mW}\n", encoding="utf-8"
    )
    with pytest.raises(PartError, match="std unit"):
        lib.load_part("detectors/photodiode_generic")


def test_placeholder_report_on_templates():
    library = Library(ROOT / "library")
    allowed = {"measured", "vendor nominal", "model", "ideal default", "untraced", "override"}
    for path in sorted((ROOT / "templates").glob("*.yaml")):
        rows = placeholder_report(load_setup(path), library)
        assert rows, path.name
        assert {r.status for r in rows} <= allowed, path.name
        assert any(r.placeholder for r in rows), path.name
    rows = placeholder_report(load_setup(ROOT / "templates" / "laser_power_meter.yaml"), library)
    by = {(r.owner, r.parameter): r for r in rows}
    assert by[("nd", "optical_density")].status == "untraced"
    assert by[("laser", "power")].placeholder


def test_placeholder_report_shows_measured_values(lib):
    save_calibration(lib.part_dir("filters/nd_filter_od1"), [od_record(date="2026-10-01")])
    rows = placeholder_report(load_setup(ROOT / "templates" / "laser_power_meter.yaml"), lib)
    nd = next(r for r in rows if r.owner == "nd")
    assert nd.status == "measured" and nd.record == "od_cal" and not nd.placeholder
    assert "1.05 +/- 0.02" in nd.value


def test_uncertainty_propagation_matches_the_analytic_case(lib):
    setup = load_setup(ROOT / "templates" / "laser_power_meter.yaml")
    save_calibration(lib.part_dir("filters/nd_filter_od1"), [od_record(std=0.02)])
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", ["nd.optical_density"], n=400, seed=3)
    p = r.nominal
    assert p == pytest.approx(5e-3 * 10**-1.05, rel=1e-9)
    analytic = p * math.log(10) * 0.02
    assert r.contributions["overrides:nd.optical_density"] == pytest.approx(analytic, rel=1e-3)
    assert r.std == pytest.approx(analytic, rel=0.15)
    assert r.mean == pytest.approx(p, rel=0.01) and r.failures == 0
    assert "counting noise" in r.note
    again = propagate_uncertainty(
        setup, lib, "probe:pm.power_W", ["nd.optical_density"], n=400, seed=3
    )
    assert np.array_equal(again.samples, r.samples)
    both = propagate_uncertainty(
        setup,
        lib,
        "probe:pm.power_W",
        ["nd.optical_density", Uncertain("overrides:laser.power", 0.05, "mW", label="power")],
        n=400,
        seed=4,
    )
    rel = math.hypot(0.05 / 5.0, math.log(10) * 0.02)
    assert both.linear_std == pytest.approx(both.nominal * rel, rel=1e-3)
    assert both.std == pytest.approx(both.nominal * rel, rel=0.15)
    ranked = both.ranked()
    assert ranked[0][0] == "overrides:nd.optical_density" and ranked[1][0] == "power"
    assert sum(f for _, _, f in ranked) == pytest.approx(1.0)


def test_offsets_and_unsupported_parameters(lib):
    setup = load_setup(ROOT / "templates" / "malus_law.yaml")
    save_calibration(
        lib.part_dir("polarization/hwp_810"),
        [
            CalibrationRecord(
                id="zero",
                parameter="zero_offset",
                value=Scalar.model_validate("0.5 deg"),
                std=Scalar.model_validate("0.2 deg"),
            )
        ],
    )
    u = from_calibration(setup, lib, "hwp.zero_offset")
    assert u.target == "settings:hwp.angle" and u.unit == "deg"
    assert u.mean == pytest.approx(11.75) and u.std == pytest.approx(0.2)
    rows = placeholder_report(setup, lib)
    assert any(r.parameter == "zero_offset" and r.status == "measured" for r in rows)
    r = propagate_uncertainty(setup, lib, "probe:pm.power_W", [u], n=50, seed=0)
    assert r.std > 0 and r.contributions["settings:hwp.angle"] > 0
    with pytest.raises(UncertaintyError, match="supports"):
        propagate_uncertainty(
            setup, lib, "probe:pm.power_W", [Uncertain("link:s1.length", 1, "mm")]
        )
    with pytest.raises(UncertaintyError, match="no scalar calibration record"):
        from_calibration(setup, lib, "pol.transmission")


def test_library_manager_lists_calibration_records_read_only(qtbot, lib):
    from sequoia.gui.library_manager import LibraryManager

    ref = "filters/nd_filter_od1"
    save_calibration(
        lib.part_dir(ref),
        [
            od_record(id="old", value=1.02, date="2025-01-01"),
            od_record(id="new", value=1.05, date="2026-10-01", conditions="633 nm"),
            od_record(id="model", value=0.9, source="model"),
        ],
    )
    manager = LibraryManager(lib)
    qtbot.addWidget(manager)
    assert manager.open_ref(ref)
    rows = {
        manager.calibration.topLevelItem(i).text(5): manager.calibration.topLevelItem(i)
        for i in range(manager.calibration.topLevelItemCount())
    }
    assert rows["new"].text(4).startswith("in use")
    assert rows["old"].text(4).startswith("superseded")
    assert "conditions: 633 nm" in rows["new"].toolTip(0)
    assert manager.fingerprint.text().endswith(lib.part_fingerprint(ref)[:16])
    assert manager.draft["optical_density"] in (1.0, "1.0", 1)
    assert manager.save()
    assert lib.load_part(ref, calibrated=False).params.optical_density.value == 1.0
    assert manager.open_ref("probes/power_meter")
    assert "no calibration records" in manager.calibration.topLevelItem(0).text(2)
