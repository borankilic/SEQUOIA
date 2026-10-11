import math
from pathlib import Path

import numpy as np
import pytest
from optics import LASER, basic_library, fiber, fs, inst, probe, put, run

import sequoia.components.builtin  # noqa: F401
from sequoia.library.store import Library

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lib(tmp_path):
    return basic_library(Library(tmp_path))


def chain(library, lens_params, power="100 mW", waist="1 mm"):
    put(library, "laser2", "Laser", **{**LASER, "power": power, "waist_radius": waist})
    put(library, "lens", "Lens", **lens_params)
    return run(
        library,
        [inst("l", "laser2"), inst("x", "lens"), inst("m", "mirror")],
        [fs("a", "l.out", "x.in", "50 mm"), fs("b", "x.out", "m.in", "50 mm")],
    )


def test_wavefront_error_degrades_m2_but_keeps_local_beam(lib):
    clean = chain(lib, {"focal_length": "1 m"})
    dirty = chain(lib, {"focal_length": "1 m", "wavefront_error": 0.2})
    a, b = clean.links["a"].end.mode, dirty.links["a"].end.mode
    assert a.m2_x == pytest.approx(1.0)
    c_start, d_start = clean.links["b"].start.mode, dirty.links["b"].start.mode
    assert d_start.w_x == pytest.approx(c_start.w_x, rel=1e-9)
    assert d_start.radius_x == pytest.approx(c_start.radius_x, rel=1e-9)
    assert d_start.m2_x == pytest.approx(math.sqrt(1 + (2 * math.pi * 0.2) ** 2))
    assert dirty.links["b"].end.mode.w_x > clean.links["b"].end.mode.w_x
    assert b.m2_x == pytest.approx(1.0)


def test_damage_threshold_warnings(lib):
    area_cm2 = math.pi * 0.1**2 / 2
    intensity = 0.1 / area_cm2
    quiet = chain(lib, {"focal_length": "1 m", "damage_threshold_cw": intensity * 10})
    assert not [i for i in quiet.issues if i.code.startswith("damage")]
    close = chain(lib, {"focal_length": "1 m", "damage_threshold_cw": intensity * 1.5})
    assert any(i.code == "damage_threshold_close" for i in close.issues)
    over = chain(lib, {"focal_length": "1 m", "damage_threshold_cw": intensity * 0.5})
    assert any(i.code == "damage_threshold_exceeded" for i in over.issues)


def test_multi_order_plate_is_more_wavelength_sensitive():
    from sequoia.components.polarization import WavePlate, WavePlateParams

    def retardance_at(order, wavelength):
        plate = WavePlate(WavePlateParams(retardance=0.5, order=order, design_wavelength="800 nm"))
        from sequoia.components.base import Context
        from sequoia.light.gaussian import GaussianMode
        from sequoia.light.spectrum import Spectrum
        from sequoia.light.state import Beam

        spectrum = Spectrum.line(wavelength, 1e-12) if hasattr(Spectrum, "line") else None
        assert spectrum is not None
        beam = Beam.launch("s", spectrum, GaussianMode.from_waist(wavelength, 1e-3))
        out = plate.propagate("in", beam, Context())[0][1]
        j = out.jones[0]
        return np.angle(j[1, 1] / j[0, 0]) / (2 * np.pi)

    base = retardance_at(0, 800e-9)
    shifted_zero = retardance_at(0, 805e-9)
    shifted_multi = retardance_at(10, 805e-9)
    assert abs(abs(base) - 0.5) < 1e-6
    d0 = abs(((shifted_zero - base) + 0.5) % 1.0 - 0.5)
    d10 = abs(((shifted_multi - base) + 0.5) % 1.0 - 0.5)
    assert d10 > 5 * d0


def test_pm_fiber_acts_as_a_retarder_with_crosstalk(lib):
    put(
        lib,
        "pm",
        "OpticalFiber",
        kind="pm",
        mode_field_diameter="5 um",
        beat_length="2 mm",
        polarization_extinction_db=20.0,
    )
    put(lib, "cpl", "FiberCoupler", focal_length="11 mm", clear_aperture="9 mm")
    put(lib, "hwp", "WavePlate", retardance=0.5, design_wavelength="780 nm")
    result = run(
        lib,
        [
            inst("l", "laser"),
            inst("w", "hwp", {"angle": "22.5 deg"}),
            inst("c", "cpl"),
            inst("r", "cpl", {"mode": "to_free"}),
            inst("m", "mirror"),
        ],
        [
            fs("a", "l.out", "w.in", "50 mm"),
            fs("b", "w.out", "c.in", "20 mm"),
            fiber("f", "c.out", "r.in", "pm", length="1.25 m"),
            fs("g", "r.out", "m.in", "10 mm"),
        ],
        [probe("pol", "g", "probes/pol")],
    )
    state = result.probes["pol"]
    assert state["dop"] > 0.9
    # 22.5 degree HWP rotates H to 45 degrees; 1.25 m / 2 mm = 625 waves of retardance (integer):
    # the fiber leaves diagonal light diagonal, with a small rotation from the crosstalk
    assert abs(result.probes["pol"]["azimuth_rad"] - math.pi / 4) < 0.15


def test_catalog_generator_parts_load_and_simulate():
    library = Library(ROOT / "library")
    refs = [i.ref for i in library.index()]
    assert len(refs) > 100
    types = {i.type for i in library.index()}
    for needed in ("SpectralFilter", "NDFilter", "Lens", "Objective", "Mirror", "WavePlate"):
        assert needed in types
    for ref in refs:
        library.load_part(ref)
