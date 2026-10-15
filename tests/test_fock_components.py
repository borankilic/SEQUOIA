import math
from pathlib import Path

import numpy as np
import pytest
from optics import basic_library, fs, inst, power, probe, put, run
from scipy import constants

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import get_component
from sequoia.engines.classical import simulate
from sequoia.engines.gaussian import simulate_gaussian
from sequoia.engines.rates import simulate_rates
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, save_setup, setup_from_dict

ROOT = Path(__file__).parents[1]
LIBRARY = Library(ROOT / "library")
LAM = 810e-9
RATE = 80e6
PHOTON = constants.h * constants.c / LAM


@pytest.fixture
def lib(tmp_path):
    lib = basic_library(Library(tmp_path / "lib"))
    for n in (0, 1, 3):
        put(
            lib,
            f"fock{n}",
            "FockSource",
            wavelength="810 nm",
            photons_per_pulse=n,
            repetition_rate="80 MHz",
            waist_radius="1 mm",
        )
    put(lib, "shifter", "PhaseShifter")
    put(lib, "laser810", "Laser", wavelength="810 nm", power="1 mW", waist_radius="1 mm")
    return lib


def mzi(lib, shifter=None, arm="100 mm", laser="laser810"):
    instances = [
        inst("src", laser),
        inst("bs1", "bs"),
        inst("m1", "mirror"),
        inst("m2", "mirror"),
        inst("bs2", "bs"),
        inst("d1", "pd"),
        inst("d2", "pd"),
    ]
    links = [
        fs("in", "src.out", "bs1.in_a"),
        fs("a1", "bs1.out_a", "m1.in"),
        fs("a2", "m1.out", "bs2.in_a"),
        fs("b2", "m2.out", "bs2.in_b"),
        fs("o1", "bs2.out_a", "d1.in"),
        fs("o2", "bs2.out_b", "d2.in"),
    ]
    if shifter is None:
        links.append(fs("b1", "bs1.out_b", "m2.in", arm))
    else:
        instances.append(inst("ps", "shifter", shifter))
        links += [fs("b1", "bs1.out_b", "ps.in", arm), fs("b1b", "ps.out", "m2.in", "0 mm")]
    return run(lib, instances, links, [probe("p1", "o1"), probe("p2", "o2")])


def test_library_parts_load_and_simulate():
    for ref, cls in (
        ("sources/fock_1photon_810", "FockSource"),
        ("sources/fock_2photon_810", "FockSource"),
        ("polarization/phase_shifter_ideal", "PhaseShifter"),
    ):
        assert LIBRARY.exists(ref)
        assert LIBRARY.load_part(ref).type == cls
    assert LIBRARY.load_part("sources/fock_2photon_810").params.photons_per_pulse == 2
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "sources/fock_1photon_810"),
                inst("ps", "polarization/phase_shifter_ideal", {"phase": "90 deg"}),
                inst("pd", "detectors/photodiode_generic"),
            ],
            "links": [fs("a", "src.out", "ps.in"), fs("b", "ps.out", "pd.in")],
            "probes": [probe("p", "b", "probes/power_meter")],
        }
    )
    result = simulate(setup, LIBRARY)
    assert result.probes["p"]["power_W"] == pytest.approx(PHOTON * RATE, rel=1e-6)
    assert not [i for i in result.issues if i.severity == "error"]


@pytest.mark.parametrize("n", [0, 1, 3])
def test_mean_power_is_photon_number_times_energy_times_rate(lib, n):
    result = run(
        lib,
        [inst("src", f"fock{n}"), inst("pd", "pd")],
        [fs("a", "src.out", "pd.in")],
        [probe("p", "a")],
    )
    p = power(result)
    assert math.isfinite(p)
    assert p == pytest.approx(n * PHOTON * RATE, rel=1e-9, abs=0.0)
    assert np.isfinite(result.links["a"].end.power)


def test_fock_occupation(lib):
    cls = get_component("FockSource")
    comp = cls(
        cls.Params.model_validate(
            {
                "wavelength": "810 nm",
                "photons_per_pulse": 2,
                "repetition_rate": "80 MHz",
                "waist_radius": "1 mm",
                "distinguishability_tag": 3,
            }
        ),
        cls.Settings.model_validate({"polarization_angle": "90 deg"}),
    )
    n, tag, jones, wl, rep = comp.fock_occupation()
    assert (n, tag) == (2, 3)
    assert np.allclose(jones, [0.0, 1.0], atol=1e-12)
    assert wl == pytest.approx(LAM)
    assert rep == pytest.approx(RATE)


def test_fock_source_splits_5050_over_two_photodiodes(lib):
    result = run(
        lib,
        [inst("src", "fock3"), inst("bs", "bs"), inst("d1", "pd"), inst("d2", "pd")],
        [fs("a", "src.out", "bs.in_a"), fs("t", "bs.out_a", "d1.in"), fs("r", "bs.out_b", "d2.in")],
        [probe("p1", "t"), probe("p2", "r")],
    )
    total = 3 * PHOTON * RATE
    assert power(result, "p1") == pytest.approx(total / 2)
    assert power(result, "p2") == pytest.approx(total / 2)
    assert result.readouts["d1"]["incident_power_W"] == pytest.approx(total / 2)
    assert not [i for i in result.issues if i.severity != "info"]


def test_fock_source_is_ignored_by_pair_engines(lib):
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "fock1"),
                inst("bs", "bs"),
                inst("d1", "pd"),
                inst("d2", "pd"),
            ],
            "links": [
                fs("a", "src.out", "bs.in_a"),
                fs("t", "bs.out_a", "d1.in"),
                fs("r", "bs.out_b", "d2.in"),
            ],
        }
    )
    rates = simulate_rates(setup, lib)
    assert rates.sources == {}
    assert not [i for i in rates.issues if i.severity == "error"]
    gauss = simulate_gaussian(setup, lib)
    assert not [i for i in gauss.issues if i.severity == "error"]


@pytest.mark.parametrize(
    ("axis", "expected"),
    [("both", [1j, 1j]), ("H", [1j, 1.0]), ("V", [1.0, 1j])],
)
def test_phase_shifter_jones(lib, axis, expected):
    comp = get_component("PhaseShifter")
    c = comp(
        comp.Params.model_validate({}),
        comp.Settings.model_validate({"phase": "90 deg", "axis": axis}),
    )
    from sequoia.components.base import Context
    from sequoia.light.gaussian import GaussianMode
    from sequoia.light.spectrum import Spectrum
    from sequoia.light.state import Beam

    beam = Beam.launch("s", Spectrum.line(LAM, 0.0), GaussianMode.from_waist(LAM, 1e-3))
    ctx = Context(issues=[])
    ctx.subject = "'ps'"
    ((_, out),) = c.propagate("in", beam, ctx)
    assert np.allclose(np.diag(out.jones[0]), expected, atol=1e-12)
    assert abs(out.jones[0, 0, 1]) + abs(out.jones[0, 1, 0]) == 0.0


def test_phase_shifter_polarization_axis_rotates_diagonal_input(lib):
    result = run(
        lib,
        [
            inst("src", "laser810", {"polarization_angle": "45 deg"}),
            inst("ps", "shifter", {"phase": "180 deg", "axis": "H"}),
            inst("pd", "pd"),
        ],
        [fs("a", "src.out", "ps.in"), fs("b", "ps.out", "pd.in")],
    )
    assert result.links["b"].end.power == pytest.approx(1e-3)
    st = result.links["b"].end.stokes
    assert st[2] / st[0] == pytest.approx(-1.0, abs=3e-3)


def split(result):
    return power(result, "p1"), power(result, "p2")


def test_mzi_phase_shifter_pi_swaps_ports(lib):
    p1, p2 = split(mzi(lib))
    assert p2 == pytest.approx(1e-3)
    assert p1 == pytest.approx(0.0, abs=1e-12)
    p1, p2 = split(mzi(lib, {"phase": "180 deg"}))
    assert p1 == pytest.approx(1e-3)
    assert p2 == pytest.approx(0.0, abs=1e-12)


def test_mzi_phase_shifter_half_pi_gives_5050(lib):
    p1, p2 = split(mzi(lib, {"phase": "90 deg"}))
    assert p1 == pytest.approx(0.5e-3)
    assert p2 == pytest.approx(0.5e-3)


def test_half_wavelength_path_equals_pi_phase_shifter(lib):
    by_shifter = split(mzi(lib, {"phase": "180 deg"}))
    by_path = split(mzi(lib, arm=f"{0.1 + LAM / 2!r} m"))
    assert by_path[0] == pytest.approx(by_shifter[0], abs=1e-9 * 1e-3)
    assert by_path[1] == pytest.approx(by_shifter[1], abs=1e-9 * 1e-3)


def test_template_mzi_with_phase_shifter_yaml_round_trip(tmp_path):
    setup = load_setup(ROOT / "templates/mach_zehnder.yaml")
    raw = {
        "instances": [
            *[{"id": i.id, "part": i.part, "settings": {}} for i in setup.instances],
            {
                "id": "ps",
                "part": "polarization/phase_shifter_ideal",
                "settings": {"phase": "45 deg", "axis": "V"},
            },
            {
                "id": "fs",
                "part": "sources/fock_2photon_810",
                "settings": {"polarization_angle": "30 deg"},
            },
        ],
        "links": [],
    }
    setup = setup_from_dict(raw)
    path = tmp_path / "s.yaml"
    save_setup(setup, path)
    again = load_setup(path)
    ps = next(i for i in again.instances if i.id == "ps")
    assert ps.part == "polarization/phase_shifter_ideal"
    assert ps.settings["axis"] == "V"
    assert [i.id for i in again.instances] == [i.id for i in setup.instances]


def test_new_parts_in_library_listing():
    refs = {str(r) for r in LIBRARY.list()} if hasattr(LIBRARY, "list") else set()
    for ref in (
        "sources/fock_1photon_810",
        "sources/fock_2photon_810",
        "polarization/phase_shifter_ideal",
    ):
        assert ref in refs or LIBRARY.exists(ref)
