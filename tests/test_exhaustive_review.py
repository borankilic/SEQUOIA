"""Broad conditions review: every catalogue part, extreme values, document fuzzing, file I/O."""

import math
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import PortDirection, PortKind, get_component
from sequoia.engines.classical import simulate
from sequoia.engines.rates import simulate_rates
from sequoia.errors import SequoiaError
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")
PARTS = LIB.index()
PASSIVE_FS = {
    "Lens",
    "Objective",
    "Mirror",
    "BeamSplitter",
    "DichroicMirror",
    "Iris",
    "BeamDump",
    "WavePlate",
    "Polarizer",
    "PBS",
    "SpectralFilter",
    "NDFilter",
}


def laser_for(wavelength_nm: float) -> dict:
    return {
        "id": "laser",
        "part": "sources/laser_780_cw",
        "overrides": {"wavelength": f"{wavelength_nm} nm", "power": "5 mW"},
    }


def design_wavelength(part) -> float:
    p = part.params
    for name in ("design_wavelength", "wavelength"):
        v = getattr(p, name, None)
        if v is not None and hasattr(v, "si"):
            return v.si * 1e9
    for name in ("transmission", "reflectance", "transmittance", "efficiency", "responsivity"):
        v = getattr(p, name, None)
        if hasattr(v, "x_si") and len(v.x) > 1:
            x = np.asarray(v.x_si) * 1e9
            y = np.asarray(v.y_si)
            return float(x[int(np.argmax(y))]) if np.ptp(y) > 0 else float(np.median(x))
    return 810.0


def chain_for(info):
    part = LIB.load_part(info.ref)
    wl = min(max(design_wavelength(part), 350.0), 1650.0)
    comp = get_component(part.type)(part.params)
    ports = comp.ports()
    ins = [p for p in ports if p.direction == PortDirection.IN and p.kind != PortKind.ELECTRICAL]
    outs = [p for p in ports if p.direction == PortDirection.OUT and p.kind != PortKind.ELECTRICAL]
    return part, wl, ins, outs


@pytest.mark.parametrize("info", [i for i in PARTS if i.type in PASSIVE_FS], ids=lambda i: i.ref)
def test_every_passive_optic_transmits_a_beam_physically(info):
    part, wl, ins, outs = chain_for(info)
    instances = [laser_for(wl), {"id": "x", "part": info.ref}]
    links = [
        {
            "id": "a",
            "kind": "free_space",
            "from": "laser.out",
            "to": f"x.{ins[0].name}",
            "length": "50 mm",
        }
    ]
    probes = []
    for k, out in enumerate(outs):
        instances.append({"id": f"d{k}", "part": "freespace/beam_dump"})
        links.append(
            {
                "id": f"o{k}",
                "kind": "free_space",
                "from": f"x.{out.name}",
                "to": f"d{k}.in",
                "length": "50 mm",
            }
        )
        probes.append({"id": f"p{k}", "part": "probes/power_meter", "link": f"o{k}"})
    setup = setup_from_dict({"instances": instances, "links": links, "probes": probes})
    result = simulate(setup, LIB)
    errors = [i for i in result.issues if i.severity == "error"]
    assert not errors, errors
    total_in = result.links["a"].end.power
    total_out = 0.0
    for k in range(len(outs)):
        light = result.links[f"o{k}"].end
        if light is None:
            continue
        assert math.isfinite(light.power) and light.power >= 0.0
        m = light.mode
        assert math.isfinite(m.w_x) and m.w_x > 0 and m.m2_x >= 1.0 - 1e-9
        total_out += light.power
    assert total_out <= total_in * (1 + 1e-6)


FIBERS = [i for i in PARTS if i.type == "OpticalFiber"]


@pytest.mark.parametrize("info", FIBERS, ids=lambda i: i.ref)
def test_every_fiber_carries_light_in_a_launch_chain(info):
    part = LIB.load_part(info.ref)
    wl = 810.0
    cutoff = getattr(part.params, "cutoff_wavelength", None)
    if cutoff is not None:
        wl = max(cutoff.si * 1e9 + 40.0, 500.0)
    wl = min(wl, 1600.0)
    setup = setup_from_dict(
        {
            "instances": [
                laser_for(wl),
                {"id": "c", "part": "fiber/coupler_f11mm"},
                {"id": "r", "part": "fiber/coupler_f11mm", "settings": {"mode": "to_free"}},
                {"id": "d", "part": "freespace/beam_dump"},
            ],
            "links": [
                {
                    "id": "a",
                    "kind": "free_space",
                    "from": "laser.out",
                    "to": "c.in",
                    "length": "50 mm",
                },
                {
                    "id": "f",
                    "kind": "fiber",
                    "from": "c.out",
                    "to": "r.in",
                    "fiber": info.ref,
                    "length": "2 m",
                    "connector_loss_db": 0.3,
                },
                {"id": "g", "kind": "free_space", "from": "r.out", "to": "d.in", "length": "30 mm"},
            ],
        }
    )
    result = simulate(setup, LIB)
    assert not [i for i in result.issues if i.severity == "error"]
    eta = result.links["f"].info["coupling_efficiency"]
    assert 0.0 <= eta <= 1.0 + 1e-9
    assert result.links["g"].end.power <= result.links["a"].end.power


DETECTORS = [i for i in PARTS if i.type in ("Photodiode", "SPAD", "SNSPD")]


@pytest.mark.parametrize("info", DETECTORS, ids=lambda i: i.ref)
def test_every_detector_reads_out_or_reports_a_clear_problem(info):
    part = LIB.load_part(info.ref)
    wl = min(max(design_wavelength(part), 400.0), 1600.0)
    comp = get_component(part.type)(part.params)
    in_port = next(p for p in comp.ports() if p.direction == PortDirection.IN)
    instances = [laser_for(wl), {"id": "det", "part": info.ref}]
    if in_port.kind == PortKind.FIBER:
        instances += [
            {"id": "c", "part": "fiber/coupler_f11mm"},
        ]
        links = [
            {"id": "a", "kind": "free_space", "from": "laser.out", "to": "c.in", "length": "50 mm"},
            {
                "id": "f",
                "kind": "fiber",
                "from": "c.out",
                "to": f"det.{in_port.name}",
                "fiber": "fiber/smf_780",
                "length": "1 m",
            },
        ]
    else:
        links = [
            {
                "id": "a",
                "kind": "free_space",
                "from": "laser.out",
                "to": f"det.{in_port.name}",
                "length": "50 mm",
            }
        ]
    if part.type in ("SPAD", "SNSPD"):
        instances.append({"id": "tag", "part": "electronics/tagger_generic_4ch"})
        links.append({"id": "cab", "kind": "cable", "from": "det.out", "to": "tag.ch1"})
    setup = setup_from_dict({"instances": instances, "links": links})
    result = simulate(setup, LIB)
    assert not [i for i in result.issues if i.severity == "error"]
    values = result.readouts.get("det", {})
    assert all(math.isfinite(v) for v in values.values() if isinstance(v, int | float))


def base_setup(overrides=None, length="100 mm"):
    return setup_from_dict(
        {
            "instances": [
                {"id": "laser", "part": "sources/laser_780_cw", "overrides": overrides or {}},
                {"id": "lens", "part": "freespace/lens_f100mm"},
                {"id": "pd", "part": "detectors/photodiode_generic"},
            ],
            "links": [
                {
                    "id": "a",
                    "kind": "free_space",
                    "from": "laser.out",
                    "to": "lens.in",
                    "length": length,
                },
                {
                    "id": "b",
                    "kind": "free_space",
                    "from": "lens.out",
                    "to": "pd.in",
                    "length": length,
                },
            ],
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"power": "1 kW"},
        {"power": "1 fW"},
        {"waist_radius": "1 um"},
        {"waist_radius": "50 mm"},
        {"m_squared": 50},
        {"wavelength": "200 nm"},
        {"wavelength": "5 um"},
        {"polarization_extinction_ratio": 1},
    ],
)
def test_extreme_laser_values_do_not_crash(overrides):
    result = simulate(base_setup(overrides), LIB)
    for state in result.links.values():
        if state.end is not None:
            assert math.isfinite(state.end.power)
            assert math.isfinite(state.end.mode.w_x)


@pytest.mark.parametrize("length", ["0 mm", "1 um", "100 km"])
def test_extreme_lengths(length):
    result = simulate(base_setup(None, length), LIB)
    assert all(math.isfinite(s.end.power) for s in result.links.values() if s.end is not None)


@pytest.mark.parametrize(
    "bad",
    [
        {"power": "-1 mW"},
        {"waist_radius": "0 mm"},
        {"m_squared": 0.2},
        {"power": "ten"},
        {"wavelength": "810"},
    ],
)
def test_invalid_values_are_reported_not_raised(bad):
    try:
        setup = base_setup(bad)
    except SequoiaError:
        return
    try:
        result = simulate(setup, LIB)
    except SequoiaError:
        return
    assert any(i.severity == "error" for i in result.issues)


def test_corrupt_and_foreign_files_give_clean_errors(tmp_path):
    cases = {
        "empty.yaml": "",
        "list.yaml": "- 1\n- 2\n",
        "binary.yaml": "\x00\x01\x02",
        "badyaml.yaml": "instances: [unclosed\n",
        "unknown_key.yaml": "frobnicate: 1\n",
        "bad_link.yaml": "links:\n  - {id: l, kind: carrier_pigeon, from: a.b, to: c.d}\n",
    }
    for name, content in cases.items():
        path = tmp_path / name
        path.write_text(content)
        with pytest.raises(SequoiaError):
            load_setup(path)
    with pytest.raises(SequoiaError):
        load_setup(tmp_path / "missing.yaml")


def test_missing_parts_and_ports_are_diagnosed():
    setup = setup_from_dict(
        {
            "instances": [
                {"id": "a", "part": "does/not_exist"},
                {"id": "l", "part": "sources/laser_780_cw"},
            ],
            "links": [
                {"id": "x", "kind": "free_space", "from": "l.out", "to": "a.in", "length": "1 m"}
            ],
        }
    )
    with pytest.raises(SequoiaError):
        simulate(setup, LIB)


@pytest.mark.parametrize("seed", range(6))
def test_document_fuzz_keeps_invariants(seed, tmp_path):
    from sequoia.gui.document import SetupDocument

    rng = random.Random(seed)
    doc = SetupDocument(LIB)
    parts = [
        i.ref
        for i in PARTS
        if i.type
        in (
            "Laser",
            "Lens",
            "Mirror",
            "BeamSplitter",
            "SpectralFilter",
            "WavePlate",
            "PBS",
            "Photodiode",
            "BeamDump",
            "Polarizer",
            "FiberCoupler",
            "OpticalFiber",
        )
        and i.type != "OpticalFiber"
    ]
    for _ in range(120):
        ids = [i.id for i in doc.setup.instances]
        action = rng.choice(
            ["add", "add", "link", "move", "rotate", "remove", "undo", "redo", "setting", "convert"]
        )
        try:
            if action == "add":
                doc.add_instance(rng.choice(parts), [rng.randint(-30, 30), rng.randint(-20, 20)])
            elif action == "link" and len(ids) >= 2:
                a, b = rng.sample(ids, 2)
                pa = [p.name for p in doc.ports(a) if p.direction == PortDirection.OUT]
                pb = [p.name for p in doc.ports(b) if p.direction == PortDirection.IN]
                if pa and pb:
                    doc.add_link("free_space", f"{a}.{pa[0]}", f"{b}.{pb[0]}")
            elif action == "move" and ids:
                doc.move_instance(rng.choice(ids), [rng.randint(-30, 30), rng.randint(-20, 20)])
            elif action == "rotate" and ids:
                doc.rotate_instance(rng.choice(ids), rng.choice([0, 90, 180, 270, 45]))
            elif action == "remove" and ids:
                doc.remove_instance(rng.choice(ids))
            elif action == "undo":
                doc.undo_stack.undo()
            elif action == "redo":
                doc.undo_stack.redo()
            elif action == "setting" and ids:
                doc.set_setting(rng.choice(ids), "angle", f"{rng.randint(0, 180)} deg")
            elif action == "convert":
                links = [x.id for x in doc.setup.links if x.kind == "free_space"]
                if links:
                    doc.convert_link_to_fiber(rng.choice(links))
        except ValueError:
            pass
        ids_now = {i.id for i in doc.setup.instances}
        for link in doc.setup.links:
            assert link.src.split(".")[0] in ids_now and link.dst.split(".")[0] in ids_now
    path = tmp_path / "fuzz.yaml"
    doc.save(path)
    again = load_setup(path)
    assert setup_to_dict(again) == setup_to_dict(doc.setup)
    try:
        simulate(doc.setup, LIB)
    except SequoiaError:
        pass


def test_all_templates_run_all_engines_briefly():
    from sequoia.engines.gaussian import UnsupportedSourceError, simulate_gaussian
    from sequoia.engines.montecarlo import simulate_timetags
    from sequoia.engines.rates import has_rate_devices

    for path in sorted((ROOT / "templates").glob("*.yaml")):
        setup = load_setup(path)
        result = simulate(setup, LIB)
        assert not [i for i in result.issues if i.severity == "error"], path.name
        if has_rate_devices(setup, LIB, None, result.network):
            rates = simulate_rates(setup, LIB, classical=result)
            assert not [i for i in rates.issues if i.severity == "error"], path.name
            fock = any(d.get("fock_hz", 0.0) > 0 for d in rates.detectors.values())
            if fock:
                with pytest.raises(UnsupportedSourceError):
                    simulate_timetags(setup, LIB, 1e-4, seed=1)
            tags = simulate_timetags(
                setup, LIB, 0.05 if not fock else 1e-4, seed=1, fock_as_poisson=fock
            )
            assert all(np.all(np.diff(t) >= 0) for t in tags.channels.values())
            if any(a.get("type") for a in setup.analyses):
                simulate_gaussian(setup, LIB, fock_as_poisson=fock)


def test_yaml_roundtrip_of_every_setup_and_template():
    for path in sorted(
        list((ROOT / "setups").glob("*.yaml")) + list((ROOT / "templates").glob("*.yaml"))
    ):
        setup = load_setup(path)
        again = setup_from_dict(yaml.safe_load(yaml.safe_dump(setup_to_dict(setup))))
        assert setup_to_dict(again) == setup_to_dict(setup), path.name
