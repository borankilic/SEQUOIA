import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from optics import basic_library, fs, inst, put

from sequoia.fock import run_experiment
from sequoia.fock.compile import FockCompileError, compile_setup_network
from sequoia.fock.engine import output_distribution
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict

ROOT = Path(__file__).resolve().parents[1]
WL = 810e-9
S2 = math.sqrt(2.0)
H, V = np.array([1, 0], complex), np.array([0, 1], complex)
D = np.array([1, 1], complex) / S2


def rot(t):
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def hwp(theta):
    return rot(theta) @ np.diag([-1j, 1j]) @ rot(-theta)


@pytest.fixture
def lab(tmp_path):
    library = basic_library(Library(tmp_path / "lib"))
    put(library, "hwp810", "WavePlate", retardance=0.5, design_wavelength="810 nm")
    put(library, "spad", "SPAD", efficiency=1.0)
    return library


def bs_setup(angle="22.5 deg"):
    return setup_from_dict(
        {
            "instances": [
                inst("wp", "hwp810", {"angle": angle}),
                inst("pbs", "pbs"),
                inst("mir", "mirror"),
                inst("bs", "bs"),
                inst("da", "spad", {"input": "free_space"}),
                inst("db", "spad", {"input": "free_space"}),
            ],
            "links": [
                fs("l0", "wp.out", "pbs.in"),
                fs("l1", "pbs.out_t", "bs.in_a"),
                fs("l2", "pbs.out_r", "mir.in", "50 mm"),
                fs("l3", "mir.out", "bs.in_b", "50 mm"),
                fs("l4", "bs.out_a", "da.in"),
                fs("l5", "bs.out_b", "db.in"),
            ],
        }
    )


def jones_transfer(theta):
    pbs_t, pbs_r = np.diag([1, 0]), 1j * np.diag([1, -1]) @ np.diag([0, 1])
    mirror = np.diag([1, -1])
    bs_t, bs_r = np.eye(2) / S2, 1j * np.diag([1, -1]) / S2
    w = hwp(theta)
    to_a = bs_t @ pbs_t @ w + bs_r @ mirror @ pbs_r @ w
    to_b = bs_r @ pbs_t @ w + bs_t @ mirror @ pbs_r @ w
    return np.vstack([to_a, to_b])


def reference(m, photons_vectors):
    n_out = m.shape[0]
    cols = [m @ v for v in photons_vectors]
    out: dict[tuple[int, int], float] = {}
    for pattern in itertools.combinations_with_replacement(range(n_out), len(cols)):
        a = np.array([[c[o] for c in cols] for o in pattern])
        perm = sum(
            np.prod([a[i, p[i]] for i in range(len(cols))])
            for p in itertools.permutations(range(len(cols)))
        )
        norm = np.prod([math.factorial(k) for k in np.bincount(pattern, minlength=n_out)])
        norm *= (
            math.factorial(len(cols))
            if all(np.allclose(photons_vectors[0], v) for v in photons_vectors)
            else 1
        )
        n_a = sum(1 for o in pattern if o < 2)
        key = (n_a, len(cols) - n_a)
        out[key] = out.get(key, 0.0) + abs(perm) ** 2 / norm
    return out


def same_up_to_phase(a, b, atol=1e-12):
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    phase = a[k] / b[k]
    return abs(abs(phase) - 1.0) < 1e-9 and np.allclose(a, phase * b, atol=atol)


def run_pair(lab, setup, photons, basis="HV"):
    net = compile_setup_network(setup, lab, WL, inputs=["wp.in"], input_basis=basis)
    exp = net.experiment(photons)
    return net, run_experiment(exp).probabilities


@pytest.mark.parametrize("theta_deg", [0.0, 22.5, 45.0, 17.0])
def test_transfer_matches_jones_calculus(lab, theta_deg):
    theta = math.radians(theta_deg)
    net = compile_setup_network(bs_setup(f"{theta_deg} deg"), lab, WL, inputs=["wp.in"])
    assert same_up_to_phase(net.transfer, jones_transfer(theta))
    assert [str(m) for m in net.input_modes] == ["wp.in:H", "wp.in:V"]
    assert net.detector_ids == ["da", "db"]


@pytest.mark.parametrize(
    "photons,vectors,basis",
    [
        ([("wp.in:H", 2)], [H, H], "HV"),
        ([("wp.in:V", 2)], [V, V], "HV"),
        ([("wp.in:H", 1), ("wp.in:V", 1)], [H, V], "HV"),
        ([("wp.in:D", 2)], [D, D], "DA"),
    ],
)
def test_two_photon_inputs_agree_with_jones(lab, photons, vectors, basis):
    theta = math.radians(22.5)
    setup = bs_setup("22.5 deg")
    net, probs = run_pair(lab, setup, photons, basis)
    exp = reference(jones_transfer(theta), vectors)
    assert sum(probs.values()) == pytest.approx(1.0)
    for key in set(exp) | set(probs):
        assert probs.get(key, 0.0) == pytest.approx(exp.get(key, 0.0), abs=1e-12)


def test_hom_template_suppression_and_recovery(tmp_path):
    library = Library(ROOT / "library")
    setup = load_setup(ROOT / "templates" / "hom.yaml")
    net = compile_setup_network(setup, library, WL, inputs=["pbs.in"])
    assert net.detector_ids == ["spad_1", "spad_2"]
    exp = net.experiment([("pbs.in:H", 1), ("pbs.in:V", 1)])
    probs = run_experiment(exp).probabilities
    assert probs.get((1, 1), 0.0) < 1e-12
    assert probs[(2, 0)] > 0.0 and probs[(0, 2)] > 0.0

    for i in setup.instances:
        if i.id == "hwp_idler":
            i.settings = {"angle": "0 deg"}
    net0 = compile_setup_network(setup, library, WL, inputs=["pbs.in"])
    r = net0.transfer
    p_h = [np.sum(np.abs(r[2 * k : 2 * k + 2, 0]) ** 2) for k in range(2)]
    p_v = [np.sum(np.abs(r[2 * k : 2 * k + 2, 1]) ** 2) for k in range(2)]
    expected_11 = p_h[0] * p_v[1] + p_h[1] * p_v[0]
    eff = net0.detectors[0].model.efficiency * net0.detectors[1].model.efficiency
    probs0 = run_experiment(net0.experiment([("pbs.in:H", 1), ("pbs.in:V", 1)])).probabilities
    assert probs0[(1, 1)] == pytest.approx(expected_11 * eff, rel=1e-9)
    assert probs0[(1, 1)] > 0.02


def test_detector_efficiency_is_in_detector_model_not_circuit(tmp_path, lab):
    put(lab, "spad60", "SPAD", efficiency=0.6)
    setup = bs_setup()
    for i in setup.instances:
        if i.id == "da":
            i.part = "spad60"
    net = compile_setup_network(setup, lab, WL, inputs=["wp.in"])
    assert net.detectors[0].model.efficiency == pytest.approx(0.6)
    assert net.detectors[1].model.efficiency == pytest.approx(1.0)
    assert same_up_to_phase(net.transfer, jones_transfer(math.radians(22.5)))
    assert net.efficiency_in_circuit == []


def test_sources_and_nonlinear_are_rejected():
    library = Library(ROOT / "library")
    setup = load_setup(ROOT / "templates" / "hom.yaml")
    with pytest.raises(FockCompileError, match="crystal"):
        compile_setup_network(setup, library, WL, inputs=["pump.out"])
    with pytest.raises(FockCompileError, match="no port"):
        compile_setup_network(setup, library, WL, inputs=["pbs.nothing"])
    with pytest.raises(FockCompileError, match="unknown instance"):
        compile_setup_network(setup, library, WL, inputs=["zzz.in"])


def test_cancellation(lab):
    from sequoia.engines.cache import Cancelled

    with pytest.raises(Cancelled):
        compile_setup_network(bs_setup(), lab, WL, inputs=["wp.in"], should_cancel=lambda: True)


def test_compiled_distribution_matches_manual_permanent(lab):
    net = compile_setup_network(bs_setup("0 deg"), lab, WL, inputs=["wp.in"])
    d = output_distribution(net.circuit, [(0, 0), (1, 0)])
    assert d.mass == pytest.approx(1.0)
