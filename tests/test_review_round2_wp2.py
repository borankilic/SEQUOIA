import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis import metrics as M
from sequoia.analysis.chsh import run_chsh
from sequoia.engines.rates import simulate_rates
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")
V = 0.97


@pytest.fixture(scope="module")
def bell():
    return simulate_rates(load_setup(ROOT / "setups" / "bell_test.yaml"), LIB)


def test_bell_source_pair_state_is_the_werner_state(bell):
    st = bell.pair_states[("s_a1", "s_b1")]
    assert st["source"] == "source" and st["basis"] == "HH,HV,VH,VV"
    assert "signal on s_a1" in st["photon_a"] and "idler on s_b1" in st["photon_b"]
    assert "conventions.md" in st["frame"]
    assert st["rate_hz"] == pytest.approx(bell.sources["source"]["rate_hz"], rel=1e-9)
    assert np.allclose(st["rho"], M.werner_state(V), atol=1e-9)
    assert st["bell_fidelity"]["phi+"] == pytest.approx((1 + 3 * V) / 4, abs=1e-7)
    for name in ("phi-", "psi+", "psi-"):
        assert st["bell_fidelity"][name] == pytest.approx((1 - V) / 4, abs=1e-7)
    assert st["concurrence"] == pytest.approx((3 * V - 1) / 2, abs=1e-9)
    assert st["purity"] == pytest.approx((1 + 3 * V**2) / 4, abs=1e-9)
    for s in (st["stokes_a"], st["stokes_b"]):
        assert s[0] == pytest.approx(1.0) and np.linalg.norm(s[1:]) < 1e-9


def test_bell_conditional_states(bell):
    cond = bell.pair_states[("s_a1", "s_b1")]["conditional"]
    expected = {"H": (V, 0, 0), "V": (-V, 0, 0), "D": (0, V, 0), "A": (0, -V, 0)}
    expected |= {"R": (0, 0, V), "L": (0, 0, -V)}
    for label, s in expected.items():
        assert cond[label]["probability"] == pytest.approx(0.5, abs=1e-9)
        assert np.allclose(cond[label]["stokes_b"], (1.0, *s), atol=1e-9), label
    rho_h = cond["H"]["rho_b"]
    assert rho_h[0, 0].real == pytest.approx((1 + V) / 2)


def test_pair_state_orientation_and_series_links(bell):
    ab, ba = bell.pair_states[("s_a1", "s_b1")], bell.pair_states[("s_b1", "s_a1")]
    assert np.allclose(ba["rho"], M.swap_qubits(ab["rho"]))
    assert "idler on s_b1" in ba["photon_a"]
    assert ("s_a1", "s_a2") not in bell.pair_states
    hbt = simulate_rates(load_setup(ROOT / "setups" / "hbt_heralded.yaml"), LIB)
    assert ("s_collect", "s_lp") not in hbt.pair_states
    assert ("s_pbs", "s_signal") not in hbt.pair_states


def test_pair_state_through_the_analysers_matches_chsh(bell):
    src = bell.pair_states[("s_a1", "s_b1")]["rho"]
    after = bell.pair_states[("s_a3", "s_b3")]
    h, v = M.KETS["H"], M.KETS["V"]
    e_plates = M.correlation_from_state(after["rho"], h, h)
    assert e_plates == pytest.approx(M.correlation_from_angles(src, 0.0, math.radians(22.5)))
    p_hh = np.real(np.kron(h, h).conj() @ after["rho"] @ np.kron(h, h))
    split = bell.pair_states[("s_at", "s_bt")]
    assert split["rate_hz"] == pytest.approx(after["rate_hz"] * p_hh, rel=1e-6)
    det = bell.pair_states[("f_at", "f_bt")]
    assert det["rate_hz"] < split["rate_hz"]
    assert det["concurrence"] < 1e-9 and det["purity"] == pytest.approx(1.0, abs=1e-9)
    assert np.real(np.kron(v, v).conj() @ det["rho"] @ np.kron(v, v)) < 1e-9
    result = run_chsh(
        load_setup(ROOT / "setups" / "bell_test.yaml"),
        LIB,
        "settings:hwp_a.angle",
        "settings:hwp_b.angle",
        [0.0, 22.5],
        [11.25, 33.75],
        quantity="true_hz",
    )
    for i, ha in enumerate((0.0, 22.5)):
        for j, hb in enumerate((11.25, 33.75)):
            e = M.correlation_from_angles(src, math.radians(2 * ha), math.radians(2 * hb))
            assert result.correlations[i, j] == pytest.approx(e, abs=2e-3)
    assert M.chsh_from_state(src) == pytest.approx(result.s, rel=2e-3)


def _bell_dict():
    return yaml.safe_load((ROOT / "setups" / "bell_test.yaml").read_text())


def test_mirror_on_one_arm_turns_phi_plus_into_phi_minus():
    raw = _bell_dict()
    raw["instances"].append({"id": "m", "part": "freespace/mirror_ideal", "position": [3, -4]})
    for link in raw["links"]:
        if link["id"] == "s_a1":
            link["to"] = "m.in"
    raw["links"].append(
        {"id": "s_m", "kind": "free_space", "from": "m.out", "to": "hwp_a.in", "length": "50 mm"}
    )
    r = simulate_rates(setup_from_dict(raw), LIB)
    assert r.pair_states[("s_a1", "s_b1")]["bell_fidelity"]["phi+"] == pytest.approx(
        (1 + 3 * V) / 4
    )
    assert r.pair_states[("s_m", "s_b1")]["bell_fidelity"]["phi-"] == pytest.approx((1 + 3 * V) / 4)


@pytest.mark.parametrize("state", ["phi-", "psi+", "psi-"])
def test_other_bell_inputs_reach_the_source_links(state):
    raw = _bell_dict()
    raw["instances"][0]["overrides"] = {"visibility": 1.0, "state": state}
    r = simulate_rates(setup_from_dict(raw), LIB)
    st = r.pair_states[("s_a1", "s_b1")]
    assert st["bell_fidelity"][state] == pytest.approx(1.0, abs=1e-9)
    assert st["concurrence"] == pytest.approx(1.0, abs=1e-9)


def test_type_two_product_pairs_have_no_entanglement():
    hbt = simulate_rates(load_setup(ROOT / "setups" / "hbt_heralded.yaml"), LIB)
    st = hbt.pair_states[("s_signal", "s_idler")]
    assert "signal on s_signal" in st["photon_a"] and "idler on s_idler" in st["photon_b"]
    # pbs_cube_810 has a finite extinction ratio: a little idler leaks into the signal arm
    assert st["purity"] == pytest.approx(1.0, abs=1e-4)
    assert st["concurrence"] == pytest.approx(0.0, abs=1e-3)
    assert st["rho"][1, 1].real == pytest.approx(1.0, abs=1e-4)
    assert st["conditional"]["V"]["probability"] < 1e-4
    hom = simulate_rates(load_setup(ROOT / "setups" / "hom.yaml"), LIB)
    st = hom.pair_states[("delay_arm", "s_idler_b")]
    assert st["purity"] == pytest.approx(1.0, abs=1e-6)
    assert st["concurrence"] == pytest.approx(0.0, abs=1e-6)
    assert st["rho"][0, 0].real == pytest.approx(1.0, abs=1e-6)
    assert st["conditional"]["V"]["probability"] < 1e-9
    assert np.isnan(st["conditional"]["V"]["stokes_b"]).all()


def test_conditional_state_helpers():
    psi = M.bell_state("psi-")
    m, p = M.conditional_state(psi, M.KETS["D"])
    assert p == pytest.approx(0.5)
    assert np.allclose(M.stokes_of(m), (1, 0, -1, 0))
    ra, rb = M.reduced_states(np.kron(M.KETS["H"], M.KETS["R"]))
    assert np.allclose(M.stokes_of(ra), (1, 1, 0, 0))
    assert np.allclose(M.stokes_of(rb), (1, 0, 0, -1))
