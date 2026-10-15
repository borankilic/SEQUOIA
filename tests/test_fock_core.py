import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import comb, stirling2

from sequoia.engines.cache import Cancelled
from sequoia.fock import (
    Circuit,
    DetectorLayout,
    DetectorSpec,
    Experiment,
    FockError,
    FockResourceError,
    IdealPNR,
    Limits,
    Loss,
    PseudoPNR,
    Threshold,
    condition,
    load_experiment,
    marginal,
    mixed_distribution,
    output_distribution,
    run_experiment,
    sample,
    save_experiment,
    simplify_losses,
)
from sequoia.fock.circuit import bs_matrix, db_to_loss
from sequoia.fock.cli import main as fock_main
from sequoia.fock.engine import permanents
from sequoia.fock.import_perceval_gui import FockImportError, import_perceval_gui

GUI_DIR = Path(__file__).resolve().parents[2] / "percevalGUI-v0 0 11"


def brute_perm(m):
    n = m.shape[0]
    return sum(np.prod([m[i, p[i]] for i in range(n)]) for p in itertools.permutations(range(n)))


def random_circuit(rng, n=4, depth=6, lossy=True):
    c = Circuit(n)
    for _ in range(depth):
        a = int(rng.integers(0, n - 1))
        c.beam_splitter(
            a,
            a + 1,
            float(rng.uniform(0, np.pi)),
            phi_tl=float(rng.uniform(0, 2 * np.pi)),
            phi_br=float(rng.uniform(0, 2 * np.pi)),
        )
        c.phase_shifter(int(rng.integers(0, n)), float(rng.uniform(0, 2 * np.pi)))
        if lossy:
            c.loss(int(rng.integers(0, n)), float(rng.uniform(0, 0.5)))
    return c


def dilation_reference(t, photons):
    n = t.shape[0]

    def herm_sqrt(h):
        w, v = np.linalg.eigh(h)
        return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T

    d1 = herm_sqrt(np.eye(n) - t @ t.conj().T)
    d2 = herm_sqrt(np.eye(n) - t.conj().T @ t)
    u = np.block([[t, d1], [d2, -t.conj().T]])
    assert np.allclose(u.conj().T @ u, np.eye(2 * n), atol=1e-6)
    cols = [m for m, _ in photons]
    k = len(cols)
    out = {}
    for pattern in itertools.combinations_with_replacement(range(2 * n), k):
        a = u[np.ix_(pattern, cols)]
        norm = np.prod([math.factorial(c) for c in np.bincount(pattern, minlength=2 * n)])
        norm *= np.prod([math.factorial(c) for c in np.bincount(cols, minlength=n)])
        key = tuple(np.bincount([p for p in pattern if p < n], minlength=n).tolist())
        out[key] = out.get(key, 0.0) + abs(brute_perm(a)) ** 2 / norm
    return out


def test_permanent_matches_bruteforce():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5):
        a = rng.normal(size=(7, n, n)) + 1j * rng.normal(size=(7, n, n))
        got = permanents(a)
        ref = [brute_perm(m) for m in a]
        assert np.allclose(got, ref)


def test_beam_splitter_convention_and_phases():
    m = bs_matrix(math.pi / 2)
    assert np.allclose(m, np.array([[1, 1j], [1j, 1]]) / math.sqrt(2))
    full = bs_matrix(0.7, 0.1, 0.2, 0.3, 0.4)
    assert np.allclose(full.conj().T @ full, np.eye(2))
    assert full[0, 0] == pytest.approx(np.exp(1j * 0.4) * math.cos(0.35))
    assert full[1, 0] == pytest.approx(1j * np.exp(1j * 0.5) * math.sin(0.35))
    assert np.allclose(bs_matrix(math.pi / 2, convention="H"), np.array([[1, 1], [1, -1]]) / 2**0.5)
    assert np.allclose(
        bs_matrix(math.pi / 2, convention="Ry"), np.array([[1, -1], [1, 1]]) / 2**0.5
    )


def test_transfer_matrix_unitary_and_contractive():
    rng = np.random.default_rng(1)
    lossless = random_circuit(rng, lossy=False)
    t = lossless.matrix()
    assert np.allclose(t.conj().T @ t, np.eye(4), atol=1e-12)
    assert lossless.is_lossless()
    lossy = random_circuit(rng)
    s = np.linalg.svd(lossy.matrix(), compute_uv=False)
    assert s.max() <= 1.0 + 1e-12 and s.min() < 1.0
    with pytest.raises(FockError):
        Circuit(2).block((0, 1), 2.0 * np.eye(2))


def test_hom_dip_and_tags():
    c = Circuit(2).beam_splitter(0, 1)
    same = output_distribution(c, [(0, "a"), (1, "a")]).probs
    assert same.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-15)
    assert same[(2, 0)] == pytest.approx(0.5) and same[(0, 2)] == pytest.approx(0.5)
    diff = output_distribution(c, [(0, "a"), (1, "b")]).probs
    assert diff[(1, 1)] == pytest.approx(0.5)
    assert diff[(2, 0)] == pytest.approx(0.25) and diff[(0, 2)] == pytest.approx(0.25)


def test_partial_indistinguishability_and_brightness():
    c = Circuit(2).beam_splitter(0, 1)
    d = mixed_distribution(c, [(0, 0), (1, 0)], indistinguishability=0.8)
    assert d.probs[(1, 1)] == pytest.approx(0.5 * (1 - 0.8))
    assert d.mass == pytest.approx(1.0)
    b = mixed_distribution(c, [(0, 0), (1, 0)], brightness=0.7)
    assert b.probs[(0, 0)] == pytest.approx(0.09)
    assert b.by_photon_number()[1] == pytest.approx(2 * 0.7 * 0.3)
    assert b.mass == pytest.approx(1.0)


def test_native_matches_unitary_dilation_bruteforce():
    rng = np.random.default_rng(2)
    for _ in range(3):
        c = random_circuit(rng, n=3, depth=4)
        photons = [(0, 0), (1, 0), (1, 0)]
        ref = dilation_reference(c.matrix(), photons)
        got = output_distribution(c, photons).probs
        assert sum(got.values()) == pytest.approx(1.0)
        for k in set(ref) | set(got):
            assert got.get(k, 0.0) == pytest.approx(ref.get(k, 0.0), abs=1e-7)


def test_loss_before_and_after_mixing_differ_exactly():
    before = Circuit(2).loss(0, 0.5).beam_splitter(0, 1)
    after = Circuit(2).beam_splitter(0, 1).loss(0, 0.5)
    pb = output_distribution(before, [(0, 0), (1, 0)]).probs
    pa = output_distribution(after, [(0, 0), (1, 0)]).probs
    assert pb[(2, 0)] == pytest.approx(0.25) and pb[(0, 2)] == pytest.approx(0.25)
    assert pb[(1, 0)] == pytest.approx(0.25) and pb[(0, 1)] == pytest.approx(0.25)
    assert pa[(0, 2)] == pytest.approx(0.5) and pa[(2, 0)] == pytest.approx(0.125)
    assert pa[(1, 0)] == pytest.approx(0.25) and pa[(0, 0)] == pytest.approx(0.125)
    assert sum(pb.values()) == pytest.approx(1.0) and sum(pa.values()) == pytest.approx(1.0)


def test_photon_number_conserved_when_lossless():
    rng = np.random.default_rng(3)
    c = random_circuit(rng, n=5, depth=8, lossy=False)
    d = output_distribution(c, [(0, 0), (1, 0), (2, 1), (3, 1)])
    assert set(d.by_photon_number()) == {4}
    assert d.mass == pytest.approx(1.0)


def test_simplify_losses_is_exact():
    rng = np.random.default_rng(4)
    c = random_circuit(rng, n=4, depth=6)
    c.permutation([1, 2, 3], [2, 0, 1])
    c.loss(1, 0.2)
    c.loss(1, 0.3)
    s = simplify_losses(c)
    assert np.allclose(c.matrix(), s.matrix())
    assert s.loss_channels() <= c.loss_channels()
    assert db_to_loss(3.0103) == pytest.approx(0.5, abs=1e-4)


def test_detector_response_matrices():
    for det in (IdealPNR(efficiency=0.7, dark=0.01), Threshold(0.6, 0.02), PseudoPNR(wires=5)):
        m = det.response_matrix(6)
        assert np.allclose(m.sum(axis=1), 1.0)
        assert (m >= -1e-15).all()
    thr = Threshold().response_matrix(3)
    assert thr[:, 1].tolist() == [0.0, 1.0, 1.0, 1.0]
    p = PseudoPNR(wires=4).response_matrix(2)
    assert p[2, 2] == pytest.approx(0.75) and p[2, 1] == pytest.approx(0.25)
    assert np.allclose(IdealPNR(efficiency=0.5).response_matrix(2)[2], [0.25, 0.5, 0.25])


@pytest.mark.parametrize("wires", [2, 4, 7])
def test_ppnr_equal_multiplexing_rule(wires):
    m = PseudoPNR(wires=wires).response_matrix(5)
    for n in range(1, 6):
        for k in range(1, min(n, wires) + 1):
            expected = comb(wires, k) * math.factorial(k) * stirling2(n, k) / wires**n
            assert m[n, k] == pytest.approx(expected, abs=1e-12)
    assert m[2, 2] == pytest.approx((wires - 1) / wires)


def test_ppnr_wire_efficiency_and_layered():
    m = PseudoPNR(wires=2, wire_efficiencies=(1.0, 0.5)).response_matrix(1)
    assert m[1, 1] == pytest.approx(0.75)
    lay = PseudoPNR.layered(2, 0.5).response_matrix(2)
    assert lay[2, 2] == pytest.approx(0.75)
    uneven = PseudoPNR.layered(1, 0.3).response_matrix(2)
    assert uneven[2, 2] == pytest.approx(2 * 0.3 * 0.7)


def test_detector_layout_applies_response_and_mass():
    c = Circuit(2).beam_splitter(0, 1)
    d = output_distribution(c, [(0, 0), (1, 0)])
    thr = DetectorLayout.from_modes(2, {0: Threshold(), 1: Threshold()}).apply(d)
    assert thr.probs == {(1, 0): pytest.approx(0.5), (0, 1): pytest.approx(0.5)}
    grouped = DetectorLayout.build(2, [((0, 1), IdealPNR())]).apply(d)
    assert grouped.probs == {(2,): pytest.approx(1.0)}
    eff = DetectorLayout.from_modes(2, {0: IdealPNR(efficiency=0.5)}).apply(d)
    assert eff.mass == pytest.approx(1.0)


def test_efficiency_cannot_be_applied_twice():
    c = Circuit(2).beam_splitter(0, 1).loss(0, 0.2, kind="detector")
    layout = DetectorLayout.from_modes(2, {0: IdealPNR(efficiency=0.8)})
    with pytest.raises(FockError, match="declare it once"):
        layout.check_efficiency(c)
    DetectorLayout.from_modes(2, {0: IdealPNR(efficiency=1.0)}).check_efficiency(c)


def test_conditioning_reports_success_and_normalized_distribution():
    c = Circuit(2).beam_splitter(0, 1)
    d = output_distribution(c, [(0, "a"), (1, "b")])
    cond = condition(d, {0: 2})
    assert cond.success_probability == pytest.approx(0.25)
    assert cond.probs == {(2, 0): pytest.approx(1.0)}
    assert cond.joint[(2, 0)] == pytest.approx(0.25)
    ge = condition(d, {0: (">=", 1), 1: (">=", 1)})
    assert ge.success_probability == pytest.approx(0.5)
    none = condition(d, lambda s: s[0] == 3)
    assert none.success_probability == 0.0 and none.probs == {}
    assert marginal(d, [0]) == {
        (0,): pytest.approx(0.25),
        (1,): pytest.approx(0.5),
        (2,): pytest.approx(0.25),
    }


def test_sampling_seeded_and_consistent():
    c = Circuit(2).beam_splitter(0, 1)
    d = output_distribution(c, [(0, "a"), (1, "b")])
    a = sample(d, 500, seed=7)
    b = sample(d, 500, seed=7)
    other = sample(d, 500, seed=8)
    assert a.sequence == b.sequence and a.counts == b.counts
    assert a.sequence != other.sequence
    assert sum(a.counts.values()) == 500 and len(a.sequence) == 500
    assert a.frequencies()[(1, 1)] == pytest.approx(0.5, abs=0.1)


def test_resource_limit_and_cancellation():
    rng = np.random.default_rng(5)
    c = random_circuit(rng, n=6, depth=4)
    photons = [(i % 6, 0) for i in range(5)]
    with pytest.raises(FockResourceError, match="exceeds the limit"):
        output_distribution(c, photons, limits=Limits(max_work=1e3))
    with pytest.raises(FockResourceError, match="photons exceed"):
        output_distribution(c, photons, limits=Limits(max_photons=3))
    with pytest.raises(Cancelled):
        output_distribution(c, photons, should_cancel=lambda: True)


def sample_experiment():
    c = Circuit(3).beam_splitter(0, 1).phase_shifter(1, 0.4).beam_splitter(1, 2).loss(2, 0.1)
    return Experiment(
        c,
        [1, 1, 0],
        [["a"], ["b"], []],
        detectors=[DetectorSpec((0,), Threshold(0.9)), DetectorSpec((1,), PseudoPNR(wires=3))],
        shots=200,
        seed=11,
        conditions={"herald": {0: 1}},
        name="demo",
    )


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_experiment_roundtrip_and_run(tmp_path, suffix):
    exp = sample_experiment()
    path = tmp_path / f"exp{suffix}"
    save_experiment(exp, path)
    back = load_experiment(path)
    assert np.allclose(back.circuit.matrix(), exp.circuit.matrix())
    assert back.occupations == [1, 1, 0] and back.tags[0] == ["a"]
    r1 = run_experiment(path)
    r2 = run_experiment(exp)
    assert r1.probabilities.keys() == r2.probabilities.keys()
    assert r1.samples.counts == r2.samples.counts
    assert r1.mass == pytest.approx(1.0)
    assert r1.conditionals["herald"].success_probability > 0
    assert r1.seed == 11 and r1.backend == "native" and "numpy" in r1.versions
    assert r1.resources["permanents"] > 0
    json.dumps(r1.to_dict())


def test_cli_writes_json_and_csv(tmp_path, capsys):
    path = tmp_path / "exp.yaml"
    save_experiment(sample_experiment(), path)
    out_json, out_csv = tmp_path / "r.json", tmp_path / "r.csv"
    code = fock_main([str(path), "--json", str(out_json), "--csv", str(out_csv), "--shots", "50"])
    assert code == 0
    assert "total probability" in capsys.readouterr().out
    assert json.loads(out_json.read_text())["mass"] == pytest.approx(1.0)
    assert out_csv.read_text().startswith("state,probability,counts")
    assert fock_main([str(tmp_path / "missing.yaml")]) == 1


needs_gui_dir = pytest.mark.skipif(not GUI_DIR.is_dir(), reason="Perceval GUI folder not found")


@needs_gui_dir
@pytest.mark.parametrize("name", sorted(p.name for p in GUI_DIR.glob("*.json")) or ["none"])
def test_importer_runs_on_all_gui_circuits(name):
    imp = import_perceval_gui(GUI_DIR / name)
    c = imp.circuit
    s = np.linalg.svd(c.matrix(), compute_uv=False)
    assert s.max() <= 1.0 + 1e-9
    assert not imp.unsupported
    occ = [1, 1] + [0] * (c.n_modes - 2)
    imp.experiment.occupations = occ
    r = run_experiment(imp.experiment)
    assert r.mass == pytest.approx(1.0)


@needs_gui_dir
def test_importer_4p5m_has_50_states_summing_to_one():
    imp = import_perceval_gui(GUI_DIR / "4p5m.json", [1, 1, 1, 1, 0])
    r = run_experiment(imp.experiment)
    assert len(r.raw.probs) == 50
    assert r.raw.mass == pytest.approx(1.0, abs=1e-12)
    assert len(imp.experiment.detectors) == 1


def test_importer_mirrors_gui_loss_and_flags_unsupported():
    raw = {
        "version": 2,
        "mode_losses": ["0.1", "0"],
        "db_per_cm": "0.5",
        "noise": {"brightness": "1", "g2": "0.1", "indistinguishability": "0.9"},
        "components": [
            {
                "name": "Beam Splitter Rx",
                "boxes": {
                    "mode1": "0",
                    "mode2": "1",
                    "length1": "2",
                    "length2": "0",
                    "active_db": "0.3",
                },
            },
            {"name": "PNR", "boxes": {"mode": "0", "length": "1", "efficiency": "0.5"}},
        ],
    }
    imp = import_perceval_gui(raw, strict=False)
    assert imp.warnings and "g2" in imp.warnings[0]
    t = imp.circuit.matrix()
    pre = np.diag([math.sqrt(0.9) * 10 ** (-1.0 / 20), 1.0])
    post = np.diag([10 ** (-0.8 / 20), 10 ** (-0.3 / 20)])
    assert np.allclose(t, post @ bs_matrix(math.pi / 2) @ pre)
    assert imp.experiment.detectors[0].model.efficiency == 0.5
    assert imp.experiment.indistinguishability == 0.9
    with pytest.raises(FockImportError, match="unsupported"):
        import_perceval_gui(raw)
    raw["components"].append({"name": "Wormhole", "boxes": {}})
    with pytest.raises(FockImportError, match="Wormhole"):
        import_perceval_gui(raw)


def test_loss_validation():
    with pytest.raises(FockError):
        Loss(0, 1.5)


def test_element_labels_roundtrip_and_do_not_change_the_matrix():
    from sequoia.fock import Circuit, PhaseShifter

    c = Circuit(2).beam_splitter(0, 1)
    c.add(PhaseShifter(0, 0.3, "link_1"))
    c.loss(1, 0.1, label="part_x")
    c.block([0, 1], np.eye(2) * 0.9, label="blk")
    d = c.to_dict()
    assert [e.get("label") for e in d["elements"]] == [None, "link_1", "part_x", "blk"]
    back = Circuit.from_dict(d)
    assert [e.label for e in back.elements] == [None, "link_1", "part_x", "blk"]
    assert np.allclose(back.matrix(), c.matrix())
