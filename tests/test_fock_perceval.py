import math
import sys
from pathlib import Path

import numpy as np
import pytest

pcvl = pytest.importorskip("perceval")

from sequoia.fock import (  # noqa: E402
    Circuit,
    DetectorLayout,
    IdealPNR,
    PseudoPNR,
    Threshold,
    condition,
    output_distribution,
    run_experiment,
)
from sequoia.fock.circuit import bs_matrix  # noqa: E402
from sequoia.fock.import_perceval_gui import import_perceval_gui  # noqa: E402
from sequoia.fock.perceval_bridge import (  # noqa: E402
    compare_with_perceval,
    perceval_distribution,
    perceval_mixed_distribution,
    to_perceval_processor,
)

GUI_DIR = Path(__file__).resolve().parents[2] / "percevalGUI-v0 0 11"


def random_lossy(rng, n=4):
    c = Circuit(n)
    for _ in range(8):
        a = int(rng.integers(0, n - 1))
        c.beam_splitter(
            a,
            a + 1,
            float(rng.uniform(0, np.pi)),
            phi_tl=float(rng.uniform(0, 2 * np.pi)),
            phi_bl=float(rng.uniform(0, 2 * np.pi)),
            phi_tr=float(rng.uniform(0, 2 * np.pi)),
            phi_br=float(rng.uniform(0, 2 * np.pi)),
        )
        c.phase_shifter(int(rng.integers(0, n)), float(rng.uniform(0, 2 * np.pi)))
        c.loss(int(rng.integers(0, n)), float(rng.uniform(0, 0.5)))
    c.permutation([1, 2, 3], [2, 0, 1])
    c.beam_splitter(3, 0, 1.0, convention="Ry")
    c.beam_splitter(1, 2, 0.4, convention="H")
    return c


def test_element_matrices_equal_perceval():
    for theta, phis in [(0.3, (0.1, 0.2, 0.3, 0.4)), (1.9, (0.0, 1.0, 2.0, 3.0))]:
        for conv in ("Rx", "Ry", "H"):
            maker = {"Rx": pcvl.BS.Rx, "Ry": pcvl.BS.Ry, "H": pcvl.BS.H}[conv]
            u = np.array(maker(theta, *phis).compute_unitary())
            assert np.allclose(u, bs_matrix(theta, *phis, convention=conv))
    ps = np.array(pcvl.PS(0.7).compute_unitary())
    assert ps[0, 0] == pytest.approx(np.exp(0.7j))
    perm = np.array(pcvl.PERM([2, 0, 1]).compute_unitary())
    c = Circuit(3).permutation([0, 1, 2], [2, 0, 1])
    assert np.allclose(perm, c.matrix())


def test_evidence_numbers_native_and_perceval():
    c = Circuit(2).beam_splitter(0, 1)
    for dist in (
        output_distribution(c, [(0, 0), (1, 0)]),
        perceval_distribution(c, [(0, 0), (1, 0)]),
    ):
        assert dist.probs.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-12)
        assert dist.probs[(2, 0)] == pytest.approx(0.5) and dist.probs[(0, 2)] == pytest.approx(0.5)
    for dist in (
        output_distribution(c, [(0, 0), (1, 1)]),
        perceval_distribution(c, [(0, 0), (1, 1)]),
    ):
        assert dist.probs[(1, 1)] == pytest.approx(0.5)
        assert dist.probs[(2, 0)] == pytest.approx(0.25)
    d = output_distribution(c, [(0, 0), (1, 0)])
    thr = DetectorLayout.from_modes(2, {0: Threshold()}).apply(d)
    assert thr.probs[(1, 0)] == pytest.approx(0.5) and thr.probs[(0, 2)] == pytest.approx(0.5)
    pnr = DetectorLayout.from_modes(2, {0: IdealPNR()}).apply(d)
    assert pnr.probs[(2, 0)] == pytest.approx(0.5)
    pp = DetectorLayout.from_modes(2, {0: PseudoPNR(wires=4)}).apply(d)
    assert pp.probs[(0, 2)] == pytest.approx(0.5)
    assert pp.probs[(2, 0)] == pytest.approx(0.375)
    assert pp.probs[(1, 0)] == pytest.approx(0.125)
    assert condition(d, {0: 2}).success_probability == pytest.approx(0.5)


def test_native_equals_perceval_on_random_lossy_circuits():
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(5):
        c = random_lossy(rng)
        for photons in (
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (1, 1), (1, 0)],
            [(0, 0), (1, 1), (3, 2)],
        ):
            worst = max(worst, compare_with_perceval(c, photons))
    assert worst < 1e-9


def test_native_equals_perceval_with_source_noise():
    rng = np.random.default_rng(7)
    c = random_lossy(rng)
    photons = [(0, 0), (1, 0), (2, 0)]
    from sequoia.fock import mixed_distribution

    a = mixed_distribution(c, photons, brightness=0.8, indistinguishability=0.7)
    b = perceval_mixed_distribution(c, photons, brightness=0.8, indistinguishability=0.7)
    keys = set(a.probs) | set(b.probs)
    assert max(abs(a.probs.get(k, 0) - b.probs.get(k, 0)) for k in keys) < 1e-9
    src = pcvl.Source(emission_probability=0.8, indistinguishability=0.7)
    dist = src.generate_distribution(pcvl.FockState([1, 1, 1, 0]))
    assert sum(dist.values()) == pytest.approx(1.0)


def test_layered_ppnr_matches_perceval_detector():
    from perceval.algorithm import Sampler

    for n in (1, 2, 3):
        proc = pcvl.Processor("SLOS", 1)
        proc.add(0, pcvl.BSLayeredPPNR(2, 0.3))
        proc.with_input(pcvl.BasicState([n]))
        res = Sampler(proc).probs()["results"]
        row = PseudoPNR.layered(2, 0.3).response_matrix(n)[n]
        for state, p in res.items():
            assert p == pytest.approx(row[int(state[0])], abs=1e-12)


def test_perceval_backend_through_run_experiment():
    rng = np.random.default_rng(3)
    from sequoia.fock import Experiment

    exp = Experiment(random_lossy(rng), [1, 1, 1, 0], detectors=[], shots=20, seed=1)
    native = run_experiment(exp)
    exp.backend = "perceval"
    other = run_experiment(exp)
    assert other.backend.startswith("perceval")
    keys = set(native.probabilities) | set(other.probabilities)
    diff = max(abs(native.probabilities.get(k, 0) - other.probabilities.get(k, 0)) for k in keys)
    assert diff < 1e-9
    assert other.versions["perceval"] == pcvl.__version__


def test_processor_has_expected_structure():
    c = Circuit(3).beam_splitter(0, 1).loss(1, 0.2).permutation([0, 1, 2], [1, 2, 0])
    proc = to_perceval_processor(c)
    assert proc.m == 3


def test_missing_perceval_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "perceval", None)
    with pytest.raises(ImportError, match=r"pip install 'sequoia\[perceval\]'"):
        perceval_distribution(Circuit(2).beam_splitter(0, 1), [(0, 0)])


needs_gui_dir = pytest.mark.skipif(not GUI_DIR.is_dir(), reason="Perceval GUI folder not found")


@needs_gui_dir
def test_import_4p5m_matches_perceval_gui_pipeline():
    imp = import_perceval_gui(GUI_DIR / "4p5m.json", [1, 1, 1, 1, 0])
    photons = [(0, 0), (1, 0), (2, 0), (3, 0)]
    native = run_experiment(imp.experiment)
    ref = perceval_distribution(imp.circuit, photons)
    assert len(ref.probs) == 50 and len(native.raw.probs) == 50
    assert sum(ref.probs.values()) == pytest.approx(1.0, abs=1e-12)
    assert max(abs(native.raw.probs[k] - ref.probs.get(k, 0.0)) for k in native.raw.probs) < 1e-9


@needs_gui_dir
def test_import_matches_gui_builder_on_lossy_circuits():
    sys.path.insert(0, str(GUI_DIR))
    try:
        from circuit_builder import build_experiment
        from circuit_optimizer import merge_loss_channels
        from components import FIELDS, parse_box
        from sampling import make_input_state, run
    finally:
        sys.path.remove(str(GUI_DIR))
    import json

    for name, inp in [
        ("carolan_general15mzi.json", [1, 0, 1, 0, 1, 0]),
        ("4p6mcarolan.json", [1, 1, 1, 1, 0, 0]),
    ]:
        raw = json.loads((GUI_DIR / name).read_text())
        m = len(raw["mode_losses"])
        parts = [
            (
                c["name"],
                {k: parse_box(c["boxes"][k], kind, m) for k, _, kind in FIELDS[c["name"]]},
            )
            for c in raw["components"]
        ]
        ex = build_experiment(
            [float(x) for x in raw["mode_losses"]],
            parts,
            float(raw["db_per_cm"]),
            {k: float(v) for k, v in raw["noise"].items()},
        )
        res, _ = run(
            merge_loss_channels(ex),
            "SLOS",
            make_input_state("FockState", [str(x) for x in inp]),
            "probs",
            None,
        )
        ref = {tuple(int(x) for x in k[1:-1].split(",")): v for k, v in res.items()}
        imp = import_perceval_gui(GUI_DIR / name, inp)
        got = run_experiment(imp.experiment).probabilities
        assert max(abs(got.get(k, 0.0) - ref.get(k, 0.0)) for k in set(got) | set(ref)) < 1e-9
        assert math.isclose(sum(got.values()), 1.0, abs_tol=1e-9)


def test_perceval_superposition_matches_native():
    from sequoia.fock import PureState, Term, bell_pair, output_distribution_mixture
    from sequoia.fock.perceval_bridge import perceval_mixture_distribution

    c = Circuit(4).beam_splitter(0, 2).beam_splitter(1, 3, 0.9).loss(1, 0.2)
    state = PureState(
        (Term(0.6, ((0, 0), (1, 0))), Term(0.5j, ((2, 0), (3, 0))), Term(0.3, ((0, 0), (0, 0))))
    )
    for mix in (state, bell_pair(0, 1, 2, 3, phase=0.4, visibility=0.8)):
        a = output_distribution_mixture(c, mix)
        b = perceval_mixture_distribution(c, mix)
        keys = set(a.probs) | set(b.probs)
        assert max(abs(a.probs.get(k, 0.0) - b.probs.get(k, 0.0)) for k in keys) < 1e-9
