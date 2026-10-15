import math
from collections import Counter

import numpy as np
import pytest

from sequoia.fock import (
    Circuit,
    Experiment,
    FockError,
    InputMixture,
    PureState,
    Term,
    bell_pair,
    expected_counts,
    load_experiment,
    outcome_table,
    output_distribution,
    output_distribution_mixture,
    pair_number_mixture,
    product_pair,
    run_experiment,
    save_experiment,
)
from sequoia.fock.analysis import condition
from sequoia.fock.engine import pure_state_distribution
from sequoia.fock.inputs import pair_number_pmf


def brute_force(t, state):
    """Independent expansion of creation-operator polynomials through the dilated unitary."""
    n = t.shape[0]
    defect = np.eye(n) - t.conj().T @ t
    w, v = np.linalg.eigh(defect)
    env = np.sqrt(np.clip(w, 0, None))[:, None] * v.conj().T
    u = np.vstack([t, env])
    poly: dict[tuple, complex] = {}
    for term in state.terms:
        occ = Counter(term.photons)
        coeff = term.amplitude / math.sqrt(math.prod(math.factorial(c) for c in occ.values()))
        cur = {(): coeff}
        for mode, tag in term.photons:
            nxt: dict[tuple, complex] = {}
            for key, c in cur.items():
                for k in range(u.shape[0]):
                    if abs(u[k, mode]) > 1e-15:
                        nk = tuple(sorted((*key, (k, tag))))
                        nxt[nk] = nxt.get(nk, 0j) + c * u[k, mode]
            cur = nxt
        for key, c in cur.items():
            poly[key] = poly.get(key, 0j) + c
    norm = sum(abs(t.amplitude) ** 2 for t in state.terms)
    out: dict[tuple, float] = {}
    for key, c in poly.items():
        amp2 = abs(c) ** 2 * math.prod(math.factorial(m) for m in Counter(key).values())
        sysocc = [0] * n
        for k, _ in key:
            if k < n:
                sysocc[k] += 1
        out[tuple(sysocc)] = out.get(tuple(sysocc), 0.0) + amp2 / norm
    return out


def random_transfer(rng, n, loss=0.0):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, _ = np.linalg.qr(z)
    return q * math.sqrt(1.0 - loss)


def close(a, b, tol=1e-10):
    keys = set(a) | set(b)
    return max((abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), default=0.0) < tol


@pytest.mark.parametrize("loss", [0.0, 0.3])
def test_superposition_matches_brute_force(loss):
    rng = np.random.default_rng(3)
    t = random_transfer(rng, 4, loss)
    state = PureState(
        (
            Term(0.6, ((0, 0), (1, 0), (2, 1))),
            Term(0.5j, ((0, 0), (0, 0), (3, 1))),
            Term(-0.4 + 0.2j, ((2, 0), (3, 0), (2, 1))),
        )
    )
    d = pure_state_distribution(t, state)
    assert close(d.probs, brute_force(t, state))
    assert d.mass == pytest.approx(1.0, abs=1e-10)


def test_two_class_with_bunching_input_brute_force():
    rng = np.random.default_rng(8)
    t = random_transfer(rng, 3, 0.15)
    state = PureState(
        (
            Term(1.0, ((0, 0), (0, 0), (1, 1), (1, 1))),
            Term(0.7, ((1, 0), (2, 0), (0, 1), (2, 1))),
        )
    )
    assert close(pure_state_distribution(t, state).probs, brute_force(t, state))


def test_single_term_bit_for_bit_with_legacy():
    rng = np.random.default_rng(1)
    t = random_transfer(rng, 4, 0.2)
    photons = [(0, 0), (1, 0), (2, 1)]
    legacy = output_distribution(t, photons)
    via = output_distribution_mixture(t, PureState.fock(photons))
    assert via.probs == legacy.probs
    general = pure_state_distribution(t, PureState.fock(photons))
    assert close(general.probs, legacy.probs, 1e-12)


def test_mixture_weights_and_omitted():
    t = np.eye(2, dtype=complex)
    mix = InputMixture(((0.3, PureState.fock([(0, 0)])), (0.5, PureState.fock([(1, 0)]))))
    d = output_distribution_mixture(t, mix)
    assert d.probs == {(1, 0): pytest.approx(0.3), (0, 1): pytest.approx(0.5)}
    assert d.omitted_weight == pytest.approx(0.2)
    with pytest.raises(FockError):
        InputMixture(((0.7, PureState.fock([(0, 0)])), (0.7, PureState.fock([(1, 0)]))))


def test_rejects_different_photon_numbers():
    with pytest.raises(FockError, match="same number of photons"):
        PureState((Term(1, ((0, 0),)), Term(1, ((0, 0), (1, 0)))))
    with pytest.raises(FockError, match="same number of photons"):
        PureState((Term(1, ((0, 0), (1, 0))), Term(1, ((0, 0), (1, 1)))))


def analyser(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s], [-s, c]])


def bell_probability(a, b, phi, v):
    c = lambda x: math.cos(x)  # noqa: E731
    s = lambda x: math.sin(x)  # noqa: E731
    pure = 0.5 * (
        c(a) ** 2 * c(b) ** 2
        + s(a) ** 2 * s(b) ** 2
        + 2 * c(a) * c(b) * s(a) * s(b) * math.cos(phi)
    )
    return v * pure + (1 - v) / 4


@pytest.mark.parametrize(
    "a,b,phi,v", [(0.0, 0.0, 0.0, 1.0), (0.4, 1.1, 0.7, 1.0), (0.4, 1.1, 0.7, 0.8)]
)
def test_bell_correlations(a, b, phi, v):
    circ = Circuit(4)
    circ.block((0, 1), analyser(a))
    circ.block((2, 3), analyser(b))
    mix = bell_pair(0, 1, 2, 3, phase=phi, visibility=v)
    d = output_distribution_mixture(circ, mix)
    assert d.probs.get((1, 0, 1, 0), 0.0) == pytest.approx(
        bell_probability(a, b, phi, v), abs=1e-12
    )
    assert d.mass == pytest.approx(1.0)
    assert d.probs.get((0, 1, 0, 1), 0.0) == pytest.approx(
        bell_probability(a + math.pi / 2, b + math.pi / 2, phi, v)
    )


def test_bell_fringe_visibility_scaling():
    def p_hh(a, b, v):
        circ = Circuit(4)
        circ.block((0, 1), analyser(a))
        circ.block((2, 3), analyser(b))
        d = output_distribution_mixture(circ, bell_pair(0, 1, 2, 3, visibility=v))
        return d.probs.get((1, 0, 1, 0), 0.0)

    for v in (1.0, 0.6):
        hi = p_hh(0.0, 0.0, v)
        lo = p_hh(math.pi / 2, 0.0, v)
        assert (hi - lo) / (hi + lo) == pytest.approx(v)


def test_psi_bell_through_beam_splitter_brute_force():
    circ = Circuit(4).beam_splitter(0, 2).beam_splitter(1, 3, 1.1)
    mix = bell_pair(0, 1, 2, 3, phase=0.9, kind="psi")
    d = output_distribution_mixture(circ, mix)
    assert close(d.probs, brute_force(circ.matrix(), mix.branches[0][1]))


def test_hom_one_class_vs_two_classes():
    circ = Circuit(2).beam_splitter(0, 1)
    same = output_distribution_mixture(circ, product_pair(0, 1, tag=0))
    assert same.probs.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-15)
    assert same.probs[(2, 0)] == pytest.approx(0.5)
    diff = output_distribution_mixture(
        circ, InputMixture.from_state(PureState.fock([(0, 0), (1, 1)]))
    )
    assert diff.probs[(1, 1)] == pytest.approx(0.5)


def single_photon(tag):
    return PureState.fock([(0, tag)])


@pytest.mark.parametrize(
    "k,mu,stat",
    [(1, 0.4, "thermal"), (2, 0.4, "thermal"), (1, 0.4, "poisson"), (3.0, 0.9, "thermal")],
)
def test_pair_number_weights_and_mean(k, mu, stat):
    n_max = 14
    mix = pair_number_mixture(single_photon, mu, k, n_max=n_max, statistics=stat)
    by_n: dict[int, float] = {}
    for w, s in mix.branches:
        n = sum(s.tag_counts.values())
        by_n[n] = by_n.get(n, 0.0) + w
    for n in range(n_max + 1):
        if stat == "poisson":
            ref = math.exp(-mu) * mu**n / math.factorial(n)
        else:
            ref = math.comb(n + int(k) - 1, n) * (mu / k) ** n * (1 + mu / k) ** (-(k + n))
        assert by_n.get(n, 0.0) == pytest.approx(ref, rel=1e-10)
        assert pair_number_pmf(n, mu, k, stat) == pytest.approx(ref, rel=1e-10)
    mean = sum(n * w for n, w in by_n.items())
    assert mean == pytest.approx(mu, abs=2 * mix.omitted_weight * n_max + 1e-9)
    assert mix.omitted_weight < 1e-6 or mu > 0.5


@pytest.mark.parametrize(
    "k,expected,stat,n_max",
    [(1, 2.0, "thermal", 9), (2, 1.5, "thermal", 8), (1, 1.0, "poisson", 7)],
)
def test_unheralded_g2_from_hbt(k, expected, stat, n_max):
    mu = 0.1
    mix = pair_number_mixture(single_photon, mu, k, n_max=n_max, statistics=stat)
    circ = Circuit(2).beam_splitter(0, 1)
    d = output_distribution_mixture(circ, mix)
    n1 = sum(p * s[0] for s, p in d.probs.items())
    n2 = sum(p * s[1] for s, p in d.probs.items())
    n12 = sum(p * s[0] * s[1] for s, p in d.probs.items())
    assert mix.omitted_weight < 1e-6
    assert n12 / (n1 * n2) == pytest.approx(expected, abs=2e-5)
    assert n1 + n2 == pytest.approx(mu, abs=1e-6)


def test_schmidt_modes_exact_and_noninteger_note():
    mix = pair_number_mixture(single_photon, 0.3, 2.0, n_max=2)
    two = [s for w, s in mix.branches if sum(s.tag_counts.values()) == 2]
    shapes = sorted(sorted(s.tag_counts.values()) for s in two)
    assert shapes == [[1, 1], [2]]
    assert not mix.notes
    odd = pair_number_mixture(single_photon, 0.3, 1.6, n_max=2)
    assert any("rounded" in n for n in odd.notes)
    assert sum(w for w, _ in odd.branches) + odd.omitted_weight == pytest.approx(1.0)


def test_bell_pairs_multi_pair_vs_brute_force_and_conservation():
    circ = Circuit(4).beam_splitter(0, 2).beam_splitter(1, 3)
    circ.loss(0, 0.2)
    mix = pair_number_mixture(
        lambda tag: bell_pair(0, 1, 2, 3, phase=0.3, tag=tag), 0.2, 2.0, n_max=2
    )
    d = output_distribution_mixture(circ, mix)
    ref: dict = {}
    for w, s in mix.branches:
        for k, v in brute_force(circ.matrix(), s).items():
            ref[k] = ref.get(k, 0.0) + w * v
    assert close(d.probs, ref)
    assert d.mass + d.omitted_weight == pytest.approx(1.0, abs=1e-10)


def test_lossy_mass_and_lost_states():
    circ = Circuit(2).beam_splitter(0, 1).loss(0, 0.3).loss(1, 0.3)
    state = PureState((Term(1, ((0, 0), (1, 0))), Term(1j, ((0, 0), (0, 0)))))
    d = output_distribution_mixture(circ, state)
    assert d.mass == pytest.approx(1.0, abs=1e-12)
    assert close(d.probs, brute_force(circ.matrix(), state))
    by_n = d.by_photon_number()
    assert sum(by_n.values()) == pytest.approx(1.0)
    assert d.lost_mass(2) == pytest.approx(by_n.get(0, 0.0) + by_n.get(1, 0.0))
    assert d.lost_mass(2) > 0.3


def test_expected_counts_and_outcome_table():
    circ = Circuit(2).beam_splitter(0, 1)
    d = output_distribution_mixture(circ, product_pair(0, 1))
    ec = expected_counts(d, rate_hz=1e6, duration_s=10.0)
    assert ec.per_second[(2, 0)] == pytest.approx(0.5e6)
    assert ec.total[(2, 0)] == pytest.approx(0.5e7)
    assert ec.sigma[(2, 0)] == pytest.approx(math.sqrt(0.5e7))
    ec2 = expected_counts(d, window_s=1e-9)
    assert ec2.per_second[(0, 2)] == pytest.approx(0.5e9)
    cond = expected_counts(condition(d, {0: 2}), rate_hz=100.0)
    assert cond.per_second == {(2, 0): pytest.approx(50.0)}
    rows = outcome_table(d, top=1, n_windows=1000)
    assert len(rows) == 1 and rows[0]["counts"] == pytest.approx(500.0)
    assert rows[0]["sigma"] == pytest.approx(math.sqrt(1000 * 0.5 * 0.5))
    with pytest.raises(FockError):
        expected_counts(d, duration_s=1.0)


def test_serialisation_and_experiment(tmp_path):
    circ = Circuit(4).beam_splitter(0, 2)
    mix = bell_pair(0, 1, 2, 3, phase=0.5, visibility=0.9)
    assert InputMixture.from_dict(mix.to_dict()) == mix
    exp = Experiment(circ, [0, 0, 0, 0], input_mixture=mix, rate_hz=1e6, duration_s=2.0)
    for name in ("e.yaml", "e.json"):
        save_experiment(exp, tmp_path / name)
        back = load_experiment(tmp_path / name)
        assert back.input_mixture == mix and back.rate_hz == 1e6 and back.duration_s == 2.0
    res = run_experiment(exp)
    assert res.counts is not None
    assert res.mass == pytest.approx(1.0)
    row = res.table()[0]
    assert row["expected_counts"] == pytest.approx(row["probability"] * 2e6)
    ref = output_distribution_mixture(circ, mix)
    assert close(res.probabilities, ref.probs)


def test_old_documents_still_load_and_run():
    raw = {
        "format": "sequoia-fock",
        "version": 1,
        "circuit": Circuit(2).beam_splitter(0, 1).to_dict(),
        "input": {"occupations": [1, 1], "tags": None},
        "detectors": [],
        "shots": 0,
    }
    res = run_experiment(raw)
    assert res.counts is None
    assert res.probabilities[(2, 0)] == pytest.approx(0.5)


def test_mode_out_of_range_rejected():
    with pytest.raises(FockError):
        output_distribution_mixture(Circuit(2).beam_splitter(0, 1), product_pair(0, 5))


def test_product_mixture_multiplies_weights_and_bunches_shared_modes():
    from sequoia.fock import InputMixture, PureState, output_distribution_mixture, product_mixture

    a = InputMixture(((0.5, PureState.fock([(0, 0)])), (0.25, PureState.fock([]))))
    b = InputMixture.from_state(PureState.fock([(1, 0)]))
    prod = product_mixture([a, b])
    assert prod.total_weight == pytest.approx(0.75)
    assert sorted(round(w, 12) for w, _ in prod.branches) == [0.25, 0.5]
    t = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    dist = output_distribution_mixture(t, product_mixture([a, b]))
    assert dist.probs.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-12)
    assert dist.probs[(2, 0)] == pytest.approx(0.25)
    assert dist.omitted_weight == pytest.approx(0.25)
