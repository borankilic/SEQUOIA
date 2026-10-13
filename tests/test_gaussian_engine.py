import itertools
import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.engines.cache import Cancelled
from sequoia.engines.gaussian import (
    GaussianModel,
    GaussianSource,
    _Batch,
    _detector_forms,
    _Member,
    log_generating_function,
    prepare_gaussian,
    simulate_gaussian,
)
from sequoia.engines.rates import simulate_rates
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


def example(name, scale=None, pulsed=False):
    raw = setup_to_dict(load_setup(ROOT / "setups" / f"{name}.yaml"))
    for inst in raw["instances"]:
        part = inst["part"]
        if part.startswith("sources/laser"):
            overrides = inst.setdefault("overrides", {})
            if scale is not None:
                overrides["power"] = f"{10 * scale} mW"
            if pulsed:
                overrides.update(repetition_rate="50 MHz", pulse_width="1 ps")
        if scale is not None and part.startswith("nonlinear/entangled"):
            inst.setdefault("settings", {})["pump_power"] = f"{1 * scale} mW"
    return setup_from_dict(raw)


def batch_from(j, h_s, h_i, equal=()):
    """One block: J (2a, 2b); h_s (D, a, 2, 2); h_i (D, b, 2, 2)."""
    u, s, vh = np.linalg.svd(j[None], full_matrices=False)
    a, b = j.shape[0] // 2, j.shape[1] // 2
    pairs = (
        np.zeros(len(equal), dtype=int),
        np.array([p[0] for p in equal], dtype=int),
        np.array([p[1] for p in equal], dtype=int),
    )
    batch = _Batch(u, vh.transpose(0, 2, 1), s, np.arange(a)[None], np.arange(b)[None], pairs)
    return batch, h_s, h_i


def no_click_probability(batch, h_s, h_i, scale, lam):
    forms = _detector_forms([batch], h_s, h_i)[0]
    return np.exp(log_generating_function(forms, batch.sigma, scale, np.asarray(lam, float))).real


def test_two_mode_squeezer_with_loss_matches_analytic():
    j = np.zeros((2, 2), dtype=complex)
    j[0, 0] = 1.0
    eta_s, eta_i, r = 0.6, 0.35, 0.7
    h_s = np.zeros((2, 1, 2, 2), dtype=complex)
    h_i = np.zeros((2, 1, 2, 2), dtype=complex)
    h_s[0, 0] = math.sqrt(eta_s) * np.eye(2)
    h_i[1, 0] = math.sqrt(eta_i) * np.eye(2)
    batch, h_s, h_i = batch_from(j, h_s, h_i)
    x = math.tanh(r) ** 2
    both = (1 - x) / (1 - x * (1 - eta_s) * (1 - eta_i))
    only_s = (1 - x) / (1 - x * (1 - eta_s))
    only_i = (1 - x) / (1 - x * (1 - eta_i))
    assert no_click_probability(batch, h_s, h_i, r, [1, 1]) == pytest.approx(both)
    assert no_click_probability(batch, h_s, h_i, r, [1, 0]) == pytest.approx(only_s)
    assert no_click_probability(batch, h_s, h_i, r, [0, 1]) == pytest.approx(only_i)
    assert no_click_probability(batch, h_s, h_i, r, [0, 0]) == pytest.approx(1.0)


def full_space_no_click(batch, h_s, h_i, equal, scale):
    """Independent route: a-space moments, thewalrus covariance and vacuum probability."""
    from thewalrus.quantum import density_matrix_element, reduced_gaussian

    u, v, sig = batch.u[0], batch.v[0], batch.sigma[0]
    r = scale * sig
    n, m = np.sinh(r) ** 2, np.sinh(r) * np.cosh(r)
    ms, mi = u.shape[0], v.shape[0]
    n_s = (u.conj() * n) @ u.T
    n_i = (v.conj() * n) @ v.T
    m_si = (u * m) @ v.T
    big_n = np.zeros((ms + mi, ms + mi), dtype=complex)
    big_m = np.zeros_like(big_n)
    big_n[:ms, :ms], big_n[ms:, ms:] = n_s, n_i
    big_m[:ms, ms:], big_m[ms:, :ms] = m_si, m_si.T
    a, b = ms // 2, mi // 2
    results = {}
    d_count = h_s.shape[0]
    for subset in itertools.chain.from_iterable(
        itertools.combinations(range(d_count), k) for k in range(d_count + 1)
    ):
        rows = []
        for d in subset:
            freq_rows = []
            for j in range(a):
                row = np.zeros((2, ms + mi), dtype=complex)
                row[:, 2 * j : 2 * j + 2] = h_s[d, j]
                for jj, kk in equal:
                    if jj == j:
                        row[:, ms + 2 * kk : ms + 2 * kk + 2] += h_i[d, kk]
                freq_rows.append(row)
            for k in range(b):
                if any(kk == k for _, kk in equal):
                    continue
                row = np.zeros((2, ms + mi), dtype=complex)
                row[:, ms + 2 * k : ms + 2 * k + 2] = h_i[d, k]
                freq_rows.append(row)
            rows.append(np.concatenate(freq_rows, axis=0))
        if not rows:
            results[subset] = 1.0
            continue
        h = np.concatenate(rows, axis=0)
        nb = h.conj() @ big_n @ h.T
        mb = h @ big_m @ h.T
        q = h.shape[0]
        xx = mb.real + nb.real
        pp = -mb.real + nb.real
        xp = mb.imag + nb.imag
        cov_n = np.block([[xx, xp], [xp.T, pp]])
        cov = 2.0 * (cov_n + 0.5 * np.eye(2 * q))
        mu = np.zeros(2 * q)
        mu_r, cov_r = reduced_gaussian(mu, cov, list(range(q)))
        results[subset] = float(
            np.real(density_matrix_element(mu_r, cov_r, [0] * q, [0] * q, hbar=2))
        )
    return results


def test_random_multimode_state_agrees_with_thewalrus():
    pytest.importorskip("thewalrus")
    rng = np.random.default_rng(5)
    a, b = 2, 2
    j = rng.normal(size=(2 * a, 2 * b)) + 1j * rng.normal(size=(2 * a, 2 * b))
    j /= np.linalg.norm(j)
    d_count = 2
    h_s = 0.6 * (rng.normal(size=(d_count, a, 2, 2)) + 1j * rng.normal(size=(d_count, a, 2, 2)))
    h_i = 0.6 * (rng.normal(size=(d_count, b, 2, 2)) + 1j * rng.normal(size=(d_count, b, 2, 2)))
    h_s /= 2.0
    h_i /= 2.0
    equal = [(0, 1), (1, 0)]
    batch, h_s, h_i = batch_from(j, h_s, h_i, equal)
    scale = 0.9
    reference = full_space_no_click(batch, h_s, h_i, equal, scale)
    for subset, value in reference.items():
        lam = [1.0 if d in subset else 0.0 for d in range(d_count)]
        assert no_click_probability(batch, h_s, h_i, scale, lam) == pytest.approx(value, rel=1e-8)


def test_pnr_distribution_of_a_single_detector():
    j = np.zeros((2, 2), dtype=complex)
    j[0, 0] = 1.0
    h_s = np.zeros((1, 1, 2, 2), dtype=complex)
    h_i = np.zeros((1, 1, 2, 2), dtype=complex)
    h_s[0, 0] = np.eye(2)
    batch, h_s, h_i = batch_from(j, h_s, h_i)
    r = 0.5
    x = math.tanh(r) ** 2
    forms = {"s": {}}
    member = _Member(1.0, [batch])
    forms["s"][id(member)] = _detector_forms([batch], h_s, h_i)
    src = GaussianSource("s", r**2, 1.0, False, [member], 1)
    model = GaussianModel([src], forms, ["d"], np.zeros(1), np.zeros(1))
    from sequoia.engines.gaussian import _pnr

    probs, _ = _pnr(model, ["d"], ("d",), 1.0, 40)
    expected = (1 - x) * x ** np.arange(41)
    assert np.allclose(probs, expected, atol=1e-9)


@pytest.mark.parametrize(
    ("name", "scale"), [("hbt_heralded", 0.02), ("bell_test", 0.05), ("hom", 0.02)]
)
def test_low_gain_matches_rate_engine(name, scale):
    setup = example(name, scale)
    g = simulate_gaussian(setup, LIB)
    r = simulate_rates(setup, LIB)
    for ch in g.channels:
        photons = g.mean_photons[ch] / g.window
        assert photons == pytest.approx(r.channels[ch]["incident_hz"], rel=2e-3)
        assert g.rate(ch) <= photons * (1 + 1e-9)
    for (a, b), v in r.pairs.items():
        if v["coincidence_rate_hz"] < 1.0:
            continue
        accidental = g.rate(a) * g.rate(b) * g.window
        assert g.rate(a, b) - accidental == pytest.approx(v["coincidence_rate_hz"], rel=6e-3)


def test_pump_power_raises_g2_and_lowers_car():
    setup = example("hbt_heralded", 0.02, pulsed=True)
    prepared = prepare_gaussian(setup, LIB)
    scales = [1, 3, 10, 30, 100, 300]
    cars, g2s = [], []
    for s in scales:
        result = prepared.evaluate(s)
        cars.append(result.analyses["herald_a"]["car"])
        g2s.append(result.analyses["hbt"]["g2_heralded"])
    assert all(np.diff(cars) < 0)
    assert all(np.diff(g2s) > 0)
    assert cars[1] / cars[0] == pytest.approx(1 / 3, rel=0.1)
    assert g2s[1] / g2s[0] == pytest.approx(3.0, rel=0.1)
    assert g2s[-1] < 0.6


def test_photon_number_distribution_is_consistent():
    setup = example("hbt_heralded", 0.02, pulsed=True)
    g = simulate_gaussian(
        setup,
        LIB,
        gain_scale=2000.0,
        pnr=[("tagger.ch2",), ("tagger.ch2", "tagger.ch3")],
        pnr_cutoff=10,
    )
    single = g.pnr[("tagger.ch2",)]
    meta = g.pnr_info[("tagger.ch2",)]
    assert single.sum() + meta["tail"] == pytest.approx(1.0, abs=1e-6)
    assert float(np.dot(np.arange(len(single)), single)) == pytest.approx(
        g.mean_photons["tagger.ch2"], rel=1e-3
    )
    assert single[0] == pytest.approx(1.0 - g.probability("tagger.ch2"), abs=1e-9)
    joint = g.pnr[("tagger.ch2", "tagger.ch3")]
    assert joint.sum() + g.pnr_info[("tagger.ch2", "tagger.ch3")]["tail"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert np.allclose(joint.sum(axis=1), single, atol=1e-8)
    assert single[2] > 0


def test_ensemble_and_interference_sources_run():
    g = simulate_gaussian(example("bell_test", 0.05), LIB)
    assert g.sources["source"]["members"] == 4
    tt = g.analyses["tt"]["true_hz"]
    tr = g.analyses["tr"]["true_hz"]
    assert tt / (tt + tr) == pytest.approx(
        0.5 * (1 + 0.97 * math.cos(math.radians(2 * 22.5))), abs=0.01
    )
    hom = simulate_gaussian(example("hom", 0.02), LIB)
    assert hom.analyses["hom"]["true_hz"] > 0


def test_cancellation_and_unknown_inputs():
    with pytest.raises(Cancelled):
        simulate_gaussian(example("hbt_heralded", 0.02), LIB, should_cancel=lambda: True)
    g = simulate_gaussian(example("bell_test", 0.05), LIB)
    with pytest.raises(KeyError):
        g.probability("nope")
