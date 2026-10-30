import copy
import math
from pathlib import Path

import numpy as np
import pytest
from test_review_round1_engines import doubled, hbt

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis import correlation as corr
from sequoia.engines.gaussian import (
    GaussianModel,
    GaussianSource,
    IncompatibleWindowsError,
    _Batch,
    _detector_forms,
    _Member,
    _pnr,
    gaussian_window_groups,
    prepare_gaussian,
    simulate_gaussian,
)
from sequoia.errors import SetupError
from sequoia.fock.detectors import IdealPNR, PseudoPNR, Threshold
from sequoia.library.store import Library
from sequoia.setup.io import setup_from_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")
WAIT = 180000


# pseudo-PNR dark counts: per-wire model


def wire_q(dark, n):
    return 1.0 - (1.0 - dark) ** (1.0 / n)


@pytest.mark.parametrize("wires", [1, 2, 4, 7])
@pytest.mark.parametrize("dark", [0.0, 0.05, 0.2, 0.9])
def test_pseudo_pnr_vacuum_is_binomial_in_wire_dark_clicks(wires, dark):
    m = PseudoPNR(wires=wires, dark=dark).response_matrix(3)
    q = wire_q(dark, wires)
    exact = [math.comb(wires, j) * q**j * (1 - q) ** (wires - j) for j in range(wires + 1)]
    assert np.allclose(m[0], exact, atol=1e-12)
    assert m[0, 0] == pytest.approx(1.0 - dark)
    assert np.allclose(m.sum(axis=1), 1.0)


def test_one_photon_on_four_wires_with_dark_counts():
    dark = 0.2
    q = wire_q(dark, 4)
    m = PseudoPNR(wires=4, dark=dark).response_matrix(1)
    assert m[1, 0] == pytest.approx(0.0, abs=1e-15)
    assert m[1, 1] == pytest.approx((1 - q) ** 3)
    assert m[1, 2] == pytest.approx(3 * q * (1 - q) ** 2)
    assert m[1, 2:].sum() == pytest.approx(1 - (1 - q) ** 3)
    assert m[1, 2:].sum() != pytest.approx(dark, abs=0.04)


def sampled_response(det, n, shots, rng):
    nw = det.wires
    w = np.full(nw, 1 / nw) if det.weights is None else np.asarray(det.weights)
    eff = np.ones(nw) if det.wire_efficiencies is None else np.asarray(det.wire_efficiencies)
    q = wire_q(det.dark, nw)
    fired = rng.random((shots, nw)) < q
    for _ in range(n):
        wire = rng.choice(nw, size=shots, p=w)
        hit = rng.random(shots) < det.efficiency * eff[wire]
        fired[np.arange(shots)[hit], wire[hit]] = True
    return np.bincount(fired.sum(axis=1), minlength=nw + 1) / shots


@pytest.mark.parametrize(
    "det",
    [
        PseudoPNR(wires=4, dark=0.2),
        PseudoPNR(wires=3, dark=0.1, efficiency=0.8),
        PseudoPNR(wires=2, weights=(0.9, 0.1), wire_efficiencies=(0.5, 0.8), dark=0.2),
        PseudoPNR.layered(2, 0.3, dark=0.3, efficiency=0.7),
    ],
)
def test_pseudo_pnr_matches_a_direct_wire_simulation(det):
    rng = np.random.default_rng(7)
    m = det.response_matrix(4)
    assert np.allclose(m.sum(axis=1), 1.0) and (m >= -1e-15).all()
    for n in range(5):
        assert np.allclose(m[n], sampled_response(det, n, 200_000, rng), atol=4e-3)


@pytest.mark.parametrize("eff,dark", [(1.0, 0.0), (0.6, 0.05), (0.3, 0.4)])
def test_threshold_is_the_one_wire_pseudo_pnr(eff, dark):
    t = Threshold(efficiency=eff, dark=dark).response_matrix(6)
    p = PseudoPNR(wires=1, efficiency=eff, dark=dark).response_matrix(6)
    assert np.allclose(t, p, atol=1e-12)
    n = np.arange(7)
    assert np.allclose(t[:, 1], 1 - (1 - eff) ** n * (1 - dark))


def test_ideal_pnr_keeps_the_one_extra_count_model():
    m = IdealPNR(dark=0.2).response_matrix(2)
    assert m.shape == (3, 4)
    assert np.allclose(m[0], [0.8, 0.2, 0, 0])
    assert np.allclose(m[1], [0, 0.8, 0.2, 0])
    assert np.allclose(m[2], [0, 0, 0.8, 0.2])


# R01: correlated high-gain PNR of a two-mode squeezed vacuum


def tmsv_model(squeezing, eta_a, eta_b):
    """Gaussian model of two-mode squeezers (one per entry of ``squeezing``, in the two
    polarizations of one signal and one idler frequency), detector a on the signal, b on the
    idler with power efficiencies eta_a, eta_b."""
    r = np.asarray(squeezing, float)
    k = len(r)
    sigma = np.zeros((1, 2))
    sigma[0, :k] = r / np.linalg.norm(r)
    eye = np.eye(2)[None]
    empty = np.zeros(0, dtype=np.int64)
    batch = _Batch(eye, eye, sigma, np.array([[0]]), np.array([[0]]), (empty, empty, empty))
    h_s = np.stack([math.sqrt(eta_a) * np.eye(2)[None], np.zeros((1, 2, 2))])
    h_i = np.stack([np.zeros((1, 2, 2)), math.sqrt(eta_b) * np.eye(2)[None]])
    member = _Member(1.0, [batch])
    src = GaussianSource("tmsv", float(np.sum(r**2)), 1.0, True, [member], k)
    forms = {"tmsv": {id(member): _detector_forms([batch], h_s, h_i)}}
    return GaussianModel([src], forms, ["a", "b"], np.zeros(2), np.zeros(2))


def tmsv_brute(r, eta_a, eta_b, n_max, k_max=600):
    x = math.tanh(r) ** 2
    k = np.arange(k_max)
    pk = (1 - x) * x**k
    from scipy.stats import binom

    ta = binom.pmf(np.arange(n_max + 1)[None, :], k[:, None], eta_a)
    tb = binom.pmf(np.arange(n_max + 1)[None, :], k[:, None], eta_b)
    return np.einsum("k,kn,km->nm", pk, ta, tb)


def conv2(a, b):
    n = a.shape[0]
    out = np.zeros_like(a)
    for i in range(n):
        for j in range(n):
            out[i:, j:] += a[i, j] * b[: n - i, : n - j]
    return out


@pytest.mark.parametrize("r", [0.5, 1.2, 1.8])
def test_lossless_tmsv_joint_distribution_is_diagonal_geometric(r):
    model = tmsv_model([r], 1.0, 1.0)
    probs, meta = _pnr(model, ["a", "b"], ("a", "b"), 1.0, 14)
    x = math.tanh(r) ** 2
    exact = np.diag((1 - x) * x ** np.arange(15))
    assert np.allclose(probs, exact, atol=1e-9)
    assert meta["converged"]
    assert meta["tail"] == pytest.approx(x**15, abs=1e-8)


@pytest.mark.parametrize("eta_a,eta_b", [(0.6, 0.3), (0.9, 0.95)])
def test_lossy_high_gain_tmsv_matches_brute_force(eta_a, eta_b):
    r = 1.5
    model = tmsv_model([r], eta_a, eta_b)
    probs, _ = _pnr(model, ["a", "b"], ("a", "b"), 1.0, 12)
    assert np.allclose(probs, tmsv_brute(r, eta_a, eta_b, 12), atol=1e-9)
    n_bar = math.sinh(r) ** 2
    assert model.mean_photons(0) == pytest.approx(eta_a * n_bar, rel=1e-12)
    assert model.mean_photons(1) == pytest.approx(eta_b * n_bar, rel=1e-12)


def test_two_schmidt_modes_are_the_convolution_of_two_squeezers():
    r1, r2 = 1.1, 0.6
    model = tmsv_model([r1, r2], 0.7, 0.5)
    probs, _ = _pnr(model, ["a", "b"], ("a", "b"), 1.0, 12)
    exact = conv2(tmsv_brute(r1, 0.7, 0.5, 12), tmsv_brute(r2, 0.7, 0.5, 12))
    assert np.allclose(probs, exact, atol=1e-9)


def test_high_gain_low_bins_do_not_depend_on_the_cutoff():
    model = tmsv_model([1.6], 0.8, 0.6)
    joint = [_pnr(model, ["a", "b"], ("a", "b"), 1.0, c)[0] for c in (5, 10, 20)]
    for p in joint[1:]:
        assert np.allclose(p[:6, :6], joint[0], atol=1e-10)
    single = [_pnr(model, ["a", "b"], ("a",), 1.0, c)[0] for c in (4, 16, 40)]
    for p in single[1:]:
        assert np.allclose(p[:5], single[0], atol=1e-10)


def test_moments_from_retained_bins_converge_with_the_cutoff():
    r, eta = 1.4, 0.6
    model = tmsv_model([r], eta, 0.5)
    mean = eta * math.sinh(r) ** 2
    fact2 = 2 * mean**2
    errors = []
    for cutoff in (5, 10, 20, 40, 80):
        probs, meta = _pnr(model, ["a", "b"], ("a",), 1.0, cutoff)
        n = np.arange(cutoff + 1)
        m1, m2 = float(n @ probs), float((n * (n - 1)) @ probs)
        assert m1 <= mean + 1e-9 and m2 <= fact2 + 1e-9
        assert mean - m1 >= (cutoff + 1) * meta["tail"] - 1e-9
        assert fact2 - m2 >= (cutoff + 1) * cutoff * meta["tail"] - 1e-9
        errors.append((mean - m1, fact2 - m2))
    assert all(a[0] > b[0] and a[1] > b[1] for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1][0] < 1e-8 * mean and errors[-1][1] < 1e-8 * fact2
    probs, meta = _pnr(model, ["a", "b"], ("a", "b"), 1.0, 40)
    n = np.arange(41)
    cross = float(n @ probs @ n)
    n_bar = math.sinh(r) ** 2
    exact = eta * 0.5 * (n_bar + 2 * n_bar**2)
    assert cross <= exact and cross == pytest.approx(exact, rel=1e-4)


def test_grid_change_is_reported_as_a_convergence_estimate():
    _, meta = _pnr(tmsv_model([1.0], 0.5, 0.5), ["a", "b"], ("a",), 1.0, 8)
    assert "grid_change" in meta and "alias_bound" not in meta
    assert 0.0 <= meta["grid_change"] < 1e-10


# R02 / R11: window groups, source selection and per-source pump control


def background_laser_setup():
    raw = doubled(hbt(0.02), second_pulsed=False)
    for inst in raw["instances"]:
        if inst["id"] == "b_crystal":
            inst["part"] = "freespace/lens_f200mm"
        if inst["id"] == "b_pump":
            inst.setdefault("overrides", {}).update(wavelength="810 nm", power="1 pW")
    return raw


@pytest.fixture(scope="module")
def single():
    return prepare_gaussian(setup_from_dict(hbt(0.02)), LIB)


@pytest.fixture(scope="module")
def two_sources():
    return prepare_gaussian(setup_from_dict(doubled(hbt(0.02), second_pulsed=False)), LIB)


@pytest.fixture(scope="module")
def with_background_laser():
    return prepare_gaussian(setup_from_dict(background_laser_setup()), LIB)


def test_window_groups_list_incompatible_sources_before_running(single):
    raw = doubled(hbt(0.02), second_pulsed=True)
    groups = gaussian_window_groups(setup_from_dict(raw), LIB)
    assert groups == [
        (pytest.approx(1 / 80e6, rel=1e-12), ["b_crystal"]),
        (pytest.approx(single.window, rel=1e-12), ["crystal"]),
    ]
    with pytest.raises(IncompatibleWindowsError, match="sources="):
        prepare_gaussian(setup_from_dict(raw), LIB)
    same = gaussian_window_groups(setup_from_dict(doubled(hbt(0.02), False)), LIB)
    assert len(same) == 1 and sorted(same[0][1]) == ["b_crystal", "crystal"]


def test_source_selection_restricts_the_model(single):
    raw = doubled(hbt(0.02), second_pulsed=True)
    ref = single.evaluate()
    r = simulate_gaussian(setup_from_dict(raw), LIB, sources=["crystal"])
    assert set(r.sources) == {"crystal"} and r.window == pytest.approx(single.window)
    for ch in ("tagger.ch1", "tagger.ch2", "tagger.ch3"):
        assert r.probability(ch) == pytest.approx(ref.probability(ch), rel=1e-9)
    assert r.analyses["herald_a"]["car"] == pytest.approx(ref.analyses["herald_a"]["car"])
    skipped = [i for i in r.issues if i.code == "gaussian_source_skipped"]
    assert [i.subject for i in skipped] == ["'b_crystal'"]
    bg = r.mean_photons["b_tagger.ch1"]
    assert r.probability("b_tagger.ch1") == pytest.approx(1 - math.exp(-bg), rel=1e-9)
    b = simulate_gaussian(setup_from_dict(raw), LIB, sources=["b_crystal"])
    assert set(b.sources) == {"b_crystal"} and b.window == pytest.approx(1 / 80e6)
    with pytest.raises(SetupError, match="unknown pair source"):
        prepare_gaussian(setup_from_dict(raw), LIB, sources=["nope"])


def test_independent_sources_take_their_own_pump_factor(single, two_sources):
    gain = {"crystal": 2.0, "b_crystal": 5.0}
    r = two_sources.evaluate(gain)
    assert r.source_scale == gain
    ref2, ref5 = single.evaluate(2.0), single.evaluate(5.0)
    for ch in ("tagger.ch1", "tagger.ch2", "tagger.ch3"):
        assert r.probability(ch) == pytest.approx(ref2.probability(ch), rel=1e-9)
        assert r.probability("b_" + ch) == pytest.approx(ref5.probability(ch), rel=1e-9)
    assert r.analyses["b_hbt"]["g2_heralded"] == pytest.approx(ref5.analyses["hbt"]["g2_heralded"])
    assert r.sources["b_crystal"]["pairs_per_window"] == pytest.approx(
        5.0 * single.source_info["crystal"]["pairs_per_window"]
    )
    m = two_sources.model
    assert m.pumps == {"crystal": ["pump"], "b_crystal": ["b_pump"]}
    factor = np.array([2.0] * 3 + [5.0] * 3)
    assert np.allclose(m.background(m.factors(gain)), m.dark + factor * m.leakage)
    assert np.all(m.leakage > 0)
    assert m.factors(3.0) == m.factors({"crystal": 3.0, "b_crystal": 3.0})
    with pytest.raises(ValueError, match="unknown pair source"):
        two_sources.evaluate({"nope": 2.0})


def test_background_laser_leakage_does_not_follow_the_pump_multiplier(with_background_laser):
    p = with_background_laser
    m = p.model
    assert m.pumps == {"crystal": ["pump"]}
    assert set(m.laser_leakage) == {"pump", "b_pump"}
    assert np.allclose(sum(m.laser_leakage.values()), m.leakage)
    assert np.all(m.laser_leakage["b_pump"][3:] > 0)
    low, high = p.evaluate(1.0), p.evaluate(10.0)
    bg1, bg10 = m.background(m.factors(1.0)), m.background(m.factors(10.0))
    assert np.allclose(bg10[3:], bg1[3:])
    assert np.allclose(bg10[:3] - m.dark[:3], 10 * (bg1[:3] - m.dark[:3]))
    for ch in ("b_tagger.ch1", "b_tagger.ch2"):
        assert high.probability(ch) == pytest.approx(low.probability(ch), rel=1e-12)
    assert high.probability("tagger.ch1") > 5 * low.probability("tagger.ch1")


def test_one_laser_cannot_follow_two_multipliers():
    model = tmsv_model([0.5], 1.0, 1.0)
    model.pumps = {"tmsv": ["L"], "other": ["L"]}
    model.laser_leakage = {"L": np.array([0.1, 0.0])}
    model.leakage = np.array([0.1, 0.0])
    assert np.allclose(model.background(model.factors(3.0)), [0.3, 0.0])
    with pytest.raises(ValueError, match="different multipliers"):
        model.background(model.factors({"tmsv": 2.0, "other": 3.0}))


# R06: searched versus fixed (confirmatory) coincidence centres


def correlated_tags(seed, delay=37e-9, pairs=4000, noise=20000, duration=1.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(pairs) * duration)
    a = np.sort(np.concatenate([t, rng.random(noise) * duration]))
    b = np.sort(
        np.concatenate([t + delay + rng.normal(0, 50e-12, pairs), rng.random(noise) * duration])
    )
    c = np.sort(np.concatenate([t + 2 * delay, rng.random(noise) * duration]))
    return {"a": a, "b": b, "c": c}, duration


SPECS = [
    {"name": "ab", "type": "coincidences", "channels": ["b", "a"], "window": "1 ns"},
    {"name": "h", "type": "heralded_g2", "herald": "a", "a": "b", "b": "c", "window": "1 ns"},
]


def test_search_and_fixed_modes_are_recorded():
    tags, duration = correlated_tags(1)
    searched = corr.analyse_tags(tags, duration, SPECS)
    assert searched["ab"]["center_mode"] == "search" and searched["ab"]["peak_found"]
    assert searched["ab"]["search_s"] > 0
    assert searched["ab"]["center_s"] == pytest.approx(37e-9, abs=0.3e-9)
    assert searched["h"]["center_mode"] == "search"
    fixed = corr.analyse_tags(tags, duration, [dict(SPECS[0], center="37 ns", mode="fixed")])
    assert fixed["ab"]["center_mode"] == "fixed" and fixed["ab"]["search_s"] is None
    assert fixed["ab"]["peak_found"] is None and fixed["ab"]["center_s"] == pytest.approx(37e-9)
    mixed = corr.analyse_tags(tags, duration, [dict(SPECS[1], center_a="37 ns")])
    assert mixed["h"]["center_mode"] == "mixed"
    with pytest.raises(corr.CorrelationError, match="needs an explicit"):
        corr.analyse_tags(tags, duration, [dict(SPECS[0], mode="fixed")])
    with pytest.raises(corr.CorrelationError, match="cannot use a fixed"):
        corr.analyse_tags(tags, duration, [dict(SPECS[0], center="1 ns", mode="search")])


def test_confirm_centers_freezes_a_search_for_independent_data():
    tags1, duration = correlated_tags(1)
    tags2, _ = correlated_tags(2)
    search = corr.analyse_tags(tags1, duration, SPECS)
    for specs in (corr.confirm_centers(search, SPECS), corr.confirm_centers(search)):
        assert {s["mode"] for s in specs} == {"fixed"}
        confirm = corr.analyse_tags(tags2, duration, specs)
        assert confirm["ab"]["center_mode"] == "fixed"
        assert confirm["ab"]["center_s"] == search["ab"]["center_s"]
        assert confirm["ab"]["counts"] == corr.count_in_window(
            tags2["b"], tags2["a"], 1e-9, search["ab"]["center_s"]
        )
        assert confirm["h"]["center_a_s"] == search["h"]["center_a_s"]
        assert confirm["h"]["center_b_s"] == search["h"]["center_b_s"]
        assert confirm["ab"]["confirmed_from"]["center_mode"] == "search"
    rng = np.random.default_rng(3)
    noise = {k: np.sort(rng.random(2000)) for k in ("a", "b", "c")}
    empty = corr.analyse_tags(noise, 1.0, SPECS[:1])
    assert empty["ab"]["peak_found"] is False
    with pytest.raises(corr.CorrelationError, match="no peak"):
        corr.confirm_centers(empty, SPECS[:1])


def test_saved_tags_store_the_centres_and_search_ranges(tmp_path):
    tags, duration = correlated_tags(1)
    analyses = corr.analyse_tags(tags, duration, SPECS)
    path = tmp_path / "t.npz"
    corr.write_tags(path, tags, duration, {}, {"seed": 1}, analyses=analyses)
    saved = corr.read_tag_record(path).meta["analyses"]
    assert saved["ab"]["center_mode"] == "search"
    assert saved["ab"]["center_s"] == pytest.approx(analyses["ab"]["center_s"])
    assert saved["ab"]["search_s"] == pytest.approx(analyses["ab"]["search_s"])
    assert saved["h"]["center_b_s"] == pytest.approx(analyses["h"]["center_b_s"])
    assert saved["ab"]["window_s"] == pytest.approx(1e-9)


# GUI


def settle(qtbot, tab):
    qtbot.waitUntil(lambda: not tab.runner.busy, timeout=WAIT)
    qtbot.wait(30)


def test_gaussian_tab_offers_a_source_group_instead_of_failing(qtbot):
    from sequoia.gui.gaussian_panel import GaussianTab

    setup = setup_from_dict(doubled(hbt(0.02), second_pulsed=True))
    g = GaussianTab(lambda: (setup, LIB))
    qtbot.addWidget(g)
    g.sweep.setChecked(False)
    g.list_sources()
    settle(qtbot, g)
    assert "differ" in g.groups_label.text() and not g.group_box.isHidden()
    assert g.factor_table.rowCount() == 2 and set(g.factors) == {"crystal", "b_crystal"}
    g.run()
    settle(qtbot, g)
    assert g.outcome is None and "choose one source group" in g.status.text()
    items = [g.group_box.itemData(i) for i in range(g.group_box.count())]
    assert items[0] is None
    g.group_box.setCurrentIndex(items.index(["crystal"]))
    g.scale.setValue(0.5)
    g.factors["crystal"].setValue(4.0)
    g.run()
    settle(qtbot, g)
    r = g.outcome.result
    assert set(r.sources) == {"crystal"} and r.source_scale["crystal"] == pytest.approx(2.0)
    assert r.source_scale["b_crystal"] == pytest.approx(0.5)
    assert "sources: crystal" in g.status.text() and "left out: 'b_crystal'" in g.status.text()
    assert "outdated" not in g.status.text()
    g.factors["crystal"].setValue(3.0)
    assert "outdated" in g.status.text()
    assert "dark counts" in g.scope_label.text() and "leakage" in g.scope_label.text()
    g.runner.shutdown()


def test_monte_carlo_tab_freezes_centres_from_the_last_search(qtbot, tmp_path, monkeypatch):
    from sequoia.gui import montecarlo_panel
    from sequoia.gui.montecarlo_panel import MonteCarloTab

    raw = hbt(1.0)
    setup = setup_from_dict(copy.deepcopy(raw))
    mc = MonteCarloTab(lambda: (setup, LIB))
    qtbot.addWidget(mc)
    mc.duration.setValue(0.05)
    mc.seed.setValue(1)
    mc.freeze.setChecked(True)
    mc.run()
    assert not mc.runner.busy and "run once" in mc.status.text()
    mc.freeze.setChecked(False)
    mc.run()
    settle(qtbot, mc)
    first = mc.outcome.analyses
    assert first["herald_a"]["center_mode"] == "search"
    assert "centres searched" in mc.status.text()
    mc.freeze.setChecked(True)
    mc.seed.setValue(2)
    mc.run()
    settle(qtbot, mc)
    second = mc.outcome.analyses
    assert mc.outcome.confirmatory and "fixed centres (confirmatory)" in mc.status.text()
    for name in ("herald_a", "herald_b"):
        assert second[name]["center_mode"] == "fixed"
        assert second[name]["center_s"] == first[name]["center_s"]
    assert second["hbt"]["center_a_s"] == first["hbt"]["center_a_s"]
    path = tmp_path / "mc.npz"
    monkeypatch.setattr(
        montecarlo_panel.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), "")
    )
    mc.save_tags()
    saved = corr.read_tag_record(path).meta["analyses"]
    assert saved["herald_a"]["center_mode"] == "fixed"
    assert saved["herald_a"]["center_s"] == pytest.approx(first["herald_a"]["center_s"])
    mc.runner.shutdown()
