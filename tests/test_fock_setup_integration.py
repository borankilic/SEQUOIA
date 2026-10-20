import math
from pathlib import Path

import numpy as np
import pytest

from sequoia.engines.classical import build_network
from sequoia.engines.gaussian import simulate_gaussian
from sequoia.engines.rates import simulate_rates
from sequoia.fock.circuit import BeamSplitter, Block, Circuit, PhaseShifter
from sequoia.fock.compile import compile_setup_network
from sequoia.fock.from_setup import (
    IncompatibleWindowsError,
    build_experiment_from_setup,
    extract_circuit,
    extraction_experiment,
    list_components,
    run_setup_experiment,
)
from sequoia.fock.inputs import pair_number_pmf
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Library(ROOT / "library")
LAM = 810e-9


def inst(i, part, settings=None, overrides=None, **kw):
    d = {"id": i, "part": part, "settings": settings or {}, **kw}
    if overrides:
        d["overrides"] = overrides
    return d


def fs(i, src, dst, length="100 mm"):
    return {"id": i, "kind": "free_space", "from": src, "to": dst, "length": length}


FOCK = "sources/fock_1photon_810"
BS = "freespace/beamsplitter_50_50"
SPAD = "detectors/spad_si_generic"


def two_source_hom(phase="0 deg", tag_b=0, length_b="50 mm", rep_b=None, mzi=False):
    ov_b = {"distinguishability_tag": tag_b}
    if rep_b:
        ov_b["repetition_rate"] = rep_b
    instances = [
        inst("a", FOCK),
        inst("b", FOCK, overrides=ov_b),
        inst("bs", BS),
        inst("d1", SPAD, {"input": "free_space"}),
        inst("d2", SPAD, {"input": "free_space"}),
    ]
    links = [fs("la", "a.out", "bs.in_a"), fs("lb", "b.out", "bs.in_b")]
    if mzi:
        instances += [
            inst("ps", "polarization/phase_shifter_ideal", {"phase": phase}),
            inst("bs2", BS),
        ]
        links += [
            fs("m1", "bs.out_a", "ps.in", "50 mm"),
            fs("m2", "ps.out", "bs2.in_a", length_b),
            fs("m3", "bs.out_b", "bs2.in_b", "100 mm"),
            fs("l1", "bs2.out_a", "d1.in"),
            fs("l2", "bs2.out_b", "d2.in"),
        ]
    else:
        links += [fs("l1", "bs.out_a", "d1.in"), fs("l2", "bs.out_b", "d2.in")]
    return setup_from_dict({"instances": instances, "links": links})


def detector_params(setup, det_id, window):
    net = build_network(setup, LIBRARY)
    model = net.components[det_id].detector_model()
    eta = float(np.real(model.efficiency(np.array([LAM]))[0, 0, 0]))
    dark = 1.0 - math.exp(-model.dark_count_rate * window)
    return eta, dark


def test_single_photon_on_beam_splitter_splits_evenly():
    setup = setup_from_dict(
        {
            "instances": [
                inst("f", FOCK),
                inst("bs", BS),
                inst("d1", SPAD, {"input": "free_space"}),
                inst("d2", SPAD, {"input": "free_space"}),
            ],
            "links": [
                fs("l0", "f.out", "bs.in_a"),
                fs("l1", "bs.out_a", "d1.in"),
                fs("l2", "bs.out_b", "d2.in"),
            ],
        }
    )
    se = build_experiment_from_setup(setup, LIBRARY, detectors_mode="pnr")
    res = run_setup_experiment(se)
    assert res.photon_numbers[(1, 0)] == pytest.approx(0.5)
    assert res.photon_numbers[(0, 1)] == pytest.approx(0.5)
    eta, dark = detector_params(setup, "d1", se.window_s)
    se_t = build_experiment_from_setup(setup, LIBRARY)
    det = run_setup_experiment(se_t).detected
    pc = 1.0 - (1.0 - eta) * (1.0 - dark)
    expect = 0.5 * pc * (1.0 - dark) + 0.5 * dark * (1.0 - pc)
    assert det[(1, 0)] == pytest.approx(expect, rel=1e-9)
    assert det[(0, 1)] == pytest.approx(expect, rel=1e-9)
    assert sum(det.values()) == pytest.approx(1.0)


def test_hom_bunching_equal_tags_and_distinguishable():
    se = build_experiment_from_setup(two_source_hom(), LIBRARY, detectors_mode="pnr")
    pn = run_setup_experiment(se).photon_numbers
    assert pn.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-12)
    assert pn[(2, 0)] == pytest.approx(0.5)
    se = build_experiment_from_setup(two_source_hom(tag_b=1), LIBRARY, detectors_mode="pnr")
    pn = run_setup_experiment(se).photon_numbers
    assert pn[(1, 1)] == pytest.approx(0.5)


def coincidence(setup):
    se = build_experiment_from_setup(setup, LIBRARY, detectors_mode="pnr")
    return run_setup_experiment(se).photon_numbers.get((1, 1), 0.0)


def test_single_mode_phase_before_a_beam_splitter_is_a_global_phase():
    assert coincidence(two_source_hom("0 deg", mzi=False)) == pytest.approx(0.0, abs=1e-12)
    a = build_experiment_from_setup(two_source_hom("0 deg"), LIBRARY)
    assert a.n_detectors == 2


def test_phase_shifter_and_path_length_set_the_two_photon_fringe():
    for deg in (0.0, 30.0, 90.0, 135.0):
        expected = math.cos(math.radians(deg)) ** 2
        assert coincidence(two_source_hom(f"{deg} deg", mzi=True)) == pytest.approx(
            expected, abs=1e-9
        )
        distinct = 1.0 - math.sin(math.radians(deg)) ** 2 / 2.0
        assert coincidence(two_source_hom(f"{deg} deg", tag_b=1, mzi=True)) == pytest.approx(
            distinct, abs=1e-9
        )
    quarter = f"{50.0 + LAM / 4.0 * 1e3:.9f} mm"
    assert coincidence(two_source_hom("0 deg", length_b=quarter, mzi=True)) == pytest.approx(
        0.0, abs=1e-5
    )


def test_counts_per_second_follow_probability_and_rate():
    se = build_experiment_from_setup(two_source_hom(tag_b=1), LIBRARY, duration=2.5)
    res = run_setup_experiment(se)
    assert se.rate_hz == pytest.approx(80e6)
    for s, p in res.detected.items():
        assert res.counts.per_second[s] == pytest.approx(p * 80e6)
        assert res.counts.total[s] == pytest.approx(p * 80e6 * 2.5)
        assert res.counts.sigma[s] == pytest.approx(math.sqrt(p * (1 - p) * 80e6 * 2.5))


def test_different_windows_raise_and_selection_works():
    setup = two_source_hom(rep_b="40 MHz")
    with pytest.raises(IncompatibleWindowsError, match="time windows"):
        build_experiment_from_setup(setup, LIBRARY)
    se = build_experiment_from_setup(setup, LIBRARY, sources=["a"])
    assert se.window_s == pytest.approx(1.0 / 80e6)


@pytest.fixture(scope="module")
def hbt():
    setup = load_setup(ROOT / "setups" / "hbt_heralded.yaml")
    return setup, build_experiment_from_setup(setup, LIBRARY)


def test_pair_number_weights_are_thermal_with_engine_rate(hbt):
    setup, se = hbt
    rates = simulate_rates(setup, LIBRARY)
    src = rates.sources["crystal"]
    mu = src["rate_hz"] / src["dnu_hz"]
    desc = se.inputs[0]
    assert desc.mean_pairs == pytest.approx(mu, rel=1e-9)
    by_n: dict[int, float] = {}
    for w, state in se.experiment.input_mixture.branches:
        n = len(state.terms[0].photons) // 2
        by_n[n] = by_n.get(n, 0.0) + w
    assert set(by_n) == set(range(desc.n_max + 1))
    for n, w in by_n.items():
        assert w == pytest.approx(pair_number_pmf(n, mu, desc.schmidt), rel=1e-9)
    assert se.omitted_weight == pytest.approx(1.0 - sum(by_n.values()), abs=1e-12)


def test_clicks_and_coincidences_match_gaussian_engine(hbt):
    setup, se = hbt
    res = run_setup_experiment(se)
    g = simulate_gaussian(setup, LIBRARY)
    chans = {"spad_h": "tagger.ch1", "spad_a": "tagger.ch2", "spad_b": "tagger.ch3"}
    ix = {d: i for i, d in enumerate(res.detector_ids)}
    for d, c in chans.items():
        p = sum(v for s, v in res.detected.items() if s[ix[d]] > 0)
        assert p == pytest.approx(g.probability(c), rel=5e-3)
    p_ha = sum(v for s, v in res.detected.items() if s[ix["spad_h"]] > 0 and s[ix["spad_a"]] > 0)
    assert p_ha == pytest.approx(g.probability("tagger.ch1", "tagger.ch2"), rel=5e-3)


def test_unheralded_g2_of_one_arm_is_one_plus_one_over_k():
    setup = load_setup(ROOT / "setups" / "hbt_heralded.yaml")
    for i in setup.instances:
        if i.id == "spad_h":
            i.overrides = {"dark_count_rate": "0 Hz"}
        if i.id == "pbs":
            i.part = "polarization/pbs_ideal"
    se = build_experiment_from_setup(setup, LIBRARY, detectors_mode="pnr", n_max=3)
    res = run_setup_experiment(se)
    k = se.inputs[0].schmidt
    dist: dict[int, float] = {}
    for s, p in res.detected.items():
        dist[s[0]] = dist.get(s[0], 0.0) + p
    mean = sum(n * p for n, p in dist.items())
    g2 = sum(n * (n - 1) * p for n, p in dist.items()) / mean**2
    mu = se.inputs[0].mean_pairs
    pmf = [pair_number_pmf(n, mu, k) for n in range(4)]
    truncated = (
        sum(n * (n - 1) * q for n, q in enumerate(pmf)) / sum(n * q for n, q in enumerate(pmf)) ** 2
    )
    assert g2 == pytest.approx(truncated, abs=2e-5)
    assert truncated == pytest.approx(1.0 + 1.0 / k, abs=3e-3)


def bell_with(angle_a, angle_b):
    setup = load_setup(ROOT / "setups" / "bell_test.yaml")
    for i in setup.instances:
        if i.id == "hwp_a":
            i.settings = {"angle": f"{angle_a / 2} deg"}
        if i.id == "hwp_b":
            i.settings = {"angle": f"{angle_b / 2} deg"}
    return setup


def bell_correlation(a, b):
    setup = bell_with(a, b)
    se = build_experiment_from_setup(setup, LIBRARY)
    res = run_setup_experiment(se)
    ix = {d: i for i, d in enumerate(res.detector_ids)}

    def both(x, y):
        return sum(p for s, p in res.detected.items() if s[ix[x]] > 0 and s[ix[y]] > 0)

    f = np.array(
        [
            both("snspd_at", "snspd_bt"),
            both("snspd_at", "snspd_br"),
            both("snspd_ar", "snspd_bt"),
            both("snspd_ar", "snspd_br"),
        ]
    )
    return setup, f / f.sum()


def test_bell_coincidence_fractions_match_rate_engine_and_chsh():
    setup, frac = bell_correlation(0.0, 22.5)
    rates = simulate_rates(setup, LIBRARY)
    names = {"snspd_at": "tagger.ch1", "snspd_ar": "tagger.ch2", "snspd_bt": "tagger.ch3"}
    names["snspd_br"] = "tagger.ch4"
    ref = np.array(
        [
            rates.pairs[(names[x], names[y])]["coincidence_rate_hz"]
            for x, y in (
                ("snspd_at", "snspd_bt"),
                ("snspd_at", "snspd_br"),
                ("snspd_ar", "snspd_bt"),
                ("snspd_ar", "snspd_br"),
            )
        ]
    )
    assert frac == pytest.approx(ref / ref.sum(), abs=1e-4)
    a, ap, b, bp = 0.0, 45.0, 22.5, 67.5
    s = 0.0
    for x, y, sign in ((a, b, 1), (a, bp, -1), (ap, b, 1), (ap, bp, 1)):
        _, f = bell_correlation(x, y)
        s += sign * (f[0] + f[3] - f[1] - f[2])
    assert s == pytest.approx(2.0 * math.sqrt(2.0) * 0.97, abs=1e-3)


def test_hom_at_the_dip_matches_gaussian_engine():
    setup = load_setup(ROOT / "setups" / "hom.yaml")
    for link in setup.links:
        if link.id == "delay_arm":
            link.length = type(link.length)(value=100.66, unit="mm")
    se = build_experiment_from_setup(setup, LIBRARY)
    res = run_setup_experiment(se)
    g = simulate_gaussian(setup, LIBRARY)
    single = sum(p for s, p in res.detected.items() if s[0] > 0)
    assert single == pytest.approx(g.probability("tagger.ch1"), rel=5e-3)
    both = sum(p for s, p in res.detected.items() if s[0] > 0 and s[1] > 0)
    assert both == pytest.approx(g.probability("tagger.ch1", "tagger.ch2"), rel=0.15)


def fock_pbs_setup():
    return setup_from_dict(
        {
            "instances": [
                inst("f", FOCK, {"polarization_angle": "30 deg"}),
                inst("hwp", "polarization/hwp_810", {"angle": "17 deg"}),
                inst("qwp", "polarization/qwp_810", {"angle": "-11 deg"}),
                inst("pbs", "polarization/pbs_cube_810"),
                inst("bs", BS),
                inst("m", "freespace/mirror_ideal"),
                inst("d1", SPAD, {"input": "free_space"}),
                inst("d2", SPAD, {"input": "free_space"}),
                inst("d3", SPAD, {"input": "free_space"}),
            ],
            "links": [
                fs("l0", "f.out", "hwp.in"),
                fs("l1", "hwp.out", "qwp.in", "20 mm"),
                fs("l2", "qwp.out", "pbs.in", "33 mm"),
                fs("l3", "pbs.out_t", "bs.in_a", "47 mm"),
                fs("l4", "pbs.out_r", "m.in", "61 mm"),
                fs("l5", "m.out", "bs.in_b", "29 mm"),
                fs("l6", "bs.out_a", "d1.in"),
                fs("l7", "bs.out_b", "d2.in"),
            ],
        }
    )


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "templates" / "fock_hom.yaml",
        ROOT / "templates" / "hbt_heralded.yaml",
        ROOT / "setups" / "bell_test.yaml",
        ROOT / "setups" / "hom.yaml",
        None,
    ],
    ids=["fock_hom", "hbt", "bell", "hom", "fock_pbs"],
)
def test_decomposition_equals_exact_compilation(path):
    setup = fock_pbs_setup() if path is None else load_setup(path)
    se = build_experiment_from_setup(setup, LIBRARY)
    ex = extract_circuit(setup, LIBRARY, se.plan.channels[0].wavelength)
    assert ex.verified and not ex.fallback
    assert ex.circuit.matrix().shape[0] == ex.circuit.n_modes
    assert any(e.label for e in ex.circuit.elements)
    specs = list(ex.input_modes)
    bases = []
    for c in se.plan.channels:
        bases.append(c.basis if c.inst in ex.fock_angles else None)
    ref = compile_setup_network(
        setup,
        LIBRARY,
        ex.wavelength,
        inputs=specs,
        detectors=list(ex.detector_modes),
        bases=bases,
        seed_modes=[
            next((c.mode for c in se.plan.channels if f"{c.inst}.{c.port}" == s), None)
            for s in specs
        ],
    )
    t = ex.circuit.matrix()
    rows = [m for d in ex.detector_modes for m in ex.detector_modes[d]]
    cols = [m for s in specs for m in ex.input_modes[s]]
    assert np.allclose(t[np.ix_(rows, cols)], ref.transfer, atol=1e-8)


def test_extracted_elements_are_editable_and_reproduce_the_distribution():
    setup = two_source_hom(tag_b=1)
    se = build_experiment_from_setup(setup, LIBRARY, detectors_mode="pnr")
    exp, ex = extraction_experiment(se)
    kinds = {type(e) for e in exp.circuit.elements}
    assert BeamSplitter in kinds and PhaseShifter in kinds or kinds == {BeamSplitter}
    assert all(not isinstance(e, Block) for e in exp.circuit.elements)
    assert {e.label for e in exp.circuit.elements} >= {"bs"}
    from sequoia.fock.experiment import run_experiment

    direct = run_setup_experiment(se).detected
    got = run_experiment(exp)
    trimmed: dict[tuple, float] = {}
    for s, p in got.probabilities.items():
        trimmed[s[: len(exp.detectors)]] = trimmed.get(s[: len(exp.detectors)], 0.0) + p
    assert sum(trimmed.values()) == pytest.approx(1.0)
    assert Circuit.from_dict(exp.circuit.to_dict()).matrix() == pytest.approx(exp.circuit.matrix())
    assert trimmed[(1, 1)] == pytest.approx(direct[(1, 1)], rel=1e-9)
    assert trimmed[(0, 2)] + trimmed[(2, 0)] == pytest.approx(
        direct[(0, 2)] + direct[(2, 0)], rel=1e-9
    )


def test_component_listing_names_every_part_and_link():
    setup = load_setup(ROOT / "templates" / "fock_hom.yaml")
    rows = list_components(setup, LIBRARY)
    ids = {r.id for r in rows}
    assert {i.id for i in setup.instances} <= ids
    assert {link.id for link in setup.links} <= ids
    by_id = {r.id: r for r in rows}
    assert by_id["phase"].type == "PhaseShifter"
    assert by_id["photon_a"].category == "sources"
    assert "photons_per_pulse=1" in by_id["photon_a"].summary


def test_pulsed_source_uses_repetition_window_and_finite_schmidt_number():
    setup = load_setup(ROOT / "setups" / "hbt_heralded.yaml")
    for i in setup.instances:
        if i.id == "pump":
            i.overrides = {"repetition_rate": "80 MHz", "pulse_width": "1 ps", "power": "20 mW"}
        if i.id == "pbs":
            i.part = "polarization/pbs_ideal"
        if i.id == "spad_h":
            i.overrides = {"dark_count_rate": "0 Hz"}
    se = build_experiment_from_setup(setup, LIBRARY, detectors_mode="pnr", n_max=4)
    assert se.window_s == pytest.approx(1.0 / 80e6)
    desc = se.inputs[0]
    assert 5.0 < desc.schmidt < 40.0
    rates = simulate_rates(setup, LIBRARY)
    assert desc.mean_pairs == pytest.approx(rates.sources["crystal"]["rate_hz"] / 80e6, rel=1e-9)
    res = run_setup_experiment(se)
    dist: dict[int, float] = {}
    for s, p in res.detected.items():
        dist[s[0]] = dist.get(s[0], 0.0) + p
    mean = sum(n * p for n, p in dist.items())
    g2 = sum(n * (n - 1) * p for n, p in dist.items()) / mean**2
    pmf = [pair_number_pmf(n, desc.mean_pairs, desc.schmidt) for n in range(desc.n_max + 1)]
    truncated = (
        sum(n * (n - 1) * q for n, q in enumerate(pmf)) / sum(n * q for n, q in enumerate(pmf)) ** 2
    )
    assert g2 == pytest.approx(truncated, rel=1e-6)
    assert truncated == pytest.approx(1.0 + 1.0 / desc.schmidt, rel=5e-3)
    g = simulate_gaussian(setup, LIBRARY)
    single = sum(p for s, p in res.detected.items() if s[1] > 0)
    assert single == pytest.approx(g.probability("tagger.ch2"), rel=2e-2)
