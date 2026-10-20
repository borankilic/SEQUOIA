import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis import correlation as corr
from sequoia.analysis.chsh import format_chsh, run_chsh
from sequoia.analysis.sweep import SweepError
from sequoia.engines.montecarlo import TimeTags, simulate_timetags
from sequoia.fock import Circuit
from sequoia.fock.analysis import (
    condition,
    count_variance,
    expected_counts,
    outcome_table,
    sample,
)
from sequoia.fock.circuit import FockError
from sequoia.fock.detectors import DetectorLayout, IdealPNR, PseudoPNR, Threshold
from sequoia.fock.engine import Distribution, output_distribution
from sequoia.fock.experiment import DetectorSpec, Experiment, run_experiment
from sequoia.fock.inputs import InputMixture, PureState
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


# RR07: detector output capacity is independent of the input cutoff


@pytest.mark.parametrize(
    "det", [IdealPNR(dark=0.2), PseudoPNR(wires=4, dark=0.2), Threshold(dark=0.2)]
)
def test_dark_counts_survive_vacuum(det):
    m = det.response_matrix(0)
    assert m[0, 0] == pytest.approx(0.8)
    assert m[0, 1] == pytest.approx(0.2)
    assert np.allclose(m.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "det",
    [
        IdealPNR(efficiency=0.7, dark=0.1),
        PseudoPNR(wires=3, dark=0.1, efficiency=0.8),
        PseudoPNR(wires=2, weights=(0.9, 0.1), wire_efficiencies=(0.5, 0.8), dark=0.2),
        Threshold(efficiency=0.6, dark=0.05),
    ],
)
def test_vacuum_response_is_independent_of_the_cutoff(det):
    rows = [det.response_matrix(n) for n in range(6)]
    for m in rows:
        assert np.allclose(m.sum(axis=1), 1.0)
        assert (m >= -1e-15).all()
    width = max(m.shape[1] for m in rows)
    padded = [np.pad(m, [(0, 0), (0, width - m.shape[1])]) for m in rows]
    for n_max, m in enumerate(padded):
        for n in range(n_max + 1):
            assert np.allclose(m[n], padded[-1][n])


def test_one_photon_plus_dark_click_reads_two_on_a_pnr():
    m = IdealPNR(dark=0.2).response_matrix(1)
    assert m.shape == (2, 3)
    assert m[1, 2] == pytest.approx(0.2)
    pp = PseudoPNR(wires=4, dark=0.2).response_matrix(1)
    assert pp.shape == (2, 5) and pp[1, 2] == pytest.approx(0.2)
    lay = DetectorLayout.from_modes(1, {0: IdealPNR(dark=0.2)})
    vac = lay.apply(Distribution({(0,): 1.0}, 1))
    assert vac.probs == pytest.approx({(0,): 0.8, (1,): 0.2})
    one = lay.apply(Distribution({(1,): 1.0}, 1))
    assert one.probs == pytest.approx({(1,): 0.8, (2,): 0.2})


def test_experiment_on_vacuum_keeps_dark_counts():
    exp = Experiment(
        Circuit(2), [0, 0], detectors=[DetectorSpec((0,), PseudoPNR(wires=2, dark=0.2))]
    )
    r = run_experiment(exp)
    assert r.probabilities[(1, 0)] == pytest.approx(0.2)
    assert r.mass == pytest.approx(1.0)


# RR08: duration semantics


def test_duration_values():
    base = dict(rate_hz=100.0)
    none = run_experiment(Experiment(Circuit(1), [1], **base))
    assert none.counts.duration_s == 1.0 and none.counts.total[(1,)] == pytest.approx(100.0)
    zero = run_experiment(Experiment(Circuit(1), [1], duration_s=0.0, **base))
    assert zero.counts.duration_s == 0.0
    assert zero.counts.total[(1,)] == 0.0 and zero.counts.n_windows == 0.0
    two = run_experiment(Experiment(Circuit(1), [1], duration_s=2.0, **base))
    assert two.counts.total[(1,)] == pytest.approx(200.0)
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(FockError):
            run_experiment(Experiment(Circuit(1), [1], duration_s=bad, **base))
    for bad in (0.0, -5.0, math.nan, math.inf):
        with pytest.raises(FockError):
            run_experiment(Experiment(Circuit(1), [1], rate_hz=bad))
        with pytest.raises(FockError):
            run_experiment(Experiment(Circuit(1), [1], window_s=bad))


def test_duration_document_validation():
    doc = Experiment(Circuit(1), [1], rate_hz=100.0, duration_s=0.0).to_dict()
    assert Experiment.from_dict(doc).duration_s == 0.0
    doc["counts"]["duration_s"] = -1
    with pytest.raises(FockError):
        Experiment.from_dict(doc)
    doc["counts"]["duration_s"] = float("nan")
    with pytest.raises(FockError):
        Experiment.from_dict(doc)


# RR05: count uncertainty under a fixed number of windows


def test_binomial_uncertainty_limits():
    n = 100
    for p, expected in ((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(n) / 2)):
        ec = expected_counts({(0,): p}, rate_hz=n, duration_s=1.0)
        assert ec.sigma[(0,)] == pytest.approx(expected)
        row = outcome_table(Distribution({(0,): p}, 1), n_windows=n)[0]
        assert row["sigma"] == pytest.approx(expected)
    rare = expected_counts({(0,): 1e-6}, rate_hz=1e9, duration_s=1.0)
    assert rare.sigma[(0,)] == pytest.approx(math.sqrt(1e3), rel=1e-5)
    poisson = expected_counts({(0,): 1.0}, rate_hz=100, duration_s=1.0, acquisition="poisson")
    assert poisson.sigma[(0,)] == pytest.approx(10.0)


def test_api_and_experiment_agree_and_covariances():
    r = run_experiment(Experiment(Circuit(2).beam_splitter(0, 1), [1, 1], rate_hz=1e3))
    table = {
        row["state"]: row["sigma"]
        for row in outcome_table(Distribution(r.probabilities, 2), n_windows=1e3)
    }
    for s, sig in r.counts.sigma.items():
        assert sig == pytest.approx(table[s])
    states = list(r.probabilities)
    ec = r.counts
    direct = math.sqrt(sum(ec.covariance(a, b) for a in states for b in states))
    assert ec.sigma_sum(states) == pytest.approx(direct, abs=1e-9)
    assert ec.sigma_sum(states) == pytest.approx(0.0, abs=1e-6)
    a, b = states[:2]
    assert ec.covariance(a, b) == pytest.approx(-1e3 * ec.per_window[a] * ec.per_window[b])


def test_multinomial_spread_matches_simulation():
    probs = {(0,): 0.6, (1,): 0.3, (2,): 0.1}
    ec = expected_counts(probs, rate_hz=200, duration_s=1.0)
    rng = np.random.default_rng(4)
    draws = rng.multinomial(200, list(probs.values()), size=20000)
    for i, s in enumerate(probs):
        assert draws[:, i].std() == pytest.approx(ec.sigma[s], rel=0.03)
    summed = draws[:, 1] + draws[:, 2]
    assert summed.std() == pytest.approx(ec.sigma_sum([(1,), (2,)]), rel=0.03)


def test_conditional_counts_use_accepted_events():
    d = output_distribution(Circuit(2).beam_splitter(0, 1), [(0, 0), (1, 0)])
    c = condition(d, {0: 2})
    ec = expected_counts(c, rate_hz=1000, duration_s=1.0)
    p = c.success_probability
    assert sum(ec.total.values()) == pytest.approx(1000 * p)
    assert ec.sigma_sum(list(ec.per_window)) == pytest.approx(math.sqrt(1000 * p * (1 - p)))
    assert count_variance(0.5, 4.0) == pytest.approx(1.0)


# RR06: sampling with unresolved probability


def test_truncated_mixture_cannot_produce_unconditional_known_outcomes():
    mix = InputMixture(((0.25, PureState.fock([(0, 0)])),))
    exp = Experiment(Circuit(1), [0], input_mixture=mix, shots=100, seed=1, rate_hz=100)
    r = run_experiment(exp)
    assert r.samples.counts[(1,)] < 100
    assert r.samples.unresolved + r.samples.counts[(1,)] == 100
    assert r.samples.unresolved_probability == pytest.approx(0.75)
    assert r.to_dict()["unresolved_shots"] == r.samples.unresolved
    d = Distribution({(1,): 0.25}, 1, omitted_weight=0.75)
    with pytest.raises(FockError):
        sample(d, 100, seed=1, tail="error")
    cond = sample(d, 100, seed=1, tail="condition")
    assert cond.counts == {(1,): 100} and cond.conditioned_on == pytest.approx(0.25)
    big = sample(d, 20000, seed=2)
    assert big.unresolved / 20000 == pytest.approx(0.75, abs=0.02)


def test_full_mass_sampling_is_unchanged():
    d = output_distribution(Circuit(2).beam_splitter(0, 1), [(0, 0), (1, 0)])
    s = sample(d, 500, seed=7)
    assert s.unresolved == 0 and s.conditioned_on is None
    rng = np.random.default_rng(7)
    states = sorted(d.probs)
    p = np.array([d.probs[k] for k in states])
    idx = rng.choice(len(states), size=500, p=p / p.sum())
    assert s.sequence == [states[i] for i in idx]


def test_cli_reports_unresolved_shots(tmp_path, capsys):
    from sequoia.fock.cli import main
    from sequoia.fock.experiment import save_experiment

    mix = InputMixture(((0.25, PureState.fock([(0, 0)])),))
    path = tmp_path / "t.yaml"
    save_experiment(Experiment(Circuit(1), [0], input_mixture=mix, shots=50, seed=1), path)
    assert main([str(path)]) == 0
    assert "unresolved" in capsys.readouterr().out


# RR03: clock maps


def _tags():
    t = np.arange(1, 2001) * 1e-6
    return {"a": t, "b": t}, 0.01


SPEC = [
    {
        "type": "coincidences",
        "name": "ab",
        "channels": ["a", "b"],
        "window": "1 ns",
        "center": "0 ns",
    }
]


def test_empty_clock_map_is_not_the_legacy_list(tmp_path):
    tags, duration = _tags()
    tt = TimeTags(tags, duration, repetition_rates=[50e6, 80e6], channel_clocks={})
    assert tt.clock == {}
    out = corr.analyse_tags(tags, duration, SPEC, tt.clock)["ab"]
    assert out["accidental_model"] == "singles" and out["timing"] == "unknown"
    assert "timing_note" in out
    path = tmp_path / "t.npz"
    corr.write_tags(path, tags, duration, tt.clock)
    record = corr.read_tag_record(path)
    assert record.repetition_rates == {} and record.has_clock
    legacy = TimeTags(tags, duration, repetition_rates=[50e6, 80e6])
    assert legacy.clock == [50e6, 80e6]
    out = corr.analyse_tags(tags, duration, SPEC, legacy.clock)["ab"]
    assert out["timing"] == "legacy" and out["accidental_model"] == "sideband"


@pytest.mark.parametrize(
    "clock, ids, timing",
    [
        ({"a": 50e6, "b": 50e6}, {"a": "s", "b": "s"}, "pulsed"),
        ({"a": 50e6}, {"a": "s"}, "unknown"),
        ({"a": 50e6, "b": 80e6}, {"a": "s", "b": "t"}, "unequal"),
        ({"a": 50e6, "b": 50e6}, {"a": "s", "b": "t"}, "unsynchronized"),
        ({"a": 50e6, "b": 0.0}, {"a": "s"}, "mixed"),
        ({"a": 0.0, "b": 0.0}, {}, "cw"),
    ],
)
def test_clock_cases(clock, ids, timing, tmp_path):
    tags, duration = _tags()
    out = corr.analyse_tags(tags, duration, SPEC, clock, ids)["ab"]
    assert out["timing"] == timing
    assert out["accidental_model"] == ("sideband" if timing == "pulsed" else "singles")
    assert ("timing_note" in out) == (timing not in ("pulsed", "cw"))
    path = tmp_path / "t.npz"
    corr.write_tags(path, tags, duration, clock, clock_ids=ids)
    record = corr.read_tag_record(path)
    assert record.repetition_rates == clock and record.clock_ids == ids
    again = corr.analyse_tags(
        record.channels, duration, SPEC, record.repetition_rates, record.clock_ids
    )["ab"]
    assert again["timing"] == timing


def test_legacy_file_without_clock(tmp_path):
    tags, duration = _tags()
    path = tmp_path / "old.npz"
    np.savez(path, duration=duration, **tags)
    record = corr.read_tag_record(path)
    assert record.repetition_rates is None and not record.has_clock


def test_monte_carlo_clock_identities():
    from test_review_round1_engines import doubled, hbt

    tags = simulate_timetags(setup_from_dict(hbt(0.02, pulsed=True)), LIB, 0.005, seed=3)
    pulsed = {k: v for k, v in tags.clock.items() if v}
    assert pulsed and set(tags.clock_ids) == set(pulsed)
    assert len(set(tags.clock_ids.values())) == 1
    both = simulate_timetags(
        setup_from_dict(doubled(hbt(0.02, pulsed=True), second_pulsed=True)), LIB, 0.002, seed=3
    )
    ids = set(both.clock_ids.values())
    assert len(ids) == 2
    a = next(k for k, v in both.clock_ids.items() if v == min(ids))
    b = next(k for k, v in both.clock_ids.items() if v == max(ids))
    spec = [dict(SPEC[0], channels=[a, b])]
    out = corr.analyse_tags(both.channels, both.duration, spec, both.clock, both.clock_ids)
    assert out["ab"]["timing"] in ("unequal", "unsynchronized")


# RR04: CHSH inference


def _bell(**kw):
    setup = load_setup(ROOT / "setups" / "bell_test.yaml")
    return run_chsh(
        setup,
        LIB,
        "settings:hwp_a.angle",
        "settings:hwp_b.angle",
        [0.0, 22.5],
        [11.25, 33.75],
        **kw,
    )


@pytest.mark.parametrize("t", [0.0, -1.0, math.nan, math.inf])
def test_chsh_rejects_invalid_duration(t):
    with pytest.raises(SweepError):
        _bell(integration_time=t)


def test_chsh_baseline_raw_and_corrected():
    raw = _bell()
    assert raw.quantity == "total_hz"
    cor = _bell(quantity="true_hz")
    assert cor.s == pytest.approx(2 * math.sqrt(2) * 0.97, rel=2e-3)
    assert raw.s_corrected == pytest.approx(cor.s) and cor.s_raw == pytest.approx(raw.s)
    assert raw.s <= cor.s + 1e-12
    assert cor.s_sigma >= raw.s_sigma
    no_sb = _bell(quantity="true_hz", sidebands=0)
    assert no_sb.s_sigma <= cor.s_sigma
    assert raw.violates is True and raw.significance > 3
    assert "per setting pair" in format_chsh(raw)


def test_chsh_relabelled_outputs_keep_the_assessment():
    normal = _bell()
    flipped = _bell(outcomes=(("tr", "tt"), ("rr", "rt")))
    assert flipped.s == pytest.approx(-normal.s)
    assert flipped.violates == normal.violates
    assert flipped.significance == pytest.approx(normal.significance)


def test_chsh_without_uncertainty_makes_no_claim():
    from sequoia.analysis.chsh import ChshResult

    r = ChshResult(
        np.zeros((2, 2, 2, 2)), np.zeros((2, 2)), 2.7, None, "total_hz", 1.0, (0, 1), (0, 1), "deg"
    )
    assert r.violates is None and r.significance is None
    assert "undefined" in format_chsh(r)
    nan = ChshResult(
        np.zeros((2, 2, 2, 2)),
        np.zeros((2, 2)),
        2.7,
        math.nan,
        "total_hz",
        1.0,
        (0, 1),
        (0, 1),
        "deg",
    )
    assert nan.violates is None


def test_chsh_sigma_matches_count_ensembles():
    from sequoia.analysis.metrics import chsh_s, correlation_from_rates

    t = 1e-3
    for quantity in ("total_hz", "true_hz"):
        r = _bell(quantity=quantity, integration_time=t)
        total = r.rates if quantity == "total_hz" else None
        if total is None:
            raw = _bell(integration_time=t)
            total = raw.rates
            acc = raw.rates - r.rates
        else:
            acc = np.zeros_like(total)
        rng = np.random.default_rng(11)
        values = []
        for _ in range(4000):
            counts = rng.poisson(total * t).astype(float)
            if quantity == "true_hz":
                side = rng.poisson(acc * t, size=(2,) + acc.shape).mean(axis=0)
                counts = counts - side
            e = np.array(
                [
                    [correlation_from_rates(*counts[i, j].ravel()) for j in range(2)]
                    for i in range(2)
                ]
            )
            values.append(chsh_s(e[0, 0], e[0, 1], e[1, 0], e[1, 1]))
        assert np.std(values) == pytest.approx(r.s_sigma, rel=0.08)


# RR09 / RR10: Fock sources across engines, dead-time regime


def _fock_hom():
    return load_setup(ROOT / "templates" / "fock_hom.yaml")


def test_fock_source_is_not_reported_as_pump_leakage():
    from sequoia.engines.rates import simulate_rates

    r = simulate_rates(_fock_hom(), LIB)
    codes = {i.code for i in r.issues}
    assert "pump_leakage" not in codes and "leakage_saturation" not in codes
    assert "fock_source_mean_flux" in codes
    for d in r.detectors.values():
        assert d["leakage_hz"] == pytest.approx(0.0, abs=1e-6)
        assert d["fock_hz"] > 0


def test_gaussian_and_monte_carlo_refuse_fock_light_unless_asked():
    from sequoia.engines.gaussian import UnsupportedSourceError, simulate_gaussian

    with pytest.raises(UnsupportedSourceError):
        simulate_gaussian(_fock_hom(), LIB)
    with pytest.raises(UnsupportedSourceError):
        simulate_timetags(_fock_hom(), LIB, 1e-4, seed=1)
    tags = simulate_timetags(_fock_hom(), LIB, 1e-5, seed=1, fock_as_poisson=True)
    assert any(i.code == "fock_as_poisson" for i in tags.issues)


def test_dead_time_regime_is_flagged_before_lab_counts():
    from sequoia.fock.from_setup import build_experiment_from_setup, run_setup_experiment

    se = build_experiment_from_setup(_fock_hom(), LIB)
    res = run_setup_experiment(se)
    assert set(res.outside_dilute_regime) == set(se.detector_ids)
    assert any("dilute regime" in m for m in res.messages)
    for d, load in res.dead_time_load.items():
        k = se.detector_ids.index(d)
        p_click = sum(p for s, p in res.detected.items() if s[k] > 0)
        assert load == pytest.approx(p_click * se.rate_hz * se.dead_times[d])


# U11 / R13: bounded sampling, atomic circuit files


def test_histogram_sampling_scales_and_keeps_the_tail():
    d = Distribution({(1,): 0.25, (0,): 0.5}, 1, omitted_weight=0.25)
    s = sample(d, 10**9, seed=3, sequence=False)
    assert s.sequence == [] and s.shots == 10**9
    assert sum(s.counts.values()) + s.unresolved == 10**9
    assert s.unresolved / 1e9 == pytest.approx(0.25, abs=1e-3)
    with pytest.raises(FockError, match="counts only"):
        sample(d, 10**8, seed=3)


def test_json_circuit_write_failure_keeps_the_old_file(tmp_path, monkeypatch):
    from sequoia.fock.experiment import save_experiment

    path = tmp_path / "c.json"
    save_experiment(Experiment(Circuit(1), [1]), path)
    before = path.read_text()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        save_experiment(Experiment(Circuit(2), [1, 1]), path)
    monkeypatch.undo()
    assert path.read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.json"]


# U07: photon-number observables in sweeps


def test_fock_phase_fringe_sweep_matches_single_points():
    from sequoia.analysis.sweep import fock_metrics, linspace_axis, run_sweep, with_values

    setup = _fock_hom()
    axis = linspace_axis("settings:phase.phase", 0.0, 90.0, 4, "deg")
    res = run_sweep(setup, LIB, [axis], fock={"detectors_mode": "pnr"})
    name = "fock:P(spad_1=1,spad_2=1)"
    assert name in res.data and not res.failures
    for i, v in enumerate(axis.values):
        point = with_values(setup, [(axis.target, float(v), "deg")])
        assert res.data[name][i] == pytest.approx(
            fock_metrics(point, LIB, detectors_mode="pnr")[name]
        )
    fringe = res.data[name]
    assert fringe[0] == pytest.approx(fringe.max()) and fringe[0] > 0.2
    assert fringe[-1] == pytest.approx(0.0, abs=1e-4)
    assert "fock:P(spad_1=2,spad_2=0)" in res.data
    assert not np.isnan(res.data["fock:P(spad_1=2,spad_2=0)"]).any()
    assert "fock:P(spad_1=1,spad_2=1)" in res.to_csv()


def test_fock_delay_scan_is_refused_with_a_reason():
    from sequoia.analysis.sweep import linspace_axis, run_sweep

    setup = _fock_hom()
    link = setup.links[0].id
    axis = linspace_axis(f"link:{link}.length", 0.1, 0.2, 3, "m")
    with pytest.raises(SweepError, match="temporal"):
        run_sweep(setup, LIB, [axis], fock={})
