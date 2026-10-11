import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis import correlation as corr
from sequoia.engines.cache import Cancelled, ResourceLimitError
from sequoia.engines.montecarlo import (
    _emission_times,
    _SourceModel,
    compare_with_rates,
    dead_time_filter,
    simulate_timetags,
)
from sequoia.engines.rates import simulate_rates
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


def example(name, scale=None, spad=None, tagger=None):
    raw = setup_to_dict(load_setup(ROOT / "setups" / f"{name}.yaml"))
    for inst in raw["instances"]:
        part = inst["part"]
        if scale is not None and part.startswith("sources/laser"):
            inst.setdefault("overrides", {})["power"] = f"{10 * scale} mW"
        if scale is not None and part.startswith("nonlinear/entangled"):
            inst.setdefault("settings", {})["pump_power"] = f"{1 * scale} mW"
        if spad and part.startswith("detectors/"):
            inst.setdefault("overrides", {}).update(spad)
        if tagger and part.startswith("electronics/"):
            inst.setdefault("overrides", {}).update(tagger)
    return setup_from_dict(raw)


def check_all(tags, setup, rates):
    analyses = corr.analyse_tags(
        tags.channels, tags.duration, setup.analyses, tags.repetition_rates
    )
    rows = compare_with_rates(tags, analyses, rates)
    assert rows
    bad = [r for r in rows if not r.ok]
    assert not bad, [(r.name, r.quantity, r.monte_carlo, r.expected, round(r.z, 2)) for r in bad]
    return analyses


def test_dead_time_filter_matches_brute_force():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1.0, 5000))
    dead = 5e-4
    kept, last = [], -np.inf
    for x in t:
        if x - last >= dead:
            kept.append(x)
            last = x
    assert np.array_equal(dead_time_filter(t, dead), np.array(kept))
    assert dead_time_filter(t, 0.0) is t
    assert len(dead_time_filter(np.array([0.0, 1e-9, 2e-9]), 1e-6)) == 1


@pytest.mark.parametrize("rate", [1e5, 1e6, 5e6])
def test_dead_time_saturation_follows_non_paralyzable_formula(rate):
    rng = np.random.default_rng(2)
    duration, dead = 0.5, 100e-9
    t = np.sort(rng.uniform(0, duration, rng.poisson(rate * duration)))
    out = len(dead_time_filter(t, dead)) / duration
    expected = rate / (1.0 + rate * dead)
    assert out == pytest.approx(expected, rel=4 / math.sqrt(out * duration))


def test_detector_saturation_in_a_setup(qtbot=None):
    dark = {"dark_count_rate": "4 MHz", "dead_time": "100 ns"}
    setup = example("hbt_heralded", scale=0.001, spad=dark, tagger={"jitter_fwhm": "0 ps"})
    rates = simulate_rates(setup, LIB)
    tags = simulate_timetags(setup, LIB, 0.3, seed=4)
    for name, ch in rates.channels.items():
        n = len(tags.channels[name])
        sigma = math.sqrt(n) / tags.duration
        assert abs(n / tags.duration - ch["singles_hz"]) < 3.5 * sigma
        assert ch["survival"] < 0.75


@pytest.mark.parametrize(
    ("name", "scale", "duration"),
    [("hbt_heralded", 0.02, 60.0), ("bell_test", 0.05, 30.0), ("hom", 0.02, 60.0)],
)
def test_monte_carlo_agrees_with_rate_engine(name, scale, duration):
    setup = example(name, scale=scale)
    rates = simulate_rates(setup, LIB)
    tags = simulate_timetags(setup, LIB, duration, seed=11)
    analyses = check_all(tags, setup, rates)
    assert analyses


def test_bell_correlations_chsh(tmp_path):
    setup = example("bell_test", scale=0.05)
    tags = simulate_timetags(setup, LIB, 60.0, seed=5)
    a = corr.analyse_tags(tags.channels, tags.duration, setup.analyses)
    tt, tr, rt, rr = (a[n]["true_hz"] for n in ("tt", "tr", "rt", "rr"))
    e = (tt + rr - tr - rt) / (tt + rr + tr + rt)
    assert e == pytest.approx(0.97 * math.cos(math.radians(2 * 22.5)), abs=0.03)


def test_hom_dip_follows_rate_engine():
    base = example("hom", scale=0.02)
    for length in ("1 m", "1.0004 m"):
        raw = setup_to_dict(base)
        for link in raw["links"]:
            if link["id"] == "delay_arm":
                link["length"] = length
        setup = setup_from_dict(raw)
        rates = simulate_rates(setup, LIB)
        tags = simulate_timetags(setup, LIB, 40.0, seed=3)
        check_all(tags, setup, rates)


def test_seed_makes_tags_reproducible():
    setup = example("bell_test", scale=0.01)
    a = simulate_timetags(setup, LIB, 2.0, seed=9)
    b = simulate_timetags(setup, LIB, 2.0, seed=9)
    c = simulate_timetags(setup, LIB, 2.0, seed=10)
    for name in a.channels:
        assert np.array_equal(a.channels[name], b.channels[name])
    assert any(len(a.channels[n]) != len(c.channels[n]) for n in a.channels)


def test_tags_are_sorted_and_inside_the_record():
    setup = example("hbt_heralded", scale=0.01)
    tags = simulate_timetags(setup, LIB, 1.0, seed=1)
    for name, t in tags.channels.items():
        assert np.all(np.diff(t) >= 0), name
        assert t.min() > -1e-6 and t.max() < 1.0 + 1e-6


def test_limits_and_cancellation():
    setup = example("hbt_heralded", scale=0.01)
    with pytest.raises(ValueError):
        simulate_timetags(setup, LIB, 0.0)
    with pytest.raises(ResourceLimitError):
        simulate_timetags(setup, LIB, 1.0, max_events=10)
    with pytest.raises(Cancelled):
        simulate_timetags(setup, LIB, 1.0, should_cancel=lambda: True)


def test_afterpulsing_follows_the_branching_estimate():
    p, tau, dead = 0.2, 200e-9, 50e-9
    ap = {"afterpulse_probability": p, "afterpulse_tau": "200 ns"}
    plain = example("hbt_heralded", scale=0.02)
    with_ap = example("hbt_heralded", scale=0.02, spad=ap)
    t0 = simulate_timetags(plain, LIB, 20.0, seed=1)
    t1 = simulate_timetags(with_ap, LIB, 20.0, seed=1)
    ratio = len(t1.channels["tagger.ch2"]) / len(t0.channels["tagger.ch2"])
    survive = 1.0 - math.exp(-dead / tau)
    expected = 1.0 / (1.0 - p * (1.0 - survive))
    assert ratio == pytest.approx(expected, abs=0.01)
    delays = np.diff(t1.channels["tagger.ch2"])
    assert np.min(delays) >= dead - 3e-9


def test_pulsed_source_structure_and_thermal_statistics():
    raw = setup_to_dict(example("hbt_heralded", scale=0.5))
    for inst in raw["instances"]:
        if inst["part"].startswith("sources/laser"):
            inst["overrides"] = {
                "power": "1 mW",
                "repetition_rate": "50 MHz",
                "pulse_width": "1 ps",
            }
            inst.pop("settings", None)
    setup = setup_from_dict(raw)
    tags = simulate_timetags(setup, LIB, 1.0, seed=3)
    assert tags.repetition_rates and tags.repetition_rates[0] == pytest.approx(50e6)
    phase = np.mod(tags.channels["tagger.ch2"] + 9.8e-9, 20e-9)
    assert np.std(np.minimum(phase, 20e-9 - phase)) < 5e-9


def test_negative_binomial_pair_number():
    rng = np.random.default_rng(0)
    model = _SourceModel("s", 5e6, 50e6, [], {}, {}, {})
    duration = 0.4
    mu = model.rate / model.repetition_rate
    for k in (1.0, 4.0):
        times = _emission_times(rng, model, duration, schmidt_number=k)
        pulses = np.round(times * model.repetition_rate).astype(np.int64)
        n_pulses = int(duration * model.repetition_rate)
        counts = np.bincount(pulses, minlength=n_pulses)
        assert counts.mean() == pytest.approx(mu, rel=0.02)
        g2 = (counts * (counts - 1)).mean() / mu**2
        assert g2 == pytest.approx(1.0 + 1.0 / k, rel=0.08)
    poisson = _emission_times(rng, model, duration, None)
    counts = np.bincount(
        np.round(poisson * model.repetition_rate).astype(np.int64),
        minlength=int(duration * model.repetition_rate),
    )
    assert (counts * (counts - 1)).mean() / mu**2 == pytest.approx(1.0, rel=0.05)


def test_correlation_primitives():
    rng = np.random.default_rng(3)
    a = np.sort(rng.uniform(0, 5, 20000))
    b = np.sort(
        np.concatenate([a[::2] - 3.2e-7 + rng.normal(0, 1e-9, 10000), rng.uniform(0, 5, 20000)])
    )
    peak = corr.find_peak(a, b, 4e-9, 1e-6)
    assert peak == pytest.approx(3.2e-7, abs=1e-9)
    n = corr.count_in_window(a, b, 10e-9, peak)
    assert n >= 10000 * 0.95
    centres, counts = corr.delay_histogram(a, b, 1e-9, 5e-7, 0.0)
    brute = 0
    for x in a[:200]:
        brute += np.sum(np.abs(x - b) < 5e-7)
    sub = np.zeros(len(centres), dtype=int)
    for d in corr.delays_within(a[:200], b, 5e-7):
        sub += np.histogram(d, bins=np.linspace(-5e-7, 5e-7, len(centres) + 1))[0]
    assert sub.sum() == brute
    assert counts.sum() >= n


def test_uncorrelated_streams_give_unit_heralded_g2():
    rng = np.random.default_rng(4)
    duration = 20.0
    h = np.sort(rng.uniform(0, duration, 100000))
    a = np.sort(rng.uniform(0, duration, 400000))
    b = np.sort(rng.uniform(0, duration, 400000))
    result = corr.heralded_g2(h, a, b, duration, 20e-6, 0.0, 0.0)
    assert result["g2_heralded"] == pytest.approx(1.0, abs=4 * result["g2_sigma"])


def test_tag_file_round_trip(tmp_path):
    channels = {"c1": np.array([0.1, 0.2, 0.3]), "c2": np.array([0.15, 0.4])}
    path = tmp_path / "tags.npz"
    corr.write_tags(path, channels, 1.0)
    back, duration = corr.read_tags(path)
    assert duration == 1.0 and np.array_equal(back["c1"], channels["c1"])
    csv_path = tmp_path / "tags.csv"
    csv_path.write_text("channel,time_s\nc1,0.3\nc1,0.1\nc2,0.4\n")
    back, duration = corr.read_tags(csv_path)
    assert duration is None and list(back["c1"]) == [0.1, 0.3]
    with pytest.raises(corr.CorrelationError):
        corr.analyse_tags(
            back, 1.0, [{"type": "coincidences", "channels": ["c1", "zz"], "window": 1e-9}]
        )
