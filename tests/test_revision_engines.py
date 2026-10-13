import copy
import math
from pathlib import Path

import numpy as np
import pytest

import sequoia.components.builtin  # noqa: F401
from sequoia.analysis import correlation as corr
from sequoia.analysis.metrics import ratio
from sequoia.engines.cache import ResourceLimitError
from sequoia.engines.gaussian import (
    IncompatibleWindowsError,
    _pnr,
    prepare_gaussian,
    simulate_gaussian,
)
from sequoia.library.store import Library
from sequoia.setup.io import load_setup, setup_from_dict, setup_to_dict

ROOT = Path(__file__).resolve().parents[1]
LIB = Library(ROOT / "library")


def hbt(scale=0.02, pulsed=False):
    raw = setup_to_dict(load_setup(ROOT / "setups" / "hbt_heralded.yaml"))
    for inst in raw["instances"]:
        if inst["part"].startswith("sources/laser"):
            overrides = inst.setdefault("overrides", {})
            overrides["power"] = f"{10 * scale} mW"
            if pulsed:
                overrides.update(repetition_rate="50 MHz", pulse_width="1 ps")
    return raw


def doubled(raw, second_pulsed):
    out = copy.deepcopy(raw)
    extra = copy.deepcopy(raw)
    for inst in extra["instances"]:
        inst["id"] = "b_" + inst["id"]
        if inst["part"].startswith("sources/laser") and second_pulsed:
            inst.setdefault("overrides", {}).update(repetition_rate="80 MHz", pulse_width="1 ps")
    ids = {i["id"][2:] for i in extra["instances"]}

    def rename(ref):
        head, _, tail = ref.partition(".")
        return ("b_" + head if head in ids else head) + ("." + tail if tail else "")

    for link in extra["links"]:
        link["id"] = "b_" + link["id"]
        link["from"], link["to"] = rename(link["from"]), rename(link["to"])
    for spec in extra["analyses"]:
        spec["name"] = "b_" + spec["name"]
        for key in ("channels",):
            if key in spec:
                spec[key] = [rename(c) for c in spec[key]]
        for key in ("herald", "a", "b"):
            if key in spec:
                spec[key] = rename(spec[key])
    out["instances"] += extra["instances"]
    out["links"] += extra["links"]
    out["analyses"] += extra["analyses"]
    return out


class Thermal:
    def __init__(self, mean):
        self.mean = mean

    def no_click(self, lam, gain_scale=1.0):
        return 1.0 / (1.0 + self.mean * lam[0])


class TwoThermal:
    def __init__(self, means):
        self.means = means

    def no_click(self, lam, gain_scale=1.0):
        return np.prod([1.0 / (1.0 + m * x) for m, x in zip(self.means, lam, strict=False)])


@pytest.mark.parametrize("mean", [0.01, 1.0, 10.0, 50.0])
@pytest.mark.parametrize("cutoff", [4, 12, 30])
def test_pnr_matches_geometric_distribution_without_aliasing(mean, cutoff):
    probs, meta = _pnr(Thermal(mean), ["a"], ("a",), 1.0, cutoff)
    n = np.arange(cutoff + 1)
    exact = (1.0 / (1.0 + mean)) * (mean / (1.0 + mean)) ** n
    assert np.allclose(probs, exact, atol=1e-9)
    assert meta["converged"]
    assert meta["retained"] + meta["tail"] == pytest.approx(1.0)
    assert meta["tail"] == pytest.approx((mean / (1 + mean)) ** (cutoff + 1), abs=1e-8)


def test_pnr_two_channel_joint_is_product_of_geometrics():
    probs, meta = _pnr(TwoThermal([2.0, 0.5]), ["a", "b"], ("a", "b"), 1.0, 8)
    n = np.arange(9)
    pa = (1 / 3) * (2 / 3) ** n
    pb = (1 / 1.5) * (0.5 / 1.5) ** n
    assert np.allclose(probs, np.outer(pa, pb), atol=1e-9)
    assert meta["tail"] == pytest.approx(1 - np.outer(pa, pb).sum(), abs=1e-9)


def test_pnr_guards_runaway_joint_grids():
    with pytest.raises(ResourceLimitError):
        _pnr(TwoThermal([400.0, 400.0]), ["a", "b"], ("a", "b"), 1.0, 12)


def test_engine_pnr_reports_tail_and_cutoff_independent_vacuum():
    setup = setup_from_dict(hbt(0.02, pulsed=True))
    low = simulate_gaussian(setup, LIB, gain_scale=3000.0, pnr=[("tagger.ch2",)], pnr_cutoff=6)
    high = simulate_gaussian(setup, LIB, gain_scale=3000.0, pnr=[("tagger.ch2",)], pnr_cutoff=20)
    a, b = low.pnr[("tagger.ch2",)], high.pnr[("tagger.ch2",)]
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert a[0] == pytest.approx(1.0 - low.probability("tagger.ch2"), abs=1e-9)
    meta = low.pnr_info[("tagger.ch2",)]
    assert meta["retained"] + meta["tail"] == pytest.approx(1.0)
    assert meta["tail"] > high.pnr_info[("tagger.ch2",)]["tail"]


def test_different_source_windows_are_rejected_not_rescaled():
    raw = hbt(0.02)
    single = simulate_gaussian(setup_from_dict(raw), LIB)
    with pytest.raises(IncompatibleWindowsError):
        simulate_gaussian(setup_from_dict(doubled(raw, second_pulsed=True)), LIB)
    same = simulate_gaussian(setup_from_dict(doubled(raw, second_pulsed=False)), LIB)
    assert same.rate("tagger.ch1") == pytest.approx(single.rate("tagger.ch1"), rel=1e-9)
    assert same.rate("b_tagger.ch1") == pytest.approx(single.rate("tagger.ch1"), rel=1e-9)


def test_gaussian_without_tagged_detectors_returns_empty_result():
    raw = hbt()
    keep = {"pump", "pump_lens", "crystal"}
    raw["instances"] = [i for i in raw["instances"] if i["id"] in keep]
    raw["links"] = [
        lk
        for lk in raw["links"]
        if lk["from"].split(".")[0] in keep and lk["to"].split(".")[0] in keep
    ]
    raw["analyses"] = []
    prepared = prepare_gaussian(setup_from_dict(raw), LIB)
    result = prepared.evaluate()
    assert result.channels == []
    assert any(i.code == "gaussian_no_detectors" for i in result.issues)


def test_gain_scale_scales_leakage_but_not_dark_counts():
    prepared = prepare_gaussian(setup_from_dict(hbt(0.02)), LIB)
    model = prepared.model
    assert np.all(model.background(1.0) >= model.dark)
    assert np.allclose(model.background(10.0) - model.dark, 10.0 * model.leakage)
    assert np.allclose(model.background(0.0), model.dark)


def test_car_conventions():
    assert math.isnan(ratio(0.0, 0.0))
    assert math.isinf(ratio(3.0, 0.0))
    assert ratio(6.0, 2.0) == 3.0
    empty = corr.coincidences(np.array([]), np.array([]), 1.0, 1e-9, 0.0)
    assert math.isnan(empty["car"]) and math.isnan(empty["car_total"])


def pulsed_tags(seed=42):
    rng = np.random.default_rng(seed)
    rep, duration = 1e7, 0.01
    pulses = np.arange(int(rep * duration)) / rep
    tags = {name: pulses[rng.random(len(pulses)) < 0.1] for name in ("a", "b")}
    spec = [{"name": "ab", "type": "coincidences", "channels": ["a", "b"], "window": "1 ns"}]
    return tags, duration, rep, spec


def test_pulse_clock_survives_export_and_reload(tmp_path):
    tags, duration, rep, spec = pulsed_tags()
    spec[0]["center"] = "0 ns"
    live = corr.analyse_tags(tags, duration, spec, {"a": rep, "b": rep})["ab"]
    path = tmp_path / "t.npz"
    corr.write_tags(path, tags, duration, {"a": rep, "b": rep}, {"seed": 42})
    record = corr.read_tag_record(path)
    assert record.format_version == 2 and record.meta["seed"] == 42
    reloaded = corr.analyse_tags(record.channels, record.duration, spec, record.repetition_rates)
    for key in ("counts", "accidental_hz", "true_hz", "car"):
        assert reloaded["ab"][key] == pytest.approx(live[key])
    assert reloaded["ab"]["accidental_model"] == "sideband"
    legacy = corr.analyse_tags(record.channels, record.duration, spec)["ab"]
    assert legacy["accidental_model"] == "singles"


def test_reading_version_one_files_and_reserved_names(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, duration=1.0, a=np.array([0.1]), b=np.array([0.2]))
    record = corr.read_tag_record(path)
    assert record.format_version == 1 and not record.has_clock
    assert set(record.channels) == {"a", "b"}
    with pytest.raises(corr.CorrelationError):
        corr.write_tags(tmp_path / "x.npz", {"duration": np.zeros(1)}, 1.0)


@pytest.mark.parametrize("delay", [0.0, 10e-9, 100e-9, 3e-6])
def test_herald_peak_found_at_any_delay(delay):
    herald = np.arange(100) * 1e-4 + 1e-4
    tags = {"h": herald, "a": herald + delay, "b": herald + delay + 1e-9}
    spec = [
        {"type": "heralded_g2", "name": "g", "herald": "h", "a": "a", "b": "b", "window": "4 ns"}
    ]
    out = corr.analyse_tags(tags, 0.011, spec)["g"]
    assert out["triples"] == 100
    assert out["peak_found_a"] and out["peak_found_b"]
    assert out["center_a_s"] == pytest.approx(delay, abs=1e-9)
    explicit = dict(spec[0], center_a=f"{delay} s", center_b=f"{delay + 1e-9} s")
    fixed = corr.analyse_tags(tags, 0.011, [explicit])["g"]
    assert fixed["triples"] == 100


def test_peak_not_found_is_reported():
    rng = np.random.default_rng(1)
    a = np.sort(rng.random(500))
    b = np.sort(rng.random(500))
    out = corr.coincidences(a, b, 1.0, 1e-9, "auto")
    assert out["peak_found"] is False


def test_write_failures_leave_previous_file(tmp_path):
    path = tmp_path / "t.npz"
    corr.write_tags(path, {"a": np.array([1.0])}, 1.0)
    before = path.read_bytes()

    class Boom:
        def __array__(self, *args, **kwargs):
            raise OSError("disk full")

    with pytest.raises(OSError):
        corr.write_tags(path, {"a": Boom()}, 1.0)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp")) and len(list(tmp_path.iterdir())) == 1
    with pytest.raises(OSError):
        corr.write_tags(tmp_path / "missing" / "t.npz", {"a": np.array([1.0])}, 1.0)


def test_monte_carlo_clock_is_per_channel():
    from sequoia.engines.montecarlo import simulate_timetags

    tags = simulate_timetags(setup_from_dict(hbt(0.02, pulsed=True)), LIB, 0.01, seed=3)
    assert set(tags.clock) >= {"tagger.ch1"}
    assert all(v == pytest.approx(50e6) for v in tags.clock.values())
    cw = simulate_timetags(setup_from_dict(hbt(0.02)), LIB, 0.01, seed=3)
    assert not cw.clock
