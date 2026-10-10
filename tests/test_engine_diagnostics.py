import math
import os
import threading
from dataclasses import replace

import numpy as np
import pytest
from optics import fs, inst, put
from scipy import constants
from test_rates import (
    DETS,
    DNU,
    FREE,
    IN,
    OUT,
    PAIRS,
    TAGS,
    cable,
    codes,
    coincidence,
    cw_pair,
    det,
    mode,
    mzi_run,
    pulsed_pair,
    signal_arm_hbt_setup,
    two_arm,
    werner_pulsed_source,
)
from test_rates import lib as rates_lib  # noqa: F401

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import Component, Params, PortSpec, register
from sequoia.data import ExtrapolationWarning
from sequoia.engines import rates as rates_module
from sequoia.engines.cache import Cancelled, RunCache, record_warnings
from sequoia.engines.classical import simulate
from sequoia.engines.rates import simulate_rates
from sequoia.engines.report import _analysis_lines, format_rates_report, format_report
from sequoia.errors import SetupError
from sequoia.library.store import Library
from sequoia.light.detection import TaggerModel
from sequoia.light.pairs import PairSource
from sequoia.setup.io import setup_from_dict

C = constants.c


@pytest.fixture
def lib(request):
    return request.getfixturevalue("rates_lib")


def hist_spec(kind="g2", channels=("tag.ch2", "tag.ch3")):
    return {"name": "g", "type": kind, "channels": list(channels), "range": "100 ps", "bins": 20}


def g2_setup():
    return signal_arm_hbt_setup([hist_spec()])


def heralded_spec(window="1 ns"):
    return {
        "name": "h",
        "type": "heralded_g2",
        "herald": "tag.ch1",
        "a": "tag.ch2",
        "b": "tag.ch3",
        "window": window,
    }


def prime(pair, eta=(0.3, 0.5, 0.4)):
    PAIRS["src"] = pair
    DETS["ds"], DETS["da"], DETS["db"] = det(eta[0]), det(eta[1]), det(eta[2])
    TAGS["tag"] = TaggerModel()


def test_pulsed_gram_matrices_are_guarded_with_an_estimate_of_the_memory(lib):
    pair, _ = pulsed_pair(1e5, 80e6, 4e9, 2e9)
    prime(pair)
    setup = signal_arm_hbt_setup([hist_spec(), heralded_spec(), coincidence("c", "1 ns")])
    ok = simulate_rates(setup, lib)
    assert "memory_limit" not in codes(ok)
    assert ok.analyses["g"]["g2_per_pulse"] > 1.0
    r = simulate_rates(setup, lib, max_memory_bytes=5e6)
    limit = [i for i in r.issues if i.code == "memory_limit"]
    assert len(limit) == 2
    assert all("GiB" in i.message and "memory" in i.message for i in limit)
    assert r.analyses["g"]["g2_per_pulse"] is None
    assert "h" not in r.analyses
    assert r.analyses["c"]["true_hz"] > 0


def test_gram_space_choice_does_not_change_the_moments(lib, monkeypatch):
    pair = werner_pulsed_source(0.5)
    prime(pair)
    setup = signal_arm_hbt_setup([hist_spec(), heralded_spec("1 ns")])
    results = {}
    for space in ("row", "col"):
        monkeypatch.setattr(rates_module, "GRAM_SPACE", space)
        results[space] = simulate_rates(setup, lib)
    g_row, g_col = results["row"].analyses["g"], results["col"].analyses["g"]
    assert g_row["g2_per_pulse"] == pytest.approx(g_col["g2_per_pulse"], rel=1e-9)
    for role in ("signal", "idler"):
        for key in ("schmidt_a", "schmidt_b", "trace_a"):
            a, b = g_row["by_role"][role][key], g_col["by_role"][role][key]
            assert a == pytest.approx(b, rel=1e-9) or (math.isnan(a) and math.isnan(b))
    h_row, h_col = results["row"].analyses["h"], results["col"].analyses["h"]
    assert h_row["g2_heralded"] == pytest.approx(h_col["g2_heralded"], rel=1e-9)
    assert h_row["multimode_bunching"] == pytest.approx(h_col["multimode_bunching"], rel=1e-9)


def test_asymmetric_jsa_uses_the_smaller_space_and_matches_the_analytic_schmidt_number(lib):
    a, b = 4e9, 2e9
    pair, _ = pulsed_pair(1e5, 80e6, a, b, count=121, n_sum_count=41)
    prime(pair)
    r = simulate_rates(g2_setup(), lib)
    k = 0.5 * (a / b + b / a)
    g = r.analyses["g"]
    assert g["by_role"]["signal"]["schmidt_a"] == pytest.approx(k, rel=0.03)
    assert g["g2_per_pulse"] == pytest.approx(1 + 1 / k, rel=0.03)


def test_sparse_lattice_far_beyond_memory_fails_with_a_clear_message_not_a_crash(lib):
    n0 = 3_700_000
    n_s = np.array([n0, n0 + 200_000_000])
    amp = np.zeros((2, 2, 2, 1), dtype=complex)
    amp[0, 0, :, 0] = 1 / math.sqrt(2)
    pair = PairSource(DNU, n_s, np.array([2 * n0 + 200_000_000]), amp, 1e5, mode(), mode(), 405e-9)
    r = two_arm(
        lib,
        pair,
        det(0.5),
        det(0.5),
        analyses=[
            coincidence(),
            {
                "name": "h",
                "type": "histogram",
                "channels": ["tag.ch1", "tag.ch2"],
                "range": "1 ns",
                "bins": 10,
            },
        ],
    )
    messages = [i.message for i in r.issues if i.code == "memory_limit"]
    assert messages and all("GiB" in m for m in messages)
    assert "c" not in r.analyses and "h" not in r.analyses


def test_huge_amplitude_arrays_are_refused_before_allocation(lib):
    pair = cw_pair(1e5, flat=10)
    PAIRS["src"], DETS["ds"], DETS["di"] = pair, det(0.5), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "ds.in", "10 mm"),
                fs("b", "src.idler", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
        }
    )
    with pytest.raises(SetupError, match="memory"):
        simulate_rates(setup, lib, max_memory_bytes=1000)


def saturating_run(lib, rate, dead):
    return two_arm(
        lib,
        cw_pair(rate, flat=10),
        det(1.0, dead_time=dead),
        det(1.0, dead_time=dead),
        analyses=[coincidence(window="9 ns")],
    )


def test_dead_time_loss_above_ten_percent_is_reported_with_numbers(lib):
    r = saturating_run(lib, 1e7, 50e-9)
    issue = next(i for i in r.issues if i.code == "detector_saturation")
    assert "33.3%" in issue.message and "50 ns" in issue.message and "1e+07" in issue.message
    quiet = saturating_run(lib, 1e6, 50e-9)
    assert "detector_saturation" not in codes(quiet)
    border = saturating_run(lib, 2.2e6, 50e-9)
    loss = 1 - 1 / (1 + 2.2e6 * 50e-9)
    assert loss == pytest.approx(0.099, abs=1e-3)
    assert "detector_saturation" not in codes(border)


def leakage_run(lib, optical_density, dead):
    wl = 780e-9
    PAIRS["src"] = cw_pair(1e4, flat=10)
    DETS["ds"], DETS["di"] = det(0.5, dead_time=dead), det(0.5)
    TAGS["tag"] = TaggerModel()
    put(lib, "ndl", "NDFilter", optical_density=optical_density)
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("las", "laser"),
                inst("att", "ndl"),
                inst("bs", "bs"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "bs.in_a", "10 mm"),
                fs("b", "las.out", "att.in", "10 mm"),
                fs("c", "att.out", "bs.in_b", "10 mm"),
                fs("d", "bs.out_a", "ds.in", "10 mm"),
                fs("e", "src.idler", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
        }
    )
    photon = constants.h * C / wl
    return simulate_rates(setup, lib), 1e-3 * 10.0**-optical_density * 0.25 / photon


def test_classical_leakage_alone_saturating_a_detector_is_reported(lib):
    r, leak = leakage_run(lib, 8.0, 100e-9)
    assert r.channels["tag.ch1"]["leakage_hz"] == pytest.approx(leak, rel=1e-6)
    issue = next(i for i in r.issues if i.code == "leakage_saturation")
    loss = 100 * (1 - 1 / (1 + leak * 100e-9))
    assert f"{loss:.1f}%" in issue.message and "100 ns" in issue.message
    assert "detector_saturation" in codes(r)
    quiet, _ = leakage_run(lib, 10.0, 100e-9)
    assert "leakage_saturation" not in codes(quiet)
    assert "detector_saturation" not in codes(quiet)


def test_a_detuned_source_that_reaches_no_detector_is_flagged(lib):
    r = two_arm(lib, cw_pair(1e5, flat=10), det(1e-8), det(1e-8), analyses=[coincidence()])
    issue = next(i for i in r.issues if i.code == "no_phase_matched_pairs")
    assert "1e-06" in issue.message and "src" in issue.message
    ok = two_arm(lib, cw_pair(1e5, flat=10), det(0.5), det(0.5), analyses=[coincidence()])
    assert "no_phase_matched_pairs" not in codes(ok)


def test_car_is_nan_for_a_dark_source_and_reports_both_definitions(lib):
    r = two_arm(lib, cw_pair(0.0, flat=10), det(0.5), det(0.5), analyses=[coincidence()])
    assert math.isnan(r.analyses["c"]["car"]) and math.isnan(r.analyses["c"]["car_total"])
    r = two_arm(
        lib, cw_pair(1e6, flat=10), det(0.5), det(0.5), analyses=[coincidence(window="1 ns")]
    )
    a = r.analyses["c"]
    assert a["car"] == pytest.approx(a["true_hz"] / a["accidental_hz"])
    assert a["car_total"] == pytest.approx(a["car"] + 1.0)
    text = format_rates_report(r)
    assert "CAR = C_true/ACC" in text and "(C_true+ACC)/ACC" in text


def test_printed_peak_and_centre_are_rounded_to_a_picosecond():
    a = {
        "type": "coincidences",
        "channels": ["t.ch1", "t.ch2"],
        "window_s": 1e-9,
        "center_s": 1.082e-23,
        "peak_s": -5.374e-17,
        "window_fraction": 1.0,
        "coincidence_rate_hz": 1.0,
        "true_hz": 1.0,
        "accidental_hz": 0.1,
        "car": 10.0,
        "car_total": 11.0,
        "heralding_a_given_b": 0.5,
        "heralding_b_given_a": 0.5,
    }
    line = _analysis_lines("c", a)[0]
    assert "e-" not in line and "-0 " not in line
    assert "at 0 ns" in line and "peak 0 ns" in line


def test_multi_path_delay_beyond_half_period_is_flagged_single_path_is_not(lib):
    c_meter_per_ns = C * 1e-9
    r, _ = mzi_run(lib, 0.0, opl_gpl=(20 * c_meter_per_ns, 20 * c_meter_per_ns))
    assert "delay_aliasing" in codes(r)
    r, _ = mzi_run(lib, 0.0, opl_gpl=(3 * c_meter_per_ns, 3 * c_meter_per_ns))
    assert "delay_aliasing" not in codes(r)


def fibre_library(lib, laser_wavelength="405 nm", cutoff="730 nm"):
    put(lib, "laser405", "Laser", wavelength=laser_wavelength, power="1 mW", waist_radius="1 mm")
    put(lib, "smf", "OpticalFiber", mode_field_diameter="5 um", cutoff_wavelength=cutoff)
    put(lib, "cpl", "FiberCoupler", focal_length="9.7 mm")
    put(lib, "cpl_out", "FiberCoupler", focal_length="9.7 mm")


def fibre_setup(optical_density=0.0, with_pairs=True):
    instances = [
        inst("las", "laser405"),
        inst("att", "ndl"),
        inst("bs", "bs"),
        inst("cin", "cpl", {"mode": "to_fiber"}),
        inst("cout", "cpl_out", {"mode": "to_free"}),
        inst("ds", "det_s"),
        inst("di", "det_i"),
        inst("tag", "tagger"),
        inst("src", "pairbox"),
    ]
    links = [
        fs("l_pump", "las.out", "att.in", "10 mm"),
        fs("l_att", "att.out", "bs.in_b", "10 mm"),
        fs("l_sig", "src.signal", "bs.in_a", "10 mm"),
        fs("l_bs", "bs.out_a", "cin.in", "10 mm"),
        {
            "id": "fib",
            "kind": "fiber",
            "from": "cin.out",
            "to": "cout.in",
            "fiber": "smf",
            "length": "1 m",
        },
        fs("l_end", "cout.out", "ds.in", "10 mm"),
        fs("l_idl", "src.idler", "di.in", "10 mm"),
        cable("c1", "ds", "ch1"),
        cable("c2", "di", "ch2"),
    ]
    return setup_from_dict({"instances": instances, "links": links})


def test_links_report_photon_coupling_next_to_the_classical_coupling(lib):
    fibre_library(lib)
    put(lib, "ndl", "NDFilter", optical_density=0.0)
    PAIRS["src"] = cw_pair(1e5, flat=10)
    DETS["ds"], DETS["di"] = det(0.5), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = fibre_setup()
    r = simulate_rates(setup, lib)
    w_in = 810e-9 * 9.7e-3 / (math.pi * 1e-3)
    w_fiber = 2.5e-6
    eta_photon = (2 * w_in * w_fiber / (w_in**2 + w_fiber**2)) ** 2
    w_pump = 405e-9 * 9.7e-3 / (math.pi * 1e-3)
    eta_pump = (2 * w_pump * w_fiber / (w_pump**2 + w_fiber**2)) ** 2
    assert eta_pump == pytest.approx(0.64, abs=0.01)
    link = r.links["fib"]
    assert link["coupling_efficiency"]["signal"] == pytest.approx(eta_photon, rel=1e-3)
    assert link["photons_hz"] == pytest.approx(1e5 * 0.5 * eta_photon, rel=1e-3)
    assert r.classical.links["fib"].info["coupling_efficiency"] == pytest.approx(eta_pump, rel=2e-2)
    classical_text = format_report(r.classical)
    assert "classical light at 405 nm" in classical_text
    text = format_rates_report(r)
    assert "Photon links" in text and "fib" in text
    row = next(line for line in text.splitlines() if line.strip().startswith("fib"))
    assert f"{100 * eta_photon:.2f}%" in row


def test_fiber_multimode_warning_only_for_real_classical_light(lib):
    fibre_library(lib, laser_wavelength="780 nm", cutoff="800 nm")
    PAIRS["src"] = cw_pair(1e5, flat=10)
    DETS["ds"], DETS["di"] = det(0.5), det(0.5)
    TAGS["tag"] = TaggerModel()
    put(lib, "ndl", "NDFilter", optical_density=0.0)
    real = simulate(fibre_setup(), lib)
    assert "fiber_multimode" in [i.code for i in real.issues]
    put(lib, "ndl", "NDFilter", optical_density=12.0)
    faint = simulate(fibre_setup(), lib)
    assert "fiber_multimode" not in [i.code for i in faint.issues]
    kept = simulate(fibre_setup(), lib, negligible_power_fraction=1e-14)
    assert "fiber_multimode" in [i.code for i in kept.issues]


def test_cli_resolves_a_relative_library_from_the_setup_location(
    lib, tmp_path, monkeypatch, capsys
):
    from sequoia.cli import _run
    from sequoia.setup.io import save_setup

    PAIRS["src"] = cw_pair(1e5)
    DETS["ds"], DETS["di"] = det(0.5), det(0.4)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "ds.in", "10 mm"),
                fs("b", "src.idler", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence("hit", window="500 ps")],
        }
    )
    project = tmp_path / "project"
    (project / "setups").mkdir(parents=True)
    os.symlink(tmp_path / "lib", project / "library")
    path = project / "setups" / "x.yaml"
    save_setup(setup, path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert _run(str(path), "library") == 0
    assert "Photon rates" in capsys.readouterr().out
    assert _run(str(path), "missing_library") == 1
    assert "missing_library" in capsys.readouterr().err


def test_recorders_of_concurrent_threads_never_see_each_others_warnings():
    import warnings

    results = {}
    barrier = threading.Barrier(4)

    def worker(index):
        barrier.wait()
        with record_warnings() as caught:
            for k in range(300):
                warnings.warn(f"thread {index} item {k}", ExtrapolationWarning, stacklevel=1)
        results[index] = [w.message for w in caught]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for index, messages in results.items():
        assert len(messages) == 300
        assert all(m.startswith(f"thread {index} ") for m in messages)


def test_cancellation_between_stages_raises_cancelled(lib):
    prime(cw_pair(1e5, flat=10))
    setup = signal_arm_hbt_setup([hist_spec("histogram", ("tag.ch1", "tag.ch2"))])
    with pytest.raises(Cancelled):
        simulate(setup, lib, should_cancel=lambda: True)
    with pytest.raises(Cancelled):
        simulate_rates(setup, lib, should_cancel=lambda: True)
    polls = []

    def count():
        polls.append(1)
        return False

    simulate_rates(setup, lib, should_cancel=count)
    assert len(polls) >= 8
    for stop_at in (3, 6, len(polls) - 2):
        seen = []

        def stop(limit=stop_at, seen=seen):
            seen.append(1)
            return len(seen) >= limit

        with pytest.raises(Cancelled):
            simulate_rates(setup, lib, should_cancel=stop)


CALLS: list[int] = []


class PumpedParams(Params):
    key: str = "x"


class RTPumpedBox(Component):
    Params = PumpedParams

    def ports(self):
        return (
            PortSpec("in", FREE, IN),
            PortSpec("signal", FREE, OUT),
            PortSpec("idler", FREE, OUT),
        )

    def emit_pairs(self, lights, ctx):
        from sequoia.light.pairs import PairEmission

        CALLS.append(1)
        power = lights["in"].power if "in" in lights else 0.0
        pair = PAIRS[self.params.key]
        return PairEmission(replace(pair, rate=1e8 * power), "signal", "idler")

    def emit_cache_key(self, lights):
        return ("static",)

    def emit_rate_scale(self, lights):
        return lights["in"].power if "in" in lights else 0.0


class RTUncachedBox(RTPumpedBox):
    emit_cache_key = None


def pumped_setup(power="1 mW", source="pumped"):
    return setup_from_dict(
        {
            "instances": [
                inst("las", "laser", {"power": power}),
                inst("src", source),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("p", "las.out", "src.in", "10 mm"),
                fs("a", "src.signal", "ds.in", "10 mm"),
                fs("b", "src.idler", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence()],
        }
    )


@pytest.fixture
def pumped(lib):
    register("nonlinear", "RTPumpedBox")(RTPumpedBox)
    register("nonlinear", "RTUncachedBox")(RTUncachedBox)
    put(lib, "pumped", "RTPumpedBox", key="src")
    put(lib, "uncached", "RTUncachedBox", key="src")
    CALLS.clear()
    PAIRS["src"] = cw_pair(1.0, flat=10)
    DETS["ds"], DETS["di"] = det(0.5), det(0.5)
    TAGS["tag"] = TaggerModel()
    return lib


def test_emission_cache_reuses_emit_pairs_and_rescales_with_the_declared_factor(pumped):
    lib = pumped
    cache = RunCache()
    first = simulate_rates(pumped_setup("1 mW"), lib, cache=cache)
    assert len(CALLS) == 1
    again = simulate_rates(pumped_setup("1 mW"), lib, cache=cache)
    double = simulate_rates(pumped_setup("2 mW"), lib, cache=cache)
    assert len(CALLS) == 1
    assert cache.emission_hits == 2 and cache.emission_misses == 1
    fresh = simulate_rates(pumped_setup("2 mW"), lib)
    assert len(CALLS) == 2
    assert again.channels["tag.ch1"]["singles_hz"] == first.channels["tag.ch1"]["singles_hz"]
    assert double.channels["tag.ch1"]["singles_hz"] == pytest.approx(
        fresh.channels["tag.ch1"]["singles_hz"], rel=1e-12
    )
    assert double.analyses["c"]["true_hz"] == pytest.approx(
        fresh.analyses["c"]["true_hz"], rel=1e-9
    )
    assert double.analyses["c"]["true_hz"] == pytest.approx(
        2 * first.analyses["c"]["true_hz"], rel=1e-9
    )


def test_emission_cache_key_includes_the_part_parameters(pumped):
    lib = pumped
    cache = RunCache()
    simulate_rates(pumped_setup(), lib, cache=cache)
    PAIRS["other"] = cw_pair(1.0, flat=5)
    put(lib, "pumped", "RTPumpedBox", key="other")
    simulate_rates(pumped_setup(), lib, cache=cache)
    assert len(CALLS) == 2


def test_components_without_a_cache_key_are_never_cached(pumped):
    cache = RunCache()
    for _ in range(2):
        simulate_rates(pumped_setup(source="uncached"), pumped, cache=cache)
    assert len(CALLS) == 2
    assert cache.emission_hits == 0


def test_part_cache_parses_each_part_once_and_notices_edits(lib, monkeypatch):
    PAIRS["src"] = cw_pair(1e5, flat=10)
    DETS["ds"], DETS["di"] = det(0.5), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = signal_arm_hbt_setup([coincidence("c", "1 ns", channels=("tag.ch1", "tag.ch2"))])
    loads = []
    original = Library.load_part

    def counting(self, ref):
        loads.append(ref)
        return original(self, ref)

    monkeypatch.setattr(Library, "load_part", counting)
    simulate_rates(setup, lib)
    uncached = len(loads)
    loads.clear()
    cache = RunCache()
    simulate_rates(setup, lib, cache=cache)
    cold = len(loads)
    loads.clear()
    simulate_rates(setup, lib, cache=cache)
    assert cold < uncached
    assert cold == len({i.part for i in setup.instances})
    assert loads == []
    put(lib, "bs", "BeamSplitter", reflectance=0.3)
    simulate_rates(setup, lib, cache=cache)
    assert loads == ["bs"]
