import math
import time
from dataclasses import replace

import numpy as np
import pytest
from optics import basic_library, fs, inst, put
from scipy import constants
from scipy.special import erf
from scipy.stats import exponnorm, norm

import sequoia.components.builtin  # noqa: F401
from sequoia.components import base
from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.engines.rates import has_rate_devices, simulate_rates
from sequoia.library.store import Library
from sequoia.light.detection import DetectorModel, TaggerModel
from sequoia.light.gaussian import GaussianMode
from sequoia.light.pairs import PairEmission, PairSource
from sequoia.setup.io import setup_from_dict

C = constants.c
FREE, ELEC, IN, OUT = PortKind.FREE_SPACE, PortKind.ELECTRICAL, PortDirection.IN, PortDirection.OUT
PAIRS: dict[str, PairSource] = {}
DETS: dict[str, DetectorModel] = {}
TAGS: dict[str, TaggerModel] = {}
DNU = 1e8
SIGMA = 5e9


class KeyParams(Params):
    key: str = "x"
    collinear: bool = False


class PhaseParams(Params):
    opl: float = 0.0
    gpl: float = 0.0


class TagSettings(Settings):
    input_delays: dict[str, float] = {}


class RTPairBox(Component):
    Params = KeyParams

    def ports(self):
        if self.params.collinear:
            return (PortSpec("out", FREE, OUT),)
        return (PortSpec("signal", FREE, OUT), PortSpec("idler", FREE, OUT))

    def emit_pairs(self, lights, ctx):
        port = "out" if self.params.collinear else None
        return PairEmission(PAIRS[self.params.key], port or "signal", port or "idler")


class RTDetector(Component):
    Params = KeyParams

    def ports(self):
        return (PortSpec("in", FREE, IN, required=True), PortSpec("out", ELEC, OUT, required=True))

    def detector_model(self):
        return DETS[self.params.key]


class RTTagger(Component):
    Params = KeyParams
    Settings = TagSettings

    def ports(self):
        return tuple(PortSpec(f"ch{i}", ELEC, IN) for i in range(1, 5))

    def tagger_model(self):
        base_model = TAGS.get(self.params.key, TaggerModel())
        return TaggerModel(
            base_model.jitter_fwhm, base_model.dead_time, dict(self.settings.input_delays)
        )


class RTSink(Component):
    Params = Params

    def ports(self):
        return (PortSpec("in", ELEC, IN),)


class RTPhase(Component):
    Params = PhaseParams

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        return [("out", beam.through(ctx.subject, None, opl=self.params.opl, gpl=self.params.gpl))]


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))
    register("nonlinear", "RTPairBox")(RTPairBox)
    register("detectors", "RTDetector")(RTDetector)
    register("electronics", "RTTagger")(RTTagger)
    register("freespace", "RTPhase")(RTPhase)
    library = basic_library(Library(tmp_path / "lib"))
    put(library, "pairbox", "RTPairBox", key="src")
    put(library, "pairbox_c", "RTPairBox", key="src", collinear=True)
    put(library, "det_s", "RTDetector", key="ds")
    put(library, "det_i", "RTDetector", key="di")
    put(library, "det_a", "RTDetector", key="da")
    put(library, "det_b", "RTDetector", key="db")
    put(library, "tagger", "RTTagger", key="tag")
    put(library, "nd", "NDFilter", optical_density=0.0)
    put(library, "polx", "Polarizer", extinction_ratio=1e12)
    put(library, "bs", "BeamSplitter")
    put(library, "pbs", "PBS")
    return library


def det(eta=1.0, **kw):
    def efficiency(wl):
        wl = np.atleast_1d(wl)
        value = eta(wl) if callable(eta) else np.full(len(wl), eta)
        return value[:, None, None] * np.eye(2, dtype=complex)[None]

    return DetectorModel("rt", efficiency, **kw)


def mode(wavelength=810e-9):
    return GaussianMode.from_waist(wavelength, 1e-3)


def cw_pair(
    rate=1e5,
    *,
    n_sum=7_400_000,
    n_c=None,
    sigma=SIGMA,
    dnu=DNU,
    span=6.0,
    state=None,
    flat=None,
    rep=None,
):
    n_c = n_sum // 2 if n_c is None else n_c
    half = flat if flat is not None else int(span * sigma / dnu)
    n_s = np.arange(n_c - half, n_c + half + 1)
    x = (n_s - n_c) * dnu
    f = np.ones(len(n_s)) if flat is not None else np.exp(-(x**2) / (4 * sigma**2))
    f = f / np.sqrt(np.sum(f**2))
    amp = np.zeros((2, 2, len(n_s), 1), dtype=complex)
    for (s, i), c in (state or {(0, 0): 1.0}).items():
        amp[s, i, :, 0] = c * f
    amp /= np.sqrt(np.sum(np.abs(amp) ** 2))
    return PairSource(
        dnu,
        n_s,
        np.array([n_sum]),
        amp,
        rate,
        mode(),
        mode(),
        405e-9,
        rep,
    )


def pulsed_pair(rate, rep, a, b, *, dnu=5e8, n_c=760_000, count=121, n_sum_count=121):
    n_s = np.arange(n_c - count // 2, n_c + count // 2 + 1)
    n_sum = np.arange(2 * n_c - n_sum_count // 2, 2 * n_c + n_sum_count // 2 + 1)
    xs = (n_s - n_c)[:, None] * dnu
    u = (n_sum[None, :] - 2 * n_c) * dnu
    xi = u - xs
    f = np.exp(-(u**2) / (2 * a**2) - ((xs - xi) ** 2) / (2 * b**2))
    f = f / np.sqrt(np.sum(f**2))
    amp = np.zeros((2, 2, len(n_s), len(n_sum)), dtype=complex)
    amp[0, 0] = f
    return PairSource(dnu, n_s, n_sum, amp, rate, mode(), mode(), 405e-9, rep), f


def schmidt_of(pair):
    n_i = pair.n_sum[None, :] - pair.n_signal[:, None]
    grid = np.unique(n_i)
    matrix = np.zeros((len(pair.n_signal), len(grid)))
    rows = np.broadcast_to(np.arange(len(pair.n_signal))[:, None], n_i.shape)
    matrix[rows, np.searchsorted(grid, n_i)] = pair.amplitude[0, 0].real
    s = np.linalg.svd(matrix, compute_uv=False)
    p = s**2 / np.sum(s**2)
    return 1.0 / np.sum(p**2)


def cable(link_id, det_id, channel, delay="0 ns"):
    return {
        "id": link_id,
        "kind": "cable",
        "from": f"{det_id}.out",
        "to": f"tag.{channel}",
        "delay": delay,
    }


def coincidence(name="c", window="2 ns", center="auto", channels=("tag.ch1", "tag.ch2")):
    return {
        "name": name,
        "type": "coincidences",
        "channels": list(channels),
        "window": window,
        "center": center,
    }


def two_arm(
    lib,
    pair,
    ds,
    di,
    *,
    sig=(),
    idl=(),
    analyses=(),
    delay_s="0 ns",
    delay_i="0 ns",
    tagger=None,
    tagger_settings=None,
    source="pairbox",
    sig_len="10 mm",
    idl_len="10 mm",
):
    PAIRS["src"], DETS["ds"], DETS["di"] = pair, ds, di
    TAGS["tag"] = tagger or TaggerModel()
    instances = [
        inst("src", source),
        inst("ds", "det_s"),
        inst("di", "det_i"),
        inst("tag", "tagger", tagger_settings or {}),
    ]
    links = []
    prev = "src.signal"
    for k, (part, settings) in enumerate(sig):
        instances.append(inst(f"s{k}", part, settings))
        links.append(fs(f"ls{k}", prev, f"s{k}.in", sig_len if k == 0 else "1 mm"))
        prev = f"s{k}.out"
    links.append(fs("ls_end", prev, "ds.in", sig_len if not sig else "1 mm"))
    prev = "src.idler"
    for k, (part, settings) in enumerate(idl):
        instances.append(inst(f"i{k}", part, settings))
        links.append(fs(f"li{k}", prev, f"i{k}.in", idl_len if k == 0 else "1 mm"))
        prev = f"i{k}.out"
    links.append(fs("li_end", prev, "di.in", idl_len if not idl else "1 mm"))
    links += [cable("c1", "ds", "ch1", delay_s), cable("c2", "di", "ch2", delay_i)]
    setup = setup_from_dict({"instances": instances, "links": links, "analyses": list(analyses)})
    return simulate_rates(setup, lib)


def codes(result):
    return [i.code for i in result.issues]


def test_flat_jsa_coincidences_and_singles_through_lossy_paths(lib):
    rate = 2e5
    eta_s, eta_i, dark_s, dark_i = 0.6, 0.35, 100.0, 50.0
    t_s, t_i = 10 ** (-0.1), 10 ** (-0.25)
    put(lib, "nd_s", "NDFilter", optical_density=0.1)
    put(lib, "nd_i", "NDFilter", optical_density=0.25)
    r = two_arm(
        lib,
        cw_pair(rate, flat=100),
        det(eta_s, dark_count_rate=dark_s),
        det(eta_i, dark_count_rate=dark_i),
        sig=[("nd_s", {})],
        idl=[("nd_i", {})],
        analyses=[coincidence(window="9 ns")],
    )
    ch1, ch2 = r.channels["tag.ch1"], r.channels["tag.ch2"]
    assert ch1["pair_singles_hz"] == pytest.approx(rate * eta_s * t_s, rel=1e-9)
    assert ch1["singles_hz"] == pytest.approx(rate * eta_s * t_s + dark_s, rel=1e-9)
    assert ch2["background_hz"] == pytest.approx(dark_i, rel=1e-9)
    assert ch2["singles_hz"] == pytest.approx(rate * eta_i * t_i + dark_i, rel=1e-9)
    a = r.analyses["c"]
    expected = rate * eta_s * eta_i * t_s * t_i
    assert a["coincidence_rate_hz"] == pytest.approx(expected, rel=1e-9)
    assert a["window_fraction"] == pytest.approx(1.0, abs=2e-3)
    assert a["true_hz"] == pytest.approx(expected, rel=3e-3)
    assert a["accidental_hz"] == pytest.approx(9e-9 * ch1["singles_hz"] * ch2["singles_hz"])
    assert a["car"] == pytest.approx(a["true_hz"] / a["accidental_hz"])
    assert a["heralding_a_given_b"] == pytest.approx(a["true_hz"] / ch2["singles_hz"])
    assert r.pairs[("tag.ch1", "tag.ch2")]["coincidence_rate_hz"] == pytest.approx(expected)
    assert not [i for i in r.issues if i.severity == "error"]


def test_polarization_dependent_efficiency_operator(lib):
    state = {(0, 0): 1.0, (1, 1): 1.0}

    def eff(wl):
        n = len(np.atleast_1d(wl))
        return np.tile(np.diag([0.8, 0.2]).astype(complex), (n, 1, 1))

    model = DetectorModel("rt", eff)
    r = two_arm(lib, cw_pair(1e5, flat=20, state=state), model, model, analyses=[coincidence()])
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(1e5 * 0.5 * (0.8 + 0.2))
    expected = 1e5 * 0.5 * (0.8 * 0.8 + 0.2 * 0.2)
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(expected, rel=1e-9)


def test_pump_leakage_and_dark_counts_in_singles(lib):
    wl = 780e-9
    photon = constants.h * C / wl
    PAIRS["src"] = cw_pair(1e4, flat=10)
    DETS["ds"], DETS["di"] = det(0.5, dark_count_rate=200.0), det(0.5)
    TAGS["tag"] = TaggerModel()
    put(lib, "ndl", "NDFilter", optical_density=14.0)
    instances = [
        inst("src", "pairbox"),
        inst("las", "laser"),
        inst("att", "ndl"),
        inst("bs", "bs"),
        inst("ds", "det_s"),
        inst("di", "det_i"),
        inst("tag", "tagger"),
    ]
    links = [
        fs("a", "src.signal", "bs.in_a", "10 mm"),
        fs("b", "las.out", "att.in", "10 mm"),
        fs("c", "att.out", "bs.in_b", "10 mm"),
        fs("d", "bs.out_a", "ds.in", "10 mm"),
        fs("e", "src.idler", "di.in", "10 mm"),
        cable("c1", "ds", "ch1"),
        cable("c2", "di", "ch2"),
    ]
    setup = setup_from_dict({"instances": instances, "links": links})
    r = simulate_rates(setup, lib)
    leak = 1e-3 * 1e-14 * 0.5 * 0.5 / photon
    ch = r.channels["tag.ch1"]
    assert ch["leakage_hz"] == pytest.approx(leak, rel=1e-6)
    assert ch["pair_singles_hz"] == pytest.approx(1e4 * 0.5 * 0.5, rel=1e-9)
    assert ch["singles_hz"] == pytest.approx(1e4 * 0.25 + 200.0 + leak, rel=1e-9)
    assert "pump_leakage" not in codes(r)
    put(lib, "ndl", "NDFilter", optical_density=12.0)
    r2 = simulate_rates(setup, lib)
    assert r2.channels["tag.ch1"]["leakage_hz"] == pytest.approx(leak * 1e2, rel=1e-6)
    assert "pump_leakage" in codes(r2)
    assert not r2.channels["tag.ch2"]["leakage_hz"]


def test_afterpulsing_adds_uncorrelated_counts(lib):
    r = two_arm(
        lib,
        cw_pair(1e5, flat=10),
        det(0.5, dark_count_rate=100.0, afterpulse_probability=0.02),
        det(0.5),
    )
    ch = r.channels["tag.ch1"]
    assert ch["afterpulse_hz"] == pytest.approx(0.02 * (5e4 + 100.0))
    assert ch["background_hz"] == pytest.approx(100.0 + 0.02 * (5e4 + 100.0))


def test_dead_time_follows_non_paralyzable_formula(lib):
    rate, eta, tau = 5e6, 0.5, 100e-9
    r = two_arm(
        lib,
        cw_pair(rate, flat=10),
        det(eta, dead_time=tau),
        det(eta, dead_time=tau),
        analyses=[coincidence(window="9 ns")],
    )
    s = rate * eta
    ch = r.channels["tag.ch1"]
    assert ch["incident_hz"] == pytest.approx(s)
    assert ch["singles_hz"] == pytest.approx(s / (1 + s * tau))
    assert ch["pair_singles_hz"] + ch["background_hz"] == pytest.approx(ch["singles_hz"])
    survival = 1 / (1 + s * tau)
    a = r.analyses["c"]
    assert a["coincidence_rate_hz"] == pytest.approx(rate * eta * eta * survival**2, rel=1e-9)


def test_tagger_dead_time_dominates_when_longer(lib):
    r = two_arm(
        lib,
        cw_pair(1e6, flat=10),
        det(0.5, dead_time=50e-9),
        det(0.5, dead_time=50e-9),
        tagger=TaggerModel(dead_time=200e-9),
    )
    assert r.channels["tag.ch1"]["dead_time_s"] == 200e-9
    assert r.channels["tag.ch1"]["singles_hz"] == pytest.approx(5e5 / (1 + 5e5 * 200e-9), rel=1e-3)


def test_accidentals_cw_use_total_singles(lib):
    r = two_arm(
        lib,
        cw_pair(1e6, flat=10),
        det(0.3, dark_count_rate=1e4),
        det(0.2, dark_count_rate=2e4),
        analyses=[coincidence(window="1 ns")],
    )
    sa, sb = 3e5 + 1e4, 2e5 + 2e4
    a = r.analyses["c"]
    assert a["accidental_hz"] == pytest.approx(sa * sb * 1e-9, rel=1e-9)
    assert a["total_hz"] == pytest.approx(a["true_hz"] + a["accidental_hz"])


def test_accidentals_pulsed_formula(lib):
    rate, rep, w = 5e6, 80e6, 2e-9
    pair = cw_pair(rate, flat=10, rep=rep)
    r = two_arm(
        lib,
        pair,
        det(0.3, dark_count_rate=1e4),
        det(0.2, dark_count_rate=2e4),
        analyses=[coincidence(window="2 ns")],
    )
    pa, pb = rate * 0.3, rate * 0.2
    ba, bb = 1e4, 2e4
    expected = pa * pb / rep + (pa * bb + ba * pb + ba * bb) * w
    assert r.analyses["c"]["accidental_hz"] == pytest.approx(expected, rel=1e-9)


def gauss_tau_sigma(sigma_nu=SIGMA):
    return 1.0 / (4 * math.pi * sigma_nu)


def fwhm_to_sigma(f):
    return f / (2 * math.sqrt(2 * math.log(2)))


def test_window_fraction_gaussian_biphoton_with_gaussian_jitters(lib):
    ja, jb, jt = 60e-12, 40e-12, 30e-12
    sigma_tot = math.sqrt(
        gauss_tau_sigma() ** 2
        + fwhm_to_sigma(ja) ** 2
        + fwhm_to_sigma(jb) ** 2
        + 2 * fwhm_to_sigma(jt) ** 2
    )
    for window in (60e-12, 150e-12, 400e-12):
        r = two_arm(
            lib,
            cw_pair(1e5),
            det(jitter_fwhm=ja),
            det(jitter_fwhm=jb),
            tagger=TaggerModel(jitter_fwhm=jt),
            analyses=[coincidence(window=f"{window * 1e12} ps")],
        )
        expected = erf(window / (2 * math.sqrt(2) * sigma_tot))
        assert r.analyses["c"]["window_fraction"] == pytest.approx(expected, abs=1e-3)
        assert r.analyses["c"]["true_hz"] == pytest.approx(1e5 * expected, rel=2e-3)


def test_window_fraction_off_centre_and_delays(lib):
    sigma_tot = math.sqrt(gauss_tau_sigma() ** 2 + 2 * fwhm_to_sigma(50e-12) ** 2)
    delay_a, delay_b, tag_delay = 3.2e-9, 1.1e-9, 0.4e-9
    d = delay_a + tag_delay - delay_b
    r = two_arm(
        lib,
        cw_pair(1e5),
        det(jitter_fwhm=50e-12),
        det(jitter_fwhm=50e-12),
        delay_s=f"{delay_a * 1e9} ns",
        delay_i=f"{delay_b * 1e9} ns",
        tagger_settings={"input_delays": {"ch1": tag_delay}},
        analyses=[
            coincidence("auto", window="200 ps"),
            coincidence("on", window="100 ps", center=f"{d * 1e9} ns"),
            coincidence("off", window="100 ps", center=f"{(d + 80e-12) * 1e9} ns"),
        ],
    )
    assert r.analyses["auto"]["center_s"] == pytest.approx(d, abs=0.5e-12)
    s2 = math.sqrt(2) * sigma_tot
    assert r.analyses["on"]["window_fraction"] == pytest.approx(erf(50e-12 / s2), abs=1e-3)
    expected = 0.5 * (erf((80e-12 + 50e-12) / s2) - erf((80e-12 - 50e-12) / s2))
    assert r.analyses["off"]["window_fraction"] == pytest.approx(expected, abs=1e-3)


def test_window_fraction_with_exponential_jitter_tail(lib):
    fraction, tau, ja = 0.2, 150e-12, 40e-12
    r = two_arm(
        lib,
        cw_pair(1e5),
        det(jitter_fwhm=ja, jitter_tail_fraction=fraction, jitter_tail_tau=tau),
        det(),
        analyses=[coincidence(window="300 ps", center="0 ns")],
    )
    sigma_a = fwhm_to_sigma(ja)
    sig_main = math.sqrt(gauss_tau_sigma() ** 2 + sigma_a**2)
    lo, hi = -150e-12, 150e-12
    gauss = norm.cdf(hi, scale=sig_main) - norm.cdf(lo, scale=sig_main)
    emg = exponnorm.cdf(hi, tau / sig_main, scale=sig_main) - exponnorm.cdf(
        lo, tau / sig_main, scale=sig_main
    )
    expected = (1 - fraction) * gauss + fraction * emg
    assert r.analyses["c"]["window_fraction"] == pytest.approx(expected, abs=2e-3)


def hom_expected(tau, reflectance, sigma=SIGMA):
    t = 1 - reflectance
    visibility = math.exp(-8 * math.pi**2 * sigma**2 * tau**2)
    return t**2 + reflectance**2 - 2 * t * reflectance * visibility


def hom_run(lib, delta, reflectance=0.5, eta=0.5, rate=1e5):
    put(lib, "bsx", "BeamSplitter", reflectance=reflectance)
    PAIRS["src"] = cw_pair(rate)
    DETS["ds"], DETS["di"] = det(eta), det(eta)
    TAGS["tag"] = TaggerModel()
    length = 100e-3 + delta * C
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("bs", "bsx"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "bs.in_a", f"{length * 1e3:.9f} mm"),
                fs("b", "src.idler", "bs.in_b", "100 mm"),
                fs("c", "bs.out_a", "ds.in", "10 mm"),
                fs("d", "bs.out_b", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence(window="3 ns")],
        }
    )
    return simulate_rates(setup, lib)


@pytest.mark.parametrize("reflectance", [0.5, 0.2])
def test_hom_dip_matches_analytic_gaussian_result(lib, reflectance):
    rate, eta = 1e5, 0.5
    for tau in (0.0, 5e-12, 15e-12, 30e-12, 60e-12, -20e-12):
        r = hom_run(lib, tau, reflectance, eta, rate)
        got = r.analyses["c"]["coincidence_rate_hz"]
        assert got == pytest.approx(
            rate * eta**2 * hom_expected(tau, reflectance), rel=2e-3, abs=1e-6
        )
    zero = hom_run(lib, 0.0, reflectance).analyses["c"]["coincidence_rate_hz"]
    far = hom_run(lib, 1e-9, reflectance).analyses["c"]["coincidence_rate_hz"]
    t = 1 - reflectance
    assert 1 - zero / far == pytest.approx(2 * t * reflectance / (t**2 + reflectance**2), abs=1e-4)
    if reflectance == 0.5:
        assert zero < 1e-6 * far


def test_hom_dip_fwhm(lib):
    half = 0.5
    tau_half = math.sqrt(-math.log(half) / (8 * math.pi**2 * SIGMA**2))
    r = hom_run(lib, tau_half)
    far = hom_run(lib, 1e-9).analyses["c"]["coincidence_rate_hz"]
    mid = r.analyses["c"]["coincidence_rate_hz"]
    assert mid / far == pytest.approx(0.5, rel=2e-3)


@pytest.mark.parametrize("a_deg,b_deg", [(0, 0), (30, 30), (0, 45), (20, 70), (45, 135), (10, -10)])
def test_bell_state_correlations_through_polarizers(lib, a_deg, b_deg):
    s = 1 / math.sqrt(2)
    rate, eta = 1e5, 0.7
    for label, state, sign in (
        ("phi+", {(0, 0): s, (1, 1): s}, +1),
        ("phi-", {(0, 0): s, (1, 1): -s}, -1),
    ):
        r = two_arm(
            lib,
            cw_pair(rate, flat=5, state=state),
            det(eta),
            det(eta),
            sig=[("polx", {"angle": f"{a_deg} deg"})],
            idl=[("polx", {"angle": f"{b_deg} deg"})],
            analyses=[coincidence()],
        )
        a, b = math.radians(a_deg), math.radians(b_deg)
        expected = rate * eta**2 * 0.5 * math.cos(a - sign * b) ** 2
        got = r.analyses["c"]["coincidence_rate_hz"]
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-9 * rate), label


def test_bell_correlation_function_is_cos_2_delta(lib):
    s = 1 / math.sqrt(2)
    state = {(0, 0): s, (1, 1): s}

    def rate_at(a, b):
        r = two_arm(
            lib,
            cw_pair(1e5, flat=5, state=state),
            det(1.0),
            det(1.0),
            sig=[("polx", {"angle": f"{a} deg"})],
            idl=[("polx", {"angle": f"{b} deg"})],
            analyses=[coincidence()],
        )
        return r.analyses["c"]["coincidence_rate_hz"]

    a, b = 22.0, -13.0
    e = rate_at(a, b) + rate_at(a + 90, b + 90) - rate_at(a, b + 90) - rate_at(a + 90, b)
    e /= rate_at(a, b) + rate_at(a + 90, b + 90) + rate_at(a, b + 90) + rate_at(a + 90, b)
    assert e == pytest.approx(math.cos(2 * math.radians(a - b)), abs=1e-9)


def test_malus_law_for_product_state_singles(lib):
    for theta in (0, 25, 60, 90):
        r = two_arm(
            lib,
            cw_pair(1e5, flat=5),
            det(1.0),
            det(1.0),
            sig=[("polx", {"angle": f"{theta} deg"})],
        )
        assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(
            1e5 * math.cos(math.radians(theta)) ** 2, abs=1e-6 * 1e5
        )


def test_singles_of_bell_state_are_unpolarized(lib):
    s = 1 / math.sqrt(2)
    r = two_arm(
        lib,
        cw_pair(1e5, flat=5, state={(0, 0): s, (1, 1): s}),
        det(1.0),
        det(1.0),
        sig=[("polx", {"angle": "37 deg"})],
    )
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(0.5e5)


def test_psi_plus_amplitudes_selected_by_polarizers(lib):
    h = 1 / math.sqrt(2)
    state = {(0, 1): h, (1, 0): h}

    def rate_at(a, b):
        r = two_arm(
            lib,
            cw_pair(1e5, flat=5, state=state),
            det(1.0),
            det(1.0),
            sig=[("polx", {"angle": f"{a} deg"})],
            idl=[("polx", {"angle": f"{b} deg"})],
            analyses=[coincidence()],
        )
        return r.analyses["c"]["coincidence_rate_hz"]

    assert rate_at(0, 0) == pytest.approx(0.0, abs=1e-6)
    assert rate_at(0, 90) == pytest.approx(0.5e5, rel=1e-9)
    assert rate_at(45, 45) == pytest.approx(0.5e5, rel=1e-9)


def mzi_run(lib, delta, opl_gpl=None, sigma=1e9, rate=1e5):
    PAIRS["src"] = cw_pair(rate, sigma=sigma, n_sum=7_400_000)
    DETS["ds"], DETS["di"] = det(1.0), det(1.0)
    TAGS["tag"] = TaggerModel()
    opl, gpl = opl_gpl if opl_gpl is not None else (delta, delta)
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("b1", "bs"),
                inst("b2", "bs"),
                inst("ph", "rt_phase"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "b1.in_a", "10 mm"),
                fs("b", "b1.out_a", "b2.in_a", "50 mm"),
                fs("c", "b1.out_b", "ph.in", "50 mm"),
                fs("d", "ph.out", "b2.in_b", "0 mm"),
                fs("e", "b2.out_a", "ds.in", "10 mm"),
                fs("f", "src.idler", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence()],
        }
    )
    put(lib, "rt_phase", "RTPhase", opl=opl, gpl=gpl)
    return simulate_rates(setup, lib), opl


def test_one_arm_interferometer_fringe_and_coherence_envelope(lib):
    sigma = 1e9
    nu0 = 7_400_000 // 2 * DNU
    for delta_over_lambda in (0.0, 0.1, 0.25, 0.5, 0.8, 1.0):
        delta = delta_over_lambda * C / nu0
        r, _ = mzi_run(lib, delta, sigma=sigma)
        expected = 1e5 * 0.5 * (1 - math.cos(2 * math.pi * nu0 * delta / C))
        assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(expected, abs=2e-4 * 1e5)
    delta = 3e-3
    r, _ = mzi_run(lib, delta, sigma=sigma)
    env = math.exp(-2 * math.pi**2 * sigma**2 * delta**2 / C**2)
    expected = 1e5 * 0.5 * (1 - env * math.cos(2 * math.pi * nu0 * delta / C))
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(expected, abs=2e-4 * 1e5)


def test_phase_uses_phase_delay_on_carrier_and_group_delay_on_envelope(lib):
    sigma = 2e9
    nu0 = 7_400_000 // 2 * DNU
    opl, gpl = 0.31e-3 + 0.37 * C / nu0, 0.31e-3 + 1.1e-3
    r, _ = mzi_run(lib, 0.0, opl_gpl=(opl, gpl), sigma=sigma)
    spectrum_delay = gpl
    env = math.exp(-2 * math.pi**2 * sigma**2 * spectrum_delay**2 / C**2)
    expected = 1e5 * 0.5 * (1 - env * math.cos(2 * math.pi * nu0 * opl / C))
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(expected, abs=3e-4 * 1e5)


def hbt_run(lib, window, rate=2e6, eta=(0.5, 0.4, 0.3), jitter=0.0, pair=None):
    PAIRS["src"] = pair if pair is not None else cw_pair(rate)
    DETS["ds"] = det(eta[0], jitter_fwhm=jitter)
    DETS["da"] = det(eta[1], jitter_fwhm=jitter)
    DETS["db"] = det(eta[2], jitter_fwhm=jitter)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("bs", "bs"),
                inst("ds", "det_s"),
                inst("da", "det_a"),
                inst("db", "det_b"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("h", "src.signal", "ds.in", "10 mm"),
                fs("i", "src.idler", "bs.in_a", "10 mm"),
                fs("a", "bs.out_a", "da.in", "10 mm"),
                fs("b", "bs.out_b", "db.in", "20 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "da", "ch2"),
                cable("c3", "db", "ch3"),
            ],
            "analyses": [
                {
                    "name": "h",
                    "type": "heralded_g2",
                    "herald": "tag.ch1",
                    "a": "tag.ch2",
                    "b": "tag.ch3",
                    "window": window,
                },
                {
                    "name": "u",
                    "type": "g2",
                    "channels": ["tag.ch2", "tag.ch3"],
                    "range": "400 ps",
                    "bins": 400,
                },
                {
                    "name": "uh",
                    "type": "g2",
                    "channels": ["tag.ch1", "tag.ch2"],
                    "range": "400 ps",
                    "bins": 40,
                },
            ],
        }
    )
    return simulate_rates(setup, lib)


def test_heralded_g2_low_rate_limit(lib):
    rate, w = 2e6, 2e-9
    r = hbt_run(lib, "2 ns", rate)
    x = rate * w
    expected = (2 * x + x**2) / (1 + x) ** 2
    assert r.analyses["h"]["g2_heralded"] == pytest.approx(expected, rel=2e-3)
    s_h = r.channels["tag.ch1"]["singles_hz"]
    assert s_h == pytest.approx(rate * 0.5)
    r2 = hbt_run(lib, "0.5 ns", rate)
    assert r2.analyses["h"]["g2_heralded"] < r.analyses["h"]["g2_heralded"] / 3.5


def test_unheralded_arm_of_cw_pair_is_thermal_and_cross_arm_is_correlated(lib):
    r = hbt_run(lib, "2 ns", 2e5)
    g = r.analyses["u"]
    assert g["g2_peak"] == pytest.approx(2.0, rel=2e-2)
    assert g["g2"][0] == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(g["true_hz"], 0.0, atol=1e-12)
    assert r.analyses["uh"]["g2_peak"] > 100


def test_cw_thermal_g2_peak_washed_out_by_jitter(lib):
    sigma_c = 1.0 / (2 * math.sqrt(2) * math.pi * SIGMA)
    ja, jb = 80e-12, 50e-12
    sigma_j = math.sqrt(fwhm_to_sigma(ja) ** 2 + fwhm_to_sigma(jb) ** 2)
    PAIRS["src"] = cw_pair(1e6, span=7.0)
    for jit in ((0.0, 0.0), (ja, jb)):
        DETS["da"], DETS["db"] = det(1.0, jitter_fwhm=jit[0]), det(1.0, jitter_fwhm=jit[1])
        DETS["ds"] = det(1.0)
        TAGS["tag"] = TaggerModel()
        setup = setup_from_dict(
            {
                "instances": [
                    inst("src", "pairbox"),
                    inst("bs", "bs"),
                    inst("ds", "det_s"),
                    inst("da", "det_a"),
                    inst("db", "det_b"),
                    inst("tag", "tagger"),
                ],
                "links": [
                    fs("h", "src.signal", "bs.in_a", "10 mm"),
                    fs("a", "bs.out_a", "da.in", "10 mm"),
                    fs("b", "bs.out_b", "db.in", "10 mm"),
                    fs("x", "src.idler", "ds.in", "10 mm"),
                    cable("c1", "ds", "ch1"),
                    cable("c2", "da", "ch2"),
                    cable("c3", "db", "ch3"),
                ],
                "analyses": [
                    {
                        "name": "g",
                        "type": "g2",
                        "channels": ["tag.ch2", "tag.ch3"],
                        "range": "400 ps",
                        "bins": 400,
                        "center": "0 ns",
                    }
                ],
            }
        )
        r = simulate_rates(setup, lib)
        g = r.analyses["g"]
        total = math.sqrt(sigma_c**2 + (sigma_j**2 if jit[0] else 0.0))
        expected_peak = 1.0 + sigma_c / total
        mid = 0.5 * (g["g2"][199] + g["g2"][200])
        assert mid == pytest.approx(expected_peak, rel=2e-3)
        tau = g["tau_s"]
        shape = 1.0 + (sigma_c / total) * np.exp(-(tau**2) / (2 * total**2))
        assert np.max(np.abs(g["g2"] - shape)) < 3e-3


@pytest.mark.parametrize("ratio", [1.0, 0.5])
def test_pulsed_unheralded_g2_is_one_plus_inverse_schmidt_number(lib, ratio):
    a, b = 4e9, 4e9 * ratio
    pair, f = pulsed_pair(1e5, 80e6, a, b)
    k = schmidt_of(pair)
    assert k == pytest.approx(0.5 * (a / b + b / a), rel=0.03)
    PAIRS["src"] = pair
    DETS["ds"], DETS["da"], DETS["db"] = det(0.3), det(0.5), det(0.4)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("bs", "bs"),
                inst("ds", "det_s"),
                inst("da", "det_a"),
                inst("db", "det_b"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("h", "src.signal", "bs.in_a", "10 mm"),
                fs("a", "bs.out_a", "da.in", "10 mm"),
                fs("b", "bs.out_b", "db.in", "10 mm"),
                fs("x", "src.idler", "ds.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "da", "ch2"),
                cable("c3", "db", "ch3"),
            ],
            "analyses": [
                {
                    "name": "g",
                    "type": "g2",
                    "channels": ["tag.ch2", "tag.ch3"],
                    "range": "2 ns",
                    "bins": 20,
                }
            ],
        }
    )
    r = simulate_rates(setup, lib)
    g = r.analyses["g"]
    assert g["g2_per_pulse"] == pytest.approx(1 + 1 / k, rel=1e-6)
    assert g["by_role"]["signal"]["schmidt_a"] == pytest.approx(k, rel=1e-6)
    assert g["by_role"]["signal"]["schmidt_b"] == pytest.approx(k, rel=1e-6)


def test_pulsed_g2_is_diluted_by_background(lib):
    pair, f = pulsed_pair(1e5, 80e6, 4e9, 2e9)
    k = schmidt_of(pair)
    PAIRS["src"] = pair
    dark = 4e3
    DETS["ds"], DETS["da"], DETS["db"] = det(0.3), det(0.5, dark_count_rate=dark), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("bs", "bs"),
                inst("ds", "det_s"),
                inst("da", "det_a"),
                inst("db", "det_b"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("h", "src.signal", "bs.in_a", "10 mm"),
                fs("a", "bs.out_a", "da.in", "10 mm"),
                fs("b", "bs.out_b", "db.in", "10 mm"),
                fs("x", "src.idler", "ds.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "da", "ch2"),
                cable("c3", "db", "ch3"),
            ],
            "analyses": [
                {
                    "name": "g",
                    "type": "g2",
                    "channels": ["tag.ch2", "tag.ch3"],
                    "range": "2 ns",
                    "bins": 20,
                }
            ],
        }
    )
    r = simulate_rates(setup, lib)
    pair_a = 1e5 * 0.5 * 0.5
    dilution = pair_a / (pair_a + dark)
    assert r.analyses["g"]["g2_per_pulse"] == pytest.approx(1 + dilution / k, rel=1e-6)


def test_non_degenerate_pair_uses_separate_signal_and_idler_lattices(lib):
    nu_s, nu_i = 3.70e14, 1.935e14
    n_sum = round((nu_s + nu_i) / DNU)
    n_c = round(nu_s / DNU)
    pair = cw_pair(1e5, n_sum=n_sum, n_c=n_c)
    ws, wi = C / (n_c * DNU), C / ((n_sum - n_c) * DNU)
    assert ws < 1e-6 < wi

    def eta(wl):
        return np.where(np.asarray(wl) < 1.0e-6, 0.8, 0.1)

    r = two_arm(lib, pair, det(eta), det(eta), analyses=[coincidence(window="1 ns")])
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(1e5 * 0.8)
    assert r.channels["tag.ch2"]["pair_singles_hz"] == pytest.approx(1e5 * 0.1)
    a = r.analyses["c"]
    assert a["coincidence_rate_hz"] == pytest.approx(1e5 * 0.08, rel=1e-9)
    sigma = gauss_tau_sigma()
    assert a["window_fraction"] == pytest.approx(erf(0.5e-9 / (math.sqrt(2) * sigma)), abs=1e-6)


def test_collinear_source_separated_by_pbs(lib):
    pair = cw_pair(1e5, flat=10, state={(0, 1): 1.0})
    PAIRS["src"], DETS["ds"], DETS["di"] = pair, det(0.6), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox_c"),
                inst("pbs", "pbs"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.out", "pbs.in", "10 mm"),
                fs("t", "pbs.out_t", "ds.in", "10 mm"),
                fs("r", "pbs.out_r", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence()],
        }
    )
    r = simulate_rates(setup, lib)
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(1e5 * 0.6)
    assert r.channels["tag.ch2"]["pair_singles_hz"] == pytest.approx(1e5 * 0.5)
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(1e5 * 0.3, rel=1e-9)


def test_diagnostics_for_unconnected_detector_and_off_peak_window(lib):
    PAIRS["src"] = cw_pair(1e5, flat=10)
    DETS["ds"], DETS["di"] = det(), det()
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
                cable("c2", "di", "ch2", "5 ns"),
            ],
            "analyses": [
                coincidence("late", window="1 ns", center="0 ns"),
                coincidence("bad", channels=("tag.ch1", "tag.ch9")),
                {"name": "z", "type": "nonsense"},
            ],
        }
    )
    r = simulate_rates(setup, lib)
    assert r.analyses["late"]["window_fraction"] < 1e-2
    assert "coincidence_outside_window" in codes(r)
    assert codes(r).count("bad_analysis") == 2
    assert "bad" not in r.analyses

    setup2 = setup_from_dict(
        {
            "instances": [inst("src", "pairbox"), inst("ds", "det_s"), inst("di", "det_i")],
            "links": [
                fs("a", "src.signal", "ds.in", "10 mm"),
                fs("b", "src.idler", "di.in", "10 mm"),
            ],
        }
    )
    with pytest.raises(Exception, match="ds.out"):
        simulate_rates(setup2, lib)


def test_detector_wired_to_a_non_tagger_is_reported(lib):
    PAIRS["src"] = cw_pair(1e5, flat=10)
    DETS["ds"], DETS["di"] = det(), det()
    TAGS["tag"] = TaggerModel()
    register("electronics", "RTSink")(RTSink)
    put(lib, "sink", "RTSink")
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
                inst("sink", "sink"),
            ],
            "links": [
                fs("a", "src.signal", "ds.in", "10 mm"),
                fs("b", "src.idler", "di.in", "10 mm"),
                {"id": "c1", "kind": "cable", "from": "ds.out", "to": "sink.in"},
                cable("c2", "di", "ch2"),
            ],
        }
    )
    r = simulate_rates(setup, lib)
    missing = [i for i in r.issues if i.code == "no_tagger_connection"]
    assert [i.subject for i in missing] == ["'ds'"]
    assert list(r.channels) == ["tag.ch2"]
    assert r.detectors["ds"]["pair_singles_hz"] == pytest.approx(1e5)


def test_has_rate_devices(lib):
    plain = setup_from_dict(
        {
            "instances": [inst("src", "laser"), inst("pd", "pd")],
            "links": [fs("a", "src.out", "pd.in")],
        }
    )
    assert not has_rate_devices(plain, lib)
    PAIRS["src"] = cw_pair(1e5, flat=3)
    DETS["ds"] = det()
    withdet = setup_from_dict({"instances": [inst("ds", "det_s")], "links": []})
    assert has_rate_devices(withdet, lib)


def test_performance_cw_5000_samples_and_pulsed_200x100(lib):
    n_s = np.arange(7_400_000 // 2 - 2500, 7_400_000 // 2 + 2500)
    x = (n_s - n_s[2500]) * DNU
    f = np.exp(-(x**2) / (4 * 1e11**2))
    amp = np.zeros((2, 2, len(n_s), 1), dtype=complex)
    amp[0, 0, :, 0] = f / np.linalg.norm(f)
    cw = PairSource(DNU, n_s, np.array([7_400_000]), amp, 1e6, mode(), mode(), 405e-9)
    start = time.perf_counter()
    r = two_arm(
        lib,
        cw,
        det(0.5, jitter_fwhm=50e-12),
        det(0.5, jitter_fwhm=50e-12),
        analyses=[
            coincidence(window="1 ns"),
            {
                "name": "h",
                "type": "histogram",
                "channels": ["tag.ch1", "tag.ch2"],
                "range": "5 ns",
                "bins": 200,
            },
        ],
    )
    t_cw = time.perf_counter() - start
    assert t_cw < 3.0
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(1e6 * 0.25)
    pair, _ = pulsed_pair(1e6, 80e6, 4e9, 2e9, count=200, n_sum_count=100)
    start = time.perf_counter()
    r = two_arm(
        lib,
        pair,
        det(0.5, jitter_fwhm=50e-12),
        det(0.5, jitter_fwhm=50e-12),
        analyses=[coincidence(window="1 ns")],
    )
    t_pulsed = time.perf_counter() - start
    assert t_pulsed < 3.0
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(1e6 * 0.25)


def test_report_and_cli_print_rates_section(lib, tmp_path, capsys):
    from sequoia.cli import _run
    from sequoia.engines.report import format_rates_report
    from sequoia.setup.io import save_setup

    PAIRS["src"] = cw_pair(1e5)
    DETS["ds"], DETS["di"] = det(0.5, dark_count_rate=10.0), det(0.4)
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
            "analyses": [
                coincidence("hit", window="500 ps"),
                {
                    "name": "hist",
                    "type": "histogram",
                    "channels": ["tag.ch1", "tag.ch2"],
                    "range": "1 ns",
                    "bins": 20,
                },
            ],
        }
    )
    text = format_rates_report(simulate_rates(setup, lib))
    for needle in ("Pair sources", "tag.ch1", "tag.ch2", "hit [coincidences]", "hist [histogram]"):
        assert needle in text
    path = tmp_path / "rates.yaml"
    save_setup(setup, path)
    assert _run(str(path), str(tmp_path / "lib")) == 0
    out = capsys.readouterr().out
    assert "Links" in out and "Photon rates" in out and "CAR" in out


def lattice_pair(state, skew=0.0, rate=1e5, half=12, n_c=3_700_000):
    n_s = np.arange(n_c - half, n_c + half + 1)
    x = (n_s - n_c) / 6.0
    f = np.exp(-(x**2) / 2) * (1 + skew * np.tanh(x))
    amp = np.zeros((2, 2, len(n_s), 1), dtype=complex)
    for (s, i), c in state.items():
        amp[s, i, :, 0] = c * f
    amp /= np.sqrt(np.sum(np.abs(amp) ** 2))
    return PairSource(DNU, n_s, np.array([2 * n_c]), amp, rate, mode(), mode(), 405e-9)


def collinear_bs_run(lib, pair, eta_s, eta_i, refl_h=0.5, refl_v=None):
    put(lib, "bsx", "BeamSplitter", reflectance=refl_h, reflectance_v=refl_v)
    PAIRS["src"], DETS["ds"], DETS["di"] = pair, det(eta_s), det(eta_i)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox_c"),
                inst("bs", "bsx"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.out", "bs.in_a", "10 mm"),
                fs("c", "bs.out_a", "ds.in", "10 mm"),
                fs("d", "bs.out_b", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence(window="9 ns")],
        }
    )
    return simulate_rates(setup, lib)


def first_quantised_expectation(pair, t_out_a, t_out_b):
    amp = pair.amplitude[:, :, :, 0]
    n_sum = int(pair.n_sum[0])
    index = {int(n): k for k, n in enumerate(pair.n_signal)}
    nb = len(pair.n_signal)
    psi = np.zeros((2, nb, 2, nb), dtype=complex)
    for j, n in enumerate(pair.n_signal):
        psi[:, j, :, index[n_sum - int(n)]] += amp[:, :, j]
    psi = psi + psi.transpose(2, 3, 0, 1)
    norm = np.linalg.norm(psi)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    p = np.sum(np.abs(psi / norm) ** 2, axis=(1, 3))
    ta, tb = np.asarray(t_out_a), np.asarray(t_out_b)
    coinc = sum(p[u, v] * (ta[u] * tb[v] + tb[u] * ta[v]) for u in range(2) for v in range(2))
    single_a = sum(p[u, v] * (ta[u] + ta[v]) for u in range(2) for v in range(2))
    single_b = sum(p[u, v] * (tb[u] + tb[v]) for u in range(2) for v in range(2))
    return coinc, single_a, single_b


@pytest.mark.parametrize("skew", [0.0, 0.6])
@pytest.mark.parametrize("refl", [(0.5, None), (0.3, 0.7)])
@pytest.mark.parametrize(
    "state",
    [
        {(0, 0): 1.0},
        {(0, 0): 1.0, (1, 1): 1.0},
        {(0, 1): 1.0, (1, 0): 1.0},
        {(0, 0): 1.0, (1, 1): 1j},
        {(0, 1): 1.0},
    ],
)
def test_same_port_indistinguishable_pairs_use_bosonic_normalisation(lib, state, refl, skew):
    rate, eta_s, eta_i = 1e5, 0.7, 0.4
    pair = lattice_pair(state, skew, rate)
    r_h, r_v = refl
    r_v = r_h if r_v is None else r_v
    r = collinear_bs_run(lib, pair, eta_s, eta_i, r_h, refl[1])
    coinc, single_a, single_b = first_quantised_expectation(pair, [1 - r_h, 1 - r_v], [r_h, r_v])
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(
        rate * eta_s * eta_i * coinc, rel=1e-9
    )
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(
        rate * eta_s * single_a, rel=1e-9
    )
    assert r.channels["tag.ch2"]["pair_singles_hz"] == pytest.approx(
        rate * eta_i * single_b, rel=1e-9
    )


def test_same_port_antisymmetric_state_has_no_pairs(lib):
    pair = lattice_pair({(0, 1): 1.0, (1, 0): -1.0})
    r = collinear_bs_run(lib, pair, 1.0, 1.0)
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(0.0, abs=1e-6)
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(0.0, abs=1e-6)
    assert "bosonic_null_state" in codes(r)


def test_pulsed_heralded_g2_is_twice_pair_number_per_pulse_and_window_independent(lib):
    rep, mu = 80e6, 0.01
    values = []
    for window in ("0.5 ns", "1 ns", "3 ns"):
        r = hbt_run(lib, window, pair=cw_pair(mu * rep, rep=rep))
        values.append(r.analyses["h"]["g2_heralded"])
    for g in values:
        assert g == pytest.approx(2 * mu, rel=0.03)
    assert max(values) / min(values) < 1.02


def gaussian_overlap_squared(w1, w2, distance, wavelength=810e-9):
    x = np.linspace(-8e-3, 8e-3, 400001)
    k = 2 * math.pi / wavelength

    def field(w0):
        zr = math.pi * w0**2 / wavelength
        w = w0 * math.sqrt(1 + (distance / zr) ** 2)
        radius = distance * (1 + (zr / distance) ** 2)
        return (2 / (math.pi * w**2)) ** 0.25 * np.exp(-(x**2) / w**2 - 0.5j * k * x**2 / radius)

    integral = np.trapezoid(np.conj(field(w1)) * field(w2), x)
    return abs(integral) ** 4


def hom_two_mode_run(lib, pair, delta=0.0):
    PAIRS["src"] = pair
    DETS["ds"], DETS["di"] = det(0.5), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("bs", "bs"),
                inst("ds", "det_s"),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "bs.in_a", f"{(100e-3 + delta * C) * 1e3:.9f} mm"),
                fs("b", "src.idler", "bs.in_b", "100 mm"),
                fs("c", "bs.out_a", "ds.in", "10 mm"),
                fs("d", "bs.out_b", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence(window="3 ns")],
        }
    )
    return simulate_rates(setup, lib)


def test_hom_visibility_is_squared_spatial_mode_overlap(lib):
    rate = 1e5
    base_pair = cw_pair(rate)
    wide = GaussianMode.from_waist(810e-9, 1e-3)
    narrow = GaussianMode.from_waist(810e-9, 0.2e-3)
    pair = replace(base_pair, signal_mode=wide, idler_mode=narrow)
    visibility = gaussian_overlap_squared(1e-3, 0.2e-3, 0.11)
    assert 0.1 < visibility < 0.2
    r = hom_two_mode_run(lib, pair)
    expected = rate * 0.5**2 * 0.5 * (1 - visibility)
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(expected, rel=2e-3)
    far = hom_two_mode_run(lib, pair, 1e-9).analyses["c"]["coincidence_rate_hz"]
    assert far == pytest.approx(rate * 0.5**2 * 0.5, rel=2e-3)
    assert "photon_mode_mismatch" in codes(r)
    matched = hom_two_mode_run(lib, base_pair)
    assert matched.analyses["c"]["coincidence_rate_hz"] < 1e-6 * rate
    assert "photon_mode_mismatch" not in codes(matched)


def aliasing_run(lib, idler_delay, center):
    length = f"{idler_delay * C * 1e3:.6f} mm"
    analyses = [coincidence("c", window="1 ns", center=center)]
    return two_arm(lib, cw_pair(1e5), det(0.5), det(0.5), idl_len=length, analyses=analyses)


def test_single_path_delay_beyond_half_period_is_flagged_and_window_fraction_is_physical(lib):
    r = aliasing_run(lib, 30e-9, "0 ns")
    assert "delay_aliasing" in codes(r)
    assert r.analyses["c"]["window_fraction"] < 1e-3
    assert "coincidence_outside_window" in codes(r)


def test_auto_centre_reports_the_unwrapped_delay(lib):
    r = aliasing_run(lib, 30e-9, "auto")
    assert "delay_aliasing" in codes(r)
    assert r.analyses["c"]["center_s"] == pytest.approx(10e-3 / C - 30e-9, abs=2e-12)
    assert r.analyses["c"]["window_fraction"] == pytest.approx(1.0, abs=1e-3)


def test_explicit_centre_at_the_long_delay_gives_full_window_fraction(lib):
    r = aliasing_run(lib, 30e-9, "-29.9667 ns")
    assert r.analyses["c"]["window_fraction"] == pytest.approx(1.0, abs=1e-3)
    assert "coincidence_outside_window" not in codes(r)


def test_delay_below_half_period_is_not_flagged(lib):
    r = aliasing_run(lib, 4e-9, "-4 ns")
    assert "delay_aliasing" not in codes(r)
    assert r.analyses["c"]["window_fraction"] == pytest.approx(1.0, abs=1e-3)


def aperture_pair_run(lib, diameter):
    put(lib, "spad_ap", "SPAD", efficiency=0.6, active_diameter=diameter)
    put(lib, "spad_in", "SPAD", efficiency=0.6)
    PAIRS["src"], DETS["di"] = cw_pair(1e5, flat=10), det(0.5)
    TAGS["tag"] = TaggerModel()
    setup = setup_from_dict(
        {
            "instances": [
                inst("src", "pairbox"),
                inst("ds", "spad_ap", {"input": "free_space"}),
                inst("di", "det_i"),
                inst("tag", "tagger"),
            ],
            "links": [
                fs("a", "src.signal", "ds.in", "10 mm"),
                fs("b", "src.idler", "di.in", "10 mm"),
                cable("c1", "ds", "ch1"),
                cable("c2", "di", "ch2"),
            ],
            "analyses": [coincidence(window="9 ns")],
        }
    )
    return simulate_rates(setup, lib)


def test_spad_active_area_clips_photon_rates(lib):
    rate = 1e5
    w = 1e-3 * math.sqrt(1 + (0.01 / (math.pi * 1e-6 / 810e-9)) ** 2)
    transmission = 1 - math.exp(-2 * (0.5e-3) ** 2 / w**2)
    r = aperture_pair_run(lib, "1 mm")
    assert r.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(
        rate * 0.6 * transmission, rel=1e-6
    )
    assert r.analyses["c"]["coincidence_rate_hz"] == pytest.approx(
        rate * 0.6 * 0.5 * transmission, rel=1e-6
    )
    wide = aperture_pair_run(lib, "20 mm")
    assert wide.channels["tag.ch1"]["pair_singles_hz"] == pytest.approx(rate * 0.6, rel=1e-6)


def test_spad_active_area_clips_classical_leakage(lib):
    put(lib, "laser_l", "Laser", wavelength="810 nm", power="1 pW", waist_radius="1 mm")
    put(lib, "spad_small", "SPAD", efficiency=0.5, active_diameter="1 mm")
    put(lib, "spad_big", "SPAD", efficiency=0.5, active_diameter="30 mm")
    leak = {}
    for part in ("spad_small", "spad_big"):
        setup = setup_from_dict(
            {
                "instances": [
                    inst("src", "laser_l"),
                    inst("ds", part, {"input": "free_space"}),
                    inst("tag", "tagger"),
                ],
                "links": [fs("a", "src.out", "ds.in", "100 mm"), cable("c1", "ds", "ch1")],
            }
        )
        PAIRS["src"] = cw_pair(1e5, flat=2)
        TAGS["tag"] = TaggerModel()
        leak[part] = simulate_rates(setup, lib).channels["tag.ch1"]["leakage_hz"]
    w = 1e-3 * math.sqrt(1 + (0.1 / (math.pi * 1e-6 / 810e-9)) ** 2)
    transmission = 1 - math.exp(-2 * (0.5e-3) ** 2 / w**2)
    assert leak["spad_small"] / leak["spad_big"] == pytest.approx(transmission, rel=1e-6)


def test_bad_channel_names_are_reported_not_crashed(lib):
    PAIRS["src"] = cw_pair(1e5, flat=3)
    r = two_arm(
        lib,
        PAIRS["src"],
        det(),
        det(),
        analyses=[
            {"name": "x", "type": "coincidences", "channels": [1, 2], "window": "1 ns"},
            {
                "name": "y",
                "type": "g2",
                "channels": ["tag.ch1", "tag.ch2"],
                "range": "1 ns",
                "bins": "many",
            },
        ],
    )
    assert codes(r).count("bad_analysis") == 2


def test_internal_errors_in_analyses_are_not_masked(lib, monkeypatch):
    from sequoia.engines import rates

    def broken(self, name, spec):
        raise TypeError("internal bug")

    monkeypatch.setattr(rates._Engine, "a_coincidences", broken)
    with pytest.raises(TypeError, match="internal bug"):
        two_arm(lib, cw_pair(1e5, flat=3), det(), det(), analyses=[coincidence()])


def monte_carlo_cascade_rate(rate, tau_first, tau_second, count=2_000_000, seed=11):
    times = np.cumsum(np.random.default_rng(seed).exponential(1.0 / rate, count))
    for tau in (tau_first, tau_second):
        kept, last = [], -1.0
        for t in times.tolist():
            if t - last >= tau:
                kept.append(t)
                last = t
        times = np.array(kept)
    return len(times) / times[-1] if len(times) else 0.0


def test_tagger_dead_time_in_series_with_detector_matches_monte_carlo(lib):
    rate, eta, tau_d, tau_t = 5e6, 1.0, 50e-9, 120e-9
    r = two_arm(
        lib,
        cw_pair(rate, flat=10),
        det(eta, dead_time=tau_d),
        det(eta, dead_time=tau_d),
        tagger=TaggerModel(dead_time=tau_t),
    )
    assert r.channels["tag.ch1"]["singles_hz"] == pytest.approx(
        monte_carlo_cascade_rate(rate * eta, tau_d, tau_t), rel=3e-3
    )


def test_shorter_tagger_dead_time_adds_no_loss_after_longer_detector_dead_time(lib):
    rate, tau_d = 5e6, 100e-9
    r = two_arm(
        lib,
        cw_pair(rate, flat=10),
        det(1.0, dead_time=tau_d),
        det(1.0, dead_time=tau_d),
        tagger=TaggerModel(dead_time=60e-9),
    )
    assert r.channels["tag.ch1"]["singles_hz"] == pytest.approx(rate / (1 + rate * tau_d))


def test_afterpulse_tau_does_not_change_rates(lib):
    out = []
    for tau in (0.0, 1e-6):
        r = two_arm(
            lib,
            cw_pair(1e5, flat=10),
            det(0.5, afterpulse_probability=0.02, afterpulse_tau=tau),
            det(0.5),
        )
        out.append(r.channels["tag.ch1"]["singles_hz"])
    assert out[0] == out[1]
