import math

import numpy as np
import pytest
import spdcalc
from m3_helpers import (
    LIBRARY,
    C,
    EmissionCache,
    band_average,
    crystal_emission,
    fiber_transmission,
    gaussian_overlap,
    load_example,
    marginals,
    part,
    photon_rate,
    remove_instance,
    run,
    set_length,
    set_overrides,
    set_settings,
    window_fraction,
)

from sequoia.analysis import metrics
from sequoia.setup.model import FreeSpaceLink

KELVIN = 273.15


@pytest.fixture(scope="module")
def emission_cache():
    mp = pytest.MonkeyPatch()
    cache = EmissionCache(mp)
    yield cache
    mp.undo()


def codes(result):
    return [i.code for i in result.issues]


def link_lengths(setup):
    return {lk.id: lk.length.si for lk in setup.links if lk.kind != "cable"}


def spdc_config(
    length, pump_waist, temperature_c, deff_pm, bandwidth_nm=0.0405, theta=90.0, period_um="auto"
):
    return f"""
crystal:
  kind: KTP
  pm_type: Type2_e_eo
  phi_deg: 0
  theta_deg: {theta}
  length_um: {length * 1e6}
  temperature_c: {temperature_c}
pump:
  wavelength_nm: 405.0
  waist_um: {pump_waist * 1e6}
  bandwidth_nm: {bandwidth_nm}
  average_power_mw: 1.0
signal:
  wavelength_nm: 810.0
  phi_deg: 0
  theta_deg: 0
  waist_um: 50.0
  waist_position_um: auto
idler:
  wavelength_nm: 810.0
  phi_deg: 180
  theta_deg: 0
  waist_um: 50.0
  waist_position_um: auto
periodic_poling:
  poling_period_um: {period_um}
deff_pm_per_volt: {deff_pm}
"""


def design_period_um(length, pump_waist, design_temperature_c, deff_pm):
    cfg = spdc_config(length, pump_waist, design_temperature_c, deff_pm)
    return spdcalc.SPDC.from_yaml(cfg).poling_period_um


def apparent_waist_q(wavelength, waist, length, position_um, temperature_c, kind="KTP"):
    index = float(
        np.mean(spdcalc.get_crystal_indices(kind, wavelength * 1e9, temperature_c + KELVIN))
    )
    z = (length / 2.0 - position_um * 1e-6) / index
    return z + 1j * math.pi * waist**2 / wavelength


def propagate_q(q, ops):
    for kind, value in ops:
        q = q + value if kind == "d" else 1.0 / (1.0 / q - 1.0 / value)
    return q


# ---------------------------------------------------------------------------
# HBT
# ---------------------------------------------------------------------------


class HBT:
    """Independent hand calculation of the hbt_heralded setup from the library part data."""

    def __init__(self, setup, result, emission):
        self.setup, self.result = setup, result
        self.lengths = link_lengths(setup)
        src = emission.source
        self.rate = result.sources["crystal"]["rate_hz"]
        self.spec_s, self.spec_i = marginals(src)
        cube = part("polarization/pbs_cube_810")
        self.tp, self.ts = cube.transmittance_p.si, cube.transmittance_s.si
        self.rs, self.rp = cube.reflectance_s.si, cube.reflectance_p.si
        self.lp = part("filters/lp_650_pumpblock").transmission
        self.bp = part("filters/bp_810_10nm").transmission
        self.spad = part("detectors/spad_si_generic")
        self.cpl_t = part("fiber/coupler_f11mm_810").transmission.si
        self.fbs = part("fiber/beamsplitter_50_50_810")
        self.fiber = part("fiber/smf_810")
        self.pump = part("sources/laser_405_cw")
        self.crystal = part("nonlinear/ppktp_type2_10mm")
        self.window = 1e-9
        f_pump = part("freespace/lens_f200mm").focal_length.si
        q = propagate_q(
            1j * math.pi * self.pump.waist_radius.si**2 / 405e-9,
            [("d", self.lengths["s_pump"]), ("f", f_pump), ("d", self.lengths["s_focus"])],
        )
        self.pump_waist_at_crystal = math.sqrt(405e-9 * q.imag / math.pi)
        self.spdc = spdcalc.SPDC.from_yaml(
            spdc_config(
                self.crystal.length.si,
                self.pump_waist_at_crystal,
                25.0,
                self.crystal.d_eff.si * 1e12,
            )
        )
        self.f_w = window_fraction(
            self.window,
            [self.spad.jitter_fwhm.si, self.spad.jitter_fwhm.si, 10e-12, 10e-12],
        )
        self.coupling = {
            "signal": self.photon_coupling(src.signal_mode.wavelength, "signal"),
            "idler": self.photon_coupling(src.idler_mode.wavelength, "idler"),
        }

    def temperature_c(self):
        return 25.0

    def to_pbs(self):
        return sum(self.lengths.get(k, 0.0) for k in ("s_lp", "s_bp", "s_pbs", "s_lp_gone"))

    def common_ops(self):
        L = self.lengths
        return [
            ("d", L["s_collect"]),
            ("f", part("freespace/lens_f200mm").focal_length.si),
            ("d", self.to_pbs()),
        ]

    def tail_ops(self, arm):
        L = self.lengths
        f = part("fiber/coupler_f11mm_810").focal_length.si
        return [("d", L["s_signal" if arm == "signal" else "s_idler"]), ("f", f), ("d", f)]

    def photon_coupling(self, wavelength, role):
        waist = self.crystal.signal_waist.si if role == "signal" else self.crystal.idler_waist.si
        position = (
            self.spdc.signal_waist_position_um
            if role == "signal"
            else self.spdc.idler_waist_position_um
        )
        q = apparent_waist_q(
            wavelength, waist, self.crystal.length.si, position, self.temperature_c()
        )
        arm = "signal" if role == "signal" else "idler"
        q = propagate_q(q, self.common_ops() + self.tail_ops(arm))
        w_f = self.fiber.mode_field_diameter.si / 2.0
        return gaussian_overlap(q, 1j * math.pi * w_f**2 / wavelength, wavelength)

    def spectral(self, role):
        spec = self.spec_s if role == "signal" else self.spec_i
        return band_average(spec, self.lp, self.bp, self.spad.efficiency)

    def fiber_t(self, ids):
        return float(np.prod([fiber_transmission(self.fiber, self.lengths[i]) for i in ids]))

    def chain(self, role, detector):
        pbs = {
            ("signal", "herald"): self.rp,
            ("idler", "herald"): self.rs,
            ("signal", "a"): self.tp,
            ("idler", "a"): self.ts,
        }[(role, detector)]
        eta = self.spectral(role) * pbs * self.cpl_t * self.coupling[role]
        if detector == "herald":
            return eta * self.fiber_t(["f_idler"])
        split = 0.5 * self.fbs.transmission.si
        return eta * self.fiber_t(["f_signal", "f_a"]) * split

    def pump_chain(self, detector, wavelength=405e-9):
        L = self.lengths
        w0 = self.pump.waist_radius.si
        f_pump = part("freespace/lens_f200mm").focal_length.si
        ops = [
            ("d", L["s_pump"]),
            ("f", f_pump),
            ("d", L["s_focus"]),
            ("d", L["s_collect"]),
            ("f", f_pump),
            ("d", self.to_pbs()),
        ]
        arm = "idler" if detector == "herald" else "signal"
        f = part("fiber/coupler_f11mm_810").focal_length.si
        ops += [("d", L["s_idler" if arm == "idler" else "s_signal"]), ("f", f), ("d", f)]
        q = propagate_q(1j * math.pi * w0**2 / wavelength, ops)
        w_f = self.fiber.mode_field_diameter.si / 2.0
        coupling = gaussian_overlap(q, 1j * math.pi * w_f**2 / wavelength, wavelength)
        per = self.pump.polarization_extinction_ratio.si
        h_frac, v_frac = per / (per + 1.0), 1.0 / (per + 1.0)
        if detector == "herald":
            pbs = h_frac * self.rp + v_frac * self.rs
        else:
            pbs = h_frac * self.tp + v_frac * self.ts
        t = float(self.lp(wavelength)) * float(self.bp(wavelength)) * pbs * self.cpl_t * coupling
        if detector == "herald":
            t *= self.fiber_t(["f_idler"])
        else:
            t *= self.fiber_t(["f_signal", "f_a"]) * 0.5 * self.fbs.transmission.si
        return (
            t
            * photon_rate(self.pump.power.si, wavelength)
            * float(self.spad.efficiency(wavelength))
        )

    def incident(self, detector):
        pair = self.rate * (self.chain("signal", detector) + self.chain("idler", detector))
        dark = self.spad.dark_count_rate.si
        leak = self.pump_chain(detector)
        return pair, dark, leak

    def survival(self, detector):
        pair, dark, leak = self.incident(detector)
        return 1.0 / (1.0 + (pair + dark + leak) * self.spad.dead_time.si)

    def singles(self, detector):
        return sum(self.incident(detector)) * self.survival(detector)

    def coincidence_all_time(self):
        s, i = "signal", "idler"
        return self.rate * (
            self.chain(s, "a") * self.chain(i, "herald")
            + self.chain(s, "herald") * self.chain(i, "a")
        )


@pytest.fixture(scope="module")
def hbt(emission_cache):
    setup = load_example("hbt_heralded")
    result = run(setup)
    emission = crystal_emission(setup, result)
    return HBT(setup, result, emission), setup, result


def test_hbt_example_has_no_unexpected_diagnostics(hbt):
    _, _, result = hbt
    bad = {
        "pump_leakage",
        "delay_aliasing",
        "pair_delay_window",
        "pair_spectral_resolution",
        "coincidence_outside_window",
        "no_tagger_connection",
        "low_coupling",
        "aperture_clipping",
        "bad_analysis",
        "path_mode_mismatch",
        "window_exceeds_period",
        "timing_resolution",
        "curve_extrapolation",
    }
    assert not bad & set(codes(result)), codes(result)


def test_hbt_pair_rate_matches_spdcalc_efficiencies(hbt):
    h, setup, result = hbt
    pump = result.classical.links["s_focus"].end
    assert math.sqrt(pump.mode.waist_x * pump.mode.waist_y) == pytest.approx(
        h.pump_waist_at_crystal, rel=1e-6
    )
    s = spdcalc.SPDC.from_yaml(
        spdc_config(h.crystal.length.si, h.pump_waist_at_crystal, 25.0, h.crystal.d_eff.si * 1e12)
    )
    rng = s.optimum_range(120)
    x, y = np.asarray(rng.x_values()), np.asarray(rng.y_values())
    jsi = np.asarray(s.joint_spectrum().jsi_range(rng))
    reference = jsi.sum() * (x[1] - x[0]) * (y[1] - y[0]) * h.pump.power.si * 1e3
    assert h.rate == pytest.approx(reference, rel=0.03)
    corrected = s.efficiencies(rng)["coincidences_hz"] * h.pump.power.si * 1e3
    assert 3.0 < h.rate / corrected < 7.0


def test_hbt_channel_singles_match_hand_calculation(hbt):
    h, _, result = hbt
    for name, det in (("tagger.ch1", "herald"), ("tagger.ch2", "a"), ("tagger.ch3", "a")):
        ch = result.channels[name]
        pair, dark, leak = h.incident(det)
        assert ch["incident_pair_hz"] == pytest.approx(pair, rel=5e-3)
        assert ch["dark_hz"] == pytest.approx(dark, rel=1e-12)
        assert ch["leakage_hz"] == pytest.approx(leak, rel=0.03)
        assert ch["survival"] == pytest.approx(h.survival(det), rel=1e-4)
        assert ch["singles_hz"] == pytest.approx(h.singles(det), rel=5e-3)


def test_hbt_coincidences_window_car_and_heralding(hbt):
    h, _, result = hbt
    ana = result.analyses["herald_a"]
    c_all = h.coincidence_all_time()
    surv = h.survival("herald") * h.survival("a")
    assert ana["window_fraction"] == pytest.approx(h.f_w, rel=2e-3)
    assert ana["coincidence_rate_hz"] == pytest.approx(c_all * surv, rel=5e-3)
    true_expected = c_all * surv * h.f_w
    assert ana["true_hz"] == pytest.approx(true_expected, rel=5e-3)
    s_h, s_a = h.singles("herald"), h.singles("a")
    acc = metrics.accidentals_cw(s_h, s_a, h.window)
    assert ana["accidental_hz"] == pytest.approx(acc, rel=5e-3)
    assert ana["car"] == pytest.approx(metrics.car(true_expected, acc), rel=1e-2)
    assert ana["car"] > 3e2
    eta_a_given_h = metrics.heralding_efficiency(true_expected, s_h)
    assert ana["heralding_b_given_a"] == pytest.approx(eta_a_given_h, rel=1e-2)
    assert ana["heralding_a_given_b"] == pytest.approx(true_expected / s_a, rel=1e-2)
    klyshko = (
        h.chain("signal", "a") * h.chain("idler", "herald")
        + h.chain("signal", "herald") * h.chain("idler", "a")
    ) / (h.chain("signal", "herald") + h.chain("idler", "herald"))
    assert ana["heralding_b_given_a"] == pytest.approx(klyshko * h.f_w * h.survival("a"), rel=1e-2)
    assert 0.2 < ana["heralding_b_given_a"] < 0.25


def test_hbt_heralded_g2_is_small_and_consistent_with_formula(hbt):
    h, _, result = hbt
    g = result.analyses["hbt"]
    ana_a, ana_b = result.analyses["herald_a"], result.analyses["herald_b"]
    s_h, s_a, s_b = (result.channels[f"tagger.ch{i}"]["singles_hz"] for i in (1, 2, 3))
    w = g["window_s"]
    r_ha = ana_a["true_hz"] + ana_a["accidental_hz"]
    r_hb = ana_b["true_hz"] + ana_b["accidental_hz"]
    b1, b2 = g["cw_bunching"], g["cw_bunching_accidental"]
    pair_a = ana_a["true_hz"]
    pair_b = ana_b["true_hz"]
    triples = (
        pair_a * (s_b * w + b1 * pair_b * w)
        + pair_b * (s_a * w + b1 * pair_a * w)
        + s_h * s_a * s_b * w**2 * (1.0 + b2)
    )
    assert 0.0 < b1 < 0.01 and 0.0 < b2 < 0.01
    assert g["triples_hz"] == pytest.approx(triples, rel=5e-3)
    assert g["g2_heralded"] == pytest.approx(
        metrics.g2_heralded(triples, s_h, r_ha, r_hb), rel=5e-3
    )
    s_h_hand, s_a_hand = h.singles("herald"), h.singles("a")
    true_hand = h.coincidence_all_time() * h.survival("herald") * h.survival("a") * h.f_w
    r_hand = true_hand + w * s_h_hand * s_a_hand
    triples_hand = 2.0 * true_hand * s_a_hand * w + s_h_hand * s_a_hand**2 * w**2
    expected = triples_hand * s_h_hand / r_hand**2
    assert g["g2_heralded"] == pytest.approx(expected, rel=2e-2)
    assert 0.0 < g["g2_heralded"] < 1e-2


def test_hbt_g2_histogram_between_split_outputs_has_no_thermal_bunching_at_jitter(hbt):
    _, _, result = hbt
    g2 = result.analyses["g2_ab"]
    assert np.all(np.isfinite(g2["g2"]))
    assert float(np.median(g2["g2"])) == pytest.approx(1.0, abs=1e-3)


def test_hbt_delay_peak_equals_fiber_group_delay_and_does_not_alias(hbt):
    h, setup, result = hbt
    ng = h.fiber.group_index
    lengths = h.lengths
    expected = ng * (lengths["f_idler"] - lengths["f_signal"] - lengths["f_a"]) / C
    peak = result.analyses["herald_a"]["peak_s"]
    assert peak == pytest.approx(expected, abs=20e-12)
    assert abs(expected) > 9e-9
    assert 1.0 / result.sources["crystal"]["dnu_hz"] > 2.0 * abs(expected)
    assert "delay_aliasing" not in codes(result)


def test_hbt_fiber_length_sweep_tracks_delay_beyond_half_period(emission_cache):
    setup = load_example("hbt_heralded")
    base = run(setup)
    period = 1.0 / base.sources["crystal"]["dnu_hz"]
    ng = part("fiber/smf_810").group_index
    for extra in (0.0, 1.0, 2.0):
        s = setup.model_copy(deep=True)
        set_length(s, "f_signal", 2.0 + extra)
        r = run(s)
        expected = -ng * (2.0 + extra + 1.0 - 1.0) / C
        assert r.analyses["herald_a"]["peak_s"] == pytest.approx(expected, abs=20e-12)
        assert abs(expected) < period / 2.0
        assert "delay_aliasing" not in codes(r)
        assert r.analyses["herald_a"]["window_fraction"] == pytest.approx(
            base.analyses["herald_a"]["window_fraction"], rel=1e-6
        )
    s = setup.model_copy(deep=True)
    set_length(s, "f_signal", 7.0)
    r = run(s)
    assert abs(ng * 7.0 / C) > period / 2.0
    assert "delay_aliasing" not in codes(r)
    assert r.analyses["herald_a"]["peak_s"] == pytest.approx(-ng * 7.0 / C, abs=20e-12)
    extra_loss = fiber_transmission(part("fiber/smf_810"), 5.0)
    assert r.analyses["herald_a"]["true_hz"] == pytest.approx(
        base.analyses["herald_a"]["true_hz"] * extra_loss, rel=1e-3
    )


def test_hbt_with_short_lattice_period_is_diagnosed_and_peak_reported_unwrapped(emission_cache):
    setup = load_example("hbt_heralded")
    set_overrides(setup, "crystal", delay_window="10 ns", max_samples=100000)
    r = run(setup)
    period = 1.0 / r.sources["crystal"]["dnu_hz"]
    assert period == pytest.approx(10e-9, rel=1e-3)
    true_delay = -part("fiber/smf_810").group_index * 2.0 / C
    assert abs(true_delay) > period / 2.0
    assert "delay_aliasing" not in codes(r)
    peak = r.analyses["herald_a"]["peak_s"]
    assert peak == pytest.approx(true_delay, abs=20e-12)
    nominal = run(load_example("hbt_heralded"))
    assert r.analyses["herald_a"]["true_hz"] == pytest.approx(
        nominal.analyses["herald_a"]["true_hz"], rel=1e-3
    )


def test_hbt_pump_leakage_without_pump_blocking_filter(emission_cache):
    setup = load_example("hbt_heralded")
    nominal = run(setup)
    assert "pump_leakage" not in codes(nominal)
    remove_instance(setup, "lp")
    setup.links.append(
        FreeSpaceLink(
            id="s_lp_gone",
            kind="free_space",
            **{"from": "collect_lens.out", "to": "bp.in"},
            length="75 mm",
        )
    )
    result = run(setup)
    leak_codes = [i for i in result.issues if i.code == "pump_leakage"]
    assert len(leak_codes) == 3
    h = HBT(setup, result, crystal_emission(setup, result))
    h.lp = lambda lam: np.ones_like(np.asarray(lam, dtype=float))

    def pump_chain_no_lp(det):
        return h.pump_chain(det)

    for name, det in (("tagger.ch1", "herald"), ("tagger.ch2", "a"), ("tagger.ch3", "a")):
        ch = result.channels[name]
        leak = pump_chain_no_lp(det)
        assert ch["leakage_hz"] == pytest.approx(leak, rel=0.03)
        assert ch["leakage_hz"] > 1e8
        assert ch["leakage_hz"] > 1e3 * nominal.channels[name]["leakage_hz"]
        assert ch["singles_hz"] < 1.0 / h.spad.dead_time.si
        assert ch["singles_hz"] > 0.7 / h.spad.dead_time.si
    ratio = (
        result.channels["tagger.ch1"]["leakage_hz"] / nominal.channels["tagger.ch1"]["leakage_hz"]
    )
    assert ratio == pytest.approx(
        1.0 / float(part("filters/lp_650_pumpblock").transmission(405e-9)), rel=0.02
    )


# ---------------------------------------------------------------------------
# HOM
# ---------------------------------------------------------------------------


def hom_pump_waist(setup):
    L = link_lengths(setup)
    f = part("freespace/lens_f200mm").focal_length.si
    laser = part("sources/laser_405_cw")
    q = propagate_q(
        1j * math.pi * laser.waist_radius.si**2 / 405e-9,
        [("d", L["s_pump"]), ("f", f), ("d", L["s_focus"])],
    )
    return math.sqrt(405e-9 * q.imag / math.pi)


def hom_engine_series(setup, delays_ps):
    base = link_lengths(setup)["delay_arm"]
    out = []
    for d in delays_ps:
        s = setup.model_copy(deep=True)
        set_length(s, "delay_arm", base + d * 1e-12 * C)
        r = run(s)
        out.append(r.analyses["hom"]["coincidence_rate_hz"])
    return np.array(out)


def hom_spdcalc_series(setup, temperature_c, bandwidth_nm, delays_ps):
    crystal = part("nonlinear/ppktp_type2_10mm")
    s = spdcalc.SPDC.from_yaml(
        spdc_config(
            crystal.length.si,
            hom_pump_waist(setup),
            temperature_c,
            crystal.d_eff.si * 1e12,
            bandwidth_nm=bandwidth_nm,
            period_um=design_period_um(
                crystal.length.si,
                hom_pump_waist(setup),
                crystal.design_temperature.si - KELVIN,
                crystal.d_eff.si * 1e12,
            ),
        )
    )
    return np.array(s.hom_rate_series([d * 1e-12 for d in delays_ps], s.optimum_range(300)))


def normalised(series, delays_ps, far=4.5, centre=2.25):
    mask = np.abs(np.asarray(delays_ps) - centre) >= far
    return series / series[mask].mean()


def dip_fwhm(delays_ps, y):
    d = np.asarray(delays_ps, dtype=float)
    k = int(np.argmin(y))
    half = 0.5 * (1.0 + y[k])
    left = np.interp(half, y[: k + 1][::-1], d[: k + 1][::-1])
    right = np.interp(half, y[k:], d[k:])
    return left, right


@pytest.fixture(scope="module")
def hom_setup():
    return load_example("hom")


def test_hom_example_runs_clean_and_dip_is_where_walkoff_puts_it(emission_cache, hom_setup):
    r = run(hom_setup)
    bad = {
        "delay_aliasing",
        "pair_delay_window",
        "pump_leakage",
        "path_mode_mismatch",
        "coincidence_outside_window",
        "bad_analysis",
        "no_tagger_connection",
    }
    assert not bad & set(codes(r)), codes(r)
    delays = np.arange(1.5, 3.01, 0.25)
    y = hom_engine_series(hom_setup, delays)
    far = hom_engine_series(hom_setup, [-8.0, 12.0]).mean()
    rel = y / far
    assert rel.min() < 0.01
    assert delays[int(np.argmin(rel))] == pytest.approx(2.25, abs=0.25)


def cw_hom_from_spdcalc_jsa(setup, temperature_c, delays_ps):
    """CW (single sum frequency) HOM rate from spdcalc's own jsa(): 1 - Re sum F(nu) F*(nu_bar)
    exp(2 pi i (nu - nu_bar) tau) / sum |F|^2, with F(nu) = jsa(nu, nu_p - nu) and the exchanged
    amplitude jsa(nu_p - nu, nu)."""
    crystal = part("nonlinear/ppktp_type2_10mm")
    waist = hom_pump_waist(setup)
    period = design_period_um(
        crystal.length.si, waist, crystal.design_temperature.si - KELVIN, crystal.d_eff.si * 1e12
    )
    s = spdcalc.SPDC.from_yaml(
        spdc_config(
            crystal.length.si,
            waist,
            temperature_c,
            crystal.d_eff.si * 1e12,
            period_um=period,
        )
    )
    js = s.joint_spectrum(spdcalc.Integrator.gauss_legendre(100))
    nu_p = C / 405e-9
    nu = np.linspace(C / 812.5e-9, C / 807.5e-9, 2500)
    w, w_bar = 2.0 * np.pi * nu, 2.0 * np.pi * (nu_p - nu)
    f = np.nan_to_num(np.array([js.jsa(a, b) for a, b in zip(w, w_bar, strict=True)]))
    g = np.nan_to_num(np.array([js.jsa(b, a) for a, b in zip(w, w_bar, strict=True)]))
    total = np.sum(np.abs(f) ** 2)
    cross = f * np.conj(g)
    return np.array(
        [
            1.0
            - np.real(np.sum(cross * np.exp(2j * np.pi * (nu - (nu_p - nu)) * d * 1e-12))) / total
            for d in delays_ps
        ]
    )


def test_hom_cw_dip_matches_spdcalc_jsa_exchange_integral(emission_cache, hom_setup):
    for temperature_c in (25.0, 27.4):
        setup = hom_setup.model_copy(deep=True)
        set_settings_temperature(setup, temperature_c)
        set_overrides(setup, "crystal", dnu="200 MHz")
        delays = np.arange(-3.0, 7.01, 0.5)
        eng = normalised(hom_engine_series(setup, delays), delays)
        ref = cw_hom_from_spdcalc_jsa(setup, temperature_c, delays)
        assert np.max(np.abs(eng - ref)) < 0.025, (temperature_c, np.max(np.abs(eng - ref)))
        l_e, r_e = dip_fwhm(delays, eng)
        l_r, r_r = dip_fwhm(delays, ref)
        assert (r_e - l_e) == pytest.approx(r_r - l_r, rel=0.03)
        assert 1.5 < r_e - l_e < 2.5
    fine = np.arange(2.0, 2.51, 0.05)
    setup = hom_setup.model_copy(deep=True)
    set_settings_temperature(setup, 25.0)
    set_overrides(setup, "crystal", dnu="200 MHz")
    far = hom_engine_series(setup, [-8.0, 12.0]).mean()
    floor = (hom_engine_series(setup, fine) / far).min()
    assert floor == pytest.approx(cw_hom_from_spdcalc_jsa(setup, 25.0, fine).min(), abs=0.025)
    assert floor == pytest.approx(0.05, abs=0.025)


def test_hom_finite_pump_bandwidth_matches_spdcalc_series(emission_cache, hom_setup):
    for bandwidth_nm, tolerance in ((0.0405, 0.01), (0.01, 0.01)):
        setup = hom_setup.model_copy(deep=True)
        set_settings_temperature(setup, 25.0)
        nu = C * bandwidth_nm * 1e-9 / (405e-9) ** 2
        set_overrides(setup, "pump", pulse_width=f"{0.441 / nu * 1e12} ps")
        delays = np.arange(-3.0, 7.01, 0.5)
        eng = normalised(hom_engine_series(setup, delays), delays)
        ref = normalised(hom_spdcalc_series(setup, 25.0, bandwidth_nm, delays), delays)
        assert np.max(np.abs(eng - ref)) < tolerance, (bandwidth_nm, np.max(np.abs(eng - ref)))
        l_e, r_e = dip_fwhm(delays, eng)
        l_r, r_r = dip_fwhm(delays, ref)
        assert (r_e - l_e) == pytest.approx(r_r - l_r, rel=0.03)
        assert eng.min() == pytest.approx(ref.min(), abs=0.01)


def test_hom_dip_depth_follows_pump_bandwidth_towards_cw_limit(emission_cache, hom_setup):
    depth = []
    for bandwidth_nm in (0.0405, 0.01, None):
        setup = hom_setup.model_copy(deep=True)
        set_settings_temperature(setup, 25.0)
        if bandwidth_nm is None:
            set_overrides(setup, "crystal", dnu="200 MHz")
        else:
            nu = C * bandwidth_nm * 1e-9 / (405e-9) ** 2
            set_overrides(setup, "pump", pulse_width=f"{0.441 / nu * 1e12} ps")
        delays = np.array([2.0, 2.25, 2.5, 8.0, -6.0])
        y = hom_engine_series(setup, delays)
        depth.append(y[:3].min() / y[3:].mean())
    assert depth[0] > depth[1] > depth[2]
    assert depth[0] == pytest.approx(0.35, abs=0.03)
    assert depth[2] < 0.06


def test_hom_degenerate_temperature_gives_full_visibility_and_orthogonal_photons_none(
    emission_cache, hom_setup
):
    setup = hom_setup.model_copy(deep=True)
    set_overrides(setup, "crystal", dnu="200 MHz")
    delays = np.arange(1.75, 2.76, 0.25)
    y = hom_engine_series(setup, delays)
    flat = hom_engine_series(setup, [-8.0, 14.0])
    assert flat[0] == pytest.approx(flat[1], rel=1e-3)
    visibility = 1.0 - y.min() / flat.mean()
    assert visibility > 0.98
    cold = setup.model_copy(deep=True)
    set_settings_temperature(cold, 25.0)
    y_cold = hom_engine_series(cold, delays)
    flat_cold = hom_engine_series(cold, [-8.0])
    assert 1.0 - y_cold.min() / flat_cold[0] == pytest.approx(0.95, abs=0.015)
    distinguishable = setup.model_copy(deep=True)
    set_settings(distinguishable, "hwp_idler", angle="0 deg")
    y_flat = hom_engine_series(distinguishable, delays)
    assert np.all(np.abs(y_flat / flat.mean() - 1.0) < 1e-3)


def set_settings_temperature(setup, temperature_c):
    set_settings(setup, "crystal", temperature=f"{temperature_c} degC")


# ---------------------------------------------------------------------------
# Bell test, CHSH and tomography
# ---------------------------------------------------------------------------

V_SOURCE = 0.97


def set_analysers(setup, a_deg, b_deg, qwp_a=0.0, qwp_b=0.0, hwp_a=None, hwp_b=None):
    set_settings(setup, "hwp_a", angle=f"{a_deg / 2.0 if hwp_a is None else hwp_a} deg")
    set_settings(setup, "hwp_b", angle=f"{b_deg / 2.0 if hwp_b is None else hwp_b} deg")
    set_settings(setup, "qwp_a", angle=f"{qwp_a} deg")
    set_settings(setup, "qwp_b", angle=f"{qwp_b} deg")


def bell_rates(setup, key="true_hz"):
    r = run(setup)
    return np.array(
        [
            [r.analyses["tt"][key], r.analyses["tr"][key]],
            [r.analyses["rt"][key], r.analyses["rr"][key]],
        ]
    )


def bell_E(setup, a_deg, b_deg, key="true_hz"):
    s = setup.model_copy(deep=True)
    set_analysers(s, a_deg, b_deg)
    c = bell_rates(s, key)
    return metrics.correlation_from_rates(c[0, 0], c[0, 1], c[1, 0], c[1, 1])


@pytest.fixture(scope="module")
def bell_setup():
    return load_example("bell_test")


def test_bell_example_runs_clean(bell_setup):
    r = run(bell_setup)
    bad = {
        "delay_aliasing",
        "pair_delay_window",
        "pump_leakage",
        "path_mode_mismatch",
        "coincidence_outside_window",
        "bad_analysis",
        "no_tagger_connection",
        "window_exceeds_period",
        "low_coupling",
        "curve_extrapolation",
    }
    assert not bad & set(codes(r)), codes(r)


def test_bell_absolute_rates_match_hand_calculation(bell_setup):
    r = run(bell_setup)
    emission = instance_emission(bell_setup)
    spec_s, spec_i = marginals(emission.source)
    snspd = part("detectors/snspd_generic")
    cpl = part("fiber/coupler_f11mm_810")
    fiber = part("fiber/smf_810")
    src = part("nonlinear/entangled_810_generic")
    L = link_lengths(bell_setup)
    f = cpl.focal_length.si
    w_f = fiber.mode_field_diameter.si / 2.0
    out = {}
    for role, spec, arm in (("signal", spec_s, "a"), ("idler", spec_i, "b")):
        lam = float(np.sum(spec[0] * spec[1]))
        w0 = src.signal_waist.si if role == "signal" else src.idler_waist.si
        ops = [
            ("d", L[f"s_{arm}1"]),
            ("d", L[f"s_{arm}2"]),
            ("d", L[f"s_{arm}3"]),
            ("d", L[f"s_{arm}t"]),
            ("f", f),
            ("d", f),
        ]
        q = propagate_q(1j * math.pi * w0**2 / lam, ops)
        coupling = gaussian_overlap(q, 1j * math.pi * w_f**2 / lam, lam)
        eta = band_average(spec, snspd.efficiency) * cpl.transmission.si * coupling
        out[role] = eta * fiber_transmission(fiber, L[f"f_{arm}t"])
    rate = src.brightness * 1.0 * 1e-3 * 1e3
    assert r.sources["source"]["rate_hz"] == pytest.approx(rate, rel=1e-12)
    dead = snspd.dead_time.si
    surv = 1.0 / (1.0 + (rate * 0.5 * out["signal"] + snspd.dark_count_rate.si) * dead)
    singles_expected = rate * 0.5 * out["signal"]
    assert r.channels["tagger.ch1"]["incident_pair_hz"] == pytest.approx(singles_expected, rel=2e-3)
    tt = r.analyses["tt"]
    # a = 0, b = 22.5 deg, phi+ with visibility V: P(++) = (1 + V cos 2(a-b)) / 4
    p_tt = 0.25 * (1.0 + V_SOURCE * math.cos(math.radians(45.0)))
    expected = rate * out["signal"] * out["idler"] * p_tt * surv * surv
    assert tt["coincidence_rate_hz"] == pytest.approx(expected, rel=3e-3)


def instance_emission(setup):
    from sequoia.components.base import Context
    from sequoia.setup.resolve import instantiate

    comp = instantiate(LIBRARY, next(i for i in setup.instances if i.id == "source"))
    return comp.emit_pairs({}, Context(subject="source"))


def test_bell_correlation_fringe_follows_cos_two_theta(bell_setup):
    b = 22.5
    for a in np.arange(0.0, 180.0, 22.5):
        e = bell_E(bell_setup, a, b)
        assert e == pytest.approx(V_SOURCE * math.cos(math.radians(2.0 * (a - b))), abs=2e-3)


def test_bell_chsh_value_equals_two_sqrt_two_times_visibility(bell_setup):
    a, ap, b, bp = 0.0, 45.0, 22.5, 67.5
    e = {
        k: bell_E(bell_setup, x, y)
        for k, (x, y) in {"ab": (a, b), "abp": (a, bp), "apb": (ap, b), "apbp": (ap, bp)}.items()
    }
    s = metrics.chsh_s(e["ab"], e["abp"], e["apb"], e["apbp"])
    assert s == pytest.approx(2.0 * math.sqrt(2.0) * V_SOURCE, abs=2e-3)
    assert s == pytest.approx(
        metrics.chsh_from_state(metrics.werner_state(V_SOURCE, "phi+")), abs=2e-3
    )
    rates = np.zeros((2, 2, 2, 2))
    for i, x in enumerate((a, ap)):
        for j, y in enumerate((b, bp)):
            s_ = bell_setup.model_copy(deep=True)
            set_analysers(s_, x, y)
            rates[i, j] = bell_rates(s_)
    assert metrics.chsh_from_rates(rates) == pytest.approx(s, abs=1e-6)
    assert s > 2.0
    # including accidentals lowers S slightly (CAR is large)
    e_tot = [
        bell_E(bell_setup, x, y, key="total_hz") for x, y in ((a, b), (a, bp), (ap, b), (ap, bp))
    ]
    s_tot = metrics.chsh_s(*e_tot)
    assert 0.0 < s - s_tot < 5e-3


def test_bell_visibility_scales_chsh_linearly(bell_setup):
    vals = []
    for v in (1.0, 0.5):
        setup = bell_setup.model_copy(deep=True)
        set_overrides(setup, "source", visibility=v)
        e = [bell_E(setup, x, y) for x, y in ((0.0, 22.5), (0.0, 67.5), (45.0, 22.5), (45.0, 67.5))]
        vals.append(metrics.chsh_s(*e))
    assert vals[0] == pytest.approx(2.0 * math.sqrt(2.0), abs=2e-3)
    assert vals[1] == pytest.approx(math.sqrt(2.0), abs=2e-3)


def tomography_rates(setup, settings):
    out = []
    for s in settings:
        t = setup.model_copy(deep=True)
        set_analysers(
            t,
            0.0,
            0.0,
            qwp_a=math.degrees(s.qwp_a),
            qwp_b=math.degrees(s.qwp_b),
            hwp_a=math.degrees(s.hwp_a),
            hwp_b=math.degrees(s.hwp_b),
        )
        out.append(run(t).analyses["tt"]["true_hz"])
    return np.array(out)


def expected_state(visibility, phase):
    psi = np.array([1.0, 0.0, 0.0, np.exp(1j * phase)]) / math.sqrt(2.0)
    return visibility * np.outer(psi, psi.conj()) + (1.0 - visibility) * np.eye(4) / 4.0


@pytest.mark.parametrize("phase_deg", [0.0, 90.0, -60.0])
def test_tomography_wave_plate_convention_and_mle_roundtrip(bell_setup, phase_deg):
    setup = bell_setup.model_copy(deep=True)
    v = 0.92
    set_overrides(setup, "source", visibility=v, phase=f"{phase_deg} deg")
    rho_true = expected_state(v, math.radians(phase_deg))
    settings = metrics.tomography_settings()
    rates = tomography_rates(setup, settings)
    probabilities = metrics.tomography_probabilities(rho_true, settings)
    scale = float(np.dot(rates, probabilities) / np.dot(probabilities, probabilities))
    assert np.max(np.abs(rates - scale * probabilities)) < 2e-3 * scale
    rho = metrics.mle_density_matrix(rates * 1.0, settings)
    assert metrics.fidelity(rho, rho_true) > 0.9995
    assert np.max(np.abs(rho - rho_true)) < 5e-3
    assert metrics.concurrence(rho) == pytest.approx(metrics.concurrence(rho_true), abs=5e-3)
    # a mirrored circular convention would give the complex-conjugate state
    if abs(math.sin(math.radians(phase_deg))) > 0.5:
        assert metrics.fidelity(rho, rho_true.conj()) < 0.8
    rng = np.random.default_rng(7)
    noisy = metrics.tomography_counts(rho_true, settings, scale * 1.0, 1.0, rng)
    rho_noisy = metrics.mle_density_matrix(noisy, settings)
    assert metrics.fidelity(rho_noisy, rho_true) > 0.99


# ---------------------------------------------------------------------------
# library and CLI
# ---------------------------------------------------------------------------


def test_every_library_part_loads():
    from m3_helpers import ROOT

    refs = [
        "/".join(p.parent.relative_to(ROOT / "library").parts)
        for p in (ROOT / "library").rglob("part.yaml")
    ]
    assert len(refs) >= 25
    for ref in refs:
        LIBRARY.load_part(ref)


@pytest.mark.parametrize("name", ["hbt_heralded", "bell_test", "hom"])
def test_cli_runs_each_example_and_prints_rates(name, capsys, emission_cache):
    from m3_helpers import ROOT

    from sequoia.cli import _run

    code = _run(str(ROOT / "setups" / f"{name}.yaml"), str(ROOT / "library"))
    out = capsys.readouterr().out
    assert code == 0
    assert "Photon rates" in out and "Analyses" in out and "Pair sources" in out
    assert "delay_aliasing" not in out
