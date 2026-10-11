import math
import re
from dataclasses import replace

import numpy as np
import pytest
import spdcalc
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import Context, get_component
from sequoia.components.nonlinear import (
    _config,
    bell_amplitude,
    bell_ensemble,
    gaussian_pair_source,
)
from sequoia.errors import SetupError
from sequoia.light.gaussian import GaussianMode
from sequoia.light.spectrum import Spectrum
from sequoia.light.state import Beam, LinkLight

C = constants.c
FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
H, V = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def pump_light(power=1e-3, wavelength=405e-9, waist=100e-6, pulsed_fwhm_hz=None, pol=(1.0, 0.0)):
    if pulsed_fwhm_hz:
        spectrum = Spectrum.gaussian(wavelength, pulsed_fwhm_hz, n=11)
    else:
        spectrum = Spectrum.line(wavelength, 0.0)
    v = np.asarray(pol, dtype=complex)
    v = v / np.linalg.norm(v)
    n = len(spectrum.wavelengths)
    coherency = spectrum.weights[:, None, None] * power * np.outer(v, v.conj())[None]
    return LinkLight(
        wavelengths=spectrum.wavelengths,
        power_spectrum=power * spectrum.weights,
        coherency=coherency.reshape(n, 2, 2),
        mode=GaussianMode.from_waist(wavelength, waist),
        paths=1,
    )


def crystal(**params):
    cls = get_component("SPDCCrystal")
    defaults = dict(
        length="10 mm",
        poling_period="auto",
        signal_waist="50 um",
        idler_waist="50 um",
        d_eff="1 pm/V",
    )
    settings = params.pop("settings", None)
    comp = cls(
        cls.Params(**{**defaults, **params}),
        None if settings is None else cls.Settings(**settings),
    )
    return comp


def emit(comp, light=None):
    ctx = Context()
    light = pump_light() if light is None else light
    emission = comp.emit_pairs({"in": light}, ctx)
    return emission, ctx


def signal_marginal(src):
    return np.sum(np.abs(src.members) ** 2, axis=(0, 1, 2, 4))


def sum_marginal(src):
    return np.sum(np.abs(src.members) ** 2, axis=(0, 1, 2, 3))


def fwhm_of(x, w):
    w = np.asarray(w, dtype=float)
    mean = np.sum(w * x) / w.sum()
    return FWHM * math.sqrt(np.sum(w * (x - mean) ** 2) / w.sum())


def polarization_density(src):
    a = src.members.reshape(len(src.members), 4, -1)
    return sum(m @ m.conj().T for m in a)


def test_registration_and_ports():
    cls = get_component("SPDCCrystal")
    assert cls.category == "nonlinear"
    names = {p.name: p for p in crystal().ports()}
    assert names["in"].required and names["in"].direction == "in"
    assert names["out"].direction == "out"
    ent = get_component("EntangledPairSource")
    assert ent.category == "nonlinear"
    assert {
        p.name
        for p in ent(
            ent.Params(center_wavelength="810 nm", bandwidth="1 nm", pair_rate="1 kHz")
        ).ports()
    } == {
        "signal",
        "idler",
    }


def test_gaussian_source_cw_normalisation_lattice_and_spectrum():
    src = gaussian_pair_source(810e-9, bandwidth=100e9, rate=3.0e4)
    assert src.rate == 3.0e4
    assert np.sum(np.abs(src.amplitude) ** 2) == pytest.approx(1.0, abs=1e-12)
    assert len(src.n_sum) == 1
    assert src.n_signal.dtype.kind == "i" and src.n_sum.dtype.kind == "i"
    assert np.all(np.diff(src.n_signal) == 1)
    assert src.pump_wavelength == pytest.approx(405e-9, rel=1e-6)
    assert fwhm_of(src.nu_signal, signal_marginal(src)) == pytest.approx(100e9, rel=0.01)
    nu_i = src.n_idler()[:, 0] * src.dnu
    assert np.allclose(src.nu_signal + nu_i, src.nu_sum[0])
    assert np.all(np.abs(src.amplitude[0, 1]) == 0.0) and np.all(np.abs(src.amplitude[1]) == 0.0)


def test_gaussian_source_pulsed_widths_and_schmidt_number():
    bandwidth, pump_bw = 200e9, 150e9
    src = gaussian_pair_source(
        810e-9, bandwidth=bandwidth, pump_bandwidth=pump_bw, repetition_rate=80e6, samples=301
    )
    assert np.sum(np.abs(src.amplitude) ** 2) == pytest.approx(1.0, abs=1e-12)
    assert src.repetition_rate == 80e6
    assert fwhm_of(src.nu_signal, signal_marginal(src)) == pytest.approx(bandwidth, rel=0.02)
    assert fwhm_of(src.nu_sum, sum_marginal(src)) == pytest.approx(pump_bw, rel=0.03)
    n_idler = src.n_idler()
    low, high = n_idler.min(), n_idler.max()
    matrix = np.zeros((len(src.n_signal), high - low + 1), dtype=complex)
    for m in range(len(src.n_sum)):
        matrix[np.arange(len(src.n_signal)), n_idler[:, m] - low] = src.amplitude[0, 0, :, m]
    lam = np.linalg.svd(matrix, compute_uv=False) ** 2
    schmidt = 1.0 / np.sum((lam / lam.sum()) ** 2)
    sigma_p = pump_bw / FWHM
    sigma_m = math.sqrt(4.0 * (bandwidth / FWHM) ** 2 - sigma_p**2)
    expected = (sigma_p**2 + sigma_m**2) / (2.0 * sigma_p * sigma_m)
    assert schmidt == pytest.approx(expected, rel=0.03)


def test_gaussian_source_polarization_argument():
    pol = np.array([[0.0, 1.0], [0.0, 0.0]])
    src = gaussian_pair_source(810e-9, bandwidth=100e9, polarization=pol)
    weights = np.sum(np.abs(src.amplitude) ** 2, axis=(2, 3))
    assert weights[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state, expected",
    [
        ("phi+", {(0, 0): 1, (1, 1): 1}),
        ("phi-", {(0, 0): 1, (1, 1): -1}),
        ("psi+", {(0, 1): 1, (1, 0): 1}),
        ("psi-", {(0, 1): 1, (1, 0): -1}),
    ],
)
def test_bell_states_exact(state, expected):
    m = bell_amplitude(state)
    target = np.zeros((2, 2), dtype=complex)
    for (i, j), v in expected.items():
        target[i, j] = v / math.sqrt(2.0)
    assert np.array_equal(m, target)
    phased = bell_amplitude(state, 0.3)
    key = max(expected, key=lambda ij: ij[0])
    assert phased[key] == pytest.approx(expected[key] * np.exp(0.3j) / math.sqrt(2.0))


@pytest.mark.parametrize("state", ["phi+", "phi-", "psi+", "psi-"])
@pytest.mark.parametrize("visibility", [1.0, 0.8, 0.25, 0.0])
def test_werner_mixing_in_entangled_source(state, visibility):
    cls = get_component("EntangledPairSource")
    comp = cls(
        cls.Params(
            center_wavelength="810 nm",
            bandwidth="0.5 nm",
            pair_rate="50 kHz",
            state=state,
            visibility=visibility,
            samples=101,
        )
    )
    emission = comp.emit_pairs({}, Context())
    src = emission.source
    assert (emission.signal_port, emission.idler_port) == ("signal", "idler")
    assert src.rate == pytest.approx(50e3)
    assert np.sum(np.abs(src.amplitude) ** 2) == pytest.approx(1.0, abs=1e-12)
    rho = polarization_density(src)
    bell = bell_amplitude(state).reshape(4)
    expected = visibility * np.outer(bell, bell.conj()) + (1.0 - visibility) * np.eye(4) / 4.0
    assert np.allclose(rho, expected, atol=1e-12)
    assert np.real(np.trace(rho @ rho)) == pytest.approx((1.0 + 3.0 * visibility**2) / 4.0)
    assert np.real(bell.conj() @ rho @ bell) == pytest.approx((1.0 + 3.0 * visibility) / 4.0)


def test_werner_copies_keep_spectral_profile():
    members = bell_ensemble("phi+", 0.5)
    assert sum(w for w, _ in members) == pytest.approx(1.0)
    src = gaussian_pair_source(810e-9, bandwidth=100e9, ensemble=members)
    assert src.is_ensemble and src.members.shape[0] == 4
    assert len(src.n_sum) == 1
    assert np.allclose(src.member_weights, [w for w, _ in members])
    assert fwhm_of(src.nu_signal, signal_marginal(src)) == pytest.approx(100e9, rel=0.01)
    with pytest.raises(ValueError):
        bell_ensemble("phi+", 1.2)


def test_entangled_source_rate_brightness_and_pulsed():
    cls = get_component("EntangledPairSource")
    params = cls.Params(
        center_wavelength="1550 nm",
        idler_wavelength="1550 nm",
        bandwidth="2 nm",
        pump_bandwidth="0.5 nm",
        repetition_rate="76 MHz",
        brightness=2.0e5,
        state="psi-",
    )
    comp = cls(params, cls.Settings(pump_power="3 mW"))
    src = comp.emit_pairs({}, Context()).source
    assert src.rate == pytest.approx(2.0e5 * 3.0)
    assert src.repetition_rate == 76e6
    assert src.pairs_per_pulse == pytest.approx(src.rate / 76e6)
    assert len(src.n_sum) > 1
    pump_fwhm = C * 0.5e-9 / (775e-9) ** 2
    assert fwhm_of(src.nu_sum, sum_marginal(src)) == pytest.approx(pump_fwhm, rel=0.05)
    assert src.pump_wavelength == pytest.approx(775e-9, rel=1e-4)
    bad = cls(cls.Params(center_wavelength="810 nm", bandwidth="1 nm"))
    with pytest.raises(SetupError):
        bad.emit_pairs({}, Context())


def test_crystal_propagate_applies_transmission():
    comp = crystal(pump_transmission=0.64)
    spectrum = Spectrum.line(405e-9)
    beam = Beam.launch("src", spectrum, GaussianMode.from_waist(405e-9, 1e-4))
    ((port, out),) = comp.propagate("in", beam, Context(subject="x"))
    assert port == "out"
    assert out.jones[0, 0, 0] == pytest.approx(0.8)
    assert out.jones[0, 1, 1] == pytest.approx(0.8)


def test_no_pump_no_emission():
    comp = crystal()
    assert comp.emit_pairs({}, Context()) is None


@pytest.fixture(scope="module")
def ppktp_cw():
    comp = crystal()
    emission, ctx = emit(comp)
    return comp, emission.source, ctx


def test_crystal_cw_state_is_normalised_on_lattice(ppktp_cw):
    _, src, _ = ppktp_cw
    assert np.sum(np.abs(src.amplitude) ** 2) == pytest.approx(1.0, abs=1e-12)
    assert len(src.n_sum) == 1
    assert src.n_signal.dtype.kind == "i" and np.all(np.diff(src.n_signal) == 1)
    assert src.pump_wavelength == pytest.approx(405e-9, rel=1e-6)
    lam = C / src.nu_signal
    assert lam.min() < 810e-9 < lam.max()
    assert src.signal_mode.wavelength == pytest.approx(810e-9, rel=2e-3)
    assert src.idler_mode.wavelength == pytest.approx(810e-9, rel=2e-3)


def test_crystal_type2_polarization_labels(ppktp_cw):
    _, src, _ = ppktp_cw
    weights = np.sum(np.abs(src.amplitude) ** 2, axis=(2, 3))
    assert weights[0, 1] == pytest.approx(1.0)


def test_crystal_extraordinary_axis_swaps_labels():
    src = emit(crystal(extraordinary_axis="V"), pump_light(pol=(0.0, 1.0)))[0].source
    weights = np.sum(np.abs(src.amplitude) ** 2, axis=(2, 3))
    assert weights[1, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "pm_type, pump_pol, labels",
    [
        ("Type0_e_ee", (1.0, 0.0), (0, 0)),
        ("Type0_o_oo", (0.0, 1.0), (1, 1)),
        ("Type1_e_oo", (1.0, 0.0), (1, 1)),
        ("Type2_e_oe", (1.0, 0.0), (1, 0)),
    ],
)
def test_crystal_polarization_labels_other_types(pm_type, pump_pol, labels):
    comp = crystal(pm_type=pm_type, length="5 mm")
    src = emit(comp, pump_light(pol=pump_pol))[0].source
    weights = np.sum(np.abs(src.amplitude) ** 2, axis=(2, 3))
    assert weights[labels] == pytest.approx(1.0)


def test_crystal_rate_linear_in_pump_power_and_deff_squared():
    r1 = emit(crystal(), pump_light(power=1e-3))[0].source.rate
    r3 = emit(crystal(), pump_light(power=3e-3))[0].source.rate
    r_d = emit(crystal(d_eff="2 pm/V"), pump_light(power=1e-3))[0].source.rate
    assert r1 > 0.0
    assert r3 / r1 == pytest.approx(3.0, rel=1e-9)
    assert r_d / r1 == pytest.approx(4.0, rel=1e-6)


def test_crystal_measured_brightness_overrides_rate():
    comp = crystal(brightness=2.5e5)
    src = emit(comp, pump_light(power=2e-3))[0].source
    assert src.rate == pytest.approx(2.5e5 * 2.0)


def test_crystal_pump_polarization_fraction_scales_rate_and_warns():
    full = emit(crystal())[0].source.rate
    emission, ctx = emit(crystal(), pump_light(pol=(1.0, 1.0)))
    assert emission.source.rate == pytest.approx(0.5 * full, rel=1e-9)
    assert any(i.code == "pump_polarization" for i in ctx.issues)
    assert emit(crystal(), pump_light(pol=(0.0, 1.0)))[0] is None


def test_crystal_rate_agrees_with_spdcalc_efficiencies_on_default_ktp():
    comp = crystal(
        material="KTP",
        pm_type="Type2_e_eo",
        length="2 mm",
        theta="52.7289 deg",
        poling_period=None,
        signal_wavelength="1550 nm",
        signal_waist="100 um",
        idler_waist="100 um",
        spectral_cutoff=1e-3,
        settings={"temperature": "20 degC"},
    )
    light = pump_light(
        power=1e-3, wavelength=775e-9, waist=100e-6, pulsed_fwhm_hz=C * 5.53e-9 / (775e-9) ** 2
    )
    src = emit(comp, light)[0].source
    ref = spdcalc.SPDC.default()
    rng = ref.optimum_range(100)
    windowed = ref.efficiencies(rng)["coincidences_hz"]
    x, y = np.asarray(rng.x_values()), np.asarray(rng.y_values())
    raw = float(
        np.asarray(ref.joint_spectrum().jsi_range(rng)).sum() * (x[1] - x[0]) * (y[1] - y[0])
    )
    assert 0.95 < src.rate / raw < 1.4
    counted = emit(
        crystal(
            material="KTP",
            pm_type="Type2_e_eo",
            length="2 mm",
            theta="52.7289 deg",
            poling_period=None,
            signal_wavelength="1550 nm",
            signal_waist="100 um",
            idler_waist="100 um",
            spectral_cutoff=1e-3,
            settings={"temperature": "20 degC"},
            spdcalc_counts_correction=True,
        ),
        light,
    )[0].source.rate
    assert 0.95 < counted / windowed < 1.4


def test_crystal_rate_independent_of_pump_bandwidth_in_linear_regime():
    cw = emit(crystal())[0].source
    pulsed = emit(crystal(), pump_light(pulsed_fwhm_hz=5e10))[0].source
    assert len(pulsed.n_sum) > 1
    assert pulsed.rate == pytest.approx(cw.rate, rel=0.05)
    assert fwhm_of(pulsed.nu_sum, sum_marginal(pulsed)) == pytest.approx(5e10, rel=0.15)


def test_crystal_pulsed_amplitude_matches_spdcalc_jsa_normalized_range():
    comp = crystal(
        material="KTP",
        pm_type="Type2_e_eo",
        length="2 mm",
        theta="52.7289 deg",
        poling_period=None,
        signal_wavelength="1550 nm",
        signal_waist="100 um",
        idler_waist="100 um",
        spectral_cutoff=1e-3,
        settings={"temperature": "20 degC"},
    )
    light = pump_light(wavelength=775e-9, waist=100e-6, pulsed_fwhm_hz=C * 5.53e-9 / (775e-9) ** 2)
    src = emit(comp, light)[0].source
    assert len(src.n_sum) > 3
    ref = spdcalc.SPDC.default()
    rng = ref.optimum_range(160)
    x, y = np.asarray(rng.x_values()), np.asarray(rng.y_values())
    grid = np.abs(np.asarray(ref.joint_spectrum().jsa_normalized_range(rng))).reshape(160, 160)
    interp = RegularGridInterpolator((y, x), grid, bounds_error=False, fill_value=None)
    w_s = 2.0 * np.pi * src.dnu * src.n_signal[:, None]
    w_i = 2.0 * np.pi * src.dnu * src.n_sum[None, :] - w_s
    w_s, w_i = np.broadcast_arrays(w_s, w_i)
    inside = (w_s >= x[0]) & (w_s <= x[-1]) & (w_i >= y[0]) & (w_i <= y[-1])
    expected = interp(np.stack([w_i[inside], w_s[inside]], axis=-1))
    mine = np.abs(src.amplitude[0, 1])
    mine = mine / mine[inside].max()
    expected = expected / expected.max()
    assert inside.sum() > 500
    assert np.max(np.abs(mine[inside] - expected)) < 0.06
    assert np.sqrt(np.mean((mine[inside] - expected) ** 2)) < 0.01


def test_crystal_temperature_tuning_follows_spdcalc_phase_matching():
    centres, roots = [], []
    for temperature in (10.0, 30.0, 50.0):
        comp = crystal(settings={"temperature": f"{temperature} degC"})
        src = emit(comp)[0].source
        w = signal_marginal(src)
        centres.append(src.nu_signal[np.argmax(w)])
        s = spdcalc.SPDC.from_yaml(
            f"""
crystal:
  kind: KTP
  pm_type: Type2_e_eo
  phi_deg: 0
  theta_deg: 90
  length_um: 10000
  temperature_c: {temperature}
pump:
  wavelength_nm: 405.0
  waist_um: 100
  bandwidth_nm: 0.05
  average_power_mw: 1.0
signal:
  wavelength_nm: 810.0
  phi_deg: 0
  theta_deg: 0
  waist_um: 50
  waist_position_um: auto
idler: auto
periodic_poling:
  poling_period_um: {comp_period()}
deff_pm_per_volt: 1
"""
        )
        wp = 2.0 * np.pi * C / 405e-9
        root = brentq(lambda ws, s=s, wp=wp: s.delta_k(ws, wp - ws)[2], 2e15, 2.6e15)
        roots.append(root / (2.0 * np.pi))
    centres, roots = np.array(centres), np.array(roots)
    assert np.all(np.diff(centres) > 0.0)
    assert np.all(np.diff(roots) > 0.0)
    assert np.allclose(np.diff(centres), np.diff(roots), rtol=0.1)
    assert np.allclose(centres, roots, atol=0.15e12)


def comp_period():
    s = spdcalc.SPDC.from_yaml(
        """
crystal:
  kind: KTP
  pm_type: Type2_e_eo
  phi_deg: 0
  theta_deg: 90
  length_um: 10000
  temperature_c: 25.0
pump:
  wavelength_nm: 405.0
  waist_um: 100
  bandwidth_nm: 0.05
  average_power_mw: 1.0
signal:
  wavelength_nm: 810.0
  phi_deg: 0
  theta_deg: 0
  waist_um: 50
  waist_position_um: auto
idler: auto
periodic_poling:
  poling_period_um: auto
deff_pm_per_volt: 1
"""
    )
    return s.poling_period_um


def test_crystal_temperature_changes_wavelength_direction_matches_known_sign():
    colder = emit(crystal(settings={"temperature": "15 degC"}))[0].source
    hotter = emit(crystal(settings={"temperature": "45 degC"}))[0].source
    lam_c = C / colder.nu_signal[np.argmax(signal_marginal(colder))]
    lam_h = C / hotter.nu_signal[np.argmax(signal_marginal(hotter))]
    assert lam_h < lam_c


def test_crystal_modes_and_pair_delay_warning():
    comp = crystal(max_samples=2000, delay_window="100 ns")
    emission, ctx = emit(comp)
    assert any(i.code == "pair_delay_window" for i in ctx.issues)
    assert len(emission.source.n_signal) <= 2100
    assert emission.source.signal_mode.waist_x == pytest.approx(50e-6, rel=1e-6)


def test_signal_shorter_than_pump_is_setup_error_not_panic():
    with pytest.raises(SetupError, match="longer than the pump"):
        emit(crystal(signal_wavelength="300 nm"))


def test_rust_panic_is_converted_to_setup_error():
    with pytest.raises(SetupError, match="spdcalc cannot evaluate"):
        emit(crystal(), pump_light(wavelength=200e-9))


def test_signal_or_idler_outside_transmission_range_is_setup_error():
    with pytest.raises(SetupError, match="transmission range"):
        emit(crystal(signal_wavelength="4 um"))


def test_pump_outside_transmission_range_warns_and_pair_state_errors_are_not_mislabelled():
    emission, ctx = emit(crystal(), pump_light(wavelength=300e-9))
    assert emission is not None
    assert any(i.code == "pump_transmission_range" for i in ctx.issues)
    _, ctx = emit(crystal(), pump_light(wavelength=250e-9))
    assert any(i.code == "pump_transmission_range" for i in ctx.issues)
    with pytest.raises(SetupError) as info:
        emit(crystal(spdcalc_counts_correction=True), pump_light(wavelength=250e-9))
    assert "spdcalc rejected" not in str(info.value)


def test_zero_pump_and_missing_pump_warn():
    ctx = Context()
    assert crystal().emit_pairs({}, ctx) is None
    assert any(i.code == "no_pump" for i in ctx.issues)
    emission, ctx = emit(crystal(), pump_light(power=0.0))
    assert emission is None
    assert any(i.code == "no_pump" for i in ctx.issues)
    emission, ctx = emit(crystal(), pump_light(pol=(0.0, 1.0)))
    assert emission is None
    assert any(i.code == "no_pump" for i in ctx.issues)


def test_d_eff_is_required_without_brightness():
    cls = get_component("SPDCCrystal")
    comp = cls(cls.Params(length="10 mm", poling_period="auto"))
    with pytest.raises(SetupError, match="d_eff"):
        comp.emit_pairs({"in": pump_light()}, Context())
    with_brightness = cls(cls.Params(length="10 mm", poling_period="auto", brightness=1e5))
    emission = with_brightness.emit_pairs({"in": pump_light()}, Context())
    assert emission.source.rate == pytest.approx(1e5)


def test_pump_waist_is_the_mode_waist_not_the_radius_at_the_port():
    focused = GaussianMode.from_waist(405e-9, 20e-6, z=5e-3)
    assert focused.w_x == pytest.approx(37.9e-6, rel=0.01)
    light = replace(pump_light(), mode=focused)
    emission, ctx = emit(crystal(), light)
    at_waist = emit(crystal(), replace(pump_light(), mode=GaussianMode.from_waist(405e-9, 20e-6)))
    at_radius = emit(
        crystal(), replace(pump_light(), mode=GaussianMode.from_waist(405e-9, focused.w_x))
    )
    assert emission.source.rate == pytest.approx(at_waist[0].source.rate, rel=1e-9)
    assert at_radius[0].source.rate / emission.source.rate == pytest.approx(1.35, abs=0.04)
    assert any(i.code == "pump_waist_position" for i in ctx.issues)
    assert not any(i.code == "pump_waist_position" for i in at_waist[1].issues)


def test_pump_waist_astigmatic_mode_uses_geometric_mean_of_waists():
    mode = GaussianMode.from_waist(405e-9, 80e-6, 125e-6)
    emission, _ = emit(crystal(), replace(pump_light(), mode=mode))
    round_ = emit(
        crystal(),
        replace(pump_light(), mode=GaussianMode.from_waist(405e-9, math.sqrt(80e-6 * 125e-6))),
    )[0]
    assert emission.source.rate == pytest.approx(round_.source.rate, rel=1e-9)


def entangled(**kw):
    cls = get_component("EntangledPairSource")
    return cls(cls.Params(center_wavelength="810 nm", pair_rate="1 kHz", **kw))


@pytest.mark.parametrize("bandwidth", ["0.3 nm", "1 nm"])
def test_entangled_default_lattice_period_covers_delay_window(bandwidth):
    ctx = Context()
    src = entangled(bandwidth=bandwidth).emit_pairs({}, ctx).source
    assert 1.0 / src.dnu >= 10e-9 * 0.999
    assert not any(i.code == "pair_delay_window" for i in ctx.issues)
    assert fwhm_of(src.nu_signal, signal_marginal(src)) == pytest.approx(
        C * float(bandwidth.split()[0]) * 1e-9 / (810e-9) ** 2, rel=0.01
    )


def test_entangled_delay_window_parameter_and_cap_warning():
    src = entangled(bandwidth="0.3 nm", delay_window="20 ns").emit_pairs({}, Context()).source
    assert 1.0 / src.dnu == pytest.approx(20e-9, rel=1e-9)
    ctx = Context()
    src = entangled(bandwidth="5 nm", max_samples=2000).emit_pairs({}, ctx).source
    (issue,) = [i for i in ctx.issues if i.code == "pair_delay_window"]
    period_ns = 1e9 / src.dnu
    assert period_ns < 10.0
    assert f"{period_ns:.3g} ns" in issue.message
    assert len(src.n_signal) <= 2100


def test_entangled_fibre_delay_does_not_alias_by_default():
    src = entangled(bandwidth="1 nm").emit_pairs({}, Context()).source
    assert 1.0 / src.dnu > 0.5e-9 * 10.0


def test_pulsed_crystal_period_warning_states_actual_number():
    comp = crystal(max_samples=4000)
    emission, ctx = emit(comp, pump_light(pulsed_fwhm_hz=1.5e12))
    (issue,) = [i for i in ctx.issues if i.code == "pair_delay_window"]
    assert f"{1e9 / emission.source.dnu:.3g} ns" in issue.message
    assert 1.0 / emission.source.dnu < 10e-9


_EPS0 = constants.epsilon_0


def _index(material, wavelength, axis, kelvin):
    return float(spdcalc.get_crystal_indices(material, wavelength * 1e9, kelvin)[axis])


def _group_index(material, wavelength, axis, kelvin):
    h = 1e-4 * wavelength
    slope = (
        _index(material, wavelength + h, axis, kelvin)
        - _index(material, wavelength - h, axis, kelvin)
    ) / (2.0 * h)
    return _index(material, wavelength, axis, kelvin) - wavelength * slope


def _overlap_closed_form(w_p, w_s, w_i):
    return 2.0 / (w_s * w_i * (1.0 / w_p**2 + 1.0 / w_s**2 + 1.0 / w_i**2))


def _phase_mismatch(material, kelvin, lam_p, lam_s, lam_i, axes, period):
    ip, is_, ii = axes
    dk = (
        2.0
        * np.pi
        * (
            _index(material, lam_p, ip, kelvin) / lam_p
            - _index(material, lam_s, is_, kelvin) / lam_s
            - _index(material, lam_i, ii, kelvin) / lam_i
        )
    )
    return dk if period is None else dk - 2.0 * np.pi / period


def _gaussian_beam_overlap(
    material, kelvin, length, lam_p, lam_s, lam_i, axes, waists, period, zpos
):
    """Brute-force J / A_0 = int d^3r g_p(r) u_s^*(r) u_i^*(r) exp(i dk z) on an (r, z) grid
    with explicit paraxial beams exp(-r^2/(w0^2 q))/q, q = 1 + i zeta (conjugated for the
    photons), the collection modes normalised to unit power at every z."""
    ip, is_, ii = axes
    w_p, w_s, w_i = waists
    n_p = _index(material, lam_p, ip, kelvin)
    n_s = _index(material, lam_s, is_, kelvin)
    n_i = _index(material, lam_i, ii, kelvin)
    z = np.linspace(-length / 2, length / 2, 601)
    r = np.linspace(0.0, 4.0 * max(w_p, w_s, w_i) * 3.0, 3000)
    rr, zz = r[None, :], z[:, None]

    def beam(w0, n, lam, z0, conj):
        zeta = (zz - z0) / (np.pi * n * w0**2 / lam)
        q = 1.0 + (-1j if conj else 1j) * zeta
        return np.exp(-(rr**2) / (w0**2 * q)) / q

    pump = beam(w_p, n_p, lam_p, 0.0, False)
    sig = beam(w_s, n_s, lam_s, zpos[0], True) * np.sqrt(2.0 / np.pi) / w_s
    idl = beam(w_i, n_i, lam_i, zpos[1], True) * np.sqrt(2.0 / np.pi) / w_i
    radial = np.trapezoid(pump * sig * idl * 2.0 * np.pi * rr, r, axis=1)
    dk = _phase_mismatch(material, kelvin, lam_p, lam_s, lam_i, axes, period)
    return np.trapezoid(radial * np.exp(1j * dk * z), z)


def coupled_mode_pair_rate(
    material,
    kelvin,
    length,
    lam_p,
    lam_s0,
    axes,
    waists,
    d,
    period=None,
    zpos=(0.0, 0.0),
    n=801,
    span=0.03,
):
    """Pairs/s per mW of the first-order perturbation theory described in
    ``test_absolute_pair_rate_first_principles`` (signal wavelength scanned +-3%)."""
    ip, is_, ii = axes
    w_p, w_s, w_i = waists
    n_p = _index(material, lam_p, ip, kelvin)
    a0_sq = 1e-3 / (np.pi * w_p**2 * n_p * _EPS0 * C)
    z = np.linspace(-length / 2, length / 2, n)
    lam = np.linspace((1.0 - span) * lam_s0, (1.0 + span) * lam_s0, n)
    rate = np.zeros(n)
    for k, ls in enumerate(lam):
        li = 1.0 / (1.0 / lam_p - 1.0 / ls)
        ns, ni = _index(material, ls, is_, kelvin), _index(material, li, ii, kelvin)
        dk = _phase_mismatch(material, kelvin, lam_p, ls, li, axes, period)
        rayleigh = [
            np.pi * nn * w**2 / lw
            for nn, w, lw in ((n_p, w_p, lam_p), (ns, w_s, ls), (ni, w_i, li))
        ]
        zp, zs, zi = z / rayleigh[0], (z - zpos[0]) / rayleigh[1], (z - zpos[1]) / rayleigh[2]
        a_p = 1.0 / (w_p**2 * (1.0 + 1j * zp))
        a_s = 1.0 / (w_s**2 * (1.0 - 1j * zs))
        a_i = 1.0 / (w_i**2 * (1.0 - 1j * zi))
        integrand = np.exp(1j * dk * z) / ((1 + 1j * zp) * (1 - 1j * zs) * (1 - 1j * zi))
        integrand = integrand * (np.pi / (a_p + a_s + a_i)) * (2.0 / np.pi) / (w_s * w_i)
        j = np.trapezoid(integrand, z)
        omega_s, omega_i = 2.0 * np.pi * C / ls, 2.0 * np.pi * C / li
        rate[k] = 2.0 * d**2 / (np.pi * C**2) * omega_s * omega_i / (ns * ni) * a0_sq * abs(j) ** 2
    domega = 2.0 * np.pi * C * np.gradient(lam) / lam**2
    return float(np.sum(rate * domega))


def _ktp_config(length, w_p, w_s, period="auto"):
    return _config(
        kind="KTP",
        pm_type="Type2_e_eo",
        theta_deg=90.0,
        length_um=length * 1e6,
        temperature_c=25.0,
        pump_nm=405.0,
        bandwidth_nm=0.0405,
        pump_waist_um=w_p * 1e6,
        signal_nm=810.0,
        signal_waist_um=w_s * 1e6,
        idler_waist_um=w_s * 1e6,
        period_um=period,
        deff=1.0,
    )


@pytest.mark.parametrize(("w_p", "w_s"), [(400e-6, 300e-6), (100e-6, 50e-6), (30e-6, 30e-6)])
def test_transverse_overlap_matches_brute_force_gaussian_beams(w_p, w_s):
    length, kelvin, lam_p = 10e-3, 298.15, 405e-9
    cfg = spdcalc.SPDC.from_yaml(_ktp_config(length, w_p, w_s))
    period = cfg.poling_period_um * 1e-6
    zpos = (cfg.signal_waist_position_um * 1e-6, cfg.idler_waist_position_um * 1e-6)
    z = np.linspace(-length / 2, length / 2, 2001)
    for lam_s in (808e-9, 810e-9, 812e-9):
        lam_i = 1.0 / (1.0 / lam_p - 1.0 / lam_s)
        brute = _gaussian_beam_overlap(
            "KTP", kelvin, length, lam_p, lam_s, lam_i, (1, 1, 2), (w_p, w_s, w_s), period, zpos
        )
        n_p, n_s, n_i = (
            _index("KTP", lam_p, 1, kelvin),
            _index("KTP", lam_s, 1, kelvin),
            _index("KTP", lam_i, 2, kelvin),
        )
        zp = z / (np.pi * n_p * w_p**2 / lam_p)
        zs = (z - zpos[0]) / (np.pi * n_s * w_s**2 / lam_s)
        zi = (z - zpos[1]) / (np.pi * n_i * w_s**2 / lam_i)
        a_p = 1.0 / (w_p**2 * (1 + 1j * zp))
        a_s = 1.0 / (w_s**2 * (1 - 1j * zs))
        a_i = 1.0 / (w_s**2 * (1 - 1j * zi))
        dk = _phase_mismatch("KTP", kelvin, lam_p, lam_s, lam_i, (1, 1, 2), period)
        closed = np.trapezoid(
            np.exp(1j * dk * z)
            / ((1 + 1j * zp) * (1 - 1j * zs) * (1 - 1j * zi))
            * (2.0 / (w_s * w_s))
            / (a_p + a_s + a_i),
            z,
        )
        assert abs(brute) == pytest.approx(abs(closed), rel=5e-3)


def test_plane_wave_pair_rate_closed_form_matches_the_numerical_overlap():
    """R = 4 d^2 w_s w_i P L I^2 / (c^2 n_s n_i pi w_p^2 n_p eps0 c |1/v_s - 1/v_i|), the
    textbook plane-wave (loose focusing) limit, I the transverse overlap of the 3 modes."""
    kelvin, length, w_p, w_s = 298.15, 10e-3, 400e-6, 300e-6
    cfg = spdcalc.SPDC.from_yaml(_ktp_config(length, w_p, w_s))
    period = cfg.poling_period_um * 1e-6
    d = 2.0 / np.pi * 1e-12
    n_p = _index("KTP", 405e-9, 1, kelvin)
    n_s, n_i = _index("KTP", 810e-9, 1, kelvin), _index("KTP", 810e-9, 2, kelvin)
    dbeta = abs(_group_index("KTP", 810e-9, 1, kelvin) - _group_index("KTP", 810e-9, 2, kelvin)) / C
    omega = 2.0 * np.pi * C / 810e-9
    overlap = _overlap_closed_form(w_p, w_s, w_s)
    closed = (
        4.0
        * d**2
        * omega**2
        * 1e-3
        * length
        * overlap**2
        / (C**2 * n_s * n_i * np.pi * w_p**2 * n_p * _EPS0 * C * dbeta)
    )
    numeric = coupled_mode_pair_rate(
        "KTP", kelvin, length, 405e-9, 810e-9, (1, 1, 2), (w_p, w_s, w_s), d, period
    )
    assert closed == pytest.approx(numeric, rel=0.06)
    assert 150.0 < closed < 260.0


@pytest.mark.parametrize(
    ("w_p", "w_s", "length"),
    [(400e-6, 300e-6, 10e-3), (100e-6, 50e-6, 10e-3), (30e-6, 30e-6, 10e-3)],
)
def test_absolute_pair_rate_first_principles(w_p, w_s, length):
    """Absolute pair rate against an independent first-principles evaluation.

    Derivation. H_I = 4 eps0 d E_p^+ E_s^- E_i^- + h.c. (the energy density -(eps0 chi/3)E^3
    with chi = 2 d; it reproduces Boyd's coupled equations dA_s/dz = 2 i w_s d A_p A_i^* /
    (n_s c) for E = A e^{i(kz - wt)} + c.c.), d being the quasi-phase-matched coefficient
    (2/pi) d_bulk. Collection mode u (int |u|^2 d^2r = 1) photon field:
    E^+ = i int dw sqrt(hbar w / (4 pi eps0 n c)) u a(w) e^{i(kz - wt)}. Fermi's golden rule
    with a classical pump A_p(r) (I = 2 n eps0 c |A_p|^2, A_0^2 = P/(pi w_p^2 n_p eps0 c)):
      R = (2 d^2 / (pi c^2)) int dw_s (w_s w_i / (n_s n_i)) |J|^2,
      J = int d^3r A_p(r) u_s^*(r) u_i^*(r) exp[i (k_p - k_s - k_i - 2 pi / Lambda) z].
    In the plane-wave limit this is R = int dw_s/(2 pi) g^2 L^2 sinc^2(dk L / 2) with the
    parametric gain g^2 = 4 w_s w_i d^2 |A_p|^2 / (n_s n_i c^2) (one vacuum photon per mode
    amplified).

    Result, 10 mm ppKTP, d_bulk = 1 pm/V, waists 100/50/50 um, 1 mW: 4.1 kHz here; spdcalc's
    JSI integral (what the engine now uses) gives 4.7 kHz (ratio 1.15; 1.12 loose focus, 1.26
    at 30 um); spdcalc's coincidences_hz (the same integral times get_counts_correction =
    0.2035) gives 0.94 kHz, i.e. 4.3x below this derivation. A golden-rule estimate that
    takes H = 2 eps0 d (half the coupling above, hence 1/4 of the rate) lands within 20% of
    the corrected figure only because the two factors cancel (0.82 in the first review). The
    pi^2/2 of the second review is not a quasi-phase-matching double count (see the bulk
    versus poled test): the correction factor is 0.1-0.2 depending on crystal and wavelengths
    and is close to 2/pi^2 at 405 -> 810 nm in KTP by accident.
    """
    kelvin = 298.15
    cfg = spdcalc.SPDC.from_yaml(_ktp_config(length, w_p, w_s))
    period = cfg.poling_period_um * 1e-6
    zpos = (cfg.signal_waist_position_um * 1e-6, cfg.idler_waist_position_um * 1e-6)
    reference = coupled_mode_pair_rate(
        "KTP",
        kelvin,
        length,
        405e-9,
        810e-9,
        (1, 1, 2),
        (w_p, w_s, w_s),
        2.0 / np.pi * 1e-12,
        period,
        zpos,
    )
    options = dict(
        length=f"{length * 1e3} mm",
        signal_waist=f"{w_s * 1e6} um",
        idler_waist=f"{w_s * 1e6} um",
        spectral_cutoff=1e-3,
        max_samples=60000,
        delay_window="100 ns",
    )
    rate = emit(crystal(**options), pump_light(power=1e-3, waist=w_p))[0].source.rate
    assert 0.9 < rate / reference < 1.4
    corrected = emit(
        crystal(spdcalc_counts_correction=True, **options), pump_light(power=1e-3, waist=w_p)
    )[0].source.rate
    assert 3.0 < reference / corrected < 6.0


def _lithium_niobate_config(kelvin, period):
    return _config(
        kind="LiNb_MgO",
        pm_type="Type1_e_oo",
        theta_deg=90.0,
        length_um=10e3,
        temperature_c=kelvin - 273.15,
        pump_nm=532.0,
        bandwidth_nm=0.0405,
        pump_waist_um=100.0,
        signal_nm=900.0,
        signal_waist_um=70.0,
        idler_waist_um=70.0,
        period_um=period,
        deff=1.0,
    )


def _lithium_niobate_phase_matching_temperature():
    lam_p, lam_s = 532e-9, 900e-9
    lam_i = 1.0 / (1.0 / lam_p - 1.0 / lam_s)
    return brentq(
        lambda t: _phase_mismatch("LiNb_MgO", t, lam_p, lam_s, lam_i, (2, 0, 0), None),
        298.0,
        400.0,
    )


def test_poled_and_bulk_jsa_peaks_differ_by_exactly_two_over_pi():
    """No double counting of the quasi-phase-matching factor: at the same geometry (non-
    critical LiNbO3:MgO) the spdcalc JSA peak of a poled crystal is 2/pi of the bulk one."""
    lam_p = 532e-9

    def peak(kelvin, period):
        js = spdcalc.SPDC.from_yaml(_lithium_niobate_config(kelvin, period)).joint_spectrum(
            spdcalc.Integrator.gauss_legendre(100)
        )
        w_p = 2.0 * np.pi * C / lam_p
        w_s = np.linspace(0.98, 1.02, 801) * 2.0 * np.pi * C / 900e-9
        return max(abs(js.jsa(x, w_p - x)) for x in w_s)

    ratio = peak(298.15, "auto") / peak(_lithium_niobate_phase_matching_temperature(), None)
    assert ratio == pytest.approx(2.0 / np.pi, rel=0.01)


def test_bulk_crystal_rate_agrees_with_first_principles_too():
    """The agreement is not specific to ppKTP: non-critically phase-matched bulk LiNbO3:MgO
    (no QPM factor), e -> o + o, 532 -> 900 + 1301 nm, 10 mm, waists 100/70/70 um."""
    kelvin = _lithium_niobate_phase_matching_temperature()
    comp = crystal(
        material="LiNb_MgO",
        pm_type="Type1_e_oo",
        poling_period=None,
        signal_wavelength="900 nm",
        signal_waist="70 um",
        idler_waist="70 um",
        design_temperature=f"{kelvin - 273.15} degC",
        settings={"temperature": f"{kelvin - 273.15} degC"},
        spectral_cutoff=1e-3,
        max_samples=100000,
        delay_window="100 ns",
    )
    rate = emit(comp, pump_light(power=1e-3, wavelength=532e-9, waist=100e-6))[0].source.rate
    ref = spdcalc.SPDC.from_yaml(_lithium_niobate_config(kelvin, None))
    reference = coupled_mode_pair_rate(
        "LiNb_MgO",
        kelvin,
        10e-3,
        532e-9,
        900e-9,
        (2, 0, 0),
        (100e-6, 70e-6, 70e-6),
        1e-12,
        None,
        (ref.signal_waist_position_um * 1e-6, ref.idler_waist_position_um * 1e-6),
        n=1601,
    )
    assert 0.6 < rate / reference < 1.3


def test_detuned_phase_matching_is_reported_with_its_magnitude():
    _, ctx = emit(crystal())
    assert "phase_matching_detuned" not in [i.code for i in ctx.issues]
    for kwargs, sign in (({"poling_period": "5 um"}, -1), ({"poling_period": "11 um"}, 1)):
        emission, ctx = emit(crystal(**kwargs))
        issue = next(i for i in ctx.issues if i.code == "phase_matching_detuned")
        assert emission is not None
        shift = float(re.search(r"\), ([+-]\d+\.\d) nm", issue.message).group(1))
        assert sign * shift > 20.0
        assert "810" in issue.message


def test_temperature_is_physically_bounded():
    with pytest.raises(SetupError, match="above any physical"):
        emit(crystal(settings={"temperature": "2000 degC"}))
    _, ctx = emit(crystal(settings={"temperature": "400 degC"}))
    assert "temperature_extrapolation" in [i.code for i in ctx.issues]
    _, ctx = emit(crystal(settings={"temperature": "60 degC"}))
    assert "temperature_extrapolation" not in [i.code for i in ctx.issues]


def test_one_hertz_lattice_is_refused_before_allocation():
    with pytest.raises(SetupError, match=r"spectral grid needs about .*max_samples"):
        emit(crystal(dnu="1 Hz"))
    with pytest.raises(SetupError, match=r"spectral grid needs about"):
        entangled(bandwidth="1 nm", dnu="1 Hz").emit_pairs({}, Context())
    with pytest.raises(ValueError, match="raise dnu"):
        gaussian_pair_source(810e-9, bandwidth=1e11, dnu=1.0)


def test_explicit_dnu_above_max_samples_warns_and_max_samples_is_respected():
    comp = crystal(max_samples=2000, dnu="200 MHz")
    emission, ctx = emit(comp)
    assert "pair_grid_exceeds_max_samples" in [i.code for i in ctx.issues]
    assert len(emission.source.n_signal) < 20000
    capped, ctx = emit(crystal(max_samples=500))
    assert len(capped.source.n_signal) <= 520


def test_dnu_must_be_positive():
    with pytest.raises(ValueError, match="dnu"):
        crystal(dnu="0 Hz")


def _ppln_type0(pump_nm, signal_nm, w_p=100e-6, w_s=70e-6, **options):
    cfg = spdcalc.SPDC.from_yaml(
        _config(
            kind="LiNb_MgO",
            pm_type="Type0_e_ee",
            theta_deg=90.0,
            length_um=10e3,
            temperature_c=25.0,
            pump_nm=pump_nm,
            bandwidth_nm=pump_nm * 1e-4,
            pump_waist_um=w_p * 1e6,
            signal_nm=signal_nm,
            signal_waist_um=w_s * 1e6,
            idler_waist_um=w_s * 1e6,
            period_um="auto",
            deff=1.0,
        )
    )
    return cfg, crystal(
        material="LiNb_MgO",
        pm_type="Type0_e_ee",
        signal_wavelength=f"{signal_nm} nm",
        signal_waist=f"{w_s * 1e6} um",
        idler_waist=f"{w_s * 1e6} um",
        spectral_cutoff=1e-3,
        max_samples=100000,
        delay_window="100 ns",
        **options,
    )


def _photon_flux(cfg, pump_nm, signal_nm, span, w_p=100e-6, w_s=70e-6):
    """Photons per second per mW emitted in the signal-wavelength window (first principles,
    every photon counted once whichever of the two roles it plays)."""
    return coupled_mode_pair_rate(
        "LiNb_MgO",
        298.15,
        10e-3,
        pump_nm * 1e-9,
        signal_nm * 1e-9,
        (2, 2, 2),
        (w_p, w_s, w_s),
        2.0 / np.pi * 1e-12,
        cfg.poling_period_um * 1e-6,
        (cfg.signal_waist_position_um * 1e-6, cfg.idler_waist_position_um * 1e-6),
        n=2401,
        span=span,
    )


def test_degenerate_same_polarisation_rate_counts_each_pair_once():
    """Type-0 775 -> 1550 + 1550 nm: signal and idler are the same mode set, so the plane-wave
    photon flux N(w) dw / 2 pi integrated over the whole ridge counts both photons of a pair;
    the pair rate is half of it. A non-degenerate type-0 (775 -> 1300 + 1918 nm, only one ridge
    inside the signal band) is not halved."""
    cfg, comp = _ppln_type0(775.0, 1550.0)
    flux = _photon_flux(cfg, 775.0, 1550.0, span=0.1)
    rate = emit(comp, pump_light(power=1e-3, wavelength=775e-9, waist=100e-6))[0].source.rate
    assert rate / flux == pytest.approx(0.5, rel=0.1)

    cfg, comp = _ppln_type0(775.0, 1300.0)
    flux = _photon_flux(cfg, 775.0, 1300.0, span=0.05)
    rate = emit(comp, pump_light(power=1e-3, wavelength=775e-9, waist=100e-6))[0].source.rate
    assert rate / flux == pytest.approx(1.0, rel=0.1)


def test_distinct_collection_modes_keep_ordered_counting():
    """With different signal and idler collection waists the two roles are different spatial
    modes: (w, w') and (w', w) are different pairs, both are counted and nothing is halved."""
    cfg = spdcalc.SPDC.from_yaml(
        _config(
            kind="LiNb_MgO",
            pm_type="Type0_e_ee",
            theta_deg=90.0,
            length_um=10e3,
            temperature_c=25.0,
            pump_nm=775.0,
            bandwidth_nm=0.0775,
            pump_waist_um=100.0,
            signal_nm=1550.0,
            signal_waist_um=50.0,
            idler_waist_um=90.0,
            period_um="auto",
            deff=1.0,
        )
    )
    reference = coupled_mode_pair_rate(
        "LiNb_MgO",
        298.15,
        10e-3,
        775e-9,
        1550e-9,
        (2, 2, 2),
        (100e-6, 50e-6, 90e-6),
        2.0 / np.pi * 1e-12,
        cfg.poling_period_um * 1e-6,
        (cfg.signal_waist_position_um * 1e-6, cfg.idler_waist_position_um * 1e-6),
        n=2401,
        span=0.1,
    )
    comp = crystal(
        material="LiNb_MgO",
        pm_type="Type0_e_ee",
        signal_wavelength="1550 nm",
        signal_waist="50 um",
        idler_waist="90 um",
        spectral_cutoff=1e-3,
        max_samples=100000,
        delay_window="100 ns",
    )
    rate = emit(comp, pump_light(power=1e-3, wavelength=775e-9, waist=100e-6))[0].source.rate
    assert rate / reference == pytest.approx(1.0, rel=0.25)
