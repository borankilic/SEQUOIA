import math
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
    return np.sum(np.abs(src.amplitude) ** 2, axis=(0, 1, 3))


def sum_marginal(src):
    return np.sum(np.abs(src.amplitude) ** 2, axis=(0, 1, 2))


def fwhm_of(x, w):
    w = np.asarray(w, dtype=float)
    mean = np.sum(w * x) / w.sum()
    return FWHM * math.sqrt(np.sum(w * (x - mean) ** 2) / w.sum())


def polarization_density(src):
    a = src.amplitude.reshape(4, -1)
    return a @ a.conj().T


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
    assert len(src.n_sum) == 4
    assert np.all(np.diff(src.n_sum) == 1)
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
    assert 0.95 < src.rate / windowed < 1.4


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
    with pytest.raises(SetupError) as info:
        emit(crystal(), pump_light(wavelength=250e-9))
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


def test_crystal_rate_against_golden_rule_estimate():
    """Independent first-principles check of the absolute pair rate.

    Lowest-order perturbation theory (golden rule) for collinear, loosely focused Gaussian
    modes, H = 2 eps0 d_eff E_p^+ E_s^- E_i^-, one-photon quantisation field
    E0 = sqrt(hbar w / (2 eps0 n^2 V)), plane-wave phase matching over the length L with
    group velocities v. After integrating over the signal frequency at fixed pump frequency
    (the CW limit), pairs per second into the two single-mode collection modes are

      R = d^2 P w_s w_i I^2 L / (2 eps0 n_p c v_s v_i (n_s n_i)^2 |1/v_s - 1/v_i|),
      I = (2/pi)^(3/2) pi / (w_p w_s w_i (1/w_p^2 + 1/w_s^2 + 1/w_i^2)),

    where I is the transverse overlap of the three Gaussian modes (w are the 1/e^2 radii),
    P is the pump power, and d = (2/pi) d_eff is the first-order quasi-phase-matching
    coefficient. For 10 mm ppKTP, d_eff = 1 pm/V, waists 100/50/50 um, 1 mW this gives about
    1150 Hz (the correction of using sqrt(n n_g) in the field normalisation lowers it to
    about 1080 Hz); spdcalc gives 943 Hz. The formula neglects focusing (Gouy phase, the
    finite confocal parameter) and the dispersion of the mode normalisation, which are
    exactly the 10-25% effects spdcalc includes, so the ratio spdcalc / estimate is expected
    in 0.8-1.3 (the same ratio is 0.78-0.83 for 2-5 mm crystals with 150-300 um waists).
    Without spdcalc's counts correction the ratio would be about 4: this bounds an error in
    the absolute normalisation, which a test of spdcalc against spdcalc cannot.
    """
    temperature = 298.15
    c, eps0 = constants.c, constants.epsilon_0

    def index(nm, axis):
        return spdcalc.get_crystal_indices("KTP", nm, temperature)[axis]

    def group_index(nm, axis):
        h = 0.05
        omega = lambda x: 2.0 * np.pi * c / (x * 1e-9)  # noqa: E731
        slope = (index(nm + h, axis) - index(nm - h, axis)) / (omega(nm + h) - omega(nm - h))
        return index(nm, axis) + omega(nm) * slope

    n_p, n_s, n_i = index(405, 1), index(810, 1), index(810, 2)
    v_s, v_i = c / group_index(810, 1), c / group_index(810, 2)
    w_s = w_i = 2.0 * np.pi * c / 810e-9
    wp, ws, wi, length, power = 100e-6, 50e-6, 50e-6, 10e-3, 1e-3
    overlap = (2.0 / np.pi) ** 1.5 * np.pi / (wp * ws * wi * (1 / wp**2 + 1 / ws**2 + 1 / wi**2))
    d = 2.0 / np.pi * 1e-12
    estimate = (
        d**2
        * power
        * w_s
        * w_i
        * overlap**2
        * length
        / (2.0 * eps0 * n_p * c * v_s * v_i * (n_s * n_i) ** 2 * abs(1.0 / v_s - 1.0 / v_i))
    )
    assert 1000.0 < estimate < 1300.0
    rate = emit(crystal(), pump_light(power=power, waist=wp))[0].source.rate
    assert 0.8 < rate / estimate < 1.3
