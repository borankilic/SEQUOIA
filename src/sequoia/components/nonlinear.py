import math
from collections.abc import Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

import numpy as np
import spdcalc
from pydantic import Field
from scipy import constants

from sequoia.components._util import diag_jones, spectral, sqrt_power
from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import Curve, Duration, Length, Power, Scalar
from sequoia.errors import SetupError
from sequoia.light.gaussian import GaussianMode
from sequoia.light.pairs import PairEmission, PairSource

FREE, IN, OUT = PortKind.FREE_SPACE, PortDirection.IN, PortDirection.OUT
Unity = Scalar(value=1.0)
_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
_TWO_PI = 2.0 * math.pi
_KELVIN = 273.15
_REFERENCE_BANDWIDTH = 1e-4

PMType = Literal["Type0_e_ee", "Type0_o_oo", "Type1_e_oo", "Type2_e_eo", "Type2_e_oe"]
BellState = Literal["phi+", "phi-", "psi+", "psi-"]

_BELL_ORDER = ("phi+", "phi-", "psi+", "psi-")


def bell_amplitude(state: BellState, phase: float = 0.0) -> np.ndarray:
    """Polarization amplitude [sigma_s, sigma_i] of a Bell state; ``phase`` multiplies the
    VV (phi) or VH (psi) term."""
    m = np.zeros((2, 2), dtype=complex)
    e = np.exp(1j * phase)
    if state == "phi+":
        m[0, 0], m[1, 1] = 1.0, e
    elif state == "phi-":
        m[0, 0], m[1, 1] = 1.0, -e
    elif state == "psi+":
        m[0, 1], m[1, 0] = 1.0, e
    elif state == "psi-":
        m[0, 1], m[1, 0] = 1.0, -e
    else:
        raise ValueError(f"unknown Bell state {state!r}")
    return m / np.sqrt(2.0)


def bell_ensemble(
    state: BellState, visibility: float = 1.0, phase: float = 0.0
) -> list[tuple[float, np.ndarray]]:
    """Werner state v|B><B| + (1-v) I/4 as an ensemble of Bell states (weights sum to 1)."""
    if not 0.0 <= visibility <= 1.0:
        raise ValueError("visibility must be in [0, 1]")
    if visibility == 1.0:
        return [(1.0, bell_amplitude(state, phase))]
    noise = (1.0 - visibility) / 4.0
    return [
        (visibility + noise if s == state else noise, bell_amplitude(s, phase)) for s in _BELL_ORDER
    ]


def _lattice_sum_entries(
    n_centre: int, copies: int, per_copy: int
) -> tuple[np.ndarray, np.ndarray]:
    half = per_copy // 2
    base = n_centre + copies * (np.arange(per_copy) - half)
    n_sum = (base[:, None] + np.arange(copies)[None, :]).reshape(-1)
    copy_index = np.tile(np.arange(copies), per_copy)
    return n_sum, copy_index


def gaussian_pair_source(
    wavelength_signal: float,
    wavelength_idler: float | None = None,
    *,
    bandwidth: float,
    rate: float = 1.0,
    pump_bandwidth: float | None = None,
    polarization: np.ndarray | None = None,
    ensemble: Sequence[tuple[float, np.ndarray]] | None = None,
    signal_mode: GaussianMode | None = None,
    idler_mode: GaussianMode | None = None,
    samples: int = 401,
    dnu: float | None = None,
    repetition_rate: float | None = None,
) -> PairSource:
    """Gaussian-JSA pair source on the integer frequency lattice.

    ``bandwidth`` is the intensity FWHM (Hz) of each photon's marginal spectrum;
    ``pump_bandwidth`` is the intensity FWHM (Hz) of the pump spectrum (None: CW, a single
    sum-frequency entry; pulsed: contiguous entries over +-4 sigma_p). The amplitude is
    exp(-u^2/(4 s_p^2)) exp(-v^2/(4 s_m^2)) with
    u = nu_s + nu_i - nu_p, v = nu_s - nu_i - (nu_s0 - nu_i0) and s_m^2 = 4 s_x^2 - s_p^2,
    so the marginal std is s_x = bandwidth / 2.3548 and the correlation is controlled by
    pump_bandwidth. ``polarization`` is a (2, 2) amplitude [sigma_s, sigma_i] (default HH);
    ``ensemble`` is a list of (weight, (2, 2) amplitude) pairs encoding a mixed state: each
    member is placed on its own interleaved sum-frequency entries so the engine, which adds
    sum-frequency entries incoherently, reproduces the mixture. An ensemble member sits at a
    sum frequency shifted by at most (members - 1) lattice steps.

    The ensemble encoding reproduces the mixed state for every quantity that is a sum over
    sum-frequency entries of per-entry terms (singles, coincidence rates, the coincidence
    window fraction, HOM interference, CW thermal g2). It is not equivalent for pulsed
    unheralded g2 / Schmidt number: there the one-photon reduced state is built coherently
    over the sum-frequency index, members share idler lattice sites and each member alone
    samples only every ``len(members)``-th sum frequency, so K and g2 - 1 come out wrong
    (about a factor 2 for visibility < 1 with a pump bandwidth).
    """
    c = constants.c
    nu_s0 = c / wavelength_signal
    nu_i0 = nu_s0 if wavelength_idler is None else c / wavelength_idler
    nu_p0 = nu_s0 + nu_i0
    sigma_x = bandwidth / _FWHM
    sigma_p = 0.0 if not pump_bandwidth else pump_bandwidth / _FWHM
    if sigma_p > 2.0 * sigma_x * (1.0 + 1e-9):
        raise ValueError("pump_bandwidth cannot exceed twice the photon bandwidth")
    sigma_m = math.sqrt(max(4.0 * sigma_x**2 - sigma_p**2, 0.0))
    if dnu is None:
        dnu = 9.0 * sigma_x / (samples - 1)
    members = list(ensemble) if ensemble is not None else None
    if members is None:
        pol = np.array([[1.0, 0.0], [0.0, 0.0]]) if polarization is None else polarization
        members = [(1.0, np.asarray(pol, dtype=complex))]
    weights = np.array([w for w, _ in members], dtype=float)
    weights = weights / weights.sum()
    copies = len(members)

    half = int(math.ceil(4.5 * sigma_x / dnu))
    n_s0 = int(round(nu_s0 / dnu))
    n_signal = n_s0 + np.arange(-half, half + 1)
    n_p0 = int(round(nu_p0 / dnu))
    if sigma_p > 0.0 and 3.0 * pump_bandwidth / dnu >= 4.0 * copies:
        per_copy = 2 * int(math.ceil(4.0 * sigma_p / (copies * dnu))) + 1
    else:
        per_copy, sigma_p = 1, 0.0
    n_sum, copy_index = _lattice_sum_entries(n_p0, copies, per_copy)

    nu_s = n_signal[:, None] * dnu
    nu_i = n_sum[None, :] * dnu - nu_s
    v = nu_s - nu_i - (nu_s0 - nu_i0)
    f = np.exp(-(v**2) / (4.0 * sigma_m**2))
    if sigma_p > 0.0:
        u = nu_s + nu_i - nu_p0
        f = f * np.exp(-(u**2) / (4.0 * sigma_p**2))
    amplitude = np.zeros((2, 2, len(n_signal), len(n_sum)), dtype=complex)
    for k, (_, pol) in enumerate(members):
        cols = np.flatnonzero(copy_index == k)
        block = f[:, cols]
        norm = np.sqrt(np.sum(block**2))
        pol = np.asarray(pol, dtype=complex)
        pol = pol / np.sqrt(np.sum(np.abs(pol) ** 2))
        amplitude[:, :, :, cols] = (
            np.sqrt(weights[k]) * pol[:, :, None, None] * (block / norm)[None, None]
        )
    if signal_mode is None:
        signal_mode = GaussianMode.from_waist(wavelength_signal, 1e-3)
    if idler_mode is None:
        idler_mode = GaussianMode.from_waist(c / nu_i0, 1e-3)
    return PairSource(
        dnu=float(dnu),
        n_signal=n_signal,
        n_sum=n_sum,
        amplitude=amplitude,
        rate=float(rate),
        signal_mode=signal_mode,
        idler_mode=idler_mode,
        pump_wavelength=c / (n_p0 * dnu),
        repetition_rate=repetition_rate,
    )


@contextmanager
def _spdcalc_panics():
    try:
        yield
    except BaseException as exc:
        if type(exc).__name__ == "PanicException":
            raise SetupError(
                f"spdcalc cannot evaluate this configuration (internal error: {exc})"
            ) from exc
        raise


class SPDCCrystalParams(Params):
    material: str = "KTP"
    pm_type: PMType = "Type2_e_eo"
    length: Length
    theta: Scalar | Literal["auto"] = Scalar(value=90.0, unit="deg")
    poling_period: Length | Literal["auto"] | None = None
    design_temperature: Scalar = Scalar(value=25.0, unit="degC")
    d_eff: Scalar | None = Field(
        default=None,
        description=(
            "bulk nonlinear coefficient of the interacting polarization components (pm/V, "
            "e.g. d24 ~ 3.9 pm/V for yyz KTP at 405 -> 810 nm). Do not pre-multiply by the "
            "2/pi quasi-phase-matching factor: spdcalc applies it itself for periodic poling. "
            "Required unless 'brightness' is given."
        ),
    )
    signal_wavelength: Length | None = None
    signal_waist: Length = Scalar(value=50e-6, unit="m")
    idler_waist: Length = Scalar(value=50e-6, unit="m")
    extraordinary_axis: Literal["H", "V"] = "H"
    pump_transmission: Scalar | Curve = Unity
    brightness: float | None = None
    delay_window: Duration = Scalar(value=10e-9, unit="s")
    max_samples: int = 20000
    dnu: Scalar | None = None
    spectral_cutoff: float = Field(
        default=0.01,
        description=(
            "the signal band kept is where |JSA|^2 along the phase-matching ridge exceeds this "
            "fraction of its peak; 0.01 drops the sinc side lobes and loses about 2.3% of the "
            "pair rate (measured, 10 mm ppKTP: 940 Hz vs 962 Hz at 1e-3)"
        ),
    )
    integration_degree: int = 100


class SPDCCrystalSettings(Settings):
    temperature: Scalar = Scalar(value=25.0, unit="degC")


def _config(
    *,
    kind: str,
    pm_type: str,
    theta_deg: float,
    length_um: float,
    temperature_c: float,
    pump_nm: float,
    bandwidth_nm: float,
    pump_waist_um: float,
    signal_nm: float,
    signal_waist_um: float,
    idler_waist_um: float,
    period_um: float | str | None,
    deff: float,
) -> str:
    idler_nm = 1.0 / (1.0 / pump_nm - 1.0 / signal_nm)
    poling = "null" if period_um is None else f"\n  poling_period_um: {period_um}"
    return f"""
crystal:
  kind: {kind}
  pm_type: {pm_type}
  phi_deg: 0
  theta_deg: {theta_deg!r}
  length_um: {length_um!r}
  temperature_c: {temperature_c!r}
pump:
  wavelength_nm: {pump_nm!r}
  waist_um: {pump_waist_um!r}
  bandwidth_nm: {bandwidth_nm!r}
  average_power_mw: 1.0
signal:
  wavelength_nm: {signal_nm!r}
  phi_deg: 0
  theta_deg: 0
  waist_um: {signal_waist_um!r}
  waist_position_um: auto
idler:
  wavelength_nm: {idler_nm!r}
  phi_deg: 180
  theta_deg: 0
  waist_um: {idler_waist_um!r}
  waist_position_um: auto
periodic_poling: {poling}
deff_pm_per_volt: {deff!r}
"""


@lru_cache(maxsize=64)
def _kappa(config: str) -> float:
    """Ratio between spdcalc's coincidence rate (Hz at 1 mW) and the plain sum of |JSA|^2 over
    the same grid times d(omega_s) d(omega_i); a constant of the configuration."""
    s = spdcalc.SPDC.from_yaml(config)
    r = s.optimum_range(40)
    x, y = np.asarray(r.x_values()), np.asarray(r.y_values())
    jsi = np.asarray(s.joint_spectrum().jsi_range(r))
    rate = s.efficiencies(r)["coincidences_hz"]
    return float(rate / (jsi.sum() * (x[1] - x[0]) * (y[1] - y[0])))


def _jsa(js, w_s: np.ndarray, w_i: np.ndarray) -> np.ndarray:
    w_s, w_i = np.broadcast_arrays(w_s, w_i)
    out = np.fromiter(
        (js.jsa(a, b) for a, b in zip(w_s.ravel(), w_i.ravel(), strict=True)),
        dtype=complex,
        count=w_s.size,
    ).reshape(w_s.shape)
    return np.nan_to_num(out)


def _ridge_hull(js, w_p: float, w_prior: float, cutoff: float) -> tuple[float, float, float, float]:
    """Signal angular-frequency interval where |JSA|^2 along w_s + w_i = w_p exceeds
    ``cutoff`` times its maximum (within 10 FWHM of the main peak), plus the peak position and
    the FWHM.
    The main peak is searched within +-12% of ``w_prior``."""
    half = 0.12 * w_prior
    for step_hz in (20e9, 5e9, 1.25e9, 0.3e9, 0.08e9):
        ws = np.arange(w_prior - half, w_prior + half, _TWO_PI * step_hz)
        p = np.abs(_jsa(js, ws, w_p - ws)) ** 2
        k = int(np.argmax(p))
        if p[k] <= 0.0:
            break
        above = p >= 0.5 * p[k]
        a = b = k
        while a > 0 and above[a - 1]:
            a -= 1
        while b < len(ws) - 1 and above[b + 1]:
            b += 1
        if b - a >= 5:
            break
    if p[k] <= 0.0:
        raise SetupError("no phase-matched pair emission found near the requested wavelength")
    fwhm = max(ws[b] - ws[a], ws[1] - ws[0])
    peak = float(ws[k])
    ws = np.linspace(peak - 10.0 * fwhm, peak + 10.0 * fwhm, 801)
    p = np.abs(_jsa(js, ws, w_p - ws)) ** 2
    keep = np.flatnonzero(p >= cutoff * p.max())
    return float(ws[keep[0]]), float(ws[keep[-1]]), peak, float(fwhm)


@register("nonlinear", "SPDCCrystal")
class SPDCCrystal(Component):
    """Bulk or periodically poled SPDC crystal driven by the pump arriving at ``in``.

    Pair rates come from spdcalc: ``counts_coincidences`` is documented as a coincidence rate
    for the configured ``average_power_mw``; we use it as pairs per second (time averaged)
    into the two Gaussian collection modes, per mW of pump polarized along the pump axis. It
    scales as d_eff^2 and linearly with power. Only the spectral window covered by the JSA
    grid is counted (spectral_cutoff of the phase-matching peak), so the rate is the pair rate
    within that band (the default cutoff of 1% costs about 2.3% of the rate).

    ``d_eff`` is the bulk coefficient: spdcalc multiplies by the 2/pi quasi-phase-matching
    factor itself. It is required unless a measured ``brightness`` is given.

    The pump waist handed to spdcalc is the waist of the incident Gaussian mode (not its
    radius at the crystal port). spdcalc places the pump waist at the crystal centre; a
    warning is issued when the real waist is displaced from the centre by more than the
    Rayleigh range in the crystal.

    Signal and idler wavelengths must lie inside the crystal transmission range (error); a
    pump outside it only produces a warning. spdcalc panics are reported as SetupError.

    Polarization labels follow ``extraordinary_axis``: the e axis is H (or V) and the o axis
    the other one; "Type2_e_eo" is pump e, signal e, idler o, i.e. signal H / idler V for
    the default. The pump power weighted by its polarization projection on the pump axis is
    used. The residual pump passes through with ``pump_transmission``.
    """

    Params = SPDCCrystalParams
    Settings = SPDCCrystalSettings

    def ports(self):
        return (PortSpec("in", FREE, IN, required=True), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        t = spectral(self.params.pump_transmission, beam.wavelengths)
        amplitude = sqrt_power(t)
        return [("out", beam.through(ctx.subject, diag_jones(amplitude, amplitude)))]

    def _resolve_crystal(self, common: dict):
        p = self.params
        design_c = p.design_temperature.si - _KELVIN
        theta = 90.0 if p.theta == "auto" else p.theta.si * 180.0 / math.pi
        period = None
        if isinstance(p.poling_period, Scalar):
            period = p.poling_period.si * 1e6
        if p.theta == "auto":
            text = _config(**common, theta_deg=theta, period_um=None, temperature_c=design_c)
            theta = spdcalc.SPDC.from_yaml(text).with_optimum_crystal_theta().crystal_theta_deg
        if p.poling_period == "auto":
            text = _config(**common, theta_deg=theta, period_um="auto", temperature_c=design_c)
            period = spdcalc.SPDC.from_yaml(text).poling_period_um
        return theta, period

    def _check_wavelengths(self, ctx, pump_nm: float, signal_nm: float) -> None:
        p = self.params
        if signal_nm <= pump_nm:
            raise SetupError(
                f"signal_wavelength ({signal_nm:.1f} nm) must be longer than the pump "
                f"wavelength ({pump_nm:.1f} nm): energy conservation"
            )
        idler_nm = 1.0 / (1.0 / pump_nm - 1.0 / signal_nm)
        try:
            with _spdcalc_panics():
                meta = spdcalc.get_crystal_meta(p.material)
        except (ValueError, KeyError) as exc:
            raise SetupError(f"unsupported crystal material {p.material!r}: {exc}") from exc
        low, high = (float(v) * 1e9 for v in meta["transmission_range"])
        for name, nm in (("signal", signal_nm), ("idler", idler_nm)):
            if not low <= nm <= high:
                raise SetupError(
                    f"{name} wavelength {nm:.1f} nm is outside the {p.material} transmission "
                    f"range [{low:.0f}, {high:.0f}] nm"
                )
        if not low <= pump_nm <= high:
            ctx.warn(
                "pump_transmission_range",
                f"pump wavelength {pump_nm:.1f} nm is outside the {p.material} transmission "
                f"range [{low:.0f}, {high:.0f}] nm; absorption is not modelled in the pair rate",
            )

    def _pump_waist(self, ctx, light, pump_nm: float) -> float:
        mode = light.mode
        waist = math.sqrt(mode.waist_x * mode.waist_y)
        kelvin = self.settings.temperature.si
        n_pump = float(np.mean(spdcalc.get_crystal_indices(self.params.material, pump_nm, kelvin)))
        past_waist = 0.5 * (mode.distance_past_waist_x + mode.distance_past_waist_y)
        offset = n_pump * past_waist + self.params.length.si / 2.0
        rayleigh = n_pump * math.pi * waist**2 / (pump_nm * 1e-9)
        if abs(offset) > rayleigh:
            ctx.warn(
                "pump_waist_position",
                f"the pump waist ({waist * 1e6:.1f} um) lies {offset * 1e3:+.1f} mm from the "
                f"crystal centre (Rayleigh range in the crystal {rayleigh * 1e3:.1f} mm); "
                "spdcalc assumes the waist at the centre, so the rate is overestimated",
            )
        return waist

    def emit_pairs(self, lights, ctx):
        light = lights.get("in")
        if light is None:
            ctx.warn("no_pump", "no pump light reaches the crystal; no pairs are emitted")
            return None
        if light.power <= 0.0:
            ctx.warn("no_pump", "the pump power at the crystal is zero; no pairs are emitted")
            return None
        p = self.params
        c = constants.c
        weights = light.power_spectrum / light.power_spectrum.sum()
        nu = c / light.wavelengths
        nu_p0 = float(np.sum(weights * nu))
        sigma_nu = float(np.sqrt(np.sum(weights * (nu - nu_p0) ** 2)))
        pump_fwhm = _FWHM * sigma_nu
        pump_nm = c / nu_p0 * 1e9
        signal_nm = (2.0 * pump_nm) if p.signal_wavelength is None else p.signal_wavelength.si * 1e9
        axis = 0 if p.extraordinary_axis == "H" else 1
        letters = p.pm_type.split("_")
        pol_of = {"e": axis, "o": 1 - axis}
        idx_pump, idx_s, idx_i = (pol_of[ch] for ch in (letters[1], letters[2][0], letters[2][1]))
        w = light.coherency.sum(axis=0)
        fraction = float(np.real(w[idx_pump, idx_pump]) / np.real(w[0, 0] + w[1, 1]))
        if fraction < 0.99:
            ctx.warn(
                "pump_polarization",
                f"only {100 * fraction:.1f}% of the pump power is polarized along the "
                f"{'e' if letters[1] == 'e' else 'o'} axis required by {p.pm_type}",
            )
        power_mw = light.power * fraction * 1e3
        if power_mw <= 0.0:
            ctx.warn(
                "no_pump",
                f"no pump power is polarized along the {'e' if letters[1] == 'e' else 'o'} axis "
                f"required by {p.pm_type}; no pairs are emitted",
            )
            return None
        if p.d_eff is None and p.brightness is None:
            raise SetupError(
                "SPDCCrystal needs 'd_eff' (bulk coefficient in pm/V, without the 2/pi "
                "quasi-phase-matching factor) or a measured 'brightness'"
            )
        self._check_wavelengths(ctx, pump_nm, signal_nm)

        try:
            with _spdcalc_panics():
                w_pump = self._pump_waist(ctx, light, pump_nm)
                common = dict(
                    kind=p.material,
                    pm_type=p.pm_type,
                    length_um=p.length.si * 1e6,
                    pump_nm=pump_nm,
                    bandwidth_nm=max(pump_fwhm * pump_nm**2 / c * 1e-9, 1e-12),
                    pump_waist_um=w_pump * 1e6,
                    signal_nm=signal_nm,
                    signal_waist_um=p.signal_waist.si * 1e6,
                    idler_waist_um=p.idler_waist.si * 1e6,
                    deff=1.0 if p.d_eff is None else p.d_eff.si * 1e12,
                )
                theta, period = self._resolve_crystal(common)
                temperature_c = self.settings.temperature.si - _KELVIN
                return self._build(
                    light,
                    ctx,
                    common,
                    theta,
                    period,
                    temperature_c,
                    nu_p0,
                    pump_fwhm,
                    power_mw,
                    (idx_s, idx_i),
                )
        except ValueError as exc:
            raise SetupError(f"spdcalc rejected the crystal configuration: {exc}") from exc

    def _build(
        self, light, ctx, common, theta, period, temperature_c, nu_p0, pump_fwhm, power_mw, pols
    ):
        idx_s, idx_i = pols
        p = self.params
        c = constants.c
        base = dict(common, theta_deg=theta, period_um=period, temperature_c=temperature_c)
        pulsed = len(light.wavelengths) > 1 and pump_fwhm > 0.0
        integrator = spdcalc.Integrator.gauss_legendre(p.integration_degree)
        wide = dict(base, bandwidth_nm=0.2 * common["pump_nm"])
        js0 = spdcalc.SPDC.from_yaml(_config(**wide)).joint_spectrum(integrator)
        w_prior = _TWO_PI * c / (common["signal_nm"] * 1e-9)
        w_p0 = _TWO_PI * nu_p0
        lo = hi = None
        shifts = (-1.5, 0.0, 1.5) if pulsed else (0.0,)
        for shift in shifts:
            a, b, pk, fwhm = _ridge_hull(
                js0, w_p0 + shift * _TWO_PI * pump_fwhm, w_prior, p.spectral_cutoff
            )
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
            if shift == 0.0:
                peak, ridge_fwhm = pk, fwhm
        span_hz = (hi - lo) / _TWO_PI
        wanted = 1.0 / p.delay_window.si
        dnu = p.dnu.si if p.dnu is not None else max(wanted, span_hz / p.max_samples)
        if pulsed and p.dnu is None and 3.0 * pump_fwhm / dnu >= 4.0:
            dnu = max(wanted, math.sqrt(span_hz * 3.0 * pump_fwhm / p.max_samples))
        if dnu > 1.001 * wanted:
            ctx.warn(
                "pair_delay_window",
                f"the spectral grid is limited to {p.max_samples} points"
                f"{' (pulsed pump: signal x sum-frequency grid)' if pulsed else ''}: the pair "
                f"state is periodic in time with period {1e9 / dnu:.3g} ns (requested "
                f"{p.delay_window.si * 1e9:.3g} ns); delays and delay differences beyond it "
                "alias. Raise max_samples (cost grows with the grid) or accept a shorter window",
            )
        if ridge_fwhm / (_TWO_PI * dnu) < 4.0:
            ctx.warn(
                "pair_spectral_resolution",
                f"the phase-matching peak ({ridge_fwhm / _TWO_PI * 1e-9:.3g} GHz FWHM) is "
                f"sampled by fewer than 4 lattice points (dnu = {dnu * 1e-9:.3g} GHz)",
            )
        n_signal = np.arange(
            int(math.floor(lo / (_TWO_PI * dnu))), int(math.ceil(hi / (_TWO_PI * dnu))) + 1
        )
        n_p0 = int(round(nu_p0 / dnu))

        if pulsed and 3.0 * pump_fwhm / dnu >= 4.0:
            half_sum = int(math.ceil(1.5 * pump_fwhm / dnu))
            n_sum = n_p0 + np.arange(-half_sum, half_sum + 1)
            config = _config(**base)
            js = spdcalc.SPDC.from_yaml(config).joint_spectrum(integrator)
            w_s = _TWO_PI * dnu * n_signal[:, None]
            w_i = _TWO_PI * dnu * n_sum[None, :] - w_s
            jsa = _jsa(js, w_s, w_i)
            weight = (_TWO_PI * dnu) ** 2
            pulsed_used = True
        else:
            n_sum = np.array([n_p0])
            ref_nm = _REFERENCE_BANDWIDTH * common["pump_nm"]
            lattice_nm = c / (n_p0 * dnu) * 1e9
            cw = dict(base, pump_nm=lattice_nm, bandwidth_nm=ref_nm)
            cw["signal_nm"] = common["signal_nm"]
            config = _config(**cw)
            js = spdcalc.SPDC.from_yaml(config).joint_spectrum(integrator)
            w_s = _TWO_PI * dnu * n_signal[:, None]
            w_i = _TWO_PI * dnu * n_sum[None, :] - w_s
            jsa = _jsa(js, w_s, w_i)
            fwhm_w = _TWO_PI * c * ref_nm * 1e-9 / (lattice_nm * 1e-9) ** 2
            weight = (_TWO_PI * dnu) * (fwhm_w / _FWHM) * math.sqrt(_TWO_PI)
            pulsed_used = False

        intensity = np.abs(jsa) ** 2
        total = float(intensity.sum())
        if total <= 0.0:
            ctx.warn("no_pairs", "spdcalc returned a vanishing joint spectrum")
            return None
        amplitude = np.zeros((2, 2, len(n_signal), len(n_sum)), dtype=complex)
        amplitude[idx_s, idx_i] = jsa / math.sqrt(total)

        if p.brightness is not None:
            rate = p.brightness * power_mw
        else:
            rate = power_mw * _kappa(config) * weight * total
        if not math.isfinite(rate) or rate < 0.0:
            raise SetupError(
                "spdcalc returned a non-finite pair rate: the pump, signal or idler wavelength "
                "is probably outside the range where the crystal model is valid"
            )

        lam_s = _TWO_PI * c / peak
        lam_i = _TWO_PI * c / (w_p0 - peak)
        reference = spdcalc.SPDC.from_yaml(_config(**base))
        kelvin = temperature_c + _KELVIN
        modes = []
        for lam, waist, position in (
            (lam_s, p.signal_waist.si, reference.signal_waist_position_um),
            (lam_i, p.idler_waist.si, reference.idler_waist_position_um),
        ):
            n_index = float(np.mean(spdcalc.get_crystal_indices(p.material, lam * 1e9, kelvin)))
            z = (p.length.si / 2.0 - position * 1e-6) / n_index
            modes.append(GaussianMode.from_waist(lam, waist, z=z))
        signal_mode, idler_mode = modes
        rep = light.repetition_rate if pulsed_used else None
        try:
            source = PairSource(
                dnu=float(dnu),
                n_signal=n_signal,
                n_sum=n_sum,
                amplitude=amplitude,
                rate=float(rate),
                signal_mode=signal_mode,
                idler_mode=idler_mode,
                pump_wavelength=c / (n_p0 * dnu),
                repetition_rate=rep,
            )
        except ValueError as exc:
            raise SetupError(f"invalid pair state for the requested configuration: {exc}") from exc
        return PairEmission(source, "out", "out")


class EntangledPairSourceParams(Params):
    pair_rate: Scalar | None = None
    brightness: float | None = None
    center_wavelength: Length
    idler_wavelength: Length | None = None
    bandwidth: Length
    pump_bandwidth: Length | None = None
    repetition_rate: Scalar | None = None
    state: BellState = "phi+"
    visibility: Scalar = Unity
    phase: Scalar = Scalar(value=0.0, unit="rad")
    signal_waist: Length = Scalar(value=0.5e-3, unit="m")
    idler_waist: Length = Scalar(value=0.5e-3, unit="m")
    delay_window: Duration = Scalar(value=10e-9, unit="s")
    samples: int | None = Field(
        default=None,
        description="upper bound on the signal lattice points (default: max_samples)",
    )
    max_samples: int = Field(
        default=20000,
        description=(
            "upper bound on signal points (CW) or signal x sum-frequency points (pulsed); "
            "when it forces a lattice spacing above 1/delay_window a warning gives the period"
        ),
    )
    dnu: Scalar | None = None


class EntangledPairSourceSettings(Settings):
    pump_power: Power = Scalar(value=1e-3, unit="W")


@register("nonlinear", "EntangledPairSource")
class EntangledPairSource(Component):
    """Black-box polarization-entangled pair source with its own pump.

    ``bandwidth`` is the wavelength FWHM of each photon's spectrum (Gaussian), centred on
    ``center_wavelength`` (signal) and ``idler_wavelength`` (default: degenerate).
    ``pump_bandwidth`` (wavelength FWHM at the pump wavelength, None = CW) sets the sum-frequency
    spread. The rate is ``pair_rate`` or ``brightness`` (pairs/s/mW) times the ``pump_power``
    setting. A visibility below 1 gives the Werner state v|B><B| + (1-v) I/4, encoded as an
    ensemble of the four Bell states on interleaved sum-frequency entries. This is exact for
    singles, coincidences, the coincidence-window fraction, HOM and CW thermal g2 (the
    members sit at sum frequencies that differ by at most three lattice steps, negligible for
    a lattice much finer than the bandwidth). It is NOT exact for pulsed unheralded g2 /
    Schmidt number (visibility < 1 with ``pump_bandwidth``): members share idler lattice
    sites and are sampled on every fourth sum frequency, so K and g2 - 1 are off by about a
    factor 2.

    The frequency lattice spacing is dnu = 1/``delay_window`` (default 10 ns, as for
    SPDCCrystal) so that the pair state is periodic in time with that period and fibre
    delays up to it do not alias. It is coarsened when the grid would exceed ``samples``
    (CW) or ``max_samples`` (signal x sum-frequency points); a ``pair_delay_window`` warning
    then states the actual period.
    """

    Params = EntangledPairSourceParams
    Settings = EntangledPairSourceSettings

    def ports(self):
        return (PortSpec("signal", FREE, OUT), PortSpec("idler", FREE, OUT))

    def _rate(self) -> float:
        p = self.params
        if p.pair_rate is not None:
            return p.pair_rate.si
        if p.brightness is not None:
            return p.brightness * self.settings.pump_power.si * 1e3
        raise SetupError("an EntangledPairSource needs 'pair_rate' or 'brightness'")

    def emit_pairs(self, lights, ctx):
        p = self.params
        c = constants.c
        lam_s = p.center_wavelength.si
        lam_i = lam_s if p.idler_wavelength is None else p.idler_wavelength.si
        lam_p = 1.0 / (1.0 / lam_s + 1.0 / lam_i)
        bandwidth = c * p.bandwidth.si / lam_s**2
        pump_bandwidth = None
        if p.pump_bandwidth is not None:
            pump_bandwidth = c * p.pump_bandwidth.si / lam_p**2
        visibility = p.visibility.si
        if not 0.0 <= visibility <= 1.0:
            raise SetupError("visibility must be between 0 and 1")
        ensemble = bell_ensemble(p.state, visibility, p.phase.si)
        sigma_x = bandwidth / _FWHM
        wanted = 1.0 / p.delay_window.si
        if p.dnu is not None:
            dnu = p.dnu.si
        else:
            cap = p.max_samples if p.samples is None else min(p.samples, p.max_samples)
            dnu = max(wanted, 9.0 * sigma_x / cap)
            if pump_bandwidth and 3.0 * pump_bandwidth / dnu >= 4.0 * len(ensemble):
                sigma_p = pump_bandwidth / _FWHM
                dnu = max(dnu, math.sqrt(9.0 * sigma_x * 8.0 * sigma_p / p.max_samples))
        if dnu > 1.001 * wanted:
            ctx.warn(
                "pair_delay_window",
                f"the spectral grid is limited ({p.samples or p.max_samples} signal points, "
                f"{p.max_samples} for signal x sum frequency): the pair state is periodic in "
                f"time with period {1e9 / dnu:.3g} ns (requested {p.delay_window.si * 1e9:.3g} "
                "ns); delays and delay differences beyond it alias",
            )
        try:
            source = gaussian_pair_source(
                lam_s,
                lam_i,
                bandwidth=bandwidth,
                rate=self._rate(),
                pump_bandwidth=pump_bandwidth,
                ensemble=ensemble,
                signal_mode=GaussianMode.from_waist(lam_s, p.signal_waist.si),
                idler_mode=GaussianMode.from_waist(lam_i, p.idler_waist.si),
                dnu=float(dnu),
                repetition_rate=None if p.repetition_rate is None else p.repetition_rate.si,
            )
        except ValueError as exc:
            raise SetupError(str(exc)) from exc
        return PairEmission(source, "signal", "idler")
