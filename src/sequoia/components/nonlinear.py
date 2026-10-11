import math
from collections.abc import Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

import numpy as np
import spdcalc
from pydantic import Field, field_validator
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
from sequoia.data import (
    Angle,
    Brightness,
    Curve,
    Dimensionless,
    Duration,
    Frequency,
    Length,
    NonlinearCoefficient,
    Power,
    Rate,
    Scalar,
    Temperature,
)
from sequoia.errors import SetupError
from sequoia.light.gaussian import GaussianMode, overlap
from sequoia.light.pairs import PairEmission, PairSource

FREE, IN, OUT = PortKind.FREE_SPACE, PortDirection.IN, PortDirection.OUT
Unity = Scalar(value=1.0)
_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
_TWO_PI = 2.0 * math.pi
_KELVIN = 273.15
_REFERENCE_BANDWIDTH = 1e-4
_MAX_GRID_CELLS = 4_000_000
_MAX_TEMPERATURE_C = 900.0
_CALIBRATED_TEMPERATURE_C = (-20.0, 250.0)

PMType = Literal["Type0_e_ee", "Type0_o_oo", "Type1_e_oo", "Type2_e_eo", "Type2_e_oe"]
BellState = Literal["phi+", "phi-", "psi+", "psi-"]

_BELL_ORDER = ("phi+", "phi-", "psi+", "psi-")


def _guard_grid(n_signal: float, n_sum: float, max_samples: int, dnu: float, explicit: bool, ctx):
    cells = n_signal * n_sum
    if cells > 10 * max_samples or cells > _MAX_GRID_CELLS:
        raise SetupError(
            f"the requested spectral grid needs about {cells:.3g} points "
            f"({n_signal:.3g} signal x {n_sum:.3g} sum-frequency) at dnu = {dnu:.3g} Hz, above "
            f"the limit of {min(10 * max_samples, _MAX_GRID_CELLS):.3g} (10 x max_samples = "
            f"{max_samples}, absolute limit {_MAX_GRID_CELLS}); raise dnu (a shorter "
            "delay_window), or raise max_samples if the memory is available"
        )
    if explicit and cells > max_samples and ctx is not None:
        ctx.warn(
            "pair_grid_exceeds_max_samples",
            f"the explicit dnu = {dnu:.3g} Hz gives about {cells:.3g} lattice points, more than "
            f"max_samples = {max_samples}",
        )


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
    ``ensemble`` is a list of (weight, (2, 2) amplitude) pairs encoding a mixed state: the
    returned ``PairSource`` has an ensemble axis (see ``PairSource``), every member shares the
    same spectral amplitude on the same contiguous lattice and carries its polarization
    amplitude times sqrt(weight). The engine adds members incoherently, so the polarization
    mixing leaves the spectral structure (Schmidt number, unheralded g2) untouched.
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

    if not dnu > 0.0:
        raise ValueError("dnu must be positive")
    half = int(math.ceil(4.5 * sigma_x / dnu))
    n_sum_count = 2 * int(math.ceil(4.0 * sigma_p / dnu)) + 1 if sigma_p > 0.0 else 1
    if (2 * half + 1) * n_sum_count * len(members) > _MAX_GRID_CELLS:
        raise ValueError(
            f"the pair lattice would need {(2 * half + 1)} signal x {n_sum_count} sum-frequency "
            f"x {len(members)} points at dnu = {dnu:.3g} Hz; raise dnu (limit {_MAX_GRID_CELLS})"
        )
    n_s0 = int(round(nu_s0 / dnu))
    n_signal = n_s0 + np.arange(-half, half + 1)
    n_p0 = int(round(nu_p0 / dnu))
    if sigma_p > 0.0 and 3.0 * pump_bandwidth / dnu >= 4.0:
        n_half = int(math.ceil(4.0 * sigma_p / dnu))
        n_sum = n_p0 + np.arange(-n_half, n_half + 1)
    else:
        n_sum, sigma_p = np.array([n_p0]), 0.0

    nu_s = n_signal[:, None] * dnu
    nu_i = n_sum[None, :] * dnu - nu_s
    v = nu_s - nu_i - (nu_s0 - nu_i0)
    f = np.exp(-(v**2) / (4.0 * sigma_m**2))
    if sigma_p > 0.0:
        u = nu_s + nu_i - nu_p0
        f = f * np.exp(-(u**2) / (4.0 * sigma_p**2))
    spectral_part = f / np.sqrt(np.sum(f**2))
    amplitude = np.zeros((len(members), 2, 2, len(n_signal), len(n_sum)), dtype=complex)
    for k, (_, pol) in enumerate(members):
        pol = np.asarray(pol, dtype=complex)
        pol = pol / np.sqrt(np.sum(np.abs(pol) ** 2))
        amplitude[k] = np.sqrt(weights[k]) * pol[:, :, None, None] * spectral_part[None, None]
    if len(members) == 1:
        amplitude = amplitude[0]
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
    theta: Angle | Literal["auto"] = Scalar(value=90.0, unit="deg")
    poling_period: Length | Literal["auto"] | None = None
    design_temperature: Temperature = Scalar(value=25.0, unit="degC")
    d_eff: NonlinearCoefficient | None = Field(
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
    pump_transmission: Dimensionless | Curve = Unity
    brightness: Brightness | None = Field(
        default=None,
        description=(
            "pairs per second (each pair counted once) per mW of pump emitted into the "
            "collection modes within the main phase-matching band, before any filter, coupling "
            "or detector loss (not a detected coincidence rate: entering one double-counts "
            "the losses). A bare number is pairs/s/mW; a quantity needs a rate-per-power unit "
            "such as '1e5 Hz/mW'. Overrides the computed rate when given"
        ),
    )
    delay_window: Duration = Scalar(value=10e-9, unit="s")
    max_samples: int = 20000
    dnu: Frequency | None = None
    spectral_cutoff: float = Field(
        default=0.01,
        description=(
            "the signal band kept is where |JSA|^2 along the phase-matching ridge exceeds this "
            "fraction of its peak; 0.01 drops the sinc side lobes and loses about 2.3% of the "
            "pair rate (measured, 10 mm ppKTP: 940 Hz vs 962 Hz at 1e-3)"
        ),
    )
    integration_degree: int = 100
    spdcalc_counts_correction: bool = Field(
        default=False,
        description=(
            "multiply the pair rate by spdcalc's get_counts_correction (about 0.2 for ppKTP "
            "405 -> 810 nm, 0.1 for LiNbO3). Off by default: the uncorrected spectral integral "
            "of spdcalc's JSI is 1.12-1.26 times an independent first-principles "
            "coupled-mode calculation for ppKTP (engine / derivation, 400/300 to 30/30 um "
            "waists; 0.9-1.0 for the other crystals tested), while the corrected figure is "
            "4-10 times below that derivation (see docs/M3_REPORT.md section 'Absolute pair "
            "rate')"
        ),
    )

    @field_validator("dnu")
    @classmethod
    def _dnu_positive(cls, v):
        if v is not None and not v.si > 0.0:
            raise ValueError("dnu must be a positive frequency spacing")
        return v


class SPDCCrystalSettings(Settings):
    temperature: Temperature = Scalar(value=25.0, unit="degC")


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


def _sig(x: float) -> float:
    return float(f"{x:.9g}")


def _same_mode(a: GaussianMode, b: GaussianMode) -> bool:
    den = math.sqrt(abs(overlap(a, a) * overlap(b, b)))
    return den > 0.0 and abs(overlap(a, b)) >= 0.99 * den


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

    Pair rates: the spectral shape and the geometry (focusing, walk-off, phase matching) come
    from spdcalc's joint spectral intensity, which is documented in pairs per second per
    (rad/s)^2 at 1 mW. The rate is its integral over the signal/idler frequencies: pairs per
    second (time averaged) into the two Gaussian collection modes, per mW of pump polarized
    along the pump axis. It scales as d_eff^2 and linearly with power. Only the spectral
    window covered by the JSA grid is counted (spectral_cutoff of the phase-matching peak),
    so the rate is the pair rate within that band (the default cutoff of 1% costs about 2.3%);
    secondary structure of spdcalc's JSA far outside the main phase-matching band is not
    included.

    The emitted ``rate`` is pairs per second, each pair counted once. spdcalc's JSI is the
    density of ordered (signal, idler) pairs. For distinguishable photons (type II, or signal
    and idler in different collection modes) every ordered pair is a different pair and the
    full plane is integrated. When both photons have the same polarization and the same
    collection mode (type 0, type I, degenerate), (w_s, w_i) and (w_i, w_s) are the same pair:
    a lattice point whose mirror point w_i <-> w_s is also on the lattice carries weight 1/2,
    so a degenerate source is half of the full-plane integral, which is the photon flux (both
    members of each pair) of the first-principles calculation divided by two. A non-degenerate
    type-0 source whose mirror ridge lies outside the signal band is not halved. Singles are
    then 2 x rate x efficiency in total, as for any pair source.

    The absolute scale was checked against an independent first-principles calculation (first
    order perturbation theory, H = 4 eps0 d E_p^+ E_s^- E_i^-, Gaussian pump and collection
    modes, longitudinal phase-matching integral; see docs/M3_REPORT.md). spdcalc's own
    ``efficiencies()['coincidences_hz']`` multiplies this integral by a "counts correction"
    of 0.1-0.2 that the independent calculation does not support (it would make the rate 4-10
    times lower than first principles); it is therefore not applied unless
    ``spdcalc_counts_correction`` is set. The remaining uncertainty of the absolute rate is
    about +-40% (PPKTP) and the unvalidated vs. measured crystal: set ``brightness``
    (pairs/s/mW emitted into the collection modes, measured) to be independent of it.

    ``d_eff`` is the bulk coefficient: spdcalc multiplies by the 2/pi quasi-phase-matching
    factor itself. It is required unless a measured ``brightness`` is given.

    Emission cache: ``emit_cache_key`` (pump wavelengths and normalised spectrum, repetition
    rate, pump mode, polarization fraction; the parameters and the temperature setting are
    keyed by the cache itself) and ``emit_rate_scale`` (pump power along the pump axis) let a
    ``RunCache`` reuse the joint spectrum when only the pump power or anything downstream of
    the crystal changes.

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

    def _check_temperature(self, ctx) -> None:
        t_c = self.settings.temperature.si - _KELVIN
        if t_c > _MAX_TEMPERATURE_C:
            raise SetupError(
                f"crystal temperature {t_c:.0f} degC is above any physical operating range "
                f"(limit {_MAX_TEMPERATURE_C:.0f} degC); the temperature-dependent Sellmeier "
                "model is meaningless there"
            )
        low, high = _CALIBRATED_TEMPERATURE_C
        if not low <= t_c <= high:
            ctx.warn(
                "temperature_extrapolation",
                f"crystal temperature {t_c:.0f} degC is outside the range ({low:.0f} to "
                f"{high:.0f} degC) in which the temperature dependence of the refractive "
                "indices is calibrated; the phase-matching wavelength is an extrapolation",
            )

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

    def _pump_fraction(self, light) -> float:
        p = self.params
        axis = 0 if p.extraordinary_axis == "H" else 1
        letter = p.pm_type.split("_")[1]
        idx = axis if letter == "e" else 1 - axis
        w = light.coherency.sum(axis=0)
        total = float(np.real(w[0, 0] + w[1, 1]))
        return float(np.real(w[idx, idx])) / total if total > 0.0 else 0.0

    def emit_cache_key(self, lights):
        """Everything but the parameters, the temperature setting and the pump power that the
        emission depends on: pump wavelengths, normalised pump spectrum, repetition rate, the
        pump mode at the crystal (q parameters and M^2), and the fraction of the power along
        the pump axis (it selects the ``pump_polarization`` diagnostic)."""
        light = lights.get("in")
        if light is None or light.power <= 0.0:
            return None
        spectrum = np.asarray(light.power_spectrum, dtype=float)
        mode = light.mode
        rep = light.repetition_rate
        return (
            tuple(_sig(x) for x in np.asarray(light.wavelengths, dtype=float)),
            tuple(_sig(x) for x in spectrum / spectrum.sum()),
            None if rep is None else _sig(rep),
            _sig(mode.wavelength),
            tuple(_sig(v) for v in (mode.q_x.real, mode.q_x.imag, mode.q_y.real, mode.q_y.imag)),
            _sig(mode.m2_x),
            _sig(mode.m2_y),
            _sig(self._pump_fraction(light)),
        )

    def emit_rate_scale(self, lights):
        """Pump power polarized along the pump axis (W): the emitted rate is linear in it."""
        light = lights.get("in")
        if light is None:
            return 1.0
        return light.power * self._pump_fraction(light)

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
        self._check_temperature(ctx)

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
        peak_nm = _TWO_PI * c / peak * 1e9
        fwhm_nm = ridge_fwhm / _TWO_PI * (peak_nm * 1e-9) ** 2 / c * 1e9
        detuning_nm = peak_nm - common["signal_nm"]
        if abs(detuning_nm) > max(5.0 * fwhm_nm, 0.5):
            idler_nm = _TWO_PI * c / (w_p0 - peak) * 1e9
            ctx.warn(
                "phase_matching_detuned",
                f"the crystal is phase matched at a signal wavelength of {peak_nm:.1f} nm "
                f"(idler {idler_nm:.1f} nm), {detuning_nm:+.1f} nm "
                f"({abs(detuning_nm) / fwhm_nm:.0f} phase-matching widths) away from the "
                "requested signal_wavelength "
                f"{common['signal_nm']:.1f} nm; check the temperature, poling_period and "
                "theta, and the filters and detectors for the wavelengths actually emitted",
            )
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
        pulsed_grid = pulsed and 3.0 * pump_fwhm / dnu >= 4.0
        n_sum_est = 2.0 * math.ceil(1.5 * pump_fwhm / dnu) + 1.0 if pulsed_grid else 1.0
        _guard_grid(
            (hi - lo) / (_TWO_PI * dnu) + 2.0, n_sum_est, p.max_samples, dnu, p.dnu is not None, ctx
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

        ordered_weight = 1.0
        if idx_s == idx_i and _same_mode(signal_mode, idler_mode):
            partner = n_sum[None, :] - n_signal[:, None]
            mirrored = (partner >= n_signal[0]) & (partner <= n_signal[-1])
            ordered_weight = np.where(mirrored, 0.5, 1.0)
        pair_total = float(np.sum(intensity * ordered_weight))

        if p.brightness is not None:
            rate = p.brightness * power_mw
        else:
            kappa = _kappa(config) if p.spdcalc_counts_correction else 1.0
            rate = power_mw * kappa * weight * pair_total
        if not math.isfinite(rate) or rate < 0.0:
            raise SetupError(
                "spdcalc returned a non-finite pair rate: the pump, signal or idler wavelength "
                "is probably outside the range where the crystal model is valid"
            )

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
    pair_rate: Rate | None = None
    brightness: Brightness | None = Field(
        default=None,
        description=(
            "pairs per second per mW of pump emitted into the collection modes, before any "
            "filter, coupling or detector loss (not a detected coincidence rate: entering one "
            "double-counts the losses). Overrides the computed rate when given"
        ),
    )
    center_wavelength: Length
    idler_wavelength: Length | None = None
    bandwidth: Length
    pump_bandwidth: Length | None = None
    repetition_rate: Frequency | None = None
    state: BellState = "phi+"
    visibility: Dimensionless = Unity
    phase: Angle = Scalar(value=0.0, unit="rad")
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
    dnu: Frequency | None = None

    @field_validator("dnu")
    @classmethod
    def _dnu_positive(cls, v):
        if v is not None and not v.si > 0.0:
            raise ValueError("dnu must be a positive frequency spacing")
        return v


class EntangledPairSourceSettings(Settings):
    pump_power: Power = Scalar(value=1e-3, unit="W")


@register("nonlinear", "EntangledPairSource")
class EntangledPairSource(Component):
    """Black-box polarization-entangled pair source with its own pump.

    ``bandwidth`` is the wavelength FWHM of each photon's spectrum (Gaussian), centred on
    ``center_wavelength`` (signal) and ``idler_wavelength`` (default: degenerate).
    ``pump_bandwidth`` (wavelength FWHM at the pump wavelength, None = CW) sets the sum-frequency
    spread. The rate (pairs per second, each pair counted once) is ``pair_rate`` or
    ``brightness`` (pairs/s/mW) times the ``pump_power`` setting; ``pump_power`` is the rate
    scale of the emission cache and not part of its key. A visibility below 1 gives the
    Werner state v|B><B| + (1-v) I/4, represented as an incoherent ensemble of the four
    Bell states (weights v + (1-v)/4 and (1-v)/4) sharing
    one spectral amplitude, on one contiguous lattice. The rate engine sums members at the
    level of density matrices, so singles, coincidences, windows, HOM, CW and pulsed
    unheralded g2 and the Schmidt number are those of the mixed state; the one-photon
    reduced state is I/2 x rho_f for every visibility, hence K and g2 do not depend on it.

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

    emit_scale_settings = ("pump_power",)

    def emit_cache_key(self, lights):
        return ()

    def emit_rate_scale(self, lights):
        p = self.params
        if p.pair_rate is not None or p.brightness is None:
            return 1.0
        return self.settings.pump_power.si

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
            if pump_bandwidth and 3.0 * pump_bandwidth / dnu >= 4.0:
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
        sigma_p = 0.0 if not pump_bandwidth else pump_bandwidth / _FWHM
        n_sum_est = (
            2.0 * math.ceil(4.0 * sigma_p / dnu) + 1.0
            if sigma_p > 0.0 and 3.0 * pump_bandwidth / dnu >= 4.0
            else 1.0
        )
        _guard_grid(
            2.0 * math.ceil(4.5 * sigma_x / dnu) + 1.0,
            n_sum_est,
            p.max_samples,
            dnu,
            p.dnu is not None,
            ctx,
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
