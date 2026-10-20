import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy import constants

from sequoia.components._util import rotation
from sequoia.components.base import (
    Component,
    Emission,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import (
    Angle,
    Current,
    Curve,
    Dimensionless,
    Duration,
    Frequency,
    Length,
    Power,
    Scalar,
)
from sequoia.errors import SetupError
from sequoia.light.gaussian import GaussianMode
from sequoia.light.spectrum import Spectrum
from sequoia.light.state import Beam, SourceLight
from sequoia.units import unit_dimension

_TIME_BANDWIDTH = {"gaussian": 0.441, "sech2": 0.315}


class LaserParams(Params):
    wavelength: Length
    power: Power
    power_vs_current: Curve | None = None
    linewidth: Length | Frequency | None = None
    polarization_extinction_ratio: Dimensionless = Scalar(value=1000.0)
    waist_radius: Length | None = None
    divergence: Angle | None = None
    m_squared: Dimensionless = Scalar(value=1.0)
    waist_offset: Length = Scalar(value=0.0, unit="m")
    repetition_rate: Frequency | None = None
    pulse_width: Duration | None = None
    pulse_shape: Literal["gaussian", "sech2"] = "gaussian"

    @model_validator(mode="after")
    def _physical(self):
        checks = (
            (self.wavelength.si > 0, "wavelength must be positive"),
            (self.power.si >= 0, "power must not be negative"),
            (
                self.waist_radius is None or self.waist_radius.si > 0,
                "waist_radius must be positive",
            ),
            (self.m_squared.si >= 1.0 - 1e-9, "m_squared must be at least 1"),
            (self.divergence is None or self.divergence.si > 0, "divergence must be positive"),
            (
                self.repetition_rate is None or self.repetition_rate.si > 0,
                "repetition_rate must be positive",
            ),
            (self.pulse_width is None or self.pulse_width.si > 0, "pulse_width must be positive"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        return self


def _launch_mode(p, wavelength: float, what: str) -> GaussianMode:
    m2 = p.m_squared.si
    if p.waist_radius is not None:
        w0 = p.waist_radius.si
    elif p.divergence is not None:
        w0 = m2 * wavelength / (math.pi * p.divergence.si)
    else:
        raise SetupError(f"{what} needs 'waist_radius' or 'divergence'")
    return GaussianMode.from_waist(wavelength, w0, m2_x=m2, z=p.waist_offset.si)


class LaserSettings(Settings):
    power: Power | None = None
    current: Current | None = None
    polarization_angle: Angle = Scalar(value=0.0, unit="deg")


@register("sources", "Laser")
class Laser(Component):
    Params = LaserParams
    Settings = LaserSettings

    def ports(self):
        return (PortSpec("out", PortKind.FREE_SPACE, PortDirection.OUT),)

    def _power(self) -> float:
        if self.settings.power is not None:
            return self.settings.power.si
        if self.params.power_vs_current is not None and self.settings.current is not None:
            return float(self.params.power_vs_current(self.settings.current.si))
        return self.params.power.si

    def _spectrum(self, wavelength: float) -> Spectrum:
        p = self.params
        fwhm_hz = 0.0
        if p.linewidth is not None:
            if unit_dimension(p.linewidth.unit) == "length":
                fwhm_hz = constants.c * p.linewidth.si / wavelength**2
            else:
                fwhm_hz = p.linewidth.si
        if p.pulse_width is not None:
            bandwidth = _TIME_BANDWIDTH[p.pulse_shape] / p.pulse_width.si
            return Spectrum.gaussian(wavelength, max(bandwidth, fwhm_hz), n=11)
        return Spectrum.line(wavelength, fwhm_hz)

    def _mode(self, wavelength: float) -> GaussianMode:
        return _launch_mode(self.params, wavelength, "a laser")

    def emit(self, ctx):
        p = self.params
        wavelength = p.wavelength.si
        spectrum = self._spectrum(wavelength)
        ratio = p.polarization_extinction_ratio.si
        r = rotation(self.settings.polarization_angle.si)
        coherency = r @ np.diag([ratio, 1.0]).astype(complex) @ r.T / (1.0 + ratio)
        source = SourceLight(
            spectrum=spectrum,
            power=self._power(),
            coherency=coherency,
            repetition_rate=None if p.repetition_rate is None else p.repetition_rate.si,
            pulse_width=None if p.pulse_width is None else p.pulse_width.si,
        )
        ctx.sources[ctx.instance_id] = source
        beam = Beam.launch(ctx.instance_id, spectrum, self._mode(wavelength))
        return Emission(source, [("out", beam)])


class FockSourceParams(Params):
    wavelength: Length
    photons_per_pulse: int = Field(default=1, ge=0)
    repetition_rate: Frequency
    pulse_width: Duration = Scalar(value=1.0, unit="ps")
    waist_radius: Length | None = None
    divergence: Angle | None = None
    m_squared: Dimensionless = Scalar(value=1.0)
    waist_offset: Length = Scalar(value=0.0, unit="m")
    distinguishability_tag: int = Field(
        default=0, ge=0, description="photons from sources with equal tags are indistinguishable"
    )

    @model_validator(mode="after")
    def _physical(self):
        checks = (
            (self.wavelength.si > 0, "wavelength must be positive"),
            (self.repetition_rate.si > 0, "repetition_rate must be positive"),
            (self.pulse_width.si > 0, "pulse_width must be positive"),
            (
                self.waist_radius is None or self.waist_radius.si > 0,
                "waist_radius must be positive",
            ),
            (self.m_squared.si >= 1.0 - 1e-9, "m_squared must be at least 1"),
            (self.divergence is None or self.divergence.si > 0, "divergence must be positive"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        return self


class FockSourceSettings(Settings):
    polarization_angle: Angle = Scalar(value=0.0, unit="deg")


@register("sources", "FockSource")
class FockSource(Component):
    """Exactly N photons per pulse; classically an attenuated pulsed beam of mean power N*h*nu*f."""

    Params = FockSourceParams
    Settings = FockSourceSettings

    def ports(self):
        return (PortSpec("out", PortKind.FREE_SPACE, PortDirection.OUT),)

    def mean_power(self) -> float:
        p = self.params
        photon_energy = constants.h * constants.c / p.wavelength.si
        return p.photons_per_pulse * photon_energy * p.repetition_rate.si

    def fock_occupation(self):
        p = self.params
        theta = self.settings.polarization_angle.si
        jones = np.array([math.cos(theta), math.sin(theta)], dtype=complex)
        return (
            p.photons_per_pulse,
            p.distinguishability_tag,
            jones,
            p.wavelength.si,
            p.repetition_rate.si,
        )

    def emit(self, ctx):
        p = self.params
        wavelength = p.wavelength.si
        bandwidth = _TIME_BANDWIDTH["gaussian"] / p.pulse_width.si
        spectrum = Spectrum.gaussian(wavelength, bandwidth, n=11)
        jones = self.fock_occupation()[2]
        source = SourceLight(
            spectrum=spectrum,
            power=self.mean_power(),
            coherency=np.outer(jones, jones.conj()),
            repetition_rate=p.repetition_rate.si,
            pulse_width=p.pulse_width.si,
            statistics="fock",
        )
        ctx.sources[ctx.instance_id] = source
        mode = _launch_mode(p, wavelength, "a Fock source")
        beam = Beam.launch(ctx.instance_id, spectrum, mode)
        return Emission(source, [("out", beam)])
