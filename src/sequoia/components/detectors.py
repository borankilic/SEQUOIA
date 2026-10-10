import warnings
from typing import Literal

import numpy as np
from pydantic import field_validator

from sequoia.components._util import rotation, spectral
from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import Curve, Duration, Length, Scalar, Surface
from sequoia.light.detection import DetectorModel
from sequoia.light.gaussian import aperture_transmission
from sequoia.units import unit_dimension


class PhotodiodeParams(Params):
    responsivity: Scalar | Curve
    active_diameter: Length | None = None
    dark_current: Scalar = Scalar(value=0.0, unit="A")


@register("detectors", "Photodiode")
class Photodiode(Component):
    Params = PhotodiodeParams

    def ports(self):
        return (PortSpec("in", PortKind.FREE_SPACE, PortDirection.IN, required=True),)

    def detect(self, light, ctx):
        arriving = light.get("in")
        if arriving is None:
            return {"photocurrent_A": self.params.dark_current.si, "incident_power_W": 0.0}
        p = self.params
        collected = 1.0
        if p.active_diameter is not None:
            collected = aperture_transmission(p.active_diameter.si / 2.0, arriving.mode)
            if collected < 0.99:
                ctx.warn(
                    "detector_overfilled",
                    f"the beam overfills the active area: {100 * (1 - collected):.1f}% is lost",
                )
        r = spectral(p.responsivity, arriving.wavelengths)
        current = float(np.sum(arriving.power_spectrum * r)) * collected + p.dark_current.si
        return {
            "photocurrent_A": current,
            "incident_power_W": arriving.power,
            "collected_fraction": collected,
        }


class DetectorConfigWarning(UserWarning):
    pass


def _check_dimension(unit: str, reference: str) -> None:
    if unit_dimension(unit) != unit_dimension(reference):
        raise ValueError(f"expected a quantity in {reference!r}-compatible units, got {unit!r}")


class SinglePhotonDetectorParams(Params):
    efficiency: Scalar | Curve
    dark_count_rate: Scalar = Scalar(value=0.0, unit="Hz")
    dead_time: Duration = Scalar(value=0.0, unit="s")
    jitter_fwhm: Duration = Scalar(value=0.0, unit="s")
    jitter_tail_fraction: Scalar = Scalar(value=0.0)
    jitter_tail_tau: Duration = Scalar(value=0.0, unit="s")
    afterpulse_probability: Scalar = Scalar(value=0.0)
    # kept for the part data; the rate engine models afterpulses as uniform background counts
    afterpulse_tau: Duration = Scalar(value=0.0, unit="s")
    active_diameter: Length | None = None

    @field_validator("efficiency")
    @classmethod
    def _efficiency_range(cls, v: Scalar | Curve) -> Scalar | Curve:
        if isinstance(v, Scalar) and not 0.0 <= v.si <= 1.0:
            raise ValueError("efficiency must be between 0 and 1")
        return v

    @field_validator("dark_count_rate")
    @classmethod
    def _rate(cls, v: Scalar) -> Scalar:
        _check_dimension(v.unit, "Hz")
        return v

    @field_validator("jitter_tail_fraction", "afterpulse_probability")
    @classmethod
    def _probability(cls, v: Scalar) -> Scalar:
        if not 0.0 <= v.si <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v


class SinglePhotonDetectorSettings(Settings):
    input: Literal["fiber", "free_space"] = "fiber"


class _SinglePhotonDetector(Component):
    Settings = SinglePhotonDetectorSettings

    def ports(self):
        kind = PortKind.FIBER if self.settings.input == "fiber" else PortKind.FREE_SPACE
        return (
            PortSpec("in", kind, PortDirection.IN, required=True),
            PortSpec("out", PortKind.ELECTRICAL, PortDirection.OUT, required=True),
        )

    def _efficiency_scalar(self, wavelengths: np.ndarray) -> np.ndarray:
        return spectral(self.params.efficiency, wavelengths)

    def _bounded_efficiency(self, wavelengths: np.ndarray) -> np.ndarray:
        eta = np.asarray(self._efficiency_scalar(wavelengths), dtype=float)
        if np.any(eta < 0.0) or np.any(eta > 1.0):
            warnings.warn(
                f"{self.type_name} efficiency leaves [0, 1] (range {eta.min():.3g} to "
                f"{eta.max():.3g}); it is clipped",
                DetectorConfigWarning,
                stacklevel=2,
            )
            eta = np.clip(eta, 0.0, 1.0)
        return eta

    def _operator(self, wavelengths: np.ndarray) -> np.ndarray:
        eta = self._bounded_efficiency(wavelengths)
        return eta[:, None, None] * np.eye(2)

    def aperture_radius(self) -> float | None:
        d = self.params.active_diameter
        if d is None or self.settings.input != "free_space":
            return None
        return d.si / 2.0

    def _check_configuration(self) -> None:
        pass

    def detector_model(self):
        self._check_configuration()
        p = self.params
        return DetectorModel(
            kind=self.type_name,
            efficiency=self._operator,
            dark_count_rate=p.dark_count_rate.si,
            dead_time=p.dead_time.si,
            jitter_fwhm=p.jitter_fwhm.si,
            jitter_tail_fraction=p.jitter_tail_fraction.si,
            jitter_tail_tau=p.jitter_tail_tau.si,
            afterpulse_probability=p.afterpulse_probability.si,
            afterpulse_tau=p.afterpulse_tau.si,
        )

    def detect(self, light, ctx):
        arriving = light.get("in")
        if arriving is None or self.settings.input != "free_space":
            return {}
        result = {"incident_power_W": arriving.power}
        d = self.params.active_diameter
        if d is not None:
            collected = aperture_transmission(d.si / 2.0, arriving.mode)
            result["collected_fraction"] = collected
            if collected < 0.99:
                ctx.warn(
                    "detector_overfilled",
                    f"the beam overfills the active area: {100 * (1 - collected):.1f}% is lost",
                )
        return result


@register("detectors", "SPAD")
class SPAD(_SinglePhotonDetector):
    Params = SinglePhotonDetectorParams


class SNSPDParams(SinglePhotonDetectorParams):
    polarization_extinction: Scalar = Scalar(value=1.0)
    polarization_axis: Scalar = Scalar(value=0.0, unit="deg")
    efficiency_vs_bias: Surface | None = None

    @field_validator("polarization_extinction")
    @classmethod
    def _extinction(cls, v: Scalar) -> Scalar:
        if v.si < 1.0:
            raise ValueError("polarization extinction is max/min efficiency and must be >= 1")
        return v


class SNSPDSettings(SinglePhotonDetectorSettings):
    bias_current: Scalar | None = None

    @field_validator("bias_current")
    @classmethod
    def _current(cls, v: Scalar | None) -> Scalar | None:
        if v is not None:
            _check_dimension(v.unit, "A")
        return v


@register("detectors", "SNSPD")
class SNSPD(_SinglePhotonDetector):
    Params = SNSPDParams
    Settings = SNSPDSettings

    def _efficiency_scalar(self, wavelengths):
        p, bias = self.params, self.settings.bias_current
        if p.efficiency_vs_bias is not None and bias is not None:
            return np.asarray(p.efficiency_vs_bias(wavelengths, bias.si), dtype=float)
        return super()._efficiency_scalar(wavelengths)

    def _check_configuration(self):
        has_surface = self.params.efficiency_vs_bias is not None
        has_bias = self.settings.bias_current is not None
        if has_surface and not has_bias:
            warnings.warn(
                "SNSPD has an efficiency-vs-bias surface but no bias_current setting; the "
                "scalar efficiency is used",
                DetectorConfigWarning,
                stacklevel=3,
            )
        elif has_bias and not has_surface:
            warnings.warn(
                "SNSPD bias_current is set but the part has no efficiency-vs-bias surface; "
                "the setting is ignored",
                DetectorConfigWarning,
                stacklevel=3,
            )

    def _operator(self, wavelengths):
        p = self.params
        eta = self._bounded_efficiency(wavelengths)
        r = rotation(p.polarization_axis.si)
        base = r @ np.diag([1.0, 1.0 / p.polarization_extinction.si]) @ r.T
        return eta[:, None, None] * base
