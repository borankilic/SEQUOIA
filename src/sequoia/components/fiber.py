from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

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
from sequoia.components.freespace import _Optic
from sequoia.data import Curve, Dimensionless, Length, Scalar
from sequoia.light.gaussian import GaussianMode, coupling_efficiency, thin_lens

FREE, FIBER, IN, OUT = PortKind.FREE_SPACE, PortKind.FIBER, PortDirection.IN, PortDirection.OUT
Unity = Scalar(value=1.0)
_PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def su2_from_rotation_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(2, dtype=complex)
    axis = vector / angle
    generator = sum(a * s for a, s in zip(axis, _PAULI, strict=True))
    return np.cos(angle / 2.0) * np.eye(2) - 1j * np.sin(angle / 2.0) * generator


def fiber_polarization(spec) -> np.ndarray:
    if spec.kind == "su2":
        return su2_from_rotation_vector(spec.rotation_vector)
    if spec.kind == "random":
        vector = Rotation.random(random_state=spec.seed).as_rotvec()
        return su2_from_rotation_vector(vector)
    return np.eye(2, dtype=complex)


class OpticalFiberParams(Params):
    kind: Literal["smf", "pm", "mmf"] = "smf"
    mode_field_diameter: Length | Curve | None = None
    cutoff_wavelength: Length | None = None
    numerical_aperture: float = 0.13
    core_diameter: Length | None = None
    attenuation_db_per_km: float | Curve = 0.0
    effective_index: float = 1.46
    group_index: float = 1.468


@register("fiber", "OpticalFiber")
class OpticalFiber(Component):
    Params = OpticalFiberParams

    def mode_field_radius(self, wavelength: float) -> float:
        p = self.params
        if p.kind == "mmf":
            if p.core_diameter is None:
                raise ValueError("a multimode fiber needs 'core_diameter'")
            return p.core_diameter.si / 2.0
        if p.mode_field_diameter is None:
            raise ValueError("a single-mode fiber needs 'mode_field_diameter'")
        mfd = p.mode_field_diameter
        return (float(mfd(wavelength)) if isinstance(mfd, Curve) else mfd.si) / 2.0

    def attenuation_db_per_m(self, wavelengths) -> np.ndarray:
        a = self.params.attenuation_db_per_km
        return spectral(a if isinstance(a, Curve) else float(a), wavelengths, 0.0) / 1000.0

    def transmit(self, beam, link, ctx, link_id: str):
        p = self.params
        wavelength = beam.mode.wavelength
        radius = self.mode_field_radius(wavelength)
        info: dict[str, float] = {}
        if p.kind == "mmf":
            a = radius
            area = 1.0 - np.exp(-2.0 * a**2 / (beam.mode.w_x * beam.mode.w_y))
            theta = max(beam.mode.divergence_x, beam.mode.divergence_y)
            eta = float(area * min(1.0, (p.numerical_aperture / theta) ** 2))
            out_mode = GaussianMode.from_waist(
                wavelength,
                a,
                m2_x=max(1.0, np.pi * a * p.numerical_aperture / wavelength),
            )
        else:
            out_mode = GaussianMode.from_waist(wavelength, radius)
            eta = coupling_efficiency(beam.mode, out_mode, beam.offset)
            if (
                p.cutoff_wavelength is not None
                and wavelength < p.cutoff_wavelength.si
                and ctx.significant(beam)
            ):
                ctx.warn(
                    "fiber_multimode",
                    f"link '{link_id}': wavelength {wavelength * 1e9:.1f} nm is below the "
                    f"cutoff of {p.cutoff_wavelength.si * 1e9:.1f} nm; the fiber is not "
                    "single-mode",
                )
        loss_db = (
            self.attenuation_db_per_m(beam.wavelengths) * link.length.si + link.connector_loss_db
        )
        transmission = 10.0 ** (-loss_db / 10.0)
        amplitude = np.sqrt(eta * transmission)
        u = fiber_polarization(link.polarization)
        jones = amplitude[:, None, None] * u[None, :, :]
        out = beam.through(
            link_id,
            jones,
            mode=out_mode,
            opl=p.effective_index * link.length.si,
            gpl=p.group_index * link.length.si,
            offset=0.0,
        )
        info.update(coupling_efficiency=float(eta), transmission=float(np.mean(transmission)))
        return out, info


class CouplerParams(Params):
    focal_length: Length
    clear_aperture: Length | None = None
    transmission: Dimensionless | Curve = Unity


class CouplerSettings(Settings):
    mode: Literal["to_fiber", "to_free"] = "to_fiber"
    defocus: Length = Scalar(value=0.0, unit="m")
    lateral_offset: Length = Scalar(value=0.0, unit="m")


@register("fiber", "FiberCoupler")
class FiberCoupler(_Optic):
    Params = CouplerParams
    Settings = CouplerSettings

    def ports(self):
        if self.settings.mode == "to_fiber":
            return (PortSpec("in", FREE, IN), PortSpec("out", FIBER, OUT))
        return (PortSpec("in", FIBER, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        p, s = self.params, self.settings
        f = p.focal_length.si
        aperture = None if p.clear_aperture is None else p.clear_aperture.si / 2.0
        t = spectral(p.transmission, beam.wavelengths)
        offset = s.lateral_offset.si
        if s.mode == "to_fiber":
            clip = self._clip(beam, ctx, aperture, ctx.subject)
            mode = beam.mode.apply(thin_lens(f)).propagate(f + s.defocus.si)
            offset = beam.offset + offset
        else:
            mode = beam.mode.propagate(f + s.defocus.si).apply(thin_lens(f))
            clip = self._clip(beam.through(ctx.subject, mode=mode), ctx, aperture, ctx.subject)
            offset = beam.offset
        amplitude = sqrt_power(t * clip)
        jones = diag_jones(amplitude, amplitude)
        out = beam.through(ctx.subject, jones, mode=mode, opl=f + s.defocus.si, offset=offset)
        return [("out", out)]


class FiberBeamSplitterParams(Params):
    split_ratio: Dimensionless | Curve = Scalar(value=0.5)
    transmission: Dimensionless | Curve = Unity


@register("fiber", "FiberBeamSplitter")
class FiberBeamSplitter(Component):
    Params = FiberBeamSplitterParams

    def ports(self):
        return (
            PortSpec("in_a", FIBER, IN),
            PortSpec("in_b", FIBER, IN),
            PortSpec("out_a", FIBER, OUT),
            PortSpec("out_b", FIBER, OUT),
        )

    def propagate(self, port, beam, ctx):
        w = beam.wavelengths
        r = np.clip(spectral(self.params.split_ratio, w), 0.0, 1.0)
        t = spectral(self.params.transmission, w)
        straight = diag_jones(np.sqrt((1.0 - r) * t), np.sqrt((1.0 - r) * t))
        crossed = diag_jones(1j * np.sqrt(r * t), 1j * np.sqrt(r * t))
        same, other = ("out_a", "out_b") if port == "in_a" else ("out_b", "out_a")
        return [
            (same, beam.through(ctx.subject, straight)),
            (other, beam.through(ctx.subject, crossed)),
        ]
