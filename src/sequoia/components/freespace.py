import math

import numpy as np

from sequoia.components._util import REFLECT, diag_jones, spectral, sqrt_power
from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import Curve, Length, Scalar
from sequoia.light.gaussian import aperture_transmission, thin_lens

FREE, IN, OUT = PortKind.FREE_SPACE, PortDirection.IN, PortDirection.OUT
Unity = Scalar(value=1.0)


class _Optic(Component):
    def _clip(self, beam, ctx, radius: float | None, label: str) -> float:
        if radius is None:
            return 1.0
        t = aperture_transmission(radius, beam.mode)
        if t < 0.99:
            ctx.warn(
                "aperture_clipping",
                f"{label} clips the beam: {100 * (1 - t):.1f}% of the power is lost "
                f"(aperture radius {radius * 1e3:.3g} mm, beam radius "
                f"{beam.mode.w_x * 1e3:.3g} x {beam.mode.w_y * 1e3:.3g} mm)",
            )
        return t


class LensParams(Params):
    focal_length: Length
    focal_length_curve: Curve | None = None
    transmission: Scalar | Curve = Unity
    clear_aperture: Length | None = None


@register("freespace", "Lens")
class Lens(_Optic):
    Params = LensParams

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def _focal(self, beam) -> float:
        curve = self.params.focal_length_curve
        if curve is not None:
            return float(curve(beam.mode.wavelength))
        return self.params.focal_length.si

    def _aperture_radius(self) -> float | None:
        a = self.params.clear_aperture
        return None if a is None else a.si / 2.0

    def propagate(self, port, beam, ctx):
        t = self._clip(beam, ctx, self._aperture_radius(), f"{ctx.subject}")
        f = self._focal(beam)
        power = spectral(self.params.transmission, beam.wavelengths) * t
        jones = diag_jones(sqrt_power(power), sqrt_power(power))
        mode = beam.mode.apply(thin_lens(f))
        return [("out", beam.through(ctx.subject, jones, mode=mode))]


class ObjectiveParams(Params):
    effective_focal_length: Length
    numerical_aperture: Scalar
    back_aperture: Length | None = None
    working_distance: Length | None = None
    transmission: Scalar | Curve = Unity


@register("freespace", "Objective")
class Objective(_Optic):
    Params = ObjectiveParams

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        p = self.params
        f = p.effective_focal_length.si
        if p.back_aperture is not None:
            radius = p.back_aperture.si / 2.0
        else:
            radius = f * p.numerical_aperture.si
        t = self._clip(beam, ctx, radius, ctx.subject)
        power = spectral(p.transmission, beam.wavelengths) * t
        jones = diag_jones(sqrt_power(power), sqrt_power(power))
        mode = beam.mode.apply(thin_lens(f))
        return [("out", beam.through(ctx.subject, jones, mode=mode))]


class MirrorParams(Params):
    reflectance: Scalar | Curve = Unity
    radius_of_curvature: Length | None = None
    clear_aperture: Length | None = None


class MirrorSettings(Settings):
    angle_of_incidence: Scalar = Scalar(value=0.0, unit="deg")


@register("freespace", "Mirror")
class Mirror(_Optic):
    Params = MirrorParams
    Settings = MirrorSettings

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        p = self.params
        aperture = None if p.clear_aperture is None else p.clear_aperture.si / 2.0
        t = self._clip(beam, ctx, aperture, ctx.subject)
        reflect = sqrt_power(spectral(p.reflectance, beam.wavelengths) * t)
        jones = diag_jones(REFLECT[0] * reflect, REFLECT[1] * reflect)
        mode = beam.mode
        if p.radius_of_curvature is not None:
            radius = p.radius_of_curvature.si
            cos_t = math.cos(self.settings.angle_of_incidence.si)
            mode = mode.apply(thin_lens(radius * cos_t / 2.0), thin_lens(radius / (2.0 * cos_t)))
        return [("out", beam.through(ctx.subject, jones, mode=mode))]


class BeamSplitterParams(Params):
    reflectance: Scalar | Curve = Scalar(value=0.5)
    reflectance_v: Scalar | Curve | None = None
    absorption: Scalar | Curve = Scalar(value=0.0)


@register("freespace", "BeamSplitter")
class BeamSplitter(Component):
    Params = BeamSplitterParams

    def ports(self):
        return (
            PortSpec("in_a", FREE, IN),
            PortSpec("in_b", FREE, IN),
            PortSpec("out_a", FREE, OUT),
            PortSpec("out_b", FREE, OUT),
        )

    def _split(self, beam):
        p = self.params
        w = beam.wavelengths
        r_h = np.clip(spectral(p.reflectance, w), 0.0, 1.0)
        r_v = r_h if p.reflectance_v is None else np.clip(spectral(p.reflectance_v, w), 0.0, 1.0)
        loss = spectral(p.absorption, w, 0.0)
        t_h = np.clip(1.0 - r_h - loss, 0.0, 1.0)
        t_v = np.clip(1.0 - r_v - loss, 0.0, 1.0)
        transmit = diag_jones(np.sqrt(t_h), np.sqrt(t_v))
        reflect = 1j * diag_jones(REFLECT[0] * np.sqrt(r_h), REFLECT[1] * np.sqrt(r_v))
        return transmit, reflect

    def propagate(self, port, beam, ctx):
        transmit, reflect = self._split(beam)
        straight, crossed = ("out_a", "out_b") if port == "in_a" else ("out_b", "out_a")
        return [
            (straight, beam.through(ctx.subject, transmit)),
            (crossed, beam.through(ctx.subject, reflect)),
        ]


class DichroicParams(Params):
    transmittance: Scalar | Curve
    reflectance: Scalar | Curve | None = None


@register("freespace", "DichroicMirror")
class DichroicMirror(Component):
    Params = DichroicParams

    def ports(self):
        return (
            PortSpec("in", FREE, IN),
            PortSpec("out_t", FREE, OUT),
            PortSpec("out_r", FREE, OUT),
        )

    def propagate(self, port, beam, ctx):
        w = beam.wavelengths
        t = np.clip(spectral(self.params.transmittance, w), 0.0, 1.0)
        if self.params.reflectance is None:
            r = 1.0 - t
        else:
            r = np.clip(spectral(self.params.reflectance, w), 0.0, 1.0)
        transmit = diag_jones(np.sqrt(t), np.sqrt(t))
        reflect = 1j * diag_jones(np.sqrt(r) * REFLECT[0], np.sqrt(r) * REFLECT[1])
        return [
            ("out_t", beam.through(ctx.subject, transmit)),
            ("out_r", beam.through(ctx.subject, reflect)),
        ]


class IrisSettings(Settings):
    diameter: Length = Scalar(value=math.inf, unit="m")


@register("freespace", "Iris")
class Iris(_Optic):
    Settings = IrisSettings

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        t = self._clip(beam, ctx, self.settings.diameter.si / 2.0, ctx.subject)
        amplitude = np.full(len(beam.wavelengths), math.sqrt(t))
        jones = diag_jones(amplitude, amplitude)
        return [("out", beam.through(ctx.subject, jones))]


@register("freespace", "BeamDump")
class BeamDump(Component):
    def ports(self):
        return (PortSpec("in", FREE, IN),)
