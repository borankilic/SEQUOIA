import numpy as np
from pydantic import Field, model_validator

from sequoia.components._util import (
    REFLECT,
    diag_jones,
    retarder,
    rotated,
    spectral,
    sqrt_power,
)
from sequoia.components.base import (
    Component,
    OpticParams,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    Settings,
    register,
)
from sequoia.data import Angle, Curve, Dimensionless, Length, Scalar

FREE, FIBER, IN, OUT = PortKind.FREE_SPACE, PortKind.FIBER, PortDirection.IN, PortDirection.OUT
Unity = Scalar(value=1.0)


class AngleSettings(Settings):
    angle: Angle = Scalar(value=0.0, unit="deg")


class WavePlateParams(OpticParams):
    retardance: Dimensionless
    design_wavelength: Length
    retardance_curve: Curve | None = None
    order: int = Field(
        default=0,
        ge=0,
        description="whole waves of retardance added to a multi-order plate (0 = zero order); "
        "higher orders are more sensitive to wavelength",
    )
    retardance_error: float = Field(
        default=0.0,
        description="as-built retardance error in waves at the design wavelength "
        "(for example 0.0033 for a lambda/300 tolerance)",
    )
    transmission: Dimensionless | Curve = Unity


@register("polarization", "WavePlate")
class WavePlate(Component):
    Params = WavePlateParams
    Settings = AngleSettings

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        p = self.params
        w = beam.wavelengths
        if p.retardance_curve is not None:
            waves = np.asarray(p.retardance_curve(w), dtype=float)
        else:
            waves = (p.retardance.si + p.order + p.retardance_error) * p.design_wavelength.si / w
        amplitude = sqrt_power(spectral(p.transmission, w))
        jones = amplitude[:, None, None] * retarder(waves, self.settings.angle.si)
        return [("out", beam.through(ctx.subject, jones))]


class PolarizerParams(OpticParams):
    extinction_ratio: Dimensionless = Scalar(value=1000.0)
    transmission: Dimensionless | Curve = Unity


@register("polarization", "Polarizer")
class Polarizer(Component):
    Params = PolarizerParams
    Settings = AngleSettings

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        p = self.params
        t = spectral(p.transmission, beam.wavelengths)
        leak = t / p.extinction_ratio.si
        jones = rotated(diag_jones(sqrt_power(t), sqrt_power(leak)), self.settings.angle.si)
        return [("out", beam.through(ctx.subject, jones))]


class PBSParams(OpticParams):
    transmittance_p: Dimensionless | Curve = Unity
    transmittance_s: Dimensionless | Curve = Scalar(value=0.0)
    reflectance_s: Dimensionless | Curve = Unity
    reflectance_p: Dimensionless | Curve = Scalar(value=0.0)

    @model_validator(mode="after")
    def _energy(self):
        for t, r, what in (
            (self.transmittance_p, self.reflectance_p, "p"),
            (self.transmittance_s, self.reflectance_s, "s"),
        ):
            if isinstance(t, Scalar) and isinstance(r, Scalar) and t.si + r.si > 1.0 + 1e-9:
                raise ValueError(
                    f"transmittance_{what} + reflectance_{what} = {t.si + r.si:.4g} exceeds 1: "
                    "a passive splitter cannot create light"
                )
        return self


@register("polarization", "PBS")
class PBS(Component):
    Params = PBSParams
    Settings = AngleSettings

    def ports(self):
        return (
            PortSpec("in", FREE, IN),
            PortSpec("out_t", FREE, OUT),
            PortSpec("out_r", FREE, OUT),
        )

    def propagate(self, port, beam, ctx):
        p, w, theta = self.params, beam.wavelengths, self.settings.angle.si
        t_h, t_v = spectral(p.transmittance_p, w), spectral(p.transmittance_s, w)
        r_h, r_v = spectral(p.reflectance_p, w), spectral(p.reflectance_s, w)
        transmit = rotated(diag_jones(sqrt_power(t_h), sqrt_power(t_v)), theta)
        reflect = rotated(
            1j * diag_jones(REFLECT[0] * sqrt_power(r_h), REFLECT[1] * sqrt_power(r_v)), theta
        )
        return [
            ("out_t", beam.through(ctx.subject, transmit)),
            ("out_r", beam.through(ctx.subject, reflect)),
        ]


class PaddleParams(Params):
    loops: list[int] = Field(default_factory=lambda: [1, 2, 1])
    loop_diameter: Length = Scalar(value=56.0, unit="mm")
    cladding_diameter: Length = Scalar(value=125.0, unit="um")
    bend_coefficient: float = 0.133


class PaddleSettings(Settings):
    angle_1: Angle = Scalar(value=0.0, unit="deg")
    angle_2: Angle = Scalar(value=0.0, unit="deg")
    angle_3: Angle = Scalar(value=0.0, unit="deg")


@register("polarization", "FiberPaddleController")
class FiberPaddleController(Component):
    Params = PaddleParams
    Settings = PaddleSettings

    def ports(self):
        return (PortSpec("in", FIBER, IN), PortSpec("out", FIBER, OUT))

    def retardance_waves(self, wavelengths, loops: int):
        p = self.params
        d, big_d = p.cladding_diameter.si, p.loop_diameter.si
        radians = 2.0 * np.pi**2 * p.bend_coefficient * d**2 * loops / (wavelengths * big_d)
        return radians / (2.0 * np.pi)

    def propagate(self, port, beam, ctx):
        angles = (self.settings.angle_1, self.settings.angle_2, self.settings.angle_3)
        jones = np.broadcast_to(np.eye(2, dtype=complex), (len(beam.wavelengths), 2, 2))
        for loops, angle in zip(self.params.loops, angles, strict=False):
            jones = retarder(self.retardance_waves(beam.wavelengths, loops), angle.si) @ jones
        return [("out", beam.through(ctx.subject, jones))]
