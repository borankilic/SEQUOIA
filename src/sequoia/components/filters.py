import numpy as np

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
from sequoia.data import Curve, Scalar

FREE, IN, OUT = PortKind.FREE_SPACE, PortDirection.IN, PortDirection.OUT


class SpectralFilterParams(Params):
    transmission: Curve
    effective_index: float = 2.0


class FilterSettings(Settings):
    angle_of_incidence: Scalar = Scalar(value=0.0, unit="deg")


@register("filters", "SpectralFilter")
class SpectralFilter(Component):
    Params = SpectralFilterParams
    Settings = FilterSettings

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        p = self.params
        sin_t = np.sin(self.settings.angle_of_incidence.si)
        shift = np.sqrt(1.0 - (sin_t / p.effective_index) ** 2)
        t = np.clip(p.transmission(beam.wavelengths / shift), 0.0, 1.0)
        amplitude = sqrt_power(t)
        return [("out", beam.through(ctx.subject, diag_jones(amplitude, amplitude)))]


class NDFilterParams(Params):
    optical_density: Scalar | Curve


@register("filters", "NDFilter")
class NDFilter(Component):
    Params = NDFilterParams

    def ports(self):
        return (PortSpec("in", FREE, IN), PortSpec("out", FREE, OUT))

    def propagate(self, port, beam, ctx):
        od = spectral(self.params.optical_density, beam.wavelengths)
        amplitude = 10.0 ** (-od / 2.0)
        return [("out", beam.through(ctx.subject, diag_jones(amplitude, amplitude)))]
