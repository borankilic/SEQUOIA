import numpy as np

from sequoia.components._util import spectral
from sequoia.components.base import (
    Component,
    Params,
    PortDirection,
    PortKind,
    PortSpec,
    register,
)
from sequoia.data import Curve, Length, Scalar
from sequoia.light.gaussian import aperture_transmission


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
