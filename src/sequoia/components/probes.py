import numpy as np

from sequoia.components.base import Component, Params, Settings, register
from sequoia.data import Length


class ProbeSettings(Settings):
    position: Length | None = None


class _Probe(Component):
    Params = Params
    Settings = ProbeSettings


@register("probes", "PowerMeter")
class PowerMeter(_Probe):
    def measure(self, light):
        result = {"power_W": light.power, "photon_rate_per_s": light.photon_rate}
        if light.mean_photons_per_pulse is not None:
            result["mean_photons_per_pulse"] = light.mean_photons_per_pulse
        return result


@register("probes", "BeamProfiler")
class BeamProfiler(_Probe):
    def measure(self, light):
        m = light.mode
        return {
            "w_x_m": m.w_x,
            "w_y_m": m.w_y,
            "radius_x_m": m.radius_x,
            "radius_y_m": m.radius_y,
            "waist_x_m": m.waist_x,
            "waist_y_m": m.waist_y,
            "distance_past_waist_x_m": m.distance_past_waist_x,
            "distance_past_waist_y_m": m.distance_past_waist_y,
            "divergence_x_rad": m.divergence_x,
            "divergence_y_rad": m.divergence_y,
            "power_W": light.power,
        }


@register("probes", "Polarimeter")
class Polarimeter(_Probe):
    def measure(self, light):
        s = light.stokes
        norm = s[0] if s[0] > 0 else 1.0
        return {
            "S0_W": float(s[0]),
            "S1": float(s[1] / norm),
            "S2": float(s[2] / norm),
            "S3": float(s[3] / norm),
            "dop": light.dop,
            "azimuth_rad": light.azimuth,
            "ellipticity_rad": light.ellipticity,
        }


@register("probes", "Spectrometer")
class Spectrometer(_Probe):
    def measure(self, light):
        total = light.power
        center = light.mean_wavelength
        rms = 0.0
        if total > 0:
            rms = float(
                np.sqrt(np.sum(light.power_spectrum * (light.wavelengths - center) ** 2) / total)
            )
        return {
            "wavelengths_m": light.wavelengths.tolist(),
            "power_W": light.power_spectrum.tolist(),
            "centroid_wavelength_m": center,
            "rms_width_m": rms,
        }
