from dataclasses import dataclass

import numpy as np
from scipy import constants


@dataclass(frozen=True)
class Spectrum:
    wavelengths: np.ndarray
    weights: np.ndarray
    width_hz: np.ndarray
    shape: str

    @classmethod
    def line(cls, wavelength: float, fwhm_hz: float = 0.0) -> "Spectrum":
        return cls(
            np.array([wavelength], dtype=float),
            np.array([1.0]),
            np.array([float(fwhm_hz)]),
            "lorentz",
        )

    @classmethod
    def gaussian(cls, wavelength: float, fwhm_hz: float, n: int = 11) -> "Spectrum":
        nu0 = constants.c / wavelength
        offsets = np.linspace(-1.5, 1.5, n) * fwhm_hz
        weights = np.exp(-4.0 * np.log(2.0) * (offsets / fwhm_hz) ** 2)
        spacing = np.full(n, 3.0 * fwhm_hz / (n - 1))
        return cls(constants.c / (nu0 + offsets), weights / weights.sum(), spacing, "rect")

    @property
    def center(self) -> float:
        return float(np.sum(self.weights * self.wavelengths))

    def coherence(self, delay_s: float) -> np.ndarray:
        if self.shape == "lorentz":
            return np.exp(-np.pi * self.width_hz * abs(delay_s))
        return np.sinc(self.width_hz * delay_s)
