from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import constants

from sequoia.light.gaussian import GaussianMode, overlap
from sequoia.light.spectrum import Spectrum


@dataclass(frozen=True)
class SourceLight:
    spectrum: Spectrum
    power: float
    coherency: np.ndarray
    repetition_rate: float | None = None
    pulse_width: float | None = None
    statistics: str = "classical"


@dataclass(frozen=True)
class Beam:
    source: str
    wavelengths: np.ndarray
    jones: np.ndarray
    opl: float
    gpl: float
    mode: GaussianMode
    offset: float = 0.0
    history: tuple[str, ...] = ()

    @classmethod
    def launch(cls, source: str, spectrum: Spectrum, mode: GaussianMode) -> "Beam":
        n = len(spectrum.wavelengths)
        jones = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
        return cls(source, spectrum.wavelengths, jones, 0.0, 0.0, mode, 0.0, (source,))

    def through(
        self,
        via: str,
        jones: np.ndarray | None = None,
        *,
        mode: GaussianMode | None = None,
        opl: float = 0.0,
        gpl: float | None = None,
        offset: float | None = None,
    ) -> "Beam":
        new_jones = self.jones if jones is None else jones @ self.jones
        return replace(
            self,
            jones=new_jones,
            mode=self.mode if mode is None else mode,
            opl=self.opl + opl,
            gpl=self.gpl + (opl if gpl is None else gpl),
            offset=self.offset if offset is None else offset,
            history=self.history + (via,),
        )


@dataclass
class LinkLight:
    wavelengths: np.ndarray
    power_spectrum: np.ndarray
    coherency: np.ndarray
    mode: GaussianMode
    paths: int
    repetition_rate: float | None = None
    pulse_width: float | None = None
    offset: float = 0.0
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def power(self) -> float:
        return float(np.sum(self.power_spectrum))

    @property
    def photon_rate(self) -> float:
        energy = constants.h * constants.c / self.wavelengths
        return float(np.sum(self.power_spectrum / energy))

    @property
    def mean_photons_per_pulse(self) -> float | None:
        if not self.repetition_rate:
            return None
        return self.photon_rate / self.repetition_rate

    @property
    def stokes(self) -> np.ndarray:
        w = self.coherency.sum(axis=0)
        return np.array(
            [
                (w[0, 0] + w[1, 1]).real,
                (w[0, 0] - w[1, 1]).real,
                2.0 * w[0, 1].real,
                -2.0 * w[0, 1].imag,
            ]
        )

    @property
    def dop(self) -> float:
        s = self.stokes
        return float(np.linalg.norm(s[1:]) / s[0]) if s[0] > 0 else 0.0

    @property
    def azimuth(self) -> float:
        s = self.stokes
        return float(0.5 * np.arctan2(s[2], s[1]))

    @property
    def ellipticity(self) -> float:
        s = self.stokes
        pol = np.linalg.norm(s[1:])
        return float(0.5 * np.arcsin(np.clip(s[3] / pol, -1.0, 1.0))) if pol > 0 else 0.0

    @property
    def mean_wavelength(self) -> float:
        if self.power <= 0:
            return float(self.mode.wavelength)
        return float(np.sum(self.power_spectrum * self.wavelengths) / self.power)


def combine(beams: list[Beam], sources: dict[str, SourceLight]) -> LinkLight | None:
    if not beams:
        return None
    groups: dict[str, list[Beam]] = defaultdict(list)
    for beam in beams:
        groups[beam.source].append(beam)
    wavelengths, powers, coherencies = [], [], []
    best_mode, best_power, best_offset = None, -1.0, 0.0
    rate = width = None
    for source_id, group in groups.items():
        src = sources[source_id]
        spec = src.spectrum
        n = len(spec.wavelengths)
        w_total = np.zeros((n, 2, 2), dtype=complex)
        for p in group:
            jw = p.jones @ src.coherency
            for q in group:
                cross = jw @ np.conj(np.swapaxes(q.jones, 1, 2))
                if q is p:
                    w_total += cross
                    continue
                phase = np.exp(1j * 2.0 * np.pi * (p.opl - q.opl) / spec.wavelengths)
                gamma = spec.coherence((p.gpl - q.gpl) / constants.c)
                w_total += cross * (phase * gamma * overlap(p.mode, q.mode))[:, None, None]
        w_total *= (src.power * spec.weights)[:, None, None]
        trace = np.real(w_total[:, 0, 0] + w_total[:, 1, 1])
        wavelengths.append(spec.wavelengths)
        powers.append(trace)
        coherencies.append(w_total)
        group_power = float(trace.sum())
        if group_power > best_power:
            best_power = group_power
            dominant = max(group, key=lambda b: np.real(np.trace(b.jones[0] @ b.jones[0].conj().T)))
            best_mode, best_offset = dominant.mode, dominant.offset
            rate, width = src.repetition_rate, src.pulse_width
    order = np.argsort(np.concatenate(wavelengths), kind="stable")
    return LinkLight(
        wavelengths=np.concatenate(wavelengths)[order],
        power_spectrum=np.concatenate(powers)[order],
        coherency=np.concatenate(coherencies)[order],
        mode=best_mode,
        paths=len(beams),
        repetition_rate=rate,
        pulse_width=width,
        offset=best_offset,
        sources=tuple(groups),
    )
