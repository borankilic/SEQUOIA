from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DetectorModel:
    """Single-photon detector as seen by the rate engine.

    ``efficiency(wavelengths)`` returns an (N, 2, 2) Hermitian positive-semidefinite array in
    the (H, V) basis: the probability that a single photon with polarization state |chi> at
    that wavelength is detected is <chi| E |chi>. A polarization-insensitive detector returns
    eta * identity. Times are in seconds, rates in Hz.
    """

    kind: str
    efficiency: Callable[[np.ndarray], np.ndarray]
    dark_count_rate: float = 0.0
    dead_time: float = 0.0
    jitter_fwhm: float = 0.0
    jitter_tail_fraction: float = 0.0
    jitter_tail_tau: float = 0.0
    afterpulse_probability: float = 0.0
    afterpulse_tau: float = 0.0


@dataclass(frozen=True)
class TaggerModel:
    """Time tagger: per-input-port delays plus shared jitter and dead time (seconds)."""

    jitter_fwhm: float = 0.0
    dead_time: float = 0.0
    input_delays: dict[str, float] = field(default_factory=dict)
