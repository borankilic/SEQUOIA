from dataclasses import dataclass

import numpy as np
from scipy import constants

from sequoia.light.gaussian import GaussianMode


@dataclass(frozen=True)
class PairSource:
    """Biphoton state emitted by a pair source, on an integer frequency lattice.

    Every frequency is an integer multiple of ``dnu``: nu = n * dnu.
    ``amplitude[sigma_s, sigma_i, j, m]`` is the amplitude for a signal photon of
    polarization sigma_s (0 = H, 1 = V) at frequency n_signal[j]*dnu and an idler photon
    of polarization sigma_i at frequency (n_sum[m] - n_signal[j])*dnu. It is normalised so
    that sum(|amplitude|**2) == 1. A CW pump has a single sum-frequency entry.
    ``rate`` is the time-averaged number of pairs per second emitted into the two modes.

    A mixed pair state is given as an ensemble: ``amplitude`` then has a leading member axis,
    ``amplitude[k, sigma_s, sigma_i, j, m]``, and member k carries the square root of its
    probability, so that its weight is ``w_k = sum |amplitude[k]|**2`` and the total norm is
    still 1. The emitted state is the incoherent mixture rho = sum_k w_k |psi_k><psi_k|
    (each emission event, i.e. each pulse of a pulsed source, is in member k with
    probability w_k); members are never added coherently, and they share the lattice vectors
    ``n_signal`` and ``n_sum``. Quantities linear in rho (singles, coincidence rates,
    correlation functions, HOM) are sums over members; the unheralded g2 and the Schmidt
    number are built from the reduced density matrices of the mixture (see the rate engine).
    """

    dnu: float
    n_signal: np.ndarray
    n_sum: np.ndarray
    amplitude: np.ndarray
    rate: float
    signal_mode: GaussianMode
    idler_mode: GaussianMode
    pump_wavelength: float
    repetition_rate: float | None = None

    def __post_init__(self) -> None:
        shape = (2, 2, len(self.n_signal), len(self.n_sum))
        if self.amplitude.shape != shape and (
            self.amplitude.ndim != 5 or self.amplitude.shape[1:] != shape
        ):
            raise ValueError(
                f"amplitude has shape {self.amplitude.shape}, expected {shape} or "
                f"(members, {shape[0]}, {shape[1]}, {shape[2]}, {shape[3]})"
            )
        norm = float(np.sum(np.abs(self.amplitude) ** 2))
        if not np.isclose(norm, 1.0, rtol=1e-6):
            raise ValueError(f"amplitude must be normalised to 1, got {norm}")
        if self.dnu <= 0 or self.rate < 0:
            raise ValueError("dnu must be positive and rate non-negative")

    @property
    def is_ensemble(self) -> bool:
        return self.amplitude.ndim == 5

    @property
    def members(self) -> np.ndarray:
        """Member amplitudes with a leading axis, (K, 2, 2, ns, nm); K = 1 for a pure state."""
        return self.amplitude if self.is_ensemble else self.amplitude[None]

    @property
    def member_weights(self) -> np.ndarray:
        return np.sum(np.abs(self.members) ** 2, axis=(1, 2, 3, 4))

    @property
    def nu_signal(self) -> np.ndarray:
        return self.n_signal * self.dnu

    @property
    def nu_sum(self) -> np.ndarray:
        return self.n_sum * self.dnu

    @property
    def wavelengths_signal(self) -> np.ndarray:
        return constants.c / self.nu_signal

    def n_idler(self) -> np.ndarray:
        return self.n_sum[None, :] - self.n_signal[:, None]

    @property
    def pairs_per_pulse(self) -> float | None:
        if self.repetition_rate is None:
            return None
        return self.rate / self.repetition_rate


@dataclass(frozen=True)
class PairEmission:
    source: PairSource
    signal_port: str
    idler_port: str
