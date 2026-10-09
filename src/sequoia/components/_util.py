import numpy as np

from sequoia.data import Curve, Scalar

REFLECT = (1.0, -1.0)


def spectral(value: Scalar | Curve | float | None, wavelengths: np.ndarray, default: float = 1.0):
    if value is None:
        return np.full(len(wavelengths), default)
    if isinstance(value, Curve):
        return np.asarray(value(wavelengths), dtype=float)
    if isinstance(value, Scalar):
        return np.full(len(wavelengths), value.si)
    return np.full(len(wavelengths), float(value))


def diag_jones(dh, dv) -> np.ndarray:
    dh, dv = np.broadcast_arrays(np.asarray(dh, dtype=complex), np.asarray(dv, dtype=complex))
    jones = np.zeros((len(dh), 2, 2), dtype=complex)
    jones[:, 0, 0] = dh
    jones[:, 1, 1] = dv
    return jones


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotated(jones: np.ndarray, theta: float) -> np.ndarray:
    if theta == 0.0:
        return jones
    r = rotation(theta)
    return r @ jones @ r.T


def retarder(waves: np.ndarray, theta: float) -> np.ndarray:
    delta = 2.0 * np.pi * np.asarray(waves, dtype=float)
    return rotated(diag_jones(np.exp(-0.5j * delta), np.exp(0.5j * delta)), theta)


def sqrt_power(power: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(power, 0.0, None))
