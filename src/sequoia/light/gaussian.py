from dataclasses import dataclass, replace

import numpy as np


def free_space(distance: float) -> np.ndarray:
    return np.array([[1.0, distance], [0.0, 1.0]])


def thin_lens(focal_length: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-1.0 / focal_length, 1.0]])


def _apply(abcd: np.ndarray, q: complex) -> complex:
    return (abcd[0, 0] * q + abcd[0, 1]) / (abcd[1, 0] * q + abcd[1, 1])


@dataclass(frozen=True)
class GaussianMode:
    wavelength: float
    q_x: complex
    q_y: complex
    m2_x: float = 1.0
    m2_y: float = 1.0

    @classmethod
    def from_waist(
        cls,
        wavelength: float,
        w0_x: float,
        w0_y: float | None = None,
        m2_x: float = 1.0,
        m2_y: float | None = None,
        z: float = 0.0,
    ) -> "GaussianMode":
        w0_y = w0_x if w0_y is None else w0_y
        m2_y = m2_x if m2_y is None else m2_y
        return cls(
            wavelength,
            z + 1j * np.pi * w0_x**2 / (wavelength * m2_x),
            z + 1j * np.pi * w0_y**2 / (wavelength * m2_y),
            m2_x,
            m2_y,
        )

    def _w(self, q: complex, m2: float) -> float:
        return float(np.sqrt(-self.wavelength * m2 / (np.pi * (1.0 / q).imag)))

    def _curvature(self, q: complex) -> float:
        inv = (1.0 / q).real
        return float(np.inf) if inv == 0.0 else float(1.0 / inv)

    @property
    def w_x(self) -> float:
        return self._w(self.q_x, self.m2_x)

    @property
    def w_y(self) -> float:
        return self._w(self.q_y, self.m2_y)

    @property
    def radius_x(self) -> float:
        return self._curvature(self.q_x)

    @property
    def radius_y(self) -> float:
        return self._curvature(self.q_y)

    @property
    def rayleigh_x(self) -> float:
        return float(self.q_x.imag)

    @property
    def rayleigh_y(self) -> float:
        return float(self.q_y.imag)

    @property
    def waist_x(self) -> float:
        return float(np.sqrt(self.wavelength * self.m2_x * self.q_x.imag / np.pi))

    @property
    def waist_y(self) -> float:
        return float(np.sqrt(self.wavelength * self.m2_y * self.q_y.imag / np.pi))

    @property
    def distance_past_waist_x(self) -> float:
        return float(self.q_x.real)

    @property
    def distance_past_waist_y(self) -> float:
        return float(self.q_y.real)

    @property
    def divergence_x(self) -> float:
        return float(self.wavelength * self.m2_x / (np.pi * self.waist_x))

    @property
    def divergence_y(self) -> float:
        return float(self.wavelength * self.m2_y / (np.pi * self.waist_y))

    def apply(self, abcd_x: np.ndarray, abcd_y: np.ndarray | None = None) -> "GaussianMode":
        abcd_y = abcd_x if abcd_y is None else abcd_y
        return replace(self, q_x=_apply(abcd_x, self.q_x), q_y=_apply(abcd_y, self.q_y))

    def degraded(self, factor: float) -> "GaussianMode":
        """Same local beam radius and curvature with the M^2 multiplied by ``factor`` (>= 1)."""
        if factor == 1.0:
            return self

        def new_q(q: complex) -> complex:
            inv = 1.0 / q
            return 1.0 / complex(inv.real, inv.imag * factor)

        return replace(
            self,
            q_x=new_q(self.q_x),
            q_y=new_q(self.q_y),
            m2_x=self.m2_x * factor,
            m2_y=self.m2_y * factor,
        )

    def propagate(self, distance: float) -> "GaussianMode":
        return self.apply(free_space(distance))

    def _equivalent_inverse_q(self, q: complex, m2: float) -> complex:
        inv = 1.0 / q
        return complex(inv.real, inv.imag / m2)


def _overlap_1d(a1: complex, a2: complex) -> complex:
    return (abs(a1.imag) * abs(a2.imag)) ** 0.25 * np.sqrt(2.0 / (1j * (a1 - np.conj(a2))))


def overlap(a: GaussianMode, b: GaussianMode) -> complex:
    ax1 = a._equivalent_inverse_q(a.q_x, a.m2_x)
    ay1 = a._equivalent_inverse_q(a.q_y, a.m2_y)
    ax2 = b._equivalent_inverse_q(b.q_x, b.m2_x)
    ay2 = b._equivalent_inverse_q(b.q_y, b.m2_y)
    return complex(_overlap_1d(ax1, ax2) * _overlap_1d(ay1, ay2))


def coupling_efficiency(beam: GaussianMode, fiber: GaussianMode, offset: float = 0.0) -> float:
    eta = abs(overlap(beam, fiber)) ** 2 / (beam.m2_x * beam.m2_y)
    if offset:
        w1 = 0.5 * (beam.w_x + beam.w_y)
        w2 = 0.5 * (fiber.w_x + fiber.w_y)
        eta *= float(np.exp(-2.0 * offset**2 / (w1**2 + w2**2)))
    return float(eta)


def aperture_transmission(radius: float, mode: GaussianMode) -> float:
    if not np.isfinite(radius):
        return 1.0
    return float(1.0 - np.exp(-2.0 * radius**2 / (mode.w_x * mode.w_y)))
