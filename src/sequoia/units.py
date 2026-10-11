import re
from functools import cache

import numpy as np
import pint
from scipy import constants

ureg = pint.UnitRegistry()
_COUNTS_PER_SECOND = re.compile(r"^([kMGm]?)cps$")

_DIMENSIONS = {
    "dimensionless": ureg.Quantity(1.0, "dimensionless").dimensionality,
    "length": ureg.Quantity(1.0, "m").dimensionality,
    "time": ureg.Quantity(1.0, "s").dimensionality,
    "power": ureg.Quantity(1.0, "W").dimensionality,
    "frequency": ureg.Quantity(1.0, "Hz").dimensionality,
    "wavenumber": ureg.Quantity(1.0, "1/m").dimensionality,
    "energy": ureg.Quantity(1.0, "J").dimensionality,
    "temperature": ureg.Quantity(1.0, "K").dimensionality,
    "current": ureg.Quantity(1.0, "A").dimensionality,
    "voltage": ureg.Quantity(1.0, "V").dimensionality,
    "nonlinear_coefficient": ureg.Quantity(1.0, "m/V").dimensionality,
}

_EXTENDED_DIMENSIONS = {
    "responsivity": ureg.Quantity(1.0, "A/W").dimensionality,
    "per_energy": ureg.Quantity(1.0, "1/J").dimensionality,
}

_ANGLE_UNIT_ROOTS = ("radian", "degree", "arcminute", "arcsecond")


def _norm(unit: str) -> str:
    u = unit.strip()
    if u in ("", "1"):
        return "dimensionless"
    return _COUNTS_PER_SECOND.sub(r"\1Hz", u)


def _magnitude(q):
    mag = q.magnitude
    return float(mag) if np.ndim(mag) == 0 else np.asarray(mag, dtype=float)


def _as_input(value):
    return float(value) if np.ndim(value) == 0 else np.asarray(value, dtype=float)


@cache
def _unit(unit: str):
    try:
        return ureg.Unit(_norm(unit))
    except Exception as exc:
        raise ValueError(f"unknown unit {unit!r}") from exc


@cache
def _base_units(unit: str):
    return ureg.Quantity(1.0, _unit(unit)).to_base_units().units


def check_unit(unit: str) -> None:
    _unit(unit)


def to_si(value, unit: str):
    return _magnitude(ureg.Quantity(_as_input(value), _unit(unit)).to_base_units())


def from_si(value, unit: str):
    return _magnitude(ureg.Quantity(_as_input(value), _base_units(unit)).to(_unit(unit)))


def unit_dimension(unit: str, *, extended: bool = False) -> str:
    """Name of the dimension of ``unit``; ``extended`` also names the responsivity (A/W) and
    the rate per power (1/J) used by typed fields, which are otherwise 'other'."""
    if _base_units(unit) == ureg.radian:
        return "angle"
    dim = ureg.Quantity(1.0, _unit(unit)).dimensionality
    references = {**_DIMENSIONS, **_EXTENDED_DIMENSIONS} if extended else _DIMENSIONS
    for name, reference in references.items():
        if dim == reference:
            return name
    return "other"


def is_plane_angle_unit(unit: str) -> bool:
    """True for radian, degree, arcminute and arcsecond (with SI prefixes, e.g. mrad, mdeg);
    a full 'turn' or a 'grad' (gradian) has the dimension of an angle but is refused."""
    container = getattr(_unit(unit), "_units", None)
    if container is None or len(container) != 1:
        return False
    ((name, exponent),) = container.items()
    return exponent == 1 and str(name).endswith(_ANGLE_UNIT_ROOTS)


def to_wavelength(x_si, dimension: str):
    x = np.asarray(x_si, dtype=float)
    if dimension == "length":
        return x
    if dimension == "frequency":
        return constants.c / x
    if dimension == "wavenumber":
        return 1.0 / x
    if dimension == "energy":
        return constants.h * constants.c / x
    raise ValueError(f"cannot convert a {dimension} axis to wavelength")
