import datetime
import re
import warnings
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)
from scipy.interpolate import RegularGridInterpolator

from sequoia.units import check_unit, to_si, unit_dimension

Source = Literal["datasheet", "raw-data", "measured", "model", "ideal-default"]
Extrapolation = Literal["warn_clamp", "clamp", "zero", "error"]


class ExtrapolationWarning(UserWarning):
    pass


class ExtrapolationError(ValueError):
    pass


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Source
    date: datetime.date | None = None
    by: str | None = None
    note: str | None = None
    reference: str | None = None


_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?inf"
_SCALAR_RE = re.compile(rf"^\s*({_NUMBER})\s*(.*?)\s*$")
_AXIS_RE = re.compile(r"^\s*(.*?)\s*(?:\((.*)\)|\[(.*)\])\s*$")


def _format_number(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


class Scalar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    unit: str = "1"
    provenance: Provenance | None = None

    _si: float = PrivateAttr(default=0.0)

    @model_validator(mode="before")
    @classmethod
    def _from_shorthand(cls, data: Any) -> Any:
        if isinstance(data, bool):
            raise ValueError("expected a number or '<number> <unit>', got a boolean")
        if isinstance(data, int | float):
            return {"value": float(data)}
        if isinstance(data, str):
            match = _SCALAR_RE.match(data)
            if not match:
                raise ValueError(f"cannot parse quantity {data!r}; expected '<number> <unit>'")
            return {"value": float(match[1]), "unit": match[2] or "1"}
        return data

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, unit: str) -> str:
        check_unit(unit)
        return unit

    def model_post_init(self, context: Any) -> None:
        self._si = to_si(self.value, self.unit)

    @property
    def si(self) -> float:
        return self._si

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        if self.provenance is not None:
            return handler(self)
        if self.unit.strip() in ("", "1"):
            return self.value
        return f"{_format_number(self.value)} {self.unit}"


def _require_dimension(dimension: str):
    def check(scalar: Scalar) -> Scalar:
        if unit_dimension(scalar.unit) != dimension:
            raise ValueError(f"expected a {dimension} quantity, got unit {scalar.unit!r}")
        return scalar

    return check


Length = Annotated[Scalar, AfterValidator(_require_dimension("length"))]
Duration = Annotated[Scalar, AfterValidator(_require_dimension("time"))]


def parse_axis(text: str) -> tuple[str, str]:
    match = _AXIS_RE.match(text)
    if not match:
        raise ValueError(f"axis {text!r} must look like 'name (unit)'")
    return match[1], (match[2] if match[2] is not None else match[3]).strip() or "1"


def format_axis(label: str, unit: str) -> str:
    return f"{label} ({unit})"


def _report_extrapolation(policy: str, message: str) -> None:
    if policy == "error":
        raise ExtrapolationError(message)
    if policy == "warn_clamp":
        warnings.warn(message, ExtrapolationWarning, stacklevel=3)


def _snap_to_range(q: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    tol = 1e-9 * (hi - lo)
    snapped = np.where((q < lo) & (q >= lo - tol), lo, q)
    snapped = np.where((snapped > hi) & (snapped <= hi + tol), hi, snapped)
    return snapped, (snapped < lo) | (snapped > hi)


def _strictly_increasing(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{name} values must be strictly increasing (sort the data and remove duplicates)"
        )


class Curve(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: list[float]
    y: list[float]
    x_label: str = "x"
    x_unit: str = "1"
    y_label: str = "y"
    y_unit: str = "1"
    provenance: Provenance | None = None
    extrapolation: Extrapolation = "warn_clamp"
    file: str | None = None

    _x_si: np.ndarray = PrivateAttr(default=None)
    _y_si: np.ndarray = PrivateAttr(default=None)

    @field_validator("x_unit", "y_unit")
    @classmethod
    def _check_unit(cls, unit: str) -> str:
        check_unit(unit)
        return unit

    @model_validator(mode="after")
    def _check_data(self) -> "Curve":
        if len(self.x) != len(self.y):
            raise ValueError(f"x has {len(self.x)} points but y has {len(self.y)}")
        if len(self.x) < 2:
            raise ValueError("a curve needs at least 2 points")
        x_si = np.asarray(to_si(np.asarray(self.x), self.x_unit))
        y_si = np.asarray(to_si(np.asarray(self.y), self.y_unit))
        _strictly_increasing(x_si, "x")
        if not np.all(np.isfinite(y_si)):
            raise ValueError("y contains non-finite values")
        x_si.flags.writeable = False
        y_si.flags.writeable = False
        self._x_si, self._y_si = x_si, y_si
        return self

    @property
    def x_si(self) -> np.ndarray:
        return self._x_si

    @property
    def y_si(self) -> np.ndarray:
        return self._y_si

    def __call__(self, x):
        lo, hi = self._x_si[0], self._x_si[-1]
        xq, outside = _snap_to_range(np.asarray(x, dtype=float), lo, hi)
        if np.any(outside):
            _report_extrapolation(
                self.extrapolation,
                f"curve '{self.y_label}' queried at x in [{xq.min():.6g}, {xq.max():.6g}] (SI) "
                f"outside its data range [{lo:.6g}, {hi:.6g}]",
            )
        edge = 0.0 if self.extrapolation == "zero" else None
        result = np.interp(xq, self._x_si, self._y_si, left=edge, right=edge)
        return float(result) if result.ndim == 0 else result

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self._content() == other._content()

    def _content(self) -> dict:
        return self.model_dump(exclude={"file"})


class Surface(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: list[float]
    y: list[float]
    z: list[list[float]]
    x_label: str = "x"
    x_unit: str = "1"
    y_label: str = "y"
    y_unit: str = "1"
    z_label: str = "z"
    z_unit: str = "1"
    provenance: Provenance | None = None
    extrapolation: Extrapolation = "warn_clamp"
    file: str | None = None

    _x_si: np.ndarray = PrivateAttr(default=None)
    _y_si: np.ndarray = PrivateAttr(default=None)
    _z_si: np.ndarray = PrivateAttr(default=None)
    _interp: RegularGridInterpolator = PrivateAttr(default=None)

    @field_validator("x_unit", "y_unit", "z_unit")
    @classmethod
    def _check_unit(cls, unit: str) -> str:
        check_unit(unit)
        return unit

    @model_validator(mode="after")
    def _check_data(self) -> "Surface":
        if len(self.x) < 2 or len(self.y) < 2:
            raise ValueError("a surface needs at least 2 points along each axis")
        z = np.asarray(self.z, dtype=float)
        if z.shape != (len(self.x), len(self.y)):
            raise ValueError(
                f"z has shape {z.shape} but x and y give ({len(self.x)}, {len(self.y)})"
            )
        x_si = np.asarray(to_si(np.asarray(self.x), self.x_unit))
        y_si = np.asarray(to_si(np.asarray(self.y), self.y_unit))
        z_si = np.asarray(to_si(z, self.z_unit))
        _strictly_increasing(x_si, "x")
        _strictly_increasing(y_si, "y")
        if not np.all(np.isfinite(z_si)):
            raise ValueError("z contains non-finite values")
        self._x_si, self._y_si, self._z_si = x_si, y_si, z_si
        self._interp = RegularGridInterpolator((x_si, y_si), z_si)
        return self

    @property
    def x_si(self) -> np.ndarray:
        return self._x_si

    @property
    def y_si(self) -> np.ndarray:
        return self._y_si

    @property
    def z_si(self) -> np.ndarray:
        return self._z_si

    def __call__(self, x, y):
        xq, yq = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        x0, x1, y0, y1 = self._x_si[0], self._x_si[-1], self._y_si[0], self._y_si[-1]
        xq, x_out = _snap_to_range(xq, x0, x1)
        yq, y_out = _snap_to_range(yq, y0, y1)
        outside = x_out | y_out
        if np.any(outside):
            _report_extrapolation(
                self.extrapolation,
                f"surface '{self.z_label}' queried outside its data range "
                f"x in [{x0:.6g}, {x1:.6g}], y in [{y0:.6g}, {y1:.6g}] (SI)",
            )
        points = np.stack([np.clip(xq, x0, x1), np.clip(yq, y0, y1)], axis=-1)
        result = self._interp(points)
        if self.extrapolation == "zero":
            result = np.where(outside, 0.0, result)
        return float(result) if result.ndim == 0 else result

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self._content() == other._content()

    def _content(self) -> dict:
        return self.model_dump(exclude={"file"})


class FileRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    provenance: Provenance | None = None
