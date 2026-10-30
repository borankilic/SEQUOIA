import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sequoia.errors import SequoiaError
from sequoia.library.importers.table import Table, _normalize_unit, read_table
from sequoia.units import from_si, to_si, unit_dimension

YKind = Literal["counts", "rate", "other"]

_COUNT_UNITS = {"counts", "count", "cts", "events"}
SI_UNITS = {
    "dimensionless": "1",
    "angle": "rad",
    "time": "s",
    "length": "m",
    "frequency": "Hz",
    "power": "W",
    "current": "A",
    "voltage": "V",
    "energy": "J",
    "temperature": "K",
    "wavenumber": "1/m",
}


class MeasuredDataError(SequoiaError):
    pass


def _unit(unit: str | None, what: str) -> str:
    if unit is None:
        raise MeasuredDataError(
            f"the unit of {what} is unknown; choose one (e.g. 'deg', 'ns', 'mm', 'counts', "
            "'Hz', 'mW', '%')"
        )
    u = _normalize_unit(unit)
    if u.lower() in _COUNT_UNITS:
        return "counts"
    try:
        dimension = unit_dimension(u)
    except ValueError as exc:
        raise MeasuredDataError(f"{what}: {exc}") from exc
    if dimension == "other":
        raise MeasuredDataError(f"{what}: unit {unit!r} is not a supported physical unit")
    return u


def dimension_of(unit: str) -> str:
    return "dimensionless" if unit == "counts" else unit_dimension(unit)


def _si(values: np.ndarray, unit: str) -> np.ndarray:
    return np.asarray(values, dtype=float) if unit == "counts" else to_si(values, unit)


def _frozen(values) -> np.ndarray | None:
    if values is None:
        return None
    out = np.array(values, dtype=float)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class MeasuredData:
    """One measured curve with its units and where it came from.

    ``x``/``y``/``sigma``/``background`` are in the units the user mapped; ``time`` is the
    acquisition time per point in seconds. Rows dropped at import are listed in ``dropped``
    with the reason; a derived curve (e.g. background-subtracted) keeps the curve it came from in
    ``original`` and the operation in ``operations``.
    """

    x: np.ndarray
    y: np.ndarray
    x_unit: str
    y_unit: str
    x_label: str = "x"
    y_label: str = "y"
    sigma: np.ndarray | None = None
    time: np.ndarray | None = None
    background: np.ndarray | None = None
    y_kind: YKind = "other"
    rows: np.ndarray | None = None
    dropped: tuple[tuple[int, str], ...] = ()
    source: dict[str, Any] = field(default_factory=dict)
    operations: tuple[dict[str, Any], ...] = ()
    original: "MeasuredData | None" = None
    warnings: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.x)

    @property
    def x_dimension(self) -> str:
        return dimension_of(self.x_unit)

    @property
    def y_dimension(self) -> str:
        return dimension_of(self.y_unit)

    @property
    def x_si(self) -> np.ndarray:
        return _si(self.x, self.x_unit)

    @property
    def y_si(self) -> np.ndarray:
        return _si(self.y, self.y_unit)

    @property
    def sigma_si(self) -> np.ndarray | None:
        return None if self.sigma is None else _si(self.sigma, self.y_unit)

    @property
    def x_si_unit(self) -> str:
        return SI_UNITS.get(self.x_dimension, "1")

    @property
    def y_si_unit(self) -> str:
        return SI_UNITS.get(self.y_dimension, "1")


def from_arrays(
    x,
    y,
    *,
    x_unit: str,
    y_unit: str,
    sigma=None,
    time=None,
    background=None,
    x_label: str = "x",
    y_label: str = "y",
    y_kind: YKind | None = None,
    source: dict[str, Any] | None = None,
) -> MeasuredData:
    """Validate measured columns; ``time`` is per point (array) or one acquisition time, in s.
    Rows with a missing value are dropped and reported; x need not be monotonic."""
    xu, yu = _unit(x_unit, "x"), _unit(y_unit, "y")
    columns = {"x": np.asarray(x, dtype=float).ravel(), "y": np.asarray(y, dtype=float).ravel()}
    n = len(columns["x"])
    if len(columns["y"]) != n:
        raise MeasuredDataError(f"x has {n} values but y has {len(columns['y'])}")
    for name, values in (("uncertainty", sigma), ("background", background)):
        if values is not None:
            columns[name] = np.asarray(values, dtype=float).ravel()
    if time is not None:
        t = np.asarray(time, dtype=float).ravel()
        columns["time"] = np.full(n, t[0]) if t.size == 1 else t
    for name, values in columns.items():
        if len(values) != n:
            raise MeasuredDataError(f"{name} has {len(values)} values, x and y have {n}")
    reasons: dict[int, str] = {}
    for name, values in columns.items():
        for i in np.flatnonzero(~np.isfinite(values)):
            reasons.setdefault(int(i), f"missing or non-numeric {name}")
    if "uncertainty" in columns:
        for i in np.flatnonzero(columns["uncertainty"] <= 0):
            reasons.setdefault(int(i), "uncertainty is zero or negative")
    if "time" in columns:
        for i in np.flatnonzero(columns["time"] <= 0):
            reasons.setdefault(int(i), "acquisition time is zero or negative")
    keep = np.array([i not in reasons for i in range(n)], dtype=bool)
    if keep.sum() < 2:
        raise MeasuredDataError(
            f"only {int(keep.sum())} usable row(s) out of {n}; a curve needs at least 2"
        )
    dimension = dimension_of(yu)
    if y_kind is None:
        y_kind = "counts" if yu == "counts" else "rate" if dimension == "frequency" else "other"
    notes = []
    if reasons:
        listed = ", ".join(str(i + 1) for i in sorted(reasons)[:10])
        more = "" if len(reasons) <= 10 else f" and {len(reasons) - 10} more"
        notes.append(
            f"dropped {len(reasons)} row(s) with unusable values (data rows {listed}{more})"
        )
    if y_kind == "counts" and np.any(columns["y"][keep] < 0):
        notes.append("negative counts: Poisson weights are not meaningful for this column")
    return MeasuredData(
        x=_frozen(columns["x"][keep]),
        y=_frozen(columns["y"][keep]),
        x_unit=xu,
        y_unit=yu,
        x_label=x_label,
        y_label=y_label,
        sigma=_frozen(columns["uncertainty"][keep]) if "uncertainty" in columns else None,
        time=_frozen(columns["time"][keep]) if "time" in columns else None,
        background=_frozen(columns["background"][keep]) if "background" in columns else None,
        y_kind=y_kind,
        rows=_frozen(np.flatnonzero(keep)),
        dropped=tuple((i + 1, r) for i, r in sorted(reasons.items())),
        source=dict(source or {}),
        warnings=tuple(notes),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_measured(
    path: Path | str,
    x: int | str,
    y: int | str,
    *,
    sigma: int | str | None = None,
    background: int | str | None = None,
    time: int | str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    time_unit: str = "s",
    acquisition_time: float | None = None,
    y_kind: YKind | None = None,
    sheet: str | None = None,
    table: Table | None = None,
) -> MeasuredData:
    """Read a CSV/TSV/XLSX table and map its columns. Units come from the headers
    ('delay (ns)') unless given; the uncertainty and background columns are in the y unit;
    ``time`` names a per-row acquisition-time column (in ``time_unit``), ``acquisition_time``
    is one time in seconds for every row. The file itself is never modified."""
    path = Path(path)
    table = table if table is not None else read_table(path, sheet)
    cx, cy = table.column(x), table.column(y)
    cs = table.column(sigma) if sigma is not None else None
    cb = table.column(background) if background is not None else None
    ct = table.column(time) if time is not None else None
    if ct is not None and acquisition_time is not None:
        raise MeasuredDataError("give either a time column or one acquisition time, not both")
    xu = _unit(x_unit or cx.unit, f"x column {cx.header!r}")
    yu = _unit(y_unit or cy.unit, f"y column {cy.header!r}")
    for col in (cs, cb):
        if col is not None and col.unit is not None and y_unit is None:
            if _unit(col.unit, col.header) != yu:
                raise MeasuredDataError(
                    f"column {col.header!r} is in {col.unit!r} but y is in {yu!r}; the "
                    "uncertainty and background columns must use the y unit"
                )
    t = None
    if ct is not None:
        tu = _unit(ct.unit or time_unit, f"time column {ct.header!r}")
        if dimension_of(tu) != "time":
            raise MeasuredDataError(f"time column {ct.header!r}: {tu!r} is not a time unit")
        t = to_si(ct.values, tu)
    elif acquisition_time is not None:
        if not (math.isfinite(acquisition_time) and acquisition_time > 0):
            raise MeasuredDataError(f"acquisition time must be positive, got {acquisition_time}")
        t = acquisition_time
    source = {
        "file": path.name,
        "path": str(path),
        "sha256": _sha256(path),
        "sheet": table.sheet,
        "header_row": table.header_row,
        "columns": {
            k: c.header
            for k, c in (("x", cx), ("y", cy), ("sigma", cs), ("background", cb), ("time", ct))
            if c is not None
        },
        "units": {"x": xu, "y": yu},
    }
    if acquisition_time is not None:
        source["acquisition_time_s"] = acquisition_time
    return from_arrays(
        cx.values,
        cy.values,
        x_unit=xu,
        y_unit=yu,
        sigma=None if cs is None else cs.values,
        background=None if cb is None else cb.values,
        time=t,
        x_label=cx.name,
        y_label=cy.name,
        y_kind=y_kind,
        source=source,
    )


def poisson_sigma(data: MeasuredData) -> np.ndarray | None:
    """Counting uncertainty sqrt(N) in the y unit; N is floored at 1 so a zero-count bin keeps
    a finite weight. Rates need the acquisition time (N = rate * t)."""
    if data.y_kind == "counts":
        return np.sqrt(np.maximum(data.y, 1.0))
    if data.y_kind == "rate" and data.time is not None:
        rate_si = data.y_si
        sigma_si = np.sqrt(np.maximum(rate_si * data.time, 1.0)) / data.time
        return from_si(sigma_si, data.y_unit)
    return None


POISSON = "Poisson (model variance)"


def poisson_model_sigma(data: MeasuredData, model_si: np.ndarray, time=None) -> np.ndarray:
    """sqrt(expected counts) from the model, floored at one count, in SI y units. Fits that
    reweight with it until convergence solve the Poisson maximum-likelihood equations; weights
    from the observed counts would bias low-count parameters (e.g. a background) downwards."""
    model_si = np.asarray(model_si, dtype=float)
    if data.y_kind == "counts":
        return np.sqrt(np.maximum(model_si, 1.0))
    t = data.time if time is None else time
    return np.sqrt(np.maximum(model_si * t, 1.0)) / t


def weights(data: MeasuredData) -> tuple[np.ndarray, str, str | None]:
    """(sigma in SI y units, weighting name, warning)."""
    if data.sigma is not None:
        return data.sigma_si, "given uncertainties", None
    poisson = poisson_sigma(data)
    if poisson is not None:
        return _si(poisson, data.y_unit), POISSON, None
    warning = (
        "no uncertainties and no counts/acquisition time: unweighted fit, parameter errors are "
        "scaled by the residual scatter and chi2 is not a goodness-of-fit test"
    )
    return np.ones(len(data)), "unweighted", warning


def subtract_background(data: MeasuredData, sidebands: int = 1) -> MeasuredData:
    """y - background, keeping the original. The background column is taken as the mean of
    ``sidebands`` equal windows, so its counting variance is B/sidebands (B in counts)."""
    if data.background is None:
        raise MeasuredDataError("no background column is mapped")
    if sidebands < 1:
        raise MeasuredDataError("sidebands must be at least 1")
    sigma_y = data.sigma if data.sigma is not None else poisson_sigma(data)
    if data.y_kind == "counts":
        sigma_b = np.sqrt(np.maximum(data.background, 0.0) / sidebands)
    elif data.y_kind == "rate" and data.time is not None:
        b_si = _si(data.background, data.y_unit)
        sigma_b = from_si(np.sqrt(np.maximum(b_si, 0.0) / (data.time * sidebands)), data.y_unit)
    else:
        sigma_b = None
    notes = list(data.warnings)
    if sigma_y is None or sigma_b is None:
        sigma = data.sigma
        notes.append("background treated as exact: no counting model for its uncertainty")
    else:
        sigma = np.sqrt(sigma_y**2 + sigma_b**2)
    operation = {
        "operation": "subtract_background",
        "column": data.source.get("columns", {}).get("background", "background"),
        "sidebands": sidebands,
        "uncertainty": "sigma_y and sqrt(B/sidebands) added in quadrature"
        if sigma is not None and sigma_b is not None
        else "unchanged",
    }
    return replace(
        data,
        y=_frozen(data.y - data.background),
        sigma=_frozen(sigma),
        background=None,
        y_label=f"{data.y_label} - background",
        operations=data.operations + (operation,),
        original=data,
        warnings=tuple(notes),
    )


@dataclass
class Comparison:
    label: str
    x: np.ndarray
    y: np.ndarray
    model: np.ndarray
    residuals: np.ndarray
    normalized: np.ndarray
    chi2: float
    dof: int
    reduced_chi2: float
    weighting: str
    y_unit: str
    warnings: list[str] = field(default_factory=list)
    scale: Any = None
    details: dict[str, Any] = field(default_factory=dict)


def compare(
    data: MeasuredData,
    model_x_si,
    model_y_si,
    *,
    label: str = "model",
    fit_scale: bool = False,
) -> Comparison:
    """Model curve (SI x and y) evaluated at the data x by linear interpolation (no
    extrapolation: points outside the model range are left out); residuals are data - model.
    ``fit_scale`` also fits one multiplicative factor between model and data (loss budget)."""
    mx = np.asarray(model_x_si, dtype=float)
    my = np.asarray(model_y_si, dtype=float)
    good = np.isfinite(mx) & np.isfinite(my)
    order = np.argsort(mx[good])
    mx, my = mx[good][order], my[good][order]
    if len(mx) == 0:
        raise MeasuredDataError(f"{label}: no finite model points")
    xs = data.x_si
    tol = 1e-12 * max(abs(mx[0]), abs(mx[-1]), 1e-300)
    inside = (xs >= mx[0] - tol) & (xs <= mx[-1] + tol)
    model_si = np.full(len(xs), np.nan)
    model_si[inside] = np.interp(xs[inside], mx, my)
    return compare_values(data, model_si, label=label, fit_scale=fit_scale)


def compare_values(
    data: MeasuredData, model_si, *, label: str = "model", fit_scale: bool = False
) -> Comparison:
    """Comparison with model values (SI y) given at each data point; NaN points are left out."""
    model_si = np.asarray(model_si, dtype=float)
    used = np.isfinite(model_si)
    notes = list(data.warnings)
    if not used.any():
        raise MeasuredDataError(f"{label}: the data x values do not overlap the model curve")
    if not used.all():
        notes.append(f"{int((~used).sum())} data point(s) outside the model range are left out")
    sigma, weighting, warning = weights(data)
    if warning:
        notes.append(warning)
    scale = None
    if fit_scale:
        from sequoia.analysis.fitting import fit, scale_model

        scale = fit(data, scale_model(model_si[used], label=label), mask=used)
        model_si = model_si * scale.values["scale"]
    if weighting == POISSON:
        sigma = poisson_model_sigma(data, np.where(used, model_si, 1.0))
    resid_si = data.y_si - model_si
    norm = resid_si / sigma
    chi2 = float(np.sum(norm[used] ** 2))
    dof = int(used.sum()) - (1 if fit_scale else 0)
    if weighting == "unweighted":
        notes.append("chi2 of an unweighted comparison is a sum of squared residuals in SI units")
    to_unit = (lambda v: v) if data.y_unit == "counts" else (lambda v: from_si(v, data.y_unit))
    return Comparison(
        label=label,
        x=data.x,
        y=data.y,
        model=np.asarray(to_unit(model_si)),
        residuals=np.asarray(to_unit(resid_si)),
        normalized=norm,
        chi2=chi2,
        dof=dof,
        reduced_chi2=chi2 / dof if dof > 0 else math.nan,
        weighting=weighting,
        y_unit=data.y_unit,
        warnings=notes,
        scale=scale,
    )


_METRIC_UNITS = {"w": "W", "hz": "Hz", "s": "s", "m": "m", "a": "A", "per_s": "Hz"}


def metric_unit(name: str) -> str:
    """SI unit of a sweep metric from its name suffix (``power_W``, ``singles_hz``)."""
    tail = name.rsplit(".", 1)[-1].lower()
    if tail.endswith("_per_s"):
        return "Hz"
    suffix = tail.rsplit("_", 1)[-1] if "_" in tail else ""
    return _METRIC_UNITS.get(suffix, "1")


def compare_sweep(
    data: MeasuredData,
    setup,
    library,
    target: str,
    unit: str,
    metric: str,
    *,
    fit_scale: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Comparison:
    """Run a SEQUOIA sweep of ``target`` (in ``unit``) at the data x values and compare
    ``metric`` with the data. A predicted rate is turned into counts with the acquisition time
    when the data are counts."""
    from sequoia.analysis.sweep import Axis, run_sweep

    axis_dimension = "dimensionless" if unit.strip() in ("", "1") else unit_dimension(unit)
    if axis_dimension != data.x_dimension:
        raise MeasuredDataError(
            f"{target} is a {axis_dimension} quantity ({unit or 'plain number'}) but the data x "
            f"is in {data.x_unit!r} ({data.x_dimension})"
        )
    m_unit = metric_unit(metric)
    m_dim = unit_dimension(m_unit)
    times = None
    if m_dim != data.y_dimension:
        if m_dim == "frequency" and data.y_kind == "counts" and data.time is not None:
            times = data.time
        else:
            hint = (
                "; map an acquisition time to compare counts with a predicted rate"
                if m_dim == "frequency" and data.y_kind == "counts"
                else ""
            )
            raise MeasuredDataError(
                f"{metric} is a {m_dim} quantity ({m_unit}) but the data y is in "
                f"{data.y_unit!r} ({data.y_dimension}){hint}"
            )
    xs = data.x_si
    unique = np.unique(xs)
    values = unique if axis_dimension == "dimensionless" else from_si(unique, unit)
    result = run_sweep(
        setup,
        library,
        [Axis(target, np.atleast_1d(values), unit)],
        [metric],
        should_cancel=should_cancel,
        progress=progress,
    )
    predicted = np.asarray(result.column(metric), dtype=float)
    model_si = predicted[np.searchsorted(unique, xs)]
    if times is not None:
        model_si = model_si * times
    cmp = compare_values(data, model_si, label=metric, fit_scale=fit_scale)
    if result.failures:
        cmp.warnings.append(f"{len(result.failures)} sweep point(s) failed and are left out")
    cmp.details = {
        "target": target,
        "unit": unit,
        "metric": metric,
        "metric_unit": m_unit,
        "counts_from_rate": times is not None,
        "failures": {str(k): v for k, v in result.failures.items()},
    }
    return cmp
