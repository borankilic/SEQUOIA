import datetime
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfcx

from sequoia.analysis.measured import (
    POISSON,
    MeasuredData,
    MeasuredDataError,
    poisson_model_sigma,
    weights,
)
from sequoia.units import from_si, to_si

RANK_RTOL = 1e-7
MAX_CORRELATION = 0.95
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True)
class FitModel:
    """A fit function of SI x with parameters in SI units.

    ``kinds`` give each parameter's dimension for scaling and display: ``x`` a position on the x
    axis, ``dx`` an x interval, ``y`` a y value, ``xy`` an area, ``y/x`` a slope, ``1`` a
    plain number. ``confounded`` maps a set of parameter names to the message shown when the
    data cannot separate them.
    """

    name: str
    params: tuple[str, ...]
    kinds: tuple[str, ...]
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    guess: Callable[[np.ndarray, np.ndarray], np.ndarray]
    x_dimensions: tuple[str, ...] | None = None
    y_dimensions: tuple[str, ...] | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    derived: dict[str, tuple[Callable[[dict[str, float]], float], str]] = field(
        default_factory=dict
    )
    confounded: dict[frozenset[str], str] = field(default_factory=dict)
    normalize: Callable[[dict[str, float]], dict[str, float]] | None = None
    description: str = ""


def _compound(a: str, op: str, b: str) -> str:
    if b == "1":
        return a
    if a == "1":
        return b if op == "*" else f"1/({b})"
    return f"({a}){op}({b})"


def _to_display(value: float, unit: str) -> float:
    if not math.isfinite(value) or unit in ("1", "counts"):
        return value
    return float(from_si(value, unit.replace("counts", "count")))


def param_unit(kind: str, x_unit: str, y_unit: str) -> str:
    """Display unit of a parameter of ``kind`` for data in ``x_unit`` and ``y_unit``."""
    return {
        "x": x_unit,
        "dx": x_unit,
        "y": y_unit,
        "xy": _compound(y_unit, "*", x_unit),
        "y/x": _compound(y_unit, "/", x_unit),
        "1": "1",
    }[kind]


def from_display(value: float, unit: str) -> float:
    if unit in ("1", "counts"):
        return value
    return float(to_si(value, unit.replace("counts", "count")))


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass
class FitResult:
    model: FitModel
    values: dict[str, float]
    errors: dict[str, float]
    fixed: dict[str, float]
    free: list[str]
    covariance: np.ndarray
    correlation: np.ndarray
    chi2: float
    dof: int
    reduced_chi2: float
    weighting: str
    residuals: np.ndarray
    normalized: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_unit: str
    y_unit: str
    derived: dict[str, tuple[float, float, str]]
    flags: list[str]
    warnings: list[str]
    unidentifiable: list[str]
    converged: bool
    fixed_sources: dict[str, str] = field(default_factory=dict)

    @property
    def identifiable(self) -> bool:
        return not self.flags

    def predict(self, x_si) -> np.ndarray:
        p = np.array([self.values[n] for n in self.model.params], dtype=float)
        return self.model.fn(np.asarray(x_si, dtype=float), p)

    def predict_display(self, x_si) -> np.ndarray:
        return np.asarray(
            [_to_display(float(v), self.y_unit) for v in np.atleast_1d(self.predict(x_si))]
        )

    def unit_of(self, kind: str) -> str:
        return param_unit(kind, self.x_unit, self.y_unit)

    def display(self, name: str) -> tuple[float, float, str]:
        """Value, standard error and unit in the units of the data."""
        if name in self.derived:
            value, se, kind = self.derived[name]
        else:
            kind = self.model.kinds[self.model.params.index(name)]
            value, se = self.values[name], self.errors[name]
        unit = self.unit_of(kind)
        return _to_display(value, unit), _to_display(se, unit), unit

    def rows(self) -> list[tuple[str, float, float, str, str]]:
        """(name, value, SE, unit, status) for every parameter and derived quantity."""
        out = []
        for name in self.model.params:
            value, se, unit = self.display(name)
            if name in self.fixed:
                status = "fixed: " + self.fixed_sources.get(name, "independently calibrated")
            elif name in self.unidentifiable:
                status = "not identifiable"
            else:
                status = "fitted"
            out.append((name, value, se, unit, status))
        for name in self.derived:
            value, se, unit = self.display(name)
            out.append((name, value, se, unit, "derived"))
        return out

    def to_record(self) -> dict[str, Any]:
        """JSON-ready record; non-finite numbers are written as null."""
        k = len(self.free)
        return _finite_or_none(
            {
                "model": self.model.name,
                "description": self.model.description,
                "weighting": self.weighting,
                "parameters": {
                    name: {
                        "value_si": self.values[name],
                        "se_si": self.errors[name],
                        "value": v,
                        "se": se,
                        "unit": unit,
                        "status": status,
                    }
                    for name, v, se, unit, status in self.rows()
                    if name in self.values
                },
                "derived": {
                    name: {"value": v, "se": se, "unit": unit}
                    for name, v, se, unit, status in self.rows()
                    if status == "derived"
                },
                "fixed": {
                    n: {"value_si": v, "source": self.fixed_sources.get(n, "")}
                    for n, v in self.fixed.items()
                },
                "free": list(self.free),
                "correlation": [
                    [float(self.correlation[i, j]) for j in range(k)] for i in range(k)
                ],
                "chi2": self.chi2,
                "dof": self.dof,
                "reduced_chi2": self.reduced_chi2,
                "converged": self.converged,
                "identifiability": list(self.flags),
                "warnings": list(self.warnings),
            }
        )


def _scales(model: FitModel, p0: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    lo, hi = float(np.min(xs)), float(np.max(xs))
    xc = 0.5 * (lo + hi)
    xspan = (hi - lo) or abs(xc) or 1.0
    yscale = float(np.max(np.abs(ys))) or 1.0
    a, b = np.zeros(len(p0)), np.ones(len(p0))
    for i, kind in enumerate(model.kinds):
        if kind == "x":
            a[i], b[i] = xc, xspan
        elif kind == "dx":
            b[i] = xspan
        elif kind == "y":
            b[i] = yscale
        elif kind == "xy":
            b[i] = xspan * yscale
        elif kind == "y/x":
            b[i] = abs(p0[i]) or yscale / xspan
        else:
            b[i] = max(abs(p0[i]), 1.0)
    return a, b


def fit(
    data: MeasuredData,
    model: FitModel,
    *,
    fixed: dict[str, float] | None = None,
    fixed_sources: dict[str, str] | None = None,
    p0: dict[str, float] | None = None,
    mask=None,
) -> FitResult:
    """Weighted least squares of ``model`` to ``data`` in SI units.

    Weights: the data's uncertainties, else Poisson with the model variance from counts (or a
    rate and its acquisition time), reweighted until it converges to the Poisson maximum
    likelihood, else unweighted with a warning (errors then scaled by the residual scatter).
    ``fixed`` parameters (SI) are held at independently calibrated values. Standard errors come
    from the covariance (J^T J)^-1 at the solution; directions the data do not constrain
    (rank-deficient Jacobian) give infinite errors and an identifiability flag.
    """
    if model.x_dimensions is not None and data.x_dimension not in model.x_dimensions:
        raise MeasuredDataError(
            f"{model.name}: x must be of dimension {' or '.join(model.x_dimensions)}; the data x "
            f"is in {data.x_unit!r} ({data.x_dimension})"
        )
    if model.y_dimensions is not None and data.y_dimension not in model.y_dimensions:
        raise MeasuredDataError(
            f"{model.name}: y must be of dimension {' or '.join(model.y_dimensions)}; the data y "
            f"is in {data.y_unit!r} ({data.y_dimension})"
        )
    fixed = dict(fixed or {})
    unknown = (set(fixed) | set(p0 or {})) - set(model.params)
    if unknown:
        raise MeasuredDataError(f"{model.name}: no parameter(s) {sorted(unknown)}")
    xs, ys = data.x_si, data.y_si
    sigma, weighting, warning = weights(data)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        xs, ys, sigma = xs[mask], ys[mask], sigma[mask]
    times = None if data.time is None else (data.time if mask is None else data.time[mask])
    notes = list(data.warnings) + ([warning] if warning else [])
    names = list(model.params)
    guess = np.asarray(model.guess(xs, ys), dtype=float)
    for n, v in (p0 or {}).items():
        guess[names.index(n)] = v
    for n, v in fixed.items():
        guess[names.index(n)] = v
    free = [n for n in names if n not in fixed]
    if not free:
        raise MeasuredDataError(f"{model.name}: every parameter is fixed, nothing to fit")
    idx = np.array([names.index(n) for n in free])
    a, b = _scales(model, guess, xs, ys)
    a, b = a[idx], b[idx]
    lower = np.array(
        [
            (model.bounds.get(n, (-np.inf, np.inf))[0] - ai) / bi
            for n, ai, bi in zip(free, a, b, strict=True)
        ]
    )
    upper = np.array(
        [
            (model.bounds.get(n, (-np.inf, np.inf))[1] - ai) / bi
            for n, ai, bi in zip(free, a, b, strict=True)
        ]
    )
    q0 = (guess[idx] - a) / b
    span = np.where(np.isfinite(upper - lower), upper - lower, 1.0)
    q0 = np.clip(q0, lower + 1e-9 * span, upper - 1e-9 * span)

    def full(q):
        p = guess.copy()
        p[idx] = a + b * q
        return p

    def resid(q):
        return (model.fn(xs, full(q)) - ys) / sigma

    with np.errstate(all="ignore"):
        res = least_squares(resid, q0, jac="3-point", bounds=(lower, upper), method="trf")
        for _ in range(20 if weighting == POISSON else 0):
            sigma = poisson_model_sigma(data, model.fn(xs, full(res.x)), times)
            previous = res.x
            res = least_squares(resid, res.x, jac="3-point", bounds=(lower, upper), method="trf")
            if np.allclose(res.x, previous, rtol=1e-9, atol=1e-12):
                break
    p = full(res.x)
    n, k = len(xs), len(free)
    chi2 = float(np.sum(res.fun**2))
    dof = n - k
    reduced = chi2 / dof if dof > 0 else math.nan
    flags: list[str] = []
    if dof <= 0:
        flags.append(f"{n} points for {k} free parameters: the fit is not overdetermined")
    if not res.success:
        notes.append(f"the optimizer did not converge: {res.message}")

    jac = np.asarray(res.jac, dtype=float)
    at_bound = (np.asarray(res.active_mask) != 0) | (res.x - lower < 1e-6) | (upper - res.x < 1e-6)
    inner = np.flatnonzero(~at_bound)
    _u, s, vt = np.linalg.svd(jac[:, inner], full_matrices=False)
    keep = s > (s[0] * RANK_RTOL if s.size and s[0] > 0 else np.inf)
    cov_q = np.zeros((k, k))
    cov_q[np.ix_(inner, inner)] = (vt[keep].T / s[keep] ** 2) @ vt[keep]
    if weighting == "unweighted" and dof > 0:
        cov_q = cov_q * reduced
    null = []
    unidentifiable: list[str] = []
    for i in np.flatnonzero(at_bound):
        unidentifiable.append(free[i])
        null.append(np.eye(k)[i])
        flags.append(
            f"{free[i]} ended at the limit of its allowed range ({p[idx[i]]:.4g} SI): the data "
            "prefer a value outside it (e.g. no tail), so its error is not meaningful; fix it "
            "or choose a simpler model"
        )
    for row in vt[~keep]:
        v = np.zeros(k)
        v[inner] = row
        null.append(v)
        involved = [free[i] for i in np.flatnonzero(np.abs(v) > 0.05)]
        for name in involved:
            if name not in unidentifiable:
                unidentifiable.append(name)
        message = next((m for group, m in model.confounded.items() if group <= set(involved)), None)
        flags.append(
            message
            or f"{', '.join(involved)} cannot be separated by these data (rank-deficient "
            "Jacobian): only a combination is constrained; fix one from an independent "
            "calibration"
        )
    cov = (cov_q * b[:, None]) * b[None, :]
    constrained = cov.copy()
    for name in unidentifiable:
        i = free.index(name)
        cov[i, :] = np.nan
        cov[:, i] = np.nan
        cov[i, i] = np.inf
    se = np.sqrt(np.abs(np.diag(cov)))
    with np.errstate(all="ignore"):
        corr = cov / np.outer(se, se)
    np.fill_diagonal(corr, 1.0)

    values = {nm: float(v) for nm, v in zip(names, p, strict=True)}
    if model.normalize is not None:
        values = model.normalize(values)
    errors = {nm: 0.0 for nm in fixed}
    errors.update({nm: float(se[i]) for i, nm in enumerate(free)})
    for i, nm in enumerate(free):
        kind = model.kinds[names.index(nm)]
        if nm in unidentifiable or kind == "x":
            continue
        if se[i] > abs(values[nm]):
            flags.append(
                f"{nm}: standard error {se[i]:.3g} exceeds |value| {abs(values[nm]):.3g} (SI); "
                "the data barely constrain it"
            )
    for i in range(k):
        for j in range(i + 1, k):
            r = corr[i, j]
            if math.isfinite(r) and abs(r) > MAX_CORRELATION:
                pair = frozenset((free[i], free[j]))
                extra = next((m for g, m in model.confounded.items() if g <= pair), None)
                flags.append(
                    f"{free[i]} and {free[j]} are strongly correlated (r = {r:+.3f}); their "
                    "separate values are poorly determined" + (f". {extra}" if extra else "")
                )

    derived = {}
    for dname, (fn, kind) in model.derived.items():
        value = float(fn(values))
        grad = np.zeros(k)
        for i, nm in enumerate(free):
            h = 1e-6 * max(abs(values[nm]), b[i])
            up, down = dict(values), dict(values)
            up[nm] += h
            down[nm] -= h
            grad[i] = (fn(up) - fn(down)) / (2 * h)
        g_q = grad * b
        norm = float(np.linalg.norm(g_q))
        along_null = any(abs(float(g_q @ v)) > 1e-6 * norm for v in null) if norm else False
        var = math.inf if along_null else float(grad @ constrained @ grad)
        derived[dname] = (value, math.sqrt(max(var, 0.0)), kind)

    resid_si = ys - model.fn(xs, p)
    return FitResult(
        model=model,
        values=values,
        errors=errors,
        fixed=fixed,
        free=free,
        covariance=cov,
        correlation=corr,
        chi2=chi2,
        dof=dof,
        reduced_chi2=reduced,
        weighting=weighting,
        residuals=np.array([_to_display(float(v), data.y_unit) for v in resid_si]),
        normalized=resid_si / sigma,
        x=data.x if mask is None else data.x[mask],
        y=data.y if mask is None else data.y[mask],
        x_unit=data.x_unit,
        y_unit=data.y_unit,
        derived=derived,
        flags=flags,
        warnings=notes,
        unidentifiable=unidentifiable,
        converged=bool(res.success),
        fixed_sources=dict(fixed_sources or {}),
    )


def fringe(harmonic: int = 2) -> FitModel:
    m = float(harmonic)

    def fn(x, p):
        amplitude, phase, offset = p
        return offset + amplitude * np.cos(m * (x - phase))

    def guess(x, y):
        basis = np.column_stack([np.ones_like(x), np.cos(m * x), np.sin(m * x)])
        (c, ca, sb), *_ = np.linalg.lstsq(basis, y, rcond=None)
        return np.array([math.hypot(ca, sb), math.atan2(sb, ca) / m, c])

    def normalize(values):
        out = dict(values)
        period = 2 * math.pi / m
        out["phase"] = (values["phase"] + period / 2) % period - period / 2
        return out

    return FitModel(
        name=f"polarization fringe (cos {harmonic}x)",
        params=("amplitude", "phase", "offset"),
        kinds=("y", "x", "y"),
        fn=fn,
        guess=guess,
        x_dimensions=("angle",),
        bounds={"amplitude": (0.0, np.inf)},
        derived={"visibility": (lambda v: v["amplitude"] / v["offset"], "1")},
        normalize=normalize,
        description=(
            f"y = offset + A cos({harmonic} (x - phase)); visibility = A / offset. Use 2 for an "
            "analyser (polarizer) angle and 4 for a half-wave-plate angle"
        ),
    )


def _emg(x, center, width, tail):
    u = x - center
    return (
        erfcx((width / tail - u / width) / math.sqrt(2.0))
        * np.exp(-(u**2) / (2 * width**2))
        / (2 * tail)
    )


def peak(shape: str = "gaussian") -> FitModel:
    if shape == "gaussian":

        def fn(x, p):
            center, width, background, area = p
            return background + area * np.exp(-((x - center) ** 2) / (2 * width**2)) / (
                width * math.sqrt(2 * math.pi)
            )

        params = ("center", "width", "background", "area")
        kinds = ("x", "dx", "y", "xy")
    elif shape == "gaussian_tail":

        def fn(x, p):
            center, width, tail, background, area = p
            return background + area * _emg(x, center, width, tail)

        params = ("center", "width", "tail", "background", "area")
        kinds = ("x", "dx", "dx", "y", "xy")
    else:
        raise ValueError(f"unknown peak shape {shape!r}; use 'gaussian' or 'gaussian_tail'")

    def guess(x, y):
        order = np.argsort(x)
        xo, yo = x[order], y[order]
        background = float(np.percentile(yo, 10))
        excess = np.clip(yo - background, 0.0, None)
        area = float(np.trapezoid(excess, xo)) or float(np.ptp(yo)) * float(np.ptp(xo)) / 10
        center = float(xo[np.argmax(yo)])
        width = math.sqrt(float(np.sum(excess * (xo - center) ** 2) / max(np.sum(excess), 1e-300)))
        width = min(max(width, float(np.ptp(xo)) / len(xo)), float(np.ptp(xo)) / 4) or 1.0
        if shape == "gaussian":
            return np.array([center, width, background, area])
        return np.array([center - width / 2, width / 2, width, background, area])

    bounds = {"width": (0.0, np.inf), "tail": (0.0, np.inf)}
    derived = {}
    if shape == "gaussian":
        derived["fwhm"] = (lambda v: FWHM_PER_SIGMA * v["width"], "dx")
    return FitModel(
        name="coincidence peak, "
        + ("Gaussian" if shape == "gaussian" else "Gaussian + exponential tail"),
        params=params,
        kinds=kinds,
        fn=fn,
        guess=guess,
        x_dimensions=("time",),
        bounds=bounds,
        derived=derived,
        description=(
            "y = background + area * G(x - center; width), width = Gaussian sigma"
            if shape == "gaussian"
            else "y = background + area * (Gaussian sigma=width convolved with exp(-t/tail)), "
            "center = Gaussian centre (the peak maximum is later)"
        ),
    )


def hom_dip(shape: str = "gaussian") -> FitModel:
    if shape == "gaussian":

        def g(u):
            return np.exp(-(u**2) / 2)

    elif shape == "sinc":
        g = np.sinc
    else:
        raise ValueError(f"unknown HOM shape {shape!r}; use 'gaussian' or 'sinc'")

    def fn(x, p):
        delay_offset, visibility, width, baseline = p
        return baseline * (1 - visibility * g((x - delay_offset) / width))

    def guess(x, y):
        order = np.argsort(x)
        xo, yo = x[order], y[order]
        baseline = float(np.percentile(yo, 80)) or float(np.max(yo)) or 1.0
        center = float(xo[np.argmin(yo)])
        depth = np.clip(baseline - yo, 0.0, None)
        width = math.sqrt(float(np.sum(depth * (xo - center) ** 2) / max(np.sum(depth), 1e-300)))
        width = min(max(width, float(np.ptp(xo)) / len(xo)), float(np.ptp(xo)) / 4) or 1.0
        return np.array([center, 1 - float(np.min(yo)) / baseline, width, baseline])

    derived = {}
    if shape == "gaussian":
        derived["fwhm"] = (lambda v: FWHM_PER_SIGMA * v["width"], "dx")
    return FitModel(
        name=f"HOM dip ({'Gaussian' if shape == 'gaussian' else 'sinc'})",
        params=("delay_offset", "visibility", "width", "baseline"),
        kinds=("x", "1", "dx", "y"),
        fn=fn,
        guess=guess,
        x_dimensions=("time", "length"),
        bounds={"width": (0.0, np.inf)},
        derived=derived,
        description=(
            "y = baseline (1 - V exp(-(x - offset)^2 / (2 width^2)))"
            if shape == "gaussian"
            else "y = baseline (1 - V sinc((x - offset) / width)), zeros at offset +/- width "
            "(rectangular spectrum)"
        ),
    )


def scale_model(model, label: str = "model") -> FitModel:
    """One multiplicative factor between a model and the data (loss-budget scale). ``model`` is
    a callable of SI x giving SI y, or the model values at the data points."""
    if callable(model):
        values = model
    else:
        fixed_values = np.asarray(model, dtype=float)

        def values(x):
            return fixed_values

    def fn(x, p):
        return p[0] * values(x)

    def guess(x, y):
        m = values(x)
        denom = float(np.sum(m * m))
        return np.array([float(np.sum(y * m)) / denom if denom else 1.0])

    return FitModel(
        name=f"scale x {label}",
        params=("scale",),
        kinds=("1",),
        fn=fn,
        guess=guess,
        description=f"y = scale * {label}; scale < 1 is extra loss not in the model",
    )


def singles() -> FitModel:
    def fn(x, p):
        brightness, efficiency, dark = p
        return brightness * efficiency * x + dark

    def guess(x, y):
        slope, dark = np.polyfit(x, y, 1)
        return np.array([slope / 0.1, 0.1, dark])

    return FitModel(
        name="singles vs pump power",
        params=("brightness", "efficiency", "dark"),
        kinds=("y/x", "1", "y"),
        fn=fn,
        guess=guess,
        x_dimensions=("power",),
        y_dimensions=("frequency",),
        bounds={"efficiency": (0.0, 1.0)},
        derived={"brightness_x_efficiency": (lambda v: v["brightness"] * v["efficiency"], "y/x")},
        confounded={
            frozenset({"brightness", "efficiency"}): (
                "brightness and total efficiency cannot be separated from singles alone: only "
                "their product is constrained; fix the efficiency from an independent "
                "calibration (e.g. the coincidence-to-singles ratio)"
            )
        },
        description="R = brightness * efficiency * P + dark",
    )


MODELS: dict[str, Callable[[], FitModel]] = {
    "Polarization fringe, analyser angle (cos 2x)": lambda: fringe(2),
    "Polarization fringe, half-wave plate angle (cos 4x)": lambda: fringe(4),
    "Coincidence peak, Gaussian": lambda: peak("gaussian"),
    "Coincidence peak, Gaussian + exponential tail": lambda: peak("gaussian_tail"),
    "HOM dip, Gaussian": lambda: hom_dip("gaussian"),
    "HOM dip, sinc": lambda: hom_dip("sinc"),
    "Singles vs pump power (brightness x efficiency)": singles,
}


def record(
    data: MeasuredData,
    result: FitResult | None = None,
    comparison=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable record of a measured-data fit or comparison with its provenance."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        software = version("sequoia")
    except PackageNotFoundError:
        software = "unknown"

    chain = []
    d = data
    while d is not None:
        chain.append(d)
        d = d.original
    base = chain[-1]
    out: dict[str, Any] = {
        "format": "sequoia-measured-comparison",
        "version": 1,
        "software": {"sequoia": software},
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "data": {
            "source": base.source,
            "x_unit": data.x_unit,
            "y_unit": data.y_unit,
            "y_kind": data.y_kind,
            "points": len(data),
            "dropped_rows": [{"row": r, "reason": why} for r, why in base.dropped],
            "operations": list(data.operations),
            "warnings": list(data.warnings),
        },
    }
    if result is not None:
        out["fit"] = result.to_record()
    if comparison is not None:
        out["comparison"] = _finite_or_none(
            {
                "label": comparison.label,
                "chi2": comparison.chi2,
                "dof": comparison.dof,
                "reduced_chi2": comparison.reduced_chi2,
                "weighting": comparison.weighting,
                "details": comparison.details,
                "warnings": comparison.warnings,
                "scale": None if comparison.scale is None else comparison.scale.to_record(),
            }
        )
    if extra:
        out.update(_finite_or_none(extra))
    return out


def table_csv(data: MeasuredData, model_y=None, residuals=None, normalized=None) -> str:
    header = [f"x ({data.x_unit})", f"y ({data.y_unit})"]
    columns = [data.x, data.y]
    if data.sigma is not None:
        header.append(f"sigma ({data.y_unit})")
        columns.append(data.sigma)
    for name, values in (
        (f"model ({data.y_unit})", model_y),
        (f"residual ({data.y_unit})", residuals),
        ("normalized residual", normalized),
    ):
        if values is not None:
            header.append(name)
            columns.append(values)
    lines = [",".join(header)]
    for row in zip(*columns, strict=True):
        lines.append(",".join("" if not math.isfinite(v) else repr(float(v)) for v in row))
    return "\n".join(lines) + "\n"
