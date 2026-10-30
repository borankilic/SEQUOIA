import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sequoia.data import Scalar
from sequoia.engines.cache import Cancelled, RunCache, check_cancel
from sequoia.engines.outcome import SimulationOutcome, compute_outcome
from sequoia.errors import SequoiaError
from sequoia.library.store import Library
from sequoia.setup.io import setup_from_dict, setup_to_dict
from sequoia.setup.model import Setup
from sequoia.units import from_si, to_si, unit_dimension

MAX_POINTS = 10_000
_C = 299_792_458.0


class SweepError(SequoiaError):
    pass


@dataclass(frozen=True)
class Axis:
    """A swept or optimised parameter.

    ``target`` is ``settings:<id>.<name>`` or ``overrides:<id>.<name>`` for an instance or probe,
    or ``link:<id>.<field>`` (``length``, ``delay``, ``connector_loss_db``). ``values`` are in
    ``unit``; an empty unit means a plain number.
    """

    target: str
    values: np.ndarray
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float).ravel())


def linspace_axis(target: str, start: float, stop: float, points: int, unit: str = "") -> Axis:
    return Axis(target, np.linspace(start, stop, int(points)), unit)


def _parse_target(target: str) -> tuple[str, str, str]:
    kind, sep, rest = target.partition(":")
    owner, dot, name = rest.rpartition(".")
    if not sep or not dot or not owner or not name or kind not in ("settings", "overrides", "link"):
        raise SweepError(
            f"cannot parse target {target!r}; use 'settings:<id>.<name>', "
            "'overrides:<id>.<name>' or 'link:<id>.<field>'"
        )
    return kind, owner, name


def _holder(raw: dict[str, Any], kind: str, owner: str) -> dict[str, Any]:
    if kind == "link":
        for item in raw.get("links", []):
            if item["id"] == owner:
                return item
        raise SweepError(f"no link '{owner}'")
    for key in ("instances", "probes"):
        for item in raw.get(key, []):
            if item["id"] == owner:
                return item.setdefault(kind, {})
    raise SweepError(f"no instance or probe '{owner}'")


def _format(value: float, unit: str):
    if not unit or unit == "1":
        return float(value)
    return f"{float(value)!r} {unit}"


def get_value(setup: Setup, target: str, unit: str = "") -> float | None:
    kind, owner, name = _parse_target(target)
    raw = _holder(setup_to_dict(setup), kind, owner)
    if name not in raw:
        return None
    value = raw[name]
    if isinstance(value, Scalar):
        return from_si(value.si, unit) if unit else value.value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            scalar = Scalar.model_validate(value)
        except ValueError:
            return None
        return from_si(scalar.si, unit) if unit else scalar.value
    return None


def check_axis_unit(setup: Setup, target: str, unit: str) -> None:
    kind, owner, name = _parse_target(target)
    value = _holder(setup_to_dict(setup), kind, owner).get(name)
    current = None
    if isinstance(value, Scalar):
        current = value.unit
    elif isinstance(value, str):
        try:
            current = Scalar.model_validate(value).unit
        except ValueError:
            current = None
    given = unit.strip()
    if current is None or current.strip() in ("", "1"):
        return
    if given in ("", "1"):
        raise SweepError(f"{target} needs a unit such as '{current}'")
    if unit_dimension(given) != unit_dimension(current):
        raise SweepError(f"unit '{given}' does not fit {target} (a quantity in '{current}')")


def with_values(setup: Setup, assignments: Sequence[tuple[str, float, str]]) -> Setup:
    raw = copy.deepcopy(setup_to_dict(setup))
    for target, value, unit in assignments:
        kind, owner, name = _parse_target(target)
        _holder(raw, kind, owner)[name] = _format(value, unit)
    return setup_from_dict(raw, source="sweep point")


def _numeric(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)


def _flatten(prefix: str, owner: str, data: Any, out: dict[str, float], depth: int = 0) -> None:
    if not isinstance(data, dict) or depth > 2:
        return
    for key, value in data.items():
        if _numeric(value):
            out[f"{prefix}:{owner}.{key}"] = float(value)
        elif isinstance(value, dict):
            _flatten(prefix, f"{owner}.{key}", value, out, depth + 1)


def metrics_of(outcome: SimulationOutcome) -> dict[str, float]:
    """Every scalar result of a run under a flat name such as ``link:f1.power_W``."""
    out: dict[str, float] = {}
    classical = outcome.classical
    if classical is not None:
        for link_id, state in classical.links.items():
            if state.end is not None:
                out[f"link:{link_id}.power_W"] = state.end.power
                out[f"link:{link_id}.dop"] = state.end.dop
                if state.end.mode is not None:
                    out[f"link:{link_id}.w_x_m"] = state.end.mode.w_x
                    out[f"link:{link_id}.w_y_m"] = state.end.mode.w_y
            for key, value in state.info.items():
                if _numeric(value):
                    out[f"link:{link_id}.{key}"] = float(value)
        for inst_id, values in classical.readouts.items():
            _flatten("readout", inst_id, values, out)
        for probe_id, values in classical.probes.items():
            _flatten("probe", probe_id, values, out)
    rates = outcome.rates
    if rates is not None:
        for name, values in rates.channels.items():
            _flatten("channel", name, values, out)
        for name, values in rates.detectors.items():
            _flatten("detector", name, values, out)
        for name, values in rates.sources.items():
            _flatten("source", name, values, out)
        for (a, b), values in rates.pairs.items():
            _flatten("pair", f"{a}~{b}", values, out)
        for name, values in rates.analyses.items():
            _flatten("analysis", name, values, out)
        for name, values in rates.links.items():
            _flatten("photons", name, values, out)
    return out


FOCK_PREFIX = "fock:"


def fock_metrics(
    setup: Setup,
    library: Library,
    *,
    detectors_mode: str = "setup",
    n_max: int | None = None,
    duration: float = 1.0,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, float]:
    """Photon-number observables of the drawn circuit (the Photon counts calculation):
    ``fock:P(d1=n1,d2=n2)`` per detector state, ``fock:P_click(d)`` per detector, the resolved
    probability and the omitted input weight. Probabilities are per detection window."""
    from sequoia.fock.from_setup import build_experiment_from_setup, run_setup_experiment

    se = build_experiment_from_setup(
        setup,
        library,
        n_max=n_max,
        duration=duration,
        detectors_mode=detectors_mode,
        should_cancel=should_cancel,
    )
    res = run_setup_experiment(se, should_cancel=should_cancel)
    ids = res.detector_ids
    out: dict[str, float] = {}
    for state, p in res.detected.items():
        label = ",".join(f"{d}={n}" for d, n in zip(ids, state, strict=True))
        out[f"{FOCK_PREFIX}P({label})"] = p
    for k, d in enumerate(ids):
        out[f"{FOCK_PREFIX}P_click({d})"] = sum(p for s, p in res.detected.items() if s[k] > 0)
    out[f"{FOCK_PREFIX}resolved_mass"] = res.mass
    out[f"{FOCK_PREFIX}omitted_weight"] = res.omitted_weight
    return out


def _check_fock_axes(axes: Sequence[Axis]) -> None:
    for axis in axes:
        kind, owner, name = _parse_target(axis.target)
        if kind == "link" or "delay" in name.lower():
            raise SweepError(
                f"{axis.target}: the photon-number (Fock) model has no temporal or spectral "
                "structure, so a delay or path-length scan would only change a path phase, not "
                "the photons' overlap; scan delays with the rate-engine analyses (HOM dip) or "
                "sweep a phase or setting instead"
            )


@dataclass
class SweepResult:
    axes: list[Axis]
    data: dict[str, np.ndarray]
    failures: dict[tuple[int, ...], str] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a.values) for a in self.axes)

    def column(self, name: str) -> np.ndarray:
        return self.data[name]

    def to_csv(self, names: Sequence[str] | None = None) -> str:
        names = list(names) if names is not None else list(self.data)
        header = [f"{a.target} [{a.unit}]" if a.unit else a.target for a in self.axes] + names
        rows = [",".join(header)]
        for index in np.ndindex(*self.shape):
            cells = [repr(float(a.values[i])) for a, i in zip(self.axes, index, strict=True)]
            cells += [repr(float(self.data[n][index])) for n in names]
            rows.append(",".join(cells))
        return "\n".join(rows) + "\n"


def evaluate(
    setup: Setup,
    library: Library,
    *,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SimulationOutcome:
    return compute_outcome(setup, library, cache, should_cancel)


def run_sweep(
    setup: Setup,
    library: Library,
    axes: Sequence[Axis],
    outputs: Sequence[str] | None = None,
    *,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fock: Mapping[str, Any] | None = None,
) -> SweepResult:
    """Run the simulation over a 1D or 2D grid of parameter values.

    ``outputs`` names (see ``metrics_of``) select the columns kept; ``None`` keeps every metric.
    Points that cannot be simulated, or lie outside the physical bounds of a parameter (see
    ``Candidate``; ``check_bounds`` refuses such an axis up front), are NaN and listed in
    ``failures``. Photon-number observables (``fock:...``, see ``fock_metrics``) are computed
    when ``fock`` (keyword arguments of ``fock_metrics``) is given or an output starts with
    ``fock:``; a state absent at a point has probability 0 there.
    """
    axes = list(axes)
    if not 1 <= len(axes) <= 2:
        raise SweepError("a sweep needs one or two axes")
    for axis in axes:
        _parse_target(axis.target)
        if len(axis.values) == 0:
            raise SweepError(f"axis {axis.target} has no values")
        if not np.all(np.isfinite(axis.values)):
            raise SweepError(f"axis {axis.target} has non-finite values")
    try:
        for a in axes:
            check_axis_unit(setup, a.target, a.unit)
        with_values(setup, [(a.target, float(a.values[0]), a.unit) for a in axes])
    except Exception as exc:
        if isinstance(exc, Cancelled):
            raise
        raise SweepError(f"{axes[0].target}: {str(exc).splitlines()[0]}") from exc
    shape = tuple(len(a.values) for a in axes)
    total = math.prod(shape)
    if total > MAX_POINTS:
        raise SweepError(f"sweep has {total} points; the limit is {MAX_POINTS}")
    cache = cache if cache is not None else RunCache()
    data: dict[str, np.ndarray] = {}
    failures: dict[tuple[int, ...], str] = {}
    wanted = list(outputs) if outputs is not None else None
    use_fock = fock is not None or any(n.startswith(FOCK_PREFIX) for n in wanted or [])
    if use_fock:
        _check_fock_axes(axes)
    fock_options = dict(fock or {})
    by_target = {c.target: c for c in candidate_targets(setup, library)}
    for count, index in enumerate(np.ndindex(*shape)):
        check_cancel(should_cancel)
        assignments = [
            (a.target, float(a.values[i]), a.unit) for a, i in zip(axes, index, strict=True)
        ]
        values: dict[str, float] = {}
        refused = next(
            (
                f"{target}: {problem}"
                for target, value, unit in assignments
                if target in by_target
                for problem in [by_target[target].violation([value], unit)]
                if problem
            ),
            None,
        )
        try:
            if refused:
                raise SweepError(refused)
            point = with_values(setup, assignments)
            outcome = evaluate(point, library, cache=cache, should_cancel=should_cancel)
            if outcome.classical is None or outcome.has_errors:
                first = next((i for i in outcome.issues if i.severity == "error"), None)
                failures[index] = first.message if first else "simulation failed"
            else:
                values = metrics_of(outcome)
            if use_fock and index not in failures:
                values.update(
                    fock_metrics(point, library, should_cancel=should_cancel, **fock_options)
                )
        except Cancelled:
            raise
        except SequoiaError as exc:
            failures[index] = str(exc)
        for name in wanted if wanted is not None else values:
            column = data.get(name)
            if column is None:
                column = data[name] = np.full(shape, np.nan)
            column[index] = values.get(name, np.nan)
        if progress is not None:
            progress(count + 1, total)
    if use_fock:
        for name, column in data.items():
            if name.startswith(FOCK_PREFIX):
                for index in np.ndindex(*shape):
                    if np.isnan(column[index]) and index not in failures:
                        column[index] = 0.0
    return SweepResult(axes, data, failures)


@dataclass(frozen=True)
class Candidate:
    """A sweepable parameter. ``lower``/``upper`` are its physical bounds in ``unit`` (None:
    unbounded; ``open_lower``: the lower bound itself is excluded); a ``period`` marks an angle
    whose values repeat, so its bounds are one period and are not enforced. ``default`` is the
    suggested sweep (start, stop, points, log-spaced)."""

    target: str
    unit: str
    current: float | None
    lower: float | None = None
    upper: float | None = None
    period: float | None = None
    open_lower: bool = False
    default: tuple[float, float, int, bool] | None = None

    def bounds_text(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        lo, hi = self.lower, self.upper
        if lo is not None and hi is not None:
            text = f"{lo:g} to {hi:g}{unit}"
        elif lo is not None:
            text = f"{'>' if self.open_lower else '>='} {lo:g}{unit}"
        elif hi is not None:
            text = f"<= {hi:g}{unit}"
        else:
            text = "unbounded"
        if self.period is not None:
            text += f", period {self.period:g}{unit}"
        return text

    def violation(self, values: Sequence[float], unit: str = "") -> str:
        if self.period is not None or (self.lower is None and self.upper is None):
            return ""
        values = np.asarray(values, dtype=float)
        unit = unit.strip()
        if unit not in ("", "1") and self.unit and unit != self.unit:
            values = np.asarray(from_si(to_si(values, unit), self.unit), dtype=float)
        for v in values:
            low = self.lower is not None and (
                v <= self.lower if self.open_lower else v < self.lower - 1e-12 * abs(self.lower)
            )
            high = self.upper is not None and v > self.upper + 1e-12 * max(abs(self.upper), 1)
            if low or high:
                shown = f"{v:g} {self.unit}".strip()
                return (
                    f"{shown} is outside the physical range ({self.bounds_text()}); "
                    "change the sweep limits"
                )
        return ""


_FRACTIONS = {
    "efficiency",
    "transmission",
    "transmittance",
    "transmittance_p",
    "transmittance_s",
    "reflectance",
    "reflectance_s",
    "reflectance_p",
    "reflectance_v",
    "split_ratio",
    "pump_transmission",
    "absorption",
    "afterpulse_probability",
    "jitter_tail_fraction",
    "visibility",
}
_AT_LEAST_ONE = {
    "m_squared",
    "extinction_ratio",
    "polarization_extinction_ratio",
    "polarization_extinction",
}
_NON_NEGATIVE = {
    "optical_density",
    "attenuation_db_per_km",
    "connector_loss_db",
    "responsivity",
    "brightness",
    "pair_rate",
    "dark_count_rate",
    "dark_current",
    "retardance",
    "photons_per_pulse",
    "linewidth",
}
_SIGNED = {"focal_length", "waist_offset", "defocus", "lateral_offset", "delay", "d_eff"}
_POSITIVE = ("wavelength", "waist", "repetition_rate", "pulse_width", "bandwidth", "window")
_HALF_TURN = {"angle", "polarization_angle", "polarization_axis"}
_LAB_TEMPERATURE_K = (1.0, 573.15)


def _si_bounds(name: str, unit: str, part_type: str | None):
    """(lower, upper, period, open_lower) in SI units (plain numbers when dimensionless)."""
    dimension = unit_dimension(unit) if unit else "dimensionless"
    if dimension == "angle":
        if name == "angle_of_incidence":
            return -math.pi / 2, math.pi / 2, None, False
        if name in _HALF_TURN:
            return 0.0, math.pi, math.pi, False
        return 0.0, 2 * math.pi, 2 * math.pi, False
    if dimension == "temperature":
        if part_type == "SPDCCrystal":
            from sequoia.components.nonlinear import _CALIBRATED_TEMPERATURE_C

            low, high = _CALIBRATED_TEMPERATURE_C
            return low + 273.15, high + 273.15, None, False
        return *_LAB_TEMPERATURE_K, None, False
    if name in _FRACTIONS:
        return 0.0, 1.0, None, False
    if name in _AT_LEAST_ONE:
        return 1.0, None, None, False
    if name in _NON_NEGATIVE:
        return 0.0, None, None, False
    if name in _SIGNED or dimension in ("dimensionless", "other", "nonlinear_coefficient"):
        return None, None, None, False
    if any(key in name for key in _POSITIVE) and "offset" not in name:
        return 0.0, None, None, True
    return 0.0, None, None, False


def _in_unit(si: float | None, unit: str) -> float | None:
    if si is None:
        return None
    return float(from_si(si, unit)) if unit else float(si)


def coherence_length(outcome: SimulationOutcome | None, link_id: str) -> float | None:
    """Coherence length (m) of the pair photons on a link, c times the Mandel coherence time
    integral(S^2) / integral(S)^2 of their spectrum; None without a photon spectrum."""
    rates = getattr(outcome, "rates", None)
    entry = rates.links.get(link_id) if rates is not None else None
    quantum = entry.get("quantum") if isinstance(entry, dict) else None
    lengths = []
    for spectrum in (quantum or {}).values():
        try:
            nu = np.asarray(spectrum["nu"], dtype=float)
            s = np.real(np.trace(np.asarray(spectrum["coherency"]), axis1=1, axis2=2))
        except (KeyError, TypeError, ValueError):
            continue
        norm = np.trapezoid(s, nu) if nu.size > 2 else 0.0
        if norm > 0:
            lengths.append(_C * float(np.trapezoid(s * s, nu)) / norm**2)
    return max(lengths) if lengths else None


def delay_links(setup: Setup, outcome: SimulationOutcome | None = None) -> list[str]:
    """Free-space links that act as a delay line: named '...delay...', else (with an outcome)
    the free-space links that carry only one photon of the pairs."""
    named = [link.id for link in setup.links if link.kind == "free_space" and "delay" in link.id]
    if named or outcome is None or outcome.rates is None:
        return named
    out = []
    for link in setup.links:
        entry = outcome.rates.links.get(link.id) if link.kind == "free_space" else None
        sources = entry.get("by_source", {}) if isinstance(entry, dict) else {}
        kinds = {k for photons in sources.values() for k in photons}
        if len(kinds) == 1:
            out.append(link.id)
    return out


def _half_wave(part) -> bool:
    retardance = getattr(getattr(part, "params", None), "retardance", None)
    return (
        getattr(part, "type", None) == "WavePlate"
        and isinstance(retardance, Scalar)
        and abs(retardance.si % 1.0 - 0.5) < 1e-6
    )


def _default(
    name: str,
    unit: str,
    current: float | None,
    bounds,
    part,
    delay: float | None,
) -> tuple[float, float, int, bool]:
    lower, upper, period, open_lower = bounds
    if period is not None:
        if name == "angle" and _half_wave(part):
            return 0.0, upper / 2, 46, False
        half_turn = math.isclose(period, _in_unit(math.pi, unit) or math.pi)
        return 0.0, upper, 91 if half_turn else 73, False
    if delay is not None and current is not None:
        start = current - delay if lower is None else max(current - delay, lower)
        return float(start), float(current + delay), 41, False
    if current is not None and current > 0 and name in ("power", "pump_power"):
        return current / 10, current * 10, 21, True
    if name in _FRACTIONS:
        return 0.0, 1.0, 21, False
    if unit and unit_dimension(unit) == "temperature" and current is not None:
        mid = to_si(current, unit)
        start, stop = _in_unit(mid - 5.0, unit), _in_unit(mid + 5.0, unit)
    elif current:
        start, stop = current - abs(current) * 0.2, current + abs(current) * 0.2
    elif lower is not None and upper is not None:
        start, stop = lower, upper
    else:
        cur = current or 0.0
        dimension = unit_dimension(unit) if unit else ""
        span = {"length": 1e-3, "time": 1e-9}.get(dimension)
        span = _in_unit(span, unit) if span is not None else 1.0
        start, stop = (cur, cur + span) if lower is not None else (cur - span, cur + span)
    if lower is not None and start < lower:
        start = lower
    if upper is not None and stop > upper:
        stop = upper
    if open_lower and lower is not None and start <= lower:
        start = lower + (stop - lower) / 20
    return start, stop, 21, False


def _candidate(
    target: str, unit: str, current, part, name: str, delay: float | None = None
) -> Candidate:
    lo, hi, period, open_lower = _si_bounds(name, unit, part.type if part is not None else None)
    bounds = (_in_unit(lo, unit), _in_unit(hi, unit), _in_unit(period, unit), open_lower)
    return Candidate(
        target,
        unit,
        current,
        *bounds,
        default=_default(name, unit, current, bounds, part, delay),
    )


def _unit_of(scalar: Scalar) -> str:
    return "" if scalar.unit.strip() in ("", "1") else scalar.unit


def candidate_targets(
    setup: Setup, library: Library, outcome: SimulationOutcome | None = None
) -> list[Candidate]:
    """Numeric parameters of ``setup`` that can be swept or optimised, with physical bounds
    and a default sweep. With ``outcome``, a delay link's default spans +/- 3 coherence lengths
    of its photons (else +/- 2 mm)."""
    from sequoia.components.base import get_component

    out: list[Candidate] = []
    raw = setup_to_dict(setup)
    placed = [(i, "instances") for i in setup.instances] + [(p, "probes") for p in setup.probes]
    for item, _group in placed:
        try:
            part = library.load_part(item.part)
            cls = get_component(part.type)
        except Exception:
            continue
        entry = next(
            (r for key in ("instances", "probes") for r in raw.get(key, []) if r["id"] == item.id),
            {},
        )
        given = entry.get("settings", {})
        for name, info in cls.Settings.model_fields.items():
            default = info.default
            target = f"settings:{item.id}.{name}"
            if isinstance(default, Scalar) and math.isfinite(default.value):
                unit = _unit_of(default)
                current = get_value(setup, target, unit) if name in given else default.value
                out.append(_candidate(target, unit, current, part, name))
            elif name in given:
                value = get_value(setup, target)
                if value is not None:
                    out.append(_candidate(target, "", value, part, name))
        for name in type(part.params).model_fields:
            value = getattr(part.params, name, None)
            if name in entry.get("overrides", {}):
                value = entry["overrides"][name]
                if isinstance(value, str):
                    try:
                        value = Scalar.model_validate(value)
                    except ValueError:
                        continue
                elif _numeric(value):
                    value = Scalar(value=float(value))
            if isinstance(value, Scalar) and math.isfinite(value.value):
                unit = _unit_of(value)
                out.append(_candidate(f"overrides:{item.id}.{name}", unit, value.value, part, name))
    delays = set(delay_links(setup, outcome))
    for link in setup.links:
        if link.kind == "cable":
            target = f"link:{link.id}.delay"
            out.append(_candidate(target, "ns", get_value(setup, target, "ns"), None, "delay"))
            continue
        if link.length is not None:
            target = f"link:{link.id}.length"
            delay = None
            if link.id in delays:
                length = coherence_length(outcome, link.id)
                delay = 3 * length * 1e3 if length is not None else 2.0
            out.append(
                _candidate(target, "mm", get_value(setup, target, "mm"), None, "length", delay)
            )
        if link.kind == "fiber":
            out.append(
                _candidate(
                    f"link:{link.id}.connector_loss_db",
                    "",
                    link.connector_loss_db,
                    None,
                    "connector_loss_db",
                )
            )
    return out


def check_bounds(setup: Setup, library: Library, axes: Sequence[Axis]) -> None:
    """Refuse axis values outside the physical bounds of their parameter (see ``Candidate``)."""
    by_target = {c.target: c for c in candidate_targets(setup, library)}
    for axis in axes:
        candidate = by_target.get(axis.target)
        problem = candidate.violation(axis.values, axis.unit) if candidate is not None else ""
        if problem:
            raise SweepError(f"{axis.target}: {problem}")


@dataclass(frozen=True)
class SweepPlan:
    task: str
    target: str
    unit: str
    start: float
    stop: float
    points: int
    log: bool
    output: str

    def axis(self) -> Axis:
        space = np.geomspace if self.log else np.linspace
        return Axis(self.target, space(self.start, self.stop, self.points), self.unit)


def default_plan(
    setup: Setup, library: Library, outcome: SimulationOutcome | None = None
) -> SweepPlan | None:
    """The recommended sweep of a recognised research setup: heralded g2(0) against pump
    power, a HOM dip against the delay line, a polarization-correlation fringe against the
    first half-wave plate, Malus' law against a wave plate or polarizer angle. None otherwise."""
    if outcome is None:
        try:
            outcome = evaluate(setup, library)
        except SequoiaError:
            return None
    metrics = metrics_of(outcome) if outcome.classical is not None else {}
    by_target = {c.target: c for c in candidate_targets(setup, library, outcome)}
    parts = {}
    for item in [*setup.instances, *setup.probes]:
        try:
            parts[item.id] = library.load_part(item.part)
        except SequoiaError:
            continue

    def of_type(*types):
        return [i for i, p in parts.items() if p.type in types]

    def plan(task, target, output):
        candidate = by_target.get(target)
        if candidate is None or candidate.default is None or output not in metrics:
            return None
        return SweepPlan(task, target, candidate.unit, *candidate.default, output)

    def first(*plans):
        return next((p for p in plans if p is not None), None)

    analyses = [(a.get("name"), a.get("type")) for a in setup.analyses]
    coincidences = [n for n, t in analyses if t == "coincidences"]
    half_wave = [i for i in of_type("WavePlate") if _half_wave(parts[i])]
    pumps = [f"settings:{i}.pump_power" for i in of_type("EntangledPairSource")]
    if of_type("SPDCCrystal"):
        pumps += [f"{k}:{i}.power" for i in of_type("Laser") for k in ("settings", "overrides")]
    for name in (n for n, t in analyses if t == "heralded_g2"):
        found = first(
            *(
                plan("heralded g2(0) vs pump power", t, f"analysis:{name}.g2_heralded")
                for t in pumps
            ),
            *(
                plan("CAR vs pump power", t, f"analysis:{c}.car")
                for t in pumps
                for c in coincidences
            ),
        )
        if found:
            return found
    for name in coincidences:
        if f"analysis:{name}.two_photon_visibility" in metrics:
            found = first(
                *(
                    plan("HOM dip", f"link:{link}.length", f"analysis:{name}.true_hz")
                    for link in delay_links(setup, outcome)
                )
            )
            if found:
                return found
    if coincidences:
        found = first(
            *(
                plan(
                    "polarization correlation fringe",
                    f"settings:{i}.angle",
                    f"analysis:{c}.true_hz",
                )
                for i in half_wave
                for c in coincidences
            )
        )
        if found:
            return found
    found = first(
        *(plan("CAR vs pump power", t, f"analysis:{c}.car") for c in coincidences for t in pumps)
    )
    if found:
        return found
    couplers = set(of_type("FiberCoupler"))
    fibers = {link.src.split(".")[0]: link.id for link in setup.links if link.kind == "fiber"}
    found = first(
        *(
            plan(
                "fiber coupling",
                f"link:{link.id}.length",
                f"link:{fibers.get(link.dst.split('.')[0])}.coupling_efficiency",
            )
            for link in setup.links
            if link.kind == "free_space" and link.dst.split(".")[0] in couplers
        )
    )
    if found:
        return found
    if of_type("Polarizer", "PBS"):
        readouts = sorted(n for n in metrics if n.startswith("probe:") and n.endswith(".power_W"))
        readouts += sorted(
            n for n in metrics if n.startswith("readout:") and n.endswith(".incident_power_W")
        )
        angles = [f"settings:{i}.angle" for i in half_wave + of_type("Polarizer")]
        return first(*(plan("Malus' law", t, r) for r in readouts[:1] for t in angles))
    return None
