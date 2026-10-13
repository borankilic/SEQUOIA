import copy
import math
from collections.abc import Callable, Sequence
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
from sequoia.units import from_si, unit_dimension

MAX_POINTS = 10_000


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
) -> SweepResult:
    """Run the simulation over a 1D or 2D grid of parameter values.

    ``outputs`` names (see ``metrics_of``) select the columns kept; ``None`` keeps every metric.
    Points that cannot be simulated are NaN and listed in ``failures``.
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
    for count, index in enumerate(np.ndindex(*shape)):
        check_cancel(should_cancel)
        assignments = [
            (a.target, float(a.values[i]), a.unit) for a, i in zip(axes, index, strict=True)
        ]
        values: dict[str, float] = {}
        try:
            point = with_values(setup, assignments)
            outcome = evaluate(point, library, cache=cache, should_cancel=should_cancel)
            if outcome.classical is None or outcome.has_errors:
                first = next((i for i in outcome.issues if i.severity == "error"), None)
                failures[index] = first.message if first else "simulation failed"
            else:
                values = metrics_of(outcome)
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
    return SweepResult(axes, data, failures)


@dataclass(frozen=True)
class Candidate:
    target: str
    unit: str
    current: float | None


def _unit_of(scalar: Scalar) -> str:
    return "" if scalar.unit.strip() in ("", "1") else scalar.unit


def candidate_targets(setup: Setup, library: Library) -> list[Candidate]:
    """Numeric parameters of ``setup`` that can be swept or optimised."""
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
            if isinstance(default, Scalar) and math.isfinite(default.value):
                out.append(
                    Candidate(
                        f"settings:{item.id}.{name}",
                        _unit_of(default),
                        get_value(setup, f"settings:{item.id}.{name}", _unit_of(default))
                        if name in given
                        else default.value,
                    )
                )
            elif name in given:
                value = get_value(setup, f"settings:{item.id}.{name}")
                if value is not None:
                    out.append(Candidate(f"settings:{item.id}.{name}", "", value))
        for name in type(part.params).model_fields:
            value = getattr(part.params, name, None)
            if name in entry.get("overrides", {}):
                value = entry["overrides"][name]
                if isinstance(value, str):
                    try:
                        value = Scalar.model_validate(value)
                    except ValueError:
                        continue
            if isinstance(value, Scalar) and math.isfinite(value.value):
                unit = _unit_of(value)
                out.append(Candidate(f"overrides:{item.id}.{name}", unit, value.value))
    for link in setup.links:
        if link.kind == "cable":
            out.append(
                Candidate(
                    f"link:{link.id}.delay", "ns", get_value(setup, f"link:{link.id}.delay", "ns")
                )
            )
            continue
        if link.length is not None:
            out.append(
                Candidate(
                    f"link:{link.id}.length", "mm", get_value(setup, f"link:{link.id}.length", "mm")
                )
            )
        if link.kind == "fiber":
            out.append(Candidate(f"link:{link.id}.connector_loss_db", "", link.connector_loss_db))
    return out
