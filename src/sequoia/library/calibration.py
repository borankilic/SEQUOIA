import csv
import datetime
import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from sequoia.data import Curve, Provenance, Scalar, format_axis, parse_axis
from sequoia.errors import PartError, SequoiaError, format_validation_error
from sequoia.units import from_si, unit_dimension
from sequoia.yamlio import dump_yaml, load_yaml

if TYPE_CHECKING:
    from sequoia.library.store import Library, Part
    from sequoia.setup.model import Setup

CALIBRATION_FILE = "calibration.yaml"
CalibrationSource = Literal["vendor nominal", "measured", "model"]

RANK = {"measured": 3, "raw-data": 3, "datasheet": 2, "vendor nominal": 2, "model": 1}
RANK_UNTRACED = 2
_PROVENANCE_SOURCE = {"measured": "measured", "vendor nominal": "datasheet", "model": "model"}
OFFSETS = {"zero_offset": "angle", "timing_offset": "time"}

CRITICAL = {
    "Laser": ("power", "wavelength"),
    "FockSource": ("photons_per_pulse",),
    "EntangledPairSource": ("brightness", "pair_rate", "visibility"),
    "SPDCCrystal": ("brightness", "pump_transmission"),
    "SPAD": ("efficiency", "dark_count_rate", "dead_time", "jitter_fwhm"),
    "SNSPD": ("efficiency", "dark_count_rate", "dead_time", "jitter_fwhm"),
    "Photodiode": ("responsivity", "dark_current"),
    "NDFilter": ("optical_density",),
    "SpectralFilter": ("transmission",),
    "FiberCoupler": ("transmission",),
    "FiberBeamSplitter": ("split_ratio", "transmission"),
    "BeamSplitter": ("reflectance",),
    "DichroicMirror": ("transmittance", "reflectance"),
    "Lens": ("transmission",),
    "Objective": ("transmission",),
    "Mirror": ("reflectance",),
    "PBS": ("transmittance_p", "reflectance_s"),
    "Polarizer": ("extinction_ratio", "transmission"),
    "WavePlate": ("retardance", "transmission"),
    "PhaseShifter": ("transmission",),
    "TimeTagger": ("jitter_fwhm",),
    "OpticalFiber": ("attenuation_db_per_km",),
}

SUPPORTED = {
    "efficiency",
    "transmission",
    "transmittance",
    "transmittance_p",
    "transmittance_s",
    "reflectance",
    "reflectance_s",
    "reflectance_p",
    "split_ratio",
    "optical_density",
    "pump_transmission",
    "attenuation_db_per_km",
    "connector_loss_db",
    "responsivity",
    "brightness",
    "power",
    "pump_power",
    "angle",
    "delay",
    "zero_offset",
    "timing_offset",
}
_UNIT_INTERVAL = {
    "efficiency",
    "transmission",
    "transmittance",
    "transmittance_p",
    "transmittance_s",
    "reflectance",
    "reflectance_s",
    "reflectance_p",
    "split_ratio",
    "pump_transmission",
}
_NON_NEGATIVE = {
    "optical_density",
    "attenuation_db_per_km",
    "connector_loss_db",
    "responsivity",
    "brightness",
    "power",
    "pump_power",
}


class CalibrationRecord(BaseModel):
    """A calibration of one part parameter, kept apart from the part's nominal values.

    Either ``value`` (with ``std`` in a compatible unit) or ``curve`` (with a per-point
    ``curve_std`` or a full ``covariance`` of its y values).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    parameter: str
    source: CalibrationSource = "measured"
    value: Scalar | None = None
    std: Scalar | None = None
    curve: Curve | None = None
    curve_std: list[float] | None = None
    covariance: list[list[float]] | None = None
    date: datetime.date | None = None
    conditions: str | None = None
    local_id: str | None = None
    measured_by: str | None = None
    method: str | None = None
    validity: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "CalibrationRecord":
        if (self.value is None) == (self.curve is None):
            raise ValueError("a calibration record needs either a 'value' or a 'curve'")
        if self.std is not None:
            if self.value is None:
                raise ValueError("'std' goes with a 'value'; a curve uses curve_std/covariance")
            if not math.isfinite(self.std.value) or self.std.value < 0:
                raise ValueError("std must be finite and non-negative")
            if unit_dimension(self.std.unit) != unit_dimension(self.value.unit):
                raise ValueError(
                    f"std unit {self.std.unit!r} does not match value unit {self.value.unit!r}"
                )
        n = len(self.curve.x) if self.curve is not None else 0
        if self.curve_std is not None and len(self.curve_std) != n:
            raise ValueError(f"curve_std has {len(self.curve_std)} values, the curve has {n}")
        if self.covariance is not None:
            cov = np.asarray(self.covariance, dtype=float)
            if cov.shape != (n, n):
                raise ValueError(f"covariance must be {n} x {n}, got {cov.shape}")
            if not np.allclose(cov, cov.T, rtol=1e-9, atol=0.0):
                raise ValueError("covariance must be symmetric")
            if np.any(np.diag(cov) < 0):
                raise ValueError("covariance has a negative variance")
        return self

    @property
    def rank(self) -> int:
        return RANK[self.source]

    @property
    def std_in_value_unit(self) -> float | None:
        if self.std is None or self.value is None:
            return None
        if self.value.unit.strip() in ("", "1"):
            return self.std.si
        return float(from_si(self.std.si, self.value.unit))

    def applied_value(self) -> Scalar | Curve:
        provenance = Provenance(
            source=_PROVENANCE_SOURCE[self.source],
            date=self.date,
            by=self.measured_by,
            note=f"calibration record {self.id}" + (f": {self.method}" if self.method else ""),
            reference=CALIBRATION_FILE,
        )
        if self.value is not None:
            return self.value.model_copy(update={"provenance": provenance})
        return self.curve.model_copy(update={"provenance": provenance})

    def summary(self) -> str:
        if self.value is not None:
            text = _text(self.value)
            if self.std is not None:
                text += f" +/- {_text(self.std)}"
            return text
        c = self.curve
        extra = " with covariance" if self.covariance else " with std" if self.curve_std else ""
        return f"curve {c.y_label} vs {c.x_label}, {len(c.x)} points{extra}"


def _read_rows(path: Path) -> list[list[float]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    out = []
    for row in rows:
        try:
            out.append([float(c) for c in row if c.strip()])
        except ValueError:
            continue
    return [r for r in out if r]


def _inside(directory: Path, name: str, where: str) -> Path:
    path = (directory / name).resolve()
    if directory.resolve() not in path.parents:
        raise PartError(f"{where}: file {name!r} must be inside the part folder")
    if not path.is_file():
        raise PartError(f"{where}: file {name!r} not found in the part folder")
    return path


def load_calibration(directory: Path, ref: str = "") -> list[CalibrationRecord]:
    path = Path(directory) / CALIBRATION_FILE
    if not path.is_file():
        return []
    try:
        raw = load_yaml(path)
    except Exception as exc:
        raise PartError(f"{ref}: cannot read {CALIBRATION_FILE}: {exc}") from exc
    entries = (raw or {}).get("records") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise PartError(f"{ref}: {CALIBRATION_FILE} must contain a 'records' list")
    records = []
    seen = set()
    for k, entry in enumerate(entries):
        where = f"{ref}: {CALIBRATION_FILE} record {k + 1}"
        if not isinstance(entry, dict):
            raise PartError(f"{where} must be a mapping")
        entry = dict(entry)
        if "curve" in entry:
            filename = str(entry["curve"])
            rows = _read_rows(_inside(Path(directory), filename, where))
            try:
                (x_label, x_unit), (y_label, y_unit) = (
                    parse_axis(entry.pop("x")),
                    parse_axis(entry.pop("y")),
                )
            except KeyError as exc:
                raise PartError(f"{where}: a curve needs 'x' and 'y' axes ({exc})") from exc
            except ValueError as exc:
                raise PartError(f"{where}: {exc}") from exc
            if any(len(r) < 2 for r in rows):
                raise PartError(f"{where}: {filename} needs x and y columns")
            try:
                entry["curve"] = Curve(
                    x=[r[0] for r in rows],
                    y=[r[1] for r in rows],
                    x_label=x_label,
                    x_unit=x_unit,
                    y_label=y_label,
                    y_unit=y_unit,
                    file=filename,
                )
            except ValidationError as exc:
                raise PartError(f"{where}: {format_validation_error(exc)}") from exc
            if all(len(r) >= 3 for r in rows) and "curve_std" not in entry:
                entry["curve_std"] = [r[2] for r in rows]
        if isinstance(entry.get("covariance"), str):
            entry["covariance"] = _read_rows(_inside(Path(directory), entry["covariance"], where))
        try:
            record = CalibrationRecord.model_validate(entry)
        except ValidationError as exc:
            raise PartError(f"{where}: {format_validation_error(exc)}") from exc
        if record.id in seen:
            raise PartError(f"{where}: duplicate record id {record.id!r}")
        seen.add(record.id)
        records.append(record)
    return records


def save_calibration(directory: Path, records: Sequence[CalibrationRecord]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for record in records:
        entry = record.model_dump(
            mode="json", exclude_none=True, exclude={"curve", "curve_std", "covariance"}
        )
        for key in ("value", "std"):
            scalar = getattr(record, key)
            if scalar is not None:
                entry[key] = scalar.model_dump(mode="json")
        if record.curve is not None:
            c = record.curve
            filename = c.file or f"{record.id}.csv"
            header = [format_axis(c.x_label, c.x_unit), format_axis(c.y_label, c.y_unit)]
            columns = [c.x, c.y]
            if record.curve_std is not None:
                header.append(f"std ({c.y_unit})")
                columns.append(record.curve_std)
            with (directory / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in zip(*columns, strict=True):
                    writer.writerow([repr(float(v)) for v in row])
            entry["curve"] = filename
            entry["x"], entry["y"] = header[0], header[1]
        if record.covariance is not None:
            filename = f"{record.id}_covariance.csv"
            with (directory / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in record.covariance:
                    writer.writerow([repr(float(v)) for v in row])
            entry["covariance"] = filename
        entries.append(entry)
    path = directory / CALIBRATION_FILE
    dump_yaml({"records": entries}, path)
    return path


def provenance_rank(value: Any) -> int:
    if isinstance(value, dict):
        provenance = value.get("provenance")
        source = provenance.get("source") if isinstance(provenance, dict) else None
    else:
        source = getattr(getattr(value, "provenance", None), "source", None)
    return RANK_UNTRACED if source is None else RANK.get(source, 0)


def select_records(
    records: Sequence[CalibrationRecord], fields: set[str], ref: str
) -> dict[str, CalibrationRecord]:
    """The record used for each parameter: highest rank (measured > vendor nominal > model),
    then the latest date."""
    best: dict[str, CalibrationRecord] = {}
    for record in records:
        if record.parameter not in fields and record.parameter not in OFFSETS:
            raise PartError(
                f"{ref}: calibration record {record.id!r} is for unknown parameter "
                f"{record.parameter!r}; this part has {sorted(fields)} (or {sorted(OFFSETS)})"
            )
        current = best.get(record.parameter)
        key = (record.rank, record.date or datetime.date.min)
        if current is None or key > (current.rank, current.date or datetime.date.min):
            best[record.parameter] = record
    return best


def fingerprint_files(directory: Path, exclude: set[str] = frozenset()) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        name = path.relative_to(directory).as_posix()
        if name in exclude or path.name.startswith(".") or name.endswith(".tmp"):
            continue
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class ParameterStatus:
    owner: str
    part: str
    parameter: str
    status: str
    value: str
    record: str | None = None
    note: str = ""

    @property
    def placeholder(self) -> bool:
        return self.status in ("model", "ideal default", "untraced")


_STATUS = {
    "measured": "measured",
    "raw-data": "measured",
    "datasheet": "vendor nominal",
    "model": "model",
    "ideal-default": "ideal default",
}


def _text(value: Any) -> str:
    if isinstance(value, Scalar):
        return f"{value.value:g} {value.unit if value.unit not in ('', '1') else ''}".strip()
    if isinstance(value, Curve):
        return f"curve, {len(value.x)} points"
    if isinstance(value, float | int):
        return f"{value:g}"
    return str(value)


def _part_rows(owner: str, part: "Part", overrides: dict[str, Any]) -> list[ParameterStatus]:
    rows = []
    for name in CRITICAL.get(part.type, ()):
        value = getattr(part.params, name, None)
        if value is None:
            continue
        record_id = part.calibrated.get(name)
        record = next((r for r in part.calibration if r.id == record_id), None)
        if name in overrides:
            rows.append(
                ParameterStatus(
                    owner,
                    part.ref,
                    name,
                    "override",
                    str(overrides[name]),
                    note="set in the setup; not traced to a calibration",
                )
            )
        elif record is not None:
            rows.append(
                ParameterStatus(
                    owner,
                    part.ref,
                    name,
                    record.source,
                    record.summary(),
                    record.id,
                    ", ".join(str(v) for v in (record.date, record.local_id) if v),
                )
            )
        else:
            provenance = getattr(value, "provenance", None)
            status = _STATUS.get(provenance.source, "untraced") if provenance else "untraced"
            rows.append(
                ParameterStatus(
                    owner,
                    part.ref,
                    name,
                    status,
                    _text(value),
                    note=(provenance.note or "") if provenance else "no provenance recorded",
                )
            )
    for record in part.calibration:
        if record.parameter in OFFSETS:
            rows.append(
                ParameterStatus(
                    owner,
                    part.ref,
                    record.parameter,
                    record.source,
                    record.summary(),
                    record.id,
                    "used by uncertainty propagation; not applied to the simulation",
                )
            )
    return rows


def placeholder_report(setup: "Setup", library: "Library") -> list[ParameterStatus]:
    """For every instance (and fiber link) of ``setup``: the status of its critical parameters
    (measured, vendor nominal, model, ideal default, untraced or a setup override)."""
    rows: list[ParameterStatus] = []
    for inst in setup.instances:
        try:
            part = library.load_part(inst.part)
        except SequoiaError as exc:
            rows.append(ParameterStatus(inst.id, inst.part, "-", "error", "", note=str(exc)))
            continue
        rows += _part_rows(inst.id, part, inst.overrides)
    for link in setup.links:
        fiber = getattr(link, "fiber", None)
        if fiber is not None:
            try:
                rows += _part_rows(link.id, library.load_part(fiber), {})
            except SequoiaError as exc:
                rows.append(ParameterStatus(link.id, fiber, "-", "error", "", note=str(exc)))
    return rows


class UncertaintyError(SequoiaError):
    pass


@dataclass(frozen=True)
class Uncertain:
    """A parameter drawn from a normal distribution: ``target`` as in sweeps
    (``overrides:<id>.<name>``, ``settings:<id>.<name>``, ``link:<id>.<field>``), mean and std
    in ``unit``; ``mean`` None means the current value. Draws are clipped to [low, high]."""

    target: str
    std: float
    unit: str = ""
    mean: float | None = None
    low: float | None = None
    high: float | None = None
    label: str | None = None


@dataclass
class UncertaintyResult:
    metric: str
    nominal: float
    mean: float
    std: float
    samples: np.ndarray
    parameters: list[Uncertain]
    means: list[float]
    contributions: dict[str, float]
    failures: int
    seed: int | None
    note: str = (
        "calibration (systematic) uncertainty only; counting noise is separate and not included"
    )

    @property
    def linear_std(self) -> float:
        return math.sqrt(sum(c * c for c in self.contributions.values()))

    def ranked(self) -> list[tuple[str, float, float]]:
        """(parameter, contribution, fraction of the linear variance), largest first."""
        total = sum(c * c for c in self.contributions.values()) or 1.0
        items = sorted(self.contributions.items(), key=lambda kv: -kv[1])
        return [(name, c, c * c / total) for name, c in items]


def _split_target(target: str) -> tuple[str, str, str]:
    kind, _, rest = target.partition(":")
    owner, _, name = rest.rpartition(".")
    return kind, owner, name


def _current(setup: "Setup", library: "Library", target: str, unit: str) -> float:
    from sequoia.analysis.sweep import get_value
    from sequoia.components.base import get_component

    value = get_value(setup, target, unit)
    if value is not None:
        return value
    kind, owner, name = _split_target(target)
    placed = next((i for i in setup.instances if i.id == owner), None)
    if placed is not None:
        if kind == "overrides":
            current = getattr(library.load_part(placed.part).params, name, None)
        else:
            cls = get_component(library.load_part(placed.part).type)
            field_info = cls.Settings.model_fields.get(name)
            current = None if field_info is None else field_info.default
        if isinstance(current, Scalar):
            return current.si if unit.strip() in ("", "1") else float(from_si(current.si, unit))
        if isinstance(current, int | float) and not isinstance(current, bool):
            return float(current)
    if kind == "link" and name == "delay":
        return 0.0
    raise UncertaintyError(f"{target}: no current scalar value to perturb")


def _record_unit(record: CalibrationRecord) -> str:
    unit = record.value.unit.strip()
    return "" if unit in ("", "1") else unit


def from_calibration(setup: "Setup", library: "Library", spec: str) -> Uncertain:
    """``<instance>.<parameter>`` with the std of the part's calibration record. A
    ``zero_offset`` perturbs the instance's ``angle`` setting and a ``timing_offset`` the delay of
    the cable leaving it; their recorded value is added to the current setting."""
    owner, _, name = spec.rpartition(".")
    placed = next((i for i in setup.instances if i.id == owner), None)
    if placed is None:
        raise UncertaintyError(f"{spec}: no instance {owner!r}")
    part = library.load_part(placed.part)
    record = select_records(part.calibration, set(type(part.params).model_fields), part.ref).get(
        name
    )
    if record is None or record.value is None or record.std is None:
        raise UncertaintyError(
            f"{spec}: part {part.ref!r} has no scalar calibration record with a std for {name!r}"
        )
    unit = _record_unit(record)
    std = record.std_in_value_unit
    if name == "zero_offset":
        target = f"settings:{owner}.angle"
        return Uncertain(
            target, std, unit, _current(setup, library, target, unit) + record.value.value
        )
    if name == "timing_offset":
        cable = next(
            (
                link
                for link in setup.links
                if link.kind == "cable" and link.src.split(".")[0] == owner
            ),
            None,
        )
        if cable is None:
            raise UncertaintyError(f"{spec}: no cable leaves {owner!r} to carry a timing offset")
        target = f"link:{cable.id}.delay"
        return Uncertain(
            target, std, unit, _current(setup, library, target, unit) + record.value.value
        )
    return Uncertain(f"overrides:{owner}.{name}", std, unit)


def _bounds(u: Uncertain) -> tuple[float, float]:
    name = _split_target(u.target)[2]
    low = (
        u.low if u.low is not None else (0.0 if name in _UNIT_INTERVAL | _NON_NEGATIVE else -np.inf)
    )
    high = u.high if u.high is not None else (1.0 if name in _UNIT_INTERVAL else np.inf)
    return low, high


def propagate_uncertainty(
    setup: "Setup",
    library: "Library",
    metric_fn: str | Callable[["Setup", "Library"], float],
    params: Sequence[Uncertain | str],
    n: int = 200,
    seed: int | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> UncertaintyResult:
    """Monte Carlo over the declared (calibration) uncertainties of a few scalar parameters.

    ``metric_fn`` is a sweep metric name (``probe:pm.power_W``) or ``fn(setup, library)``.
    ``params`` are ``Uncertain`` entries or ``<instance>.<parameter>`` names whose std comes
    from the part's calibration record. Parameters are independent normals. Contributions are
    one-at-a-time: half the metric change between mean -/+ one std of that parameter.
    """
    from sequoia.analysis.sweep import evaluate, metrics_of, with_values
    from sequoia.engines.cache import RunCache, check_cancel

    resolved = [
        p if isinstance(p, Uncertain) else from_calibration(setup, library, p) for p in params
    ]
    if not resolved:
        raise UncertaintyError("no uncertain parameters given")
    for u in resolved:
        name = _split_target(u.target)[2]
        if name not in SUPPORTED:
            raise UncertaintyError(
                f"{u.target}: uncertainty propagation supports {sorted(SUPPORTED)}"
            )
        if not (math.isfinite(u.std) and u.std >= 0):
            raise UncertaintyError(f"{u.target}: std must be finite and non-negative")
    if n < 2:
        raise UncertaintyError("n must be at least 2")
    means = [
        u.mean if u.mean is not None else _current(setup, library, u.target, u.unit)
        for u in resolved
    ]
    bounds = [_bounds(u) for u in resolved]
    cache = RunCache()

    if isinstance(metric_fn, str):
        name = metric_fn

        def f(point):
            outcome = evaluate(point, library, cache=cache, should_cancel=should_cancel)
            if outcome.has_errors:
                return math.nan
            return metrics_of(outcome).get(name, math.nan)
    else:
        name = getattr(metric_fn, "__name__", "metric")

        def f(point):
            return float(metric_fn(point, library))

    def at(values) -> float:
        check_cancel(should_cancel)
        try:
            point = with_values(
                setup, [(u.target, float(v), u.unit) for u, v in zip(resolved, values, strict=True)]
            )
            return float(f(point))
        except SequoiaError:
            return math.nan

    total = n + 2 * len(resolved) + 1
    nominal = at(means)
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n, len(resolved)))
    samples = np.empty(n)
    for i in range(n):
        values = [
            min(max(m + u.std * z, lo), hi)
            for m, u, z, (lo, hi) in zip(means, resolved, draws[i], bounds, strict=True)
        ]
        samples[i] = at(values)
        if progress is not None:
            progress(i + 1, total)
    contributions = {}
    for k, u in enumerate(resolved):
        lo_b, hi_b = bounds[k]
        lo, hi = max(means[k] - u.std, lo_b), min(means[k] + u.std, hi_b)
        if hi <= lo or u.std == 0:
            contributions[u.label or u.target] = 0.0
            continue
        down, up = list(means), list(means)
        down[k], up[k] = lo, hi
        slope = (at(up) - at(down)) / (hi - lo)
        contributions[u.label or u.target] = abs(slope) * u.std
        if progress is not None:
            progress(n + 2 * (k + 1), total)
    good = samples[np.isfinite(samples)]
    return UncertaintyResult(
        metric=name,
        nominal=nominal,
        mean=float(np.mean(good)) if good.size else math.nan,
        std=float(np.std(good, ddof=1)) if good.size > 1 else math.nan,
        samples=samples,
        parameters=resolved,
        means=means,
        contributions=contributions,
        failures=int(n - good.size),
        seed=seed,
    )
