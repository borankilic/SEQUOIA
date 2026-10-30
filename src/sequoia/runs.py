import copy
import csv
import hashlib
import io
import json
import math
import secrets
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from sequoia.errors import Issue, SequoiaError
from sequoia.library.store import Library
from sequoia.measurement import effective_analyses, resolve_setup
from sequoia.setup.io import setup_from_dict, setup_to_dict
from sequoia.setup.model import Setup

RUN_FORMAT = "sequoia-run"
RUN_VERSION = 1
MAX_SNAPSHOT_BYTES = 5_000_000
ENGINES = ("rates", "gaussian", "montecarlo", "sweep", "chsh")


class RunError(SequoiaError):
    pass


def versions() -> dict[str, str | None]:
    import scipy

    import sequoia

    return {
        "sequoia": getattr(sequoia, "__version__", "0.1.0"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _status(key: str, value: float) -> str:
    if value == math.inf:
        return "infinite"
    if value == -math.inf:
        return "negative infinite"
    k = key.lower()
    if "car" in k:
        return "undefined: no accidental estimate"
    if "g2" in k:
        return "undefined: no heralded coincidences to normalize"
    if "sigma" in k or "significance" in k:
        return "undefined: no counting uncertainty"
    return "undefined"


def _bad(value: Any) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def json_safe(value: Any, key: str = "") -> Any:
    """Plain JSON data without NaN/Infinity: a non-finite number in a mapping becomes null with
    a sibling ``<key>_status`` saying why; inside a list it becomes null and the list's key gets
    a ``_status``."""
    if isinstance(value, Issue):
        value = issue_dict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            k = str(k)
            if isinstance(v, np.generic):
                v = v.item()
            if _bad(v):
                out[k] = None
                out.setdefault(f"{k}_status", _status(k, v))
                continue
            plain = json_safe(v, k)
            out[k] = plain
            if isinstance(plain, list) and _has_null_from_bad(v):
                out.setdefault(f"{k}_status", "contains null for undefined or infinite values")
        return out
    if isinstance(value, (list, tuple)):
        return [None if _bad(_item(v)) else json_safe(v, key) for v in value]
    return value


def _item(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def _has_null_from_bad(v: Any) -> bool:
    if isinstance(v, np.ndarray):
        return v.dtype.kind == "f" and not bool(np.all(np.isfinite(v)))
    if isinstance(v, (list, tuple)):
        return any(_bad(_item(x)) or _has_null_from_bad(x) for x in v)
    return False


def issue_dict(i: Issue) -> dict[str, Any]:
    return {
        "severity": i.severity,
        "code": i.code,
        "subject": i.subject,
        "message": i.message,
        "engine": i.engine,
        "run_id": i.run_id,
    }


@dataclass(frozen=True)
class Table:
    """Data behind a plot or table: column names carry quantity, unit and denominator."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    note: str = ""

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.columns)
        for row in self.rows:
            writer.writerow([_cell(v) for v in row])
        return buf.getvalue()

    def to_json(self) -> dict[str, Any]:
        rows = [[None if _bad(_item(v)) else json_safe(v) for v in row] for row in self.rows]
        out: dict[str, Any] = {"columns": list(self.columns), "rows": rows, "note": self.note}
        if any(_bad(_item(v)) for row in self.rows for v in row):
            out["rows_status"] = "null cells are undefined (not computed) or infinite"
        return out


def _cell(v: Any) -> str:
    v = _item(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    return "" if v is None else str(v)


def table(columns: Sequence[str], rows, note: str = "") -> Table:
    return Table(tuple(columns), tuple(tuple(r) for r in rows), note)


@dataclass(frozen=True)
class RunRecord:
    """Immutable record of one run: the inputs it was computed from and what came out."""

    engine: str
    run_id: str
    created: str
    setup_yaml: str
    setup_sha256: str
    library_fingerprint: str | None
    parts: Mapping[str, str]
    measurement: Mapping[str, Any] | None
    settings: Mapping[str, Any]
    sources: Mapping[str, Any]
    limits: Mapping[str, Any]
    detectors: Mapping[str, Any]
    issues: tuple[Issue, ...]
    model: tuple[tuple[str, str], ...]
    normalization: Mapping[str, Any]
    outputs: Mapping[str, Any]
    tables: Mapping[str, Table]
    versions: Mapping[str, Any]
    label: str = ""
    part_files: Mapping[str, Mapping[str, bytes]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def setup(self) -> Setup:
        from ruamel.yaml import YAML

        return setup_from_dict(YAML(typ="safe", pure=True).load(self.setup_yaml), "run setup")

    def to_json(self) -> dict[str, Any]:
        return {
            "format": RUN_FORMAT,
            "version": RUN_VERSION,
            "engine": {"name": self.engine, **dict(self.versions)},
            "run_id": self.run_id,
            "created": self.created,
            "label": self.label,
            "setup": {"sha256": self.setup_sha256, "file": "setup.yaml"},
            "library": {"fingerprint": self.library_fingerprint, "parts": dict(self.parts)},
            "measurement": json_safe(self.measurement) if self.measurement else None,
            "settings": json_safe(self.settings),
            "sources": json_safe(self.sources),
            "limits": json_safe(self.limits),
            "detectors": json_safe(self.detectors),
            "warnings": [issue_dict(i) for i in self.issues],
            "model": [list(r) for r in self.model],
            "normalization": json_safe(self.normalization),
            "outputs": json_safe(self.outputs),
            "tables": {k: t.to_json() for k, t in self.tables.items()},
        }

    def to_json_text(self) -> str:
        return json.dumps(self.to_json(), indent=2, allow_nan=False)


def used_parts(setup: Setup) -> list[str]:
    refs = {i.part for i in setup.instances} | {p.part for p in setup.probes}
    refs |= {getattr(link, "fiber", None) for link in setup.links} - {None}
    return sorted(refs)


def _snapshot(library: Library, ref: str) -> dict[str, bytes]:
    directory = library.part_dir(ref)
    files: dict[str, bytes] = {}
    if not directory.is_dir():
        return files
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        name = path.relative_to(directory).as_posix()
        if path.name.startswith(".") or name.endswith(".tmp"):
            continue
        if path.stat().st_size > MAX_SNAPSHOT_BYTES and name != "part.yaml":
            continue
        files[name] = path.read_bytes()
    return files


def setup_yaml(setup: Setup) -> str:
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.default_flow_style = None
    yaml.width = 120
    buf = io.StringIO()
    yaml.dump(json.loads(json.dumps(setup_to_dict(setup), default=str)), buf)
    return buf.getvalue()


def detector_summary(setup: Setup, library: Library) -> dict[str, Any]:
    from sequoia.setup.resolve import instantiate

    out = {}
    for inst in setup.instances:
        try:
            comp = instantiate(library, inst)
        except SequoiaError:
            continue
        if comp.category in ("detectors", "electronics"):
            out[inst.id] = {
                "part": inst.part,
                "type": type(comp).__name__,
                "params": comp.params.model_dump(mode="json"),
            }
    return out


def calibration_text(setup: Setup, library: Library) -> str:
    from sequoia.library.calibration import placeholder_report

    try:
        rows = placeholder_report(setup, library)
    except SequoiaError:
        return ""
    if not rows:
        return ""
    generic = [r for r in rows if r.placeholder]
    if not generic:
        return f"all {len(rows)} critical values are measured or vendor values"
    names = ", ".join(f"{r.owner}.{r.parameter}" for r in generic[:4])
    more = f" and {len(generic) - 4} more" if len(generic) > 4 else ""
    return f"{len(generic)} of {len(rows)} critical values are generic placeholders ({names}{more})"


def new_run_id(engine: str) -> str:
    return f"{engine}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def make_record(
    engine: str,
    setup: Setup,
    library: Library,
    *,
    settings: Mapping[str, Any],
    outputs: Mapping[str, Any],
    tables: Mapping[str, Table] | None = None,
    sources: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    issues: Sequence[Issue] = (),
    model: Sequence[tuple[str, str]] = (),
    normalization: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    label: str = "",
) -> RunRecord:
    run_id = run_id or new_run_id(engine)
    text = setup_yaml(setup)
    refs = used_parts(setup)
    parts, snapshot = {}, {}
    for ref in refs:
        try:
            parts[ref] = library.part_fingerprint(ref)
            snapshot[ref] = MappingProxyType(_snapshot(library, ref))
        except SequoiaError:
            parts[ref] = "missing"
    try:
        fingerprint = library.fingerprint(refs) if "missing" not in parts.values() else None
    except SequoiaError:
        fingerprint = None
    m = setup.measurement
    return RunRecord(
        engine=engine,
        run_id=run_id,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        setup_yaml=text,
        setup_sha256=hashlib.sha256(text.encode()).hexdigest(),
        library_fingerprint=fingerprint,
        parts=MappingProxyType(parts),
        measurement=MappingProxyType(m.model_dump(mode="json")) if m is not None else None,
        settings=MappingProxyType(copy.deepcopy(dict(settings))),
        sources=MappingProxyType(copy.deepcopy(dict(sources or {}))),
        limits=MappingProxyType(copy.deepcopy(dict(limits or {}))),
        detectors=MappingProxyType(detector_summary(setup, library)),
        issues=tuple(i.for_run(engine, run_id) for i in issues),
        model=tuple((str(a), str(b)) for a, b in model),
        normalization=MappingProxyType(copy.deepcopy(dict(normalization or {}))),
        outputs=MappingProxyType(copy.deepcopy(dict(outputs))),
        tables=MappingProxyType(dict(tables or {})),
        versions=MappingProxyType(versions()),
        label=label,
        part_files=MappingProxyType(snapshot),
    )


# --- rate engine (base simulation) ---


def record_from_outcome(outcome, setup: Setup, library: Library, label: str = "") -> RunRecord:
    """Record of a base (rate-engine) simulation; ``setup`` must be the setup it ran on."""
    from sequoia.analysis.sweep import metrics_of
    from sequoia.engines.report import rates_card

    outputs = metrics_of(outcome) if outcome.classical is not None else {}
    tables: dict[str, Table] = {}
    rates = outcome.rates
    if rates is not None:
        tables["channels"] = table(
            ["channel", "detector", "recorded singles [Hz] (after dead time)"],
            [
                (n, c.get("detector_instance", ""), c.get("singles_hz"))
                for n, c in rates.channels.items()
            ],
        )
        rows = []
        for name, a in rates.analyses.items():
            if a.get("type") == "coincidences":
                rows.append(
                    (name, a["window_s"], a["total_hz"], a["accidental_hz"], a["true_hz"], a["car"])
                )
        if rows:
            tables["coincidences"] = table(
                [
                    "analysis",
                    "coincidence window [s] (full width)",
                    "raw coincidences [Hz]",
                    "accidentals [Hz]",
                    "coincidences minus accidentals [Hz]",
                    "CAR (true / accidental)",
                ],
                rows,
            )
    return make_record(
        "rates",
        setup,
        library,
        settings={"revision": outcome.revision},
        outputs=outputs,
        tables=tables,
        sources=dict(rates.sources) if rates is not None else {},
        issues=outcome.issues,
        model=rates_card(outcome, calibration_text(setup, library)),
        normalization={"rates": "Hz, after the rate engine's dead-time model"},
        run_id=outcome.run_id or None,
        label=label,
    )


def run_rates(setup: Setup, library: Library, *, should_cancel=None, label: str = ""):
    from sequoia.engines.cache import RunCache
    from sequoia.engines.outcome import compute_outcome

    outcome = compute_outcome(setup, library, RunCache(), should_cancel)
    return outcome, record_from_outcome(outcome, setup, library, label)


# --- Gaussian ---


@dataclass
class GaussianOutcome:
    result: Any
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    pairs: np.ndarray = field(default_factory=lambda: np.array([]))
    car: np.ndarray = field(default_factory=lambda: np.array([]))
    g2: np.ndarray = field(default_factory=lambda: np.array([]))
    groups: list[tuple[float, list[str]]] = field(default_factory=list)
    record: RunRecord | None = None
    card: list[tuple[str, str]] = field(default_factory=list)


SWEEP_POINTS = 13


def run_gaussian(
    setup: Setup,
    library: Library,
    *,
    pump_scale: float = 1.0,
    sweep: bool = False,
    sources: Sequence[str] | None = None,
    factors: Mapping[str, float] | None = None,
    should_cancel=None,
    progress=None,
    run_id: str | None = None,
) -> GaussianOutcome:
    from sequoia.engines.cache import RunCache
    from sequoia.engines.classical import simulate
    from sequoia.engines.gaussian import gaussian_window_groups, prepare_gaussian
    from sequoia.engines.report import gaussian_card

    progress = progress or (lambda *_a: None)
    factors = dict(factors or {})
    run = resolve_setup(setup, library)
    progress(0, 0)
    cache = RunCache()
    classical = simulate(run, library, should_cancel=should_cancel, cache=cache)
    groups = gaussian_window_groups(
        run, library, classical=classical, cache=cache, should_cancel=should_cancel
    )
    selected = None
    if len(groups) > 1:
        if sources is None or list(sources) not in [ids for _, ids in groups]:
            return GaussianOutcome(None, groups=groups)
        selected = list(sources)
    prepared = prepare_gaussian(
        run,
        library,
        classical=classical,
        cache=cache,
        should_cancel=should_cancel,
        sources=selected,
    )
    gain = {k: pump_scale * factors.get(k, 1.0) for k in prepared.model.pumps}
    result = prepared.evaluate(gain)
    outcome = GaussianOutcome(result, groups=groups)
    if sweep and result.sources:
        mu = next(iter(result.sources.values()))["pairs_per_window"]
        grid = np.logspace(-1, 3, SWEEP_POINTS)
        rows = []
        for i, t in enumerate(grid):
            progress(i, len(grid))
            if should_cancel is not None and should_cancel():
                break
            rows.append((t, prepared.evaluate({k: v * float(t) for k, v in gain.items()})))
        names = [n for n, a in result.analyses.items() if a["type"] == "coincidences"]
        hnames = [n for n, a in result.analyses.items() if a["type"] == "heralded_g2"]
        outcome.scales = pump_scale * np.array([t for t, _ in rows])
        outcome.pairs = np.array([t for t, _ in rows]) * mu
        if names:
            outcome.car = np.array([r.analyses[names[0]]["car"] for _, r in rows])
        if hnames:
            outcome.g2 = np.array([r.analyses[hnames[0]]["g2_heralded"] for _, r in rows])
    if not result.channels:
        return outcome
    calibration = calibration_text(setup, library)
    outcome.card = gaussian_card(result, calibration)
    settings = {
        "pump_scale": pump_scale,
        "sweep": sweep,
        "sources": selected,
        "factors": {k: v for k, v in factors.items() if v != 1.0},
    }
    outcome.record = gaussian_record(setup, library, outcome, settings, run_id)
    return outcome


def gaussian_record(setup, library, outcome: GaussianOutcome, settings, run_id=None) -> RunRecord:
    r = outcome.result
    w = r.window
    outputs: dict[str, Any] = {}
    rows = []
    for ch in r.channels:
        p = r.probability(ch)
        outputs[f"{ch}.P_click"] = p
        outputs[f"{ch}.mean_photons"] = r.mean_photons[ch]
        rows.append((ch, p, r.mean_photons[ch], p / w))
    arows = []
    for name, a in r.analyses.items():
        for key in ("car", "true_hz", "coincidence_rate_hz", "g2_heralded", "g2_at_center"):
            if key in a:
                outputs[f"{name}.{key}"] = a[key]
        arows.append(
            (
                name,
                a["type"],
                a.get("coincidence_rate_hz"),
                a.get("true_hz"),
                a.get("car"),
                a.get("g2_heralded", a.get("g2_at_center")),
            )
        )
    tables = {
        "channels": table(
            [
                "channel",
                "P(click) per statistics window",
                "mean detected photons per window",
                "click rate [Hz] (P / window)",
            ],
            rows,
        ),
        "analyses": table(
            [
                "analysis",
                "type",
                "both click in one statistics window [Hz]",
                "minus accidentals [Hz]",
                "CAR (true / accidental)",
                "g2(0)",
            ],
            arows,
        ),
    }
    if len(outcome.pairs):
        n = len(outcome.pairs)
        car = outcome.car if len(outcome.car) else np.full(n, np.nan)
        g2 = outcome.g2 if len(outcome.g2) else np.full(n, np.nan)
        tables["pump_sweep"] = table(
            [
                "pump x",
                "mean pairs per statistics window",
                "CAR (true / accidental)",
                "heralded g2(0)",
            ],
            zip(outcome.scales, outcome.pairs, car, g2, strict=True),
        )
    return make_record(
        "gaussian",
        setup,
        library,
        settings=settings,
        outputs=outputs,
        tables=tables,
        sources=r.sources,
        limits={"pnr_cutoff": 12},
        issues=r.issues,
        model=outcome.card,
        normalization={
            "denominator": "probability per statistics window",
            "statistics_window_s": w,
            "rates": "probability / statistics window [Hz]",
        },
        run_id=run_id,
    )


# --- Monte Carlo ---


@dataclass
class McOutcome:
    tags: Any
    analyses: dict[str, dict[str, Any]]
    comparisons: list = field(default_factory=list)
    note: str = ""
    confirmatory: bool = False
    record: RunRecord | None = None
    card: list[tuple[str, str]] = field(default_factory=list)
    specs: list[dict[str, Any]] = field(default_factory=list)


def run_montecarlo(
    setup: Setup,
    library: Library,
    *,
    duration_s: float,
    seed: int | None = None,
    schmidt_number: float | None = None,
    specs: Sequence[Mapping[str, Any]] | None = None,
    confirmatory: bool = False,
    should_cancel=None,
    progress=None,
    run_id: str | None = None,
) -> McOutcome:
    """Monte Carlo tags, their analyses under the measurement definition and the comparison with
    the rate engine. A missing seed is drawn here and recorded, so the run can be repeated."""
    from sequoia.engines.cache import RunCache
    from sequoia.engines.montecarlo import compare_with_rates, simulate_timetags
    from sequoia.engines.rates import simulate_rates
    from sequoia.engines.report import montecarlo_card
    from sequoia.measurement import analyse_measured_tags

    seed = secrets.randbelow(2**31) if seed is None else int(seed)
    run = resolve_setup(setup, library)
    specs = [dict(s) for s in (specs if specs is not None else effective_analyses(setup))]
    cache = RunCache()
    tags = simulate_timetags(
        run,
        library,
        duration_s,
        seed=seed,
        cache=cache,
        should_cancel=should_cancel,
        progress=progress,
        schmidt_number=schmidt_number,
    )
    analyses = analyse_measured_tags(
        run, tags.channels, tags.duration, tags.clock, tags.clock_ids, specs
    )
    rates = simulate_rates(run, library, cache=cache, should_cancel=should_cancel)
    out = McOutcome(
        tags,
        analyses,
        compare_with_rates(tags, analyses, rates),
        confirmatory=confirmatory,
        specs=specs,
    )
    out.card = montecarlo_card(
        tags, analyses, setup.measurement, schmidt_number, calibration_text(setup, library)
    )
    settings = {
        "duration_s": duration_s,
        "seed": seed,
        "schmidt_number": schmidt_number,
        "specs": specs,
        "confirmatory": confirmatory,
    }
    out.record = montecarlo_record(setup, library, out, settings, run_id)
    return out


def montecarlo_record(setup, library, out: McOutcome, settings, run_id=None) -> RunRecord:
    from sequoia.engines.report import MC_QUANTITY

    tags = out.tags
    outputs: dict[str, Any] = {f"{k}.events": v for k, v in tags.counts().items()}
    crow = []
    for c in out.comparisons:
        outputs[f"{c.name}.{c.quantity}"] = c.monte_carlo
        crow.append(
            (
                c.name,
                MC_QUANTITY.get(c.quantity, c.quantity),
                c.monte_carlo,
                c.sigma,
                c.expected,
                c.z,
            )
        )
    tables = {
        "comparison": table(
            [
                "item",
                "quantity",
                "Monte Carlo (recorded tags)",
                "sigma (Poisson counting)",
                "rate engine (expected)",
                "z",
            ],
            crow,
        )
    }
    for name, a in out.analyses.items():
        for key in ("counts", "true_hz", "accidental_hz", "car", "g2_heralded", "triples"):
            if key in a:
                outputs[f"{name}.{key}"] = a[key]
        if "tau_s" in a:
            tables[f"histogram_{name}"] = table(
                ["delay [s] (bin centre)", f"events per bin ({a['bin_width_s']:.4g} s bins)"],
                zip(np.asarray(a["tau_s"]), np.asarray(a["counts"]), strict=True),
                f"{tags.duration:g} s of tags",
            )
    return make_record(
        "montecarlo",
        setup,
        library,
        settings=settings,
        outputs=outputs,
        tables=tables,
        sources=tags.truth.get("sources", {}) if isinstance(tags.truth, dict) else {},
        limits={"duration_s": tags.duration},
        issues=tags.issues,
        model=out.card,
        normalization={
            "rates": "recorded events / acquisition duration [Hz]",
            "sigma": "Poisson counting of recorded events",
        },
        run_id=run_id,
    )


# --- sweeps ---


@dataclass
class SweepRun:
    result: Any
    record: RunRecord | None = None
    card: list[tuple[str, str]] = field(default_factory=list)


def run_sweep_record(
    setup: Setup,
    library: Library,
    axes,
    outputs: Sequence[str] | None = None,
    *,
    fock: Mapping[str, Any] | None = None,
    should_cancel=None,
    progress=None,
    run_id: str | None = None,
) -> SweepRun:
    from sequoia.analysis.sweep import run_sweep
    from sequoia.engines.cache import RunCache
    from sequoia.engines.report import sweep_card

    result = run_sweep(
        setup,
        library,
        axes,
        outputs,
        cache=RunCache(),
        should_cancel=should_cancel,
        progress=progress,
        fock=dict(fock) if fock is not None else None,
    )
    card = sweep_card(result, fock is not None, calibration_text(setup, library))
    columns = [f"{a.target} [{a.unit}]" if a.unit else a.target for a in result.axes]
    names = sorted(result.data)
    rows = []
    for index in np.ndindex(*result.shape):
        rows.append(
            [float(a.values[i]) for a, i in zip(result.axes, index, strict=True)]
            + [float(result.data[n][index]) for n in names]
            + [result.failures.get(index, "")]
        )
    tables = {"sweep": table([*columns, *[metric_label(n) for n in names], "failure"], rows)}
    settings = {
        "axes": [
            {"target": a.target, "unit": a.unit, "values": np.asarray(a.values, float).tolist()}
            for a in result.axes
        ],
        "outputs": list(outputs) if outputs is not None else None,
        "fock": dict(fock) if fock is not None else None,
    }
    record = make_record(
        "sweep",
        setup,
        library,
        settings=settings,
        outputs={n: result.data[n].ravel().tolist() for n in names},
        tables=tables,
        limits={"points": int(np.prod(result.shape))},
        model=card,
        normalization={"columns": "quantity [unit] as in the rate engine; fock: per window"},
        run_id=run_id,
    )
    return SweepRun(result, record, card)


UNIT_SUFFIX = (
    ("_hz", "Hz"),
    ("_W", "W"),
    ("_s", "s"),
    ("_m", "m"),
)


def metric_label(name: str) -> str:
    if name.startswith("fock:"):
        return f"{name} [probability per detection window]"
    for suffix, unit in UNIT_SUFFIX:
        if name.endswith(suffix):
            return f"{name} [{unit}]"
    return name


# --- CHSH ---


@dataclass
class ChshRun:
    result: Any
    record: RunRecord | None = None
    card: list[tuple[str, str]] = field(default_factory=list)


def chsh_windows(setup: Setup, names: Sequence[str]) -> dict[str, float]:
    from sequoia.analysis.correlation import _duration

    out = {}
    for spec in effective_analyses(setup):
        if spec.get("name") in names and spec.get("window") is not None:
            try:
                out[spec["name"]] = _duration(spec["window"], "window")
            except SequoiaError:
                continue
    return out


def run_chsh_record(
    setup: Setup,
    library: Library,
    *,
    target_a: str,
    target_b: str,
    settings_a: Sequence[float],
    settings_b: Sequence[float],
    unit: str = "deg",
    outcomes: Sequence[Sequence[str]] = (("tt", "tr"), ("rt", "rr")),
    quantity: str = "total_hz",
    integration_time_s: float = 10.0,
    sidebands: int = 2,
    should_cancel=None,
    run_id: str | None = None,
) -> ChshRun:
    from sequoia.analysis.chsh import run_chsh
    from sequoia.engines.cache import RunCache
    from sequoia.engines.report import chsh_card

    outcomes = [list(o) for o in outcomes]
    r = run_chsh(
        setup,
        library,
        target_a,
        target_b,
        list(settings_a),
        list(settings_b),
        unit,
        outcomes,
        quantity,
        integration_time_s,
        RunCache(),
        should_cancel,
        sidebands,
    )
    names = [n for row in outcomes for n in row]
    card = chsh_card(r, chsh_windows(setup, names), calibration_text(setup, library))
    total = r.total_rates if r.total_rates is not None else r.rates
    rows = []
    outputs: dict[str, Any] = {
        "S": r.s,
        "S_raw": r.s_raw,
        "S_accidental_subtracted": r.s_corrected,
        "S_sigma": r.s_sigma,
        "significance": r.significance,
    }
    for i in range(2):
        for j in range(2):
            counts = total[i, j].ravel() * r.integration_time
            outputs[f"E[{i}{j}]"] = float(r.correlations[i, j])
            rows.append(
                (f"{i}{j}", r.settings_a[i], r.settings_b[j], *counts, r.correlations[i, j])
            )
    labels = ("a+ b+", "a+ b-", "a- b+", "a- b-")
    t = r.integration_time
    tables = {
        "settings": table(
            [
                "setting pair",
                f"arm a parameter [{unit}]",
                f"arm b parameter [{unit}]",
                *[f"{o} raw coincidences in {t:g} s" for o in labels],
                f"E ({'raw' if quantity == 'total_hz' else 'accidentals subtracted'})",
            ],
            rows,
        )
    }
    settings = {
        "target_a": target_a,
        "target_b": target_b,
        "settings_a": list(map(float, settings_a)),
        "settings_b": list(map(float, settings_b)),
        "unit": unit,
        "outcomes": outcomes,
        "quantity": quantity,
        "integration_time_s": integration_time_s,
        "sidebands": int(sidebands),
    }
    record = make_record(
        "chsh",
        setup,
        library,
        settings=settings,
        outputs=outputs,
        tables=tables,
        model=card,
        normalization={
            "integration_time": "per setting pair (4 pairs)",
            "E": "(C++ + C-- - C+- - C-+) / sum",
            "notes": list(r.notes),
        },
        run_id=run_id,
    )
    return ChshRun(r, record, card)


# --- bundles, rerun, comparison ---


def _summary(record: RunRecord) -> str:
    lines = [
        f"# Run {record.run_id}",
        "",
        f"- engine: {record.engine} (sequoia {record.versions.get('sequoia')}, numpy "
        f"{record.versions.get('numpy')}, scipy {record.versions.get('scipy')})",
        f"- created: {record.created}",
        f"- setup: setup.yaml (sha256 {record.setup_sha256[:12]})",
        f"- library parts: library/ (fingerprint {str(record.library_fingerprint)[:12]})",
        "- settings: " + json.dumps(json_safe(record.settings))[:400],
        "",
        "## Model",
        "",
    ]
    lines += [f"- **{a}**: {b}" for a, b in record.model]
    lines += ["", "## Warnings", ""]
    lines += [f"- [{i.severity}] {i.subject}: {i.message}" for i in record.issues] or ["- none"]
    lines += ["", "## Data", ""]
    lines += [f"- data/{name}.csv: {', '.join(t.columns)}" for name, t in record.tables.items()]
    lines += [
        "",
        "Undefined values are empty CSV cells and null in run.json with a `<key>_status`.",
        "",
        "Rerun: `sequoia rerun <this folder>` (uses only the files in this folder).",
    ]
    return "\n".join(lines) + "\n"


def write_bundle(record: RunRecord, directory: str | Path) -> Path:
    """A folder that reruns without the author's library: run.json, setup.yaml, the library
    parts as they were at the run, measurement.json, data/*.csv and summary.md."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "setup.yaml").write_text(record.setup_yaml, encoding="utf-8")
    for ref, files in record.part_files.items():
        for name, data in files.items():
            target = out / "library" / ref / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if record.measurement is not None:
        (out / "measurement.json").write_text(
            json.dumps(json_safe(record.measurement), indent=2, allow_nan=False), encoding="utf-8"
        )
    data = out / "data"
    data.mkdir(exist_ok=True)
    for name, t in record.tables.items():
        (data / f"{name}.csv").write_text(t.to_csv(), encoding="utf-8")
    (out / "run.json").write_text(record.to_json_text(), encoding="utf-8")
    (out / "summary.md").write_text(_summary(record), encoding="utf-8")
    return out


def rerun(engine: str, setup: Setup, library: Library, settings: Mapping[str, Any]) -> RunRecord:
    s = dict(settings)
    if engine == "rates":
        return run_rates(setup, library)[1]
    if engine == "gaussian":
        out = run_gaussian(
            setup,
            library,
            pump_scale=s.get("pump_scale", 1.0),
            sweep=s.get("sweep", False),
            sources=s.get("sources"),
            factors=s.get("factors"),
        )
        if out.record is None:
            raise RunError("the Gaussian rerun produced no result")
        return out.record
    if engine == "montecarlo":
        return run_montecarlo(
            setup,
            library,
            duration_s=s["duration_s"],
            seed=s.get("seed"),
            schmidt_number=s.get("schmidt_number"),
            specs=s.get("specs"),
            confirmatory=s.get("confirmatory", False),
        ).record
    if engine == "sweep":
        from sequoia.analysis.sweep import Axis

        axes = [Axis(a["target"], np.asarray(a["values"], float), a["unit"]) for a in s["axes"]]
        return run_sweep_record(setup, library, axes, s.get("outputs"), fock=s.get("fock")).record
    if engine == "chsh":
        return run_chsh_record(setup, library, **s).record
    raise RunError(f"cannot rerun engine {engine!r}; known: {', '.join(ENGINES)}")


@dataclass(frozen=True)
class Comparison:
    """Side-by-side key outputs (baseline, current, difference) and the inputs that differ."""

    rows: tuple[tuple[str, Any, Any, float | None], ...]
    input_differences: tuple[str, ...]
    matches: bool
    baseline_id: str = ""
    current_id: str = ""

    def text(self) -> str:
        lines = [f"baseline {self.baseline_id} vs {self.current_id}"]
        lines += [f"input differs: {d}" for d in self.input_differences] or ["inputs identical"]
        for key, a, b, d in self.rows:
            lines.append(f"{key}: {_short(a)} | {_short(b)} | diff {_short(d)}")
        return "\n".join(lines)


def _short(v: Any) -> str:
    if isinstance(v, list):
        return f"[{len(v)} values]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return "undefined" if v is None else str(v)


def _as_json(record: "RunRecord | Mapping[str, Any]") -> dict[str, Any]:
    return record.to_json() if isinstance(record, RunRecord) else dict(record)


def _diff(a: Any, b: Any) -> float | None:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        diffs = [_diff(x, y) for x, y in zip(a, b, strict=True)]
        good = [d for d in diffs if d is not None]
        if any(
            d is None and (x is not None or y is not None)
            for d, x, y in zip(diffs, a, b, strict=True)
        ):
            return None
        return max(good, default=0.0)
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else None
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(b) - float(a))
    return 0.0 if a == b else None


def _close(a: Any, b: Any, d: float | None, rtol: float) -> bool:
    if d is None:
        return a is None and b is None
    scale = max(_magnitude(a), _magnitude(b))
    return d <= rtol * scale + 1e-300


def _magnitude(v: Any) -> float:
    if isinstance(v, list):
        return max((_magnitude(x) for x in v), default=0.0)
    if isinstance(v, int | float) and not isinstance(v, bool):
        return abs(float(v))
    return 0.0


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, list) and value and all(isinstance(x, Mapping) for x in value):
        for n, item in enumerate(value):
            key = item.get("id", item.get("name", n))
            _flatten(f"{prefix}[{key}]", item, out)
    else:
        out[prefix] = value


def _keyed(a: Mapping[str, Any], b: Mapping[str, Any], what: str) -> list[str]:
    fa: dict[str, Any] = {}
    fb: dict[str, Any] = {}
    _flatten("", a or {}, fa)
    _flatten("", b or {}, fb)
    out = []
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key) != fb.get(key):
            out.append(f"{what} {key}: {_short(fa.get(key))} -> {_short(fb.get(key))}")
    return out


def _setup_dict(record: "RunRecord | Mapping[str, Any]", text: str | None) -> dict[str, Any]:
    from ruamel.yaml import YAML

    if isinstance(record, RunRecord):
        text = record.setup_yaml
    return YAML(typ="safe", pure=True).load(text) if text else {}


def compare_records(
    baseline: "RunRecord | Mapping[str, Any]",
    current: "RunRecord | Mapping[str, Any]",
    *,
    rtol: float = 1e-9,
    baseline_setup: str | None = None,
    current_setup: str | None = None,
) -> Comparison:
    """Compare two runs. Records loaded from run.json carry their setup in a separate file;
    pass its text as ``baseline_setup`` / ``current_setup``."""
    a, b = _as_json(baseline), _as_json(current)
    inputs: list[str] = []
    if a["engine"]["name"] != b["engine"]["name"]:
        inputs.append(f"engine: {a['engine']['name']} -> {b['engine']['name']}")
    if a["setup"]["sha256"] != b["setup"]["sha256"]:
        diffs = _keyed(
            _setup_dict(baseline, baseline_setup), _setup_dict(current, current_setup), "setup"
        )
        inputs += [d for d in diffs if not d.startswith("setup measurement")] or [
            "setup: text differs"
        ]
    inputs += [
        d
        for d in _keyed(a.get("settings"), b.get("settings"), "setting")
        if not d.startswith("setting revision")
    ]
    inputs += _keyed(a.get("measurement"), b.get("measurement"), "measurement")
    if a["library"]["fingerprint"] != b["library"]["fingerprint"]:
        pa, pb = a["library"]["parts"], b["library"]["parts"]
        changed = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        inputs.append("library parts differ: " + (", ".join(changed) or "fingerprint"))
    oa, ob = a.get("outputs", {}), b.get("outputs", {})
    rows = []
    matches = True
    for key in sorted(set(oa) | set(ob)):
        if key.endswith("_status"):
            continue
        va, vb = oa.get(key), ob.get(key)
        d = _diff(va, vb)
        if not _close(va, vb, d, rtol):
            matches = False
        rows.append((key, va, vb, d))
    return Comparison(tuple(rows), tuple(inputs), matches, a.get("run_id", ""), b.get("run_id", ""))


@dataclass(frozen=True)
class RerunResult:
    original: dict[str, Any]
    record: RunRecord
    comparison: Comparison


def rerun_bundle(directory: str | Path, *, rtol: float = 1e-9) -> RerunResult:
    """Rerun an exported bundle from its own files and compare with the stored outputs."""
    from sequoia.setup.io import load_setup

    path = Path(directory)
    try:
        original = json.loads((path / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"{path}: cannot read run.json: {exc}") from exc
    if original.get("format") != RUN_FORMAT:
        raise RunError(f"{path}: not a {RUN_FORMAT} bundle")
    setup = load_setup(path / "setup.yaml")
    library = Library(path / "library")
    record = rerun(original["engine"]["name"], setup, library, original.get("settings") or {})
    text = (path / "setup.yaml").read_text(encoding="utf-8")
    comparison = compare_records(original, record, rtol=rtol, baseline_setup=text)
    return RerunResult(original, record, comparison)
