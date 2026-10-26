import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sequoia.data import Duration, Frequency
from sequoia.errors import Issue

if TYPE_CHECKING:
    from sequoia.library.store import Library
    from sequoia.setup.model import Setup

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_Id = Annotated[str, StringConstraints(pattern=rf"^{_NAME}$")]
_Endpoint = Annotated[str, StringConstraints(pattern=rf"^{_NAME}\.{_NAME}$")]
_Ref = Annotated[str, StringConstraints(pattern=rf"^{_NAME}(\.{_NAME})?$")]
_ID_RE = re.compile(rf"^{_NAME}$")
WINDOWED = ("coincidences", "heralded_g2")
ENGINE_DEAD_TIME = "non-paralyzable"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _positive(scalar, what: str) -> None:
    if scalar is not None and not (math.isfinite(scalar.si) and scalar.si > 0):
        raise ValueError(f"{what} must be finite and positive")


class PulseClock(_M):
    kind: Literal["sources", "cw", "source", "rate"] = "sources"
    source: _Id | None = None
    rate: Frequency | None = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"kind": data}
        return data

    @model_validator(mode="after")
    def _check(self) -> "PulseClock":
        if (self.kind == "source") != (self.source is not None):
            raise ValueError("kind 'source' needs 'source: <pair-source id>' (and only it)")
        if (self.kind == "rate") != (self.rate is not None):
            raise ValueError("kind 'rate' needs 'rate: <frequency>' (and only it)")
        _positive(self.rate, "rate")
        return self


class HeraldCondition(_M):
    name: _Id
    herald: Annotated[list[_Ref], Field(min_length=1)]
    veto: list[_Ref] = []
    description: str = ""


class Measurement(_M):
    """How the apparatus is read out: separate from the optics so that it is saved, shown and
    compared as one object. Times are named by what they are: the coincidence window (full
    width), the acquisition duration (per setting, repeated ``repeats`` times); repetition period
    and coherence window belong to the sources."""

    purpose: str = ""
    channels: dict[_Id, _Endpoint] = {}
    coincidence_window: Duration | None = None
    window_convention: Literal["full_width"] = "full_width"
    centers: dict[_Ref, Duration] = {}
    pulse_clock: PulseClock = PulseClock()
    acquisition_duration: Duration | None = None
    repeats: int = Field(default=1, ge=1)
    accidentals: Literal["auto", "singles", "sideband"] = "auto"
    dead_time: Literal["non-paralyzable", "none"] = ENGINE_DEAD_TIME
    conditions: list[HeraldCondition] = []

    @model_validator(mode="after")
    def _check(self) -> "Measurement":
        _positive(self.coincidence_window, "coincidence_window")
        _positive(self.acquisition_duration, "acquisition_duration")
        for key, value in self.centers.items():
            if not math.isfinite(value.si):
                raise ValueError(f"centers.{key} must be finite")
        names = [c.name for c in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("condition names must be unique")
        return self


def cable_channels(setup: "Setup") -> dict[str, str]:
    out: dict[str, str] = {}
    for link in setup.links:
        if link.kind == "cable":
            out.setdefault(link.src.split(".")[0], link.dst)
    return out


def channel_map(setup: "Setup") -> dict[str, tuple[str, str]]:
    """Detector id -> (tagger channel, 'cable' or 'measurement')."""
    out = {k: (v, "cable") for k, v in cable_channels(setup).items()}
    m = setup.measurement
    if m is not None:
        out.update({k: (v, "measurement") for k, v in m.channels.items()})
    return out


def _channel(ref: Any, mapping: Mapping[str, tuple[str, str]]) -> Any:
    if isinstance(ref, str) and _ID_RE.match(ref) and ref in mapping:
        return mapping[ref][0]
    return ref


def effective_analyses(setup: "Setup") -> list[dict[str, Any]]:
    """The setup's analyses with the measurement's defaults filled in where an analysis leaves
    them out: the coincidence window, the centres, and detector ids named instead of channels.
    Explicit analysis values always win."""
    specs = [copy.deepcopy(dict(a)) for a in setup.analyses]
    m = setup.measurement
    mapping = channel_map(setup)
    for spec in specs:
        if "channels" in spec and isinstance(spec["channels"], list):
            spec["channels"] = [_channel(c, mapping) for c in spec["channels"]]
        for key in ("herald", "a", "b"):
            if key in spec:
                spec[key] = _channel(spec[key], mapping)
        if m is None:
            continue
        kind, name = spec.get("type"), spec.get("name")
        if kind in WINDOWED and spec.get("window") is None and m.coincidence_window is not None:
            spec["window"] = m.coincidence_window.si
        if spec.get("mode") == "search":
            continue
        keys = [("center", name)] if kind == "coincidences" else []
        if kind == "heralded_g2":
            keys = [("center_a", f"{name}.a"), ("center_b", f"{name}.b")]
            keys += [("center", name)]
        for key, ref in keys:
            value = m.centers.get(ref, m.centers.get(name) if key != "center" else None)
            current = spec.get(key, "auto")
            if value is not None and isinstance(current, str) and current.lower() == "auto":
                spec[key] = value.si
    return specs


def _dead_time_instances(setup: "Setup", library: "Library") -> list[str]:
    out = []
    for inst in setup.instances:
        try:
            part = library.load_part(inst.part)
        except Exception:
            continue
        if "dead_time" in type(part.params).model_fields:
            out.append(inst.id)
    return out


def resolve_setup(setup: "Setup", library: "Library | None" = None) -> "Setup":
    """The setup an engine runs: analyses with the measurement defaults applied and, for the
    'none' dead-time model, detector and tagger dead times set to zero. Idempotent; a setup
    without a measurement keeps its analyses (copied)."""
    m = setup.measurement
    if m is None:
        return setup
    update: dict[str, Any] = {"analyses": effective_analyses(setup)}
    if m.dead_time == "none" and library is not None:
        ids = set(_dead_time_instances(setup, library))
        update["instances"] = [
            i.model_copy(update={"overrides": {**i.overrides, "dead_time": "0 s"}})
            if i.id in ids
            else i
            for i in setup.instances
        ]
    return setup.model_copy(update=update)


def acquisition_duration(setup: "Setup", default: float) -> float:
    m = setup.measurement
    return m.acquisition_duration.si if m is not None and m.acquisition_duration else default


def check_measurement(setup: "Setup") -> list[Issue]:
    m = setup.measurement
    if m is None:
        return []
    issues: list[Issue] = []

    def warn(code: str, subject: str, message: str) -> None:
        issues.append(Issue("warning", code, subject, message))

    ids = {i.id for i in setup.instances}
    cables = cable_channels(setup)
    channels = set(cables.values()) | set(m.channels.values())
    for det, ch in m.channels.items():
        if det not in ids:
            warn("measurement_reference", f"measurement channel '{det}'", "no such instance")
        elif det in cables and cables[det] != ch:
            warn(
                "measurement_channel_override",
                f"measurement channel '{det}'",
                f"mapped to {ch}, but its cable goes to {cables[det]}; analyses that name "
                f"'{det}' use {ch}",
            )
    if m.pulse_clock.kind == "source" and m.pulse_clock.source not in ids:
        warn(
            "measurement_reference",
            "measurement pulse clock",
            f"source '{m.pulse_clock.source}' is not an instance",
        )
    names = {str(a.get("name")) for a in setup.analyses}
    for key in m.centers:
        if key not in names and key.rsplit(".", 1)[0] not in names:
            warn("measurement_reference", f"measurement centre '{key}'", "no such analysis")
    for cond in m.conditions:
        for ref in [*cond.herald, *cond.veto]:
            if ref not in channels and ref not in ids:
                warn(
                    "measurement_reference",
                    f"measurement condition '{cond.name}'",
                    f"'{ref}' is neither a tagger channel nor a detector",
                )
    known = {c.name for c in m.conditions}
    for spec in setup.analyses:
        cond = spec.get("condition")
        if cond is not None and cond not in known:
            warn(
                "measurement_reference",
                f"analysis '{spec.get('name')}'",
                f"condition '{cond}' is not defined in the measurement",
            )
    if m.accidentals == "singles":
        warn(
            "measurement_singles_estimator",
            "measurement accidentals",
            "the singles-product accidental estimator is selected; for pulsed light it is not "
            "valid (use 'sideband' or 'auto')",
        )
    return issues


def measured_clock(
    setup: "Setup",
    channels: Sequence[str],
    clock: Mapping[str, float] | Sequence[float],
    clock_ids: Mapping[str, str] | None,
) -> tuple[Mapping[str, float] | Sequence[float], Mapping[str, str] | None, str]:
    """The pulse clock a tag analysis uses under the measurement definition, and a note when
    the measurement replaced the clock found in the data."""
    m = setup.measurement
    if m is None or m.pulse_clock.kind == "sources":
        return clock, clock_ids, ""
    pc = m.pulse_clock
    if pc.kind == "cw":
        return {c: 0.0 for c in channels}, None, "measurement: continuous (CW) acquisition"
    if pc.kind == "rate":
        rate = pc.rate.si
        return (
            {c: rate for c in channels},
            {c: "measurement" for c in channels},
            f"measurement: pulse clock {rate * 1e-6:.6g} MHz for every channel",
        )
    rates = clock if isinstance(clock, Mapping) else {}
    found = [rates[c] for c in channels if (clock_ids or {}).get(c) == pc.source and c in rates]
    if not found:
        return clock, clock_ids, f"measurement: clock source '{pc.source}' clocks no channel"
    return (
        {c: found[0] for c in channels},
        {c: pc.source for c in channels},
        f"measurement: every channel clocked by '{pc.source}' ({found[0] * 1e-6:.6g} MHz)",
    )


def analyse_measured_tags(
    setup: "Setup",
    channels: Mapping[str, Any],
    duration: float,
    clock: Mapping[str, float] | Sequence[float],
    clock_ids: Mapping[str, str] | None,
    specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """``analyse_tags`` under the measurement definition: effective analyses, pulse clock,
    accidental estimator and named conditions (detector ids mapped to their channels). Each
    result records the estimator and why it was used."""
    from sequoia.analysis.correlation import analyse_tags

    specs = effective_analyses(setup) if specs is None else list(specs)
    names = list(channels)
    clock, clock_ids, clock_note = measured_clock(setup, names, clock, clock_ids)
    m = setup.measurement
    choice = m.accidentals if m is not None else "auto"
    pulsed = isinstance(clock, Mapping) and any(v for v in clock.values())
    used_clock = {c: 0.0 for c in names} if choice == "singles" else clock
    mapping = channel_map(setup)
    conditions = {
        c.name: ([_channel(r, mapping) for r in c.herald], [_channel(r, mapping) for r in c.veto])
        for c in (m.conditions if m is not None else [])
    }
    out = analyse_tags(channels, duration, specs, used_clock, clock_ids, choice, conditions)
    for result in out.values():
        if result.get("type") != "coincidences":
            continue
        notes = [clock_note] if clock_note else []
        if choice == "singles":
            result["timing"] = "measurement"
            notes.append(
                "singles-product estimator chosen in the measurement definition"
                + ("; not valid for pulsed light" if pulsed else "")
            )
            result["timing_note"] = "; ".join(notes)
        elif choice == "sideband" and result.get("accidental_model") == "singles":
            notes.append(
                "sideband estimator requested but no common pulse clock: singles product used"
            )
            result["timing_note"] = "; ".join([*notes, result.get("timing_note", "")]).strip("; ")
        elif notes:
            result["timing_note"] = "; ".join([*notes, result.get("timing_note", "")]).strip("; ")
        result["accidental_choice"] = choice
    return out


def _fmt_s(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9), ("ps", 1e-12)):
        if abs(seconds) >= scale or unit == "ps":
            return f"{seconds / scale:.4g} {unit}"
    return f"{seconds:g} s"


def measurement_card(setup: "Setup", outcome=None) -> list[tuple[str, str]]:
    """Rows for a compact measurement card: what is measured and how, with every time named."""
    m = setup.measurement
    rows: list[tuple[str, str]] = []
    if m is not None and m.purpose:
        rows.append(("Purpose", m.purpose))
    mapping = channel_map(setup)
    rows.append(
        (
            "Detector -> channel",
            ", ".join(
                f"{d} -> {ch}" + (" (measurement override)" if how == "measurement" else "")
                for d, (ch, how) in sorted(mapping.items())
            )
            or "no detector is cabled to a tagger",
        )
    )
    default = m.coincidence_window if m is not None else None
    rows.append(
        (
            "Coincidence window",
            (
                f"{_fmt_s(default.si)} full width (default for analyses without one)"
                if default
                else ""
            )
            or "per analysis (full width)",
        )
    )
    for spec in effective_analyses(setup):
        if spec.get("type") not in WINDOWED:
            continue
        w = spec.get("window")
        try:
            from sequoia.analysis.correlation import _duration

            text = f"{_fmt_s(_duration(w, 'window'))} full width" if w is not None else "not set"
            centre = spec.get("center", spec.get("center_a", "auto"))
            text += (
                ", centre searched in the data"
                if isinstance(centre, str)
                else f", centre {_fmt_s(_duration(centre, 'center'))}"
            )
        except Exception:
            text = f"invalid ({w!r})"
        rows.append((f"  analysis '{spec.get('name')}'", text))
    rates = getattr(outcome, "rates", None) if outcome is not None else None
    periods = []
    if rates is not None:
        for sid, s in rates.sources.items():
            if s.get("pulsed"):
                rep = s["repetition_rate_hz"]
                periods.append(f"{sid}: {_fmt_s(1.0 / rep)} ({rep * 1e-6:.4g} MHz)")
            else:
                periods.append(f"{sid}: CW (no repetition period)")
    rows.append(("Repetition period (sources)", "; ".join(periods) or "run the simulation"))
    clock = m.pulse_clock if m is not None else PulseClock()
    rows.append(
        (
            "Pulse clock (tag analysis)",
            {
                "sources": "from the sources clocking each channel",
                "cw": "continuous (CW)",
                "source": f"source '{clock.source}'",
                "rate": f"{clock.rate.si * 1e-6:.6g} MHz (explicit)" if clock.rate else "",
            }[clock.kind],
        )
    )
    if m is not None and m.acquisition_duration is not None:
        dur = f"{_fmt_s(m.acquisition_duration.si)} per setting"
    else:
        dur = "not set (each tab uses its own duration)"
    repeats = m.repeats if m is not None else 1
    rows.append(("Acquisition duration", f"{dur}, {repeats} repeat(s)"))
    rows.append(
        (
            "Accidental estimator",
            {
                "auto": "sidebands one pulse period away with a common pulse clock, else the "
                "singles product",
                "singles": "singles product S_a S_b x window",
                "sideband": "sidebands one pulse period away; for CW light off-peak windows "
                "at +-max(10 x window, 20 ns, 5 x correlation width)",
            }[m.accidentals if m is not None else "auto"],
        )
    )
    rows.append(
        (
            "Dead-time model",
            "none (dead times set to zero)"
            if m is not None and m.dead_time == "none"
            else "non-paralyzable (engine default)",
        )
    )
    if m is not None and m.conditions:
        rows.append(
            (
                "Conditions",
                "; ".join(
                    f"{c.name}: click on {' and '.join(c.herald)}"
                    + (f", none on {' or '.join(c.veto)}" if c.veto else "")
                    for c in m.conditions
                ),
            )
        )
    return rows
