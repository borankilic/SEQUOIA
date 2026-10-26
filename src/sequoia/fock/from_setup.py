from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from sequoia.components.base import PortDirection, PortKind
from sequoia.data import Scalar
from sequoia.engines.cache import Cancelled, RunCache, check_cancel
from sequoia.engines.classical import build_network
from sequoia.fock.analysis import ExpectedCounts, Samples, expected_counts, sample
from sequoia.fock.circuit import FockError
from sequoia.fock.compile import (
    CompiledNetwork,
    FockCompileError,
    _reachable,
    compile_setup_network,
)
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import CANCELLED_SAMPLING, Experiment, FockResult, run_experiment
from sequoia.fock.extract import Extraction, extract_circuit, extraction_experiment
from sequoia.fock.records import run_record, write_run_bundle
from sequoia.fock.setup_sources import (
    MAX_N,
    MERGE_RTOL,
    PRIVATE_OFFSET,
    SOURCE_TAG_STRIDE,
    TAIL,
    WINDOW_RTOL,
    IncompatibleWindowsError,
    InputDescription,
    _discover,
    _make_plan,
    _Plan,
    list_inputs,
)
from sequoia.library.store import Library
from sequoia.setup.model import Setup

KIND_TO_MODEL = {"SPAD": "threshold", "SNSPD": "threshold"}

__all__ = [
    "MAX_N",
    "MERGE_RTOL",
    "PRIVATE_OFFSET",
    "SOURCE_TAG_STRIDE",
    "TAIL",
    "WINDOW_RTOL",
    "ComponentRow",
    "Extraction",
    "IncompatibleWindowsError",
    "InputDescription",
    "MeasurementCondition",
    "SetupExperiment",
    "SetupResult",
    "build_experiment_from_setup",
    "extract_circuit",
    "extraction_experiment",
    "list_components",
    "list_inputs",
    "measurement_conditions",
    "run_record",
    "run_setup_experiment",
    "write_run_bundle",
]


@dataclass(frozen=True)
class ComponentRow:
    id: str
    type: str
    category: str
    part: str
    summary: str
    ports: str


DILUTE_LOAD = 0.1


@dataclass
class SetupExperiment:
    experiment: Experiment
    compiled: CompiledNetwork
    input_labels: list[str]
    detector_ids: list[str]
    output_labels: list[str]
    inputs: list[InputDescription]
    notes: list[str]
    components: list[ComponentRow]
    window_s: float
    rate_hz: float
    duration_s: float
    omitted_weight: float
    plan: _Plan
    setup: Setup
    library: Library
    dead_times: dict[str, float] = field(default_factory=dict)

    @property
    def n_detectors(self) -> int:
        return len(self.detector_ids)


@dataclass
class SetupResult:
    setup_experiment: SetupExperiment
    result: FockResult
    detected: dict[tuple[int, ...], float]
    detector_ids: list[str]
    photon_numbers: dict[tuple[int, ...], float]
    per_mode: dict[tuple[int, ...], float]
    counts: ExpectedCounts
    samples: Samples | None
    omitted_weight: float
    messages: list[str]
    dead_time_load: dict[str, float] = field(default_factory=dict)
    status: str = "complete"

    @property
    def mass(self) -> float:
        return float(sum(self.detected.values()))

    @property
    def outside_dilute_regime(self) -> dict[str, float]:
        return {d: v for d, v in self.dead_time_load.items() if v > DILUTE_LOAD}


@dataclass(frozen=True)
class MeasurementCondition:
    """A named condition of the measurement on the circuit's detectors: each herald group
    (the detectors behind one channel or detector id) must click, no veto group may."""

    name: str
    description: str
    herald: tuple[tuple[str, tuple[int, ...]], ...]
    veto: tuple[tuple[str, tuple[int, ...]], ...]

    def accepts(self, state: tuple[int, ...]) -> bool:
        return all(any(state[k] > 0 for k in g) for _, g in self.herald) and not any(
            state[k] > 0 for _, g in self.veto for k in g
        )

    @property
    def rule(self) -> str:
        text = "click on " + " and ".join(r for r, _ in self.herald)
        return text + (", none on " + " or ".join(r for r, _ in self.veto) if self.veto else "")


def measurement_conditions(setup: Setup, detector_ids: Sequence[str]) -> list[MeasurementCondition]:
    """The measurement's named conditions mapped from channel or detector ids to indices of
    ``detector_ids``; a reference that is not a detector of the circuit raises ``FockError``."""
    from sequoia.measurement import channel_map

    m = setup.measurement
    if m is None:
        return []
    mapping = channel_map(setup)
    instances = {i.id for i in setup.instances}
    index = {d: k for k, d in enumerate(detector_ids)}

    def group(cond: str, ref: str) -> tuple[str, tuple[int, ...]]:
        if ref in instances:
            dets = [ref]
        else:
            dets = [d for d, (ch, _) in mapping.items() if ch == ref]
            if not dets:
                raise FockError(
                    f"measurement condition '{cond}': '{ref}' is neither a detector nor a "
                    f"tagger channel of this setup (channels: "
                    f"{', '.join(sorted({ch for ch, _ in mapping.values()})) or 'none'})"
                )
        found = tuple(index[d] for d in dets if d in index)
        if not found:
            raise FockError(
                f"measurement condition '{cond}': '{ref}' ({', '.join(dets)}) is not a "
                "single-photon detector reached by the photon sources"
            )
        return ref, found

    return [
        MeasurementCondition(
            c.name,
            c.description,
            tuple(group(c.name, r) for r in c.herald),
            tuple(group(c.name, r) for r in c.veto),
        )
        for c in m.conditions
    ]


def _fmt_value(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, Scalar):
        return f"{v.value:g}" if v.unit.strip() in ("", "1") else f"{v.value:g} {v.unit}"
    if isinstance(v, (bool, int, str)):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    if type(v).__name__ == "Curve":
        return "curve"
    return None


def list_components(setup: Setup, library: Library) -> list[ComponentRow]:
    net = build_network(setup, library)
    rows: list[ComponentRow] = []
    for inst in setup.instances:
        comp = net.components[inst.id]
        parts = []
        try:
            for k, v in comp.settings.model_dump(mode="json").items():
                if v is not None:
                    parts.append(f"{k}={v}")
        except Exception:
            parts = []
        shown = 0
        for name in type(comp.params).model_fields:
            text = _fmt_value(getattr(comp.params, name, None))
            if text is None or name in ("name", "description"):
                continue
            parts.append(f"{name}={text}")
            shown += 1
            if shown >= 5:
                break
        ports = []
        for p in comp.ports():
            tag = "" if p.kind == PortKind.ELECTRICAL else ":H,V"
            ports.append(f"{p.name}{tag} ({'in' if p.direction == PortDirection.IN else 'out'})")
        rows.append(
            ComponentRow(
                inst.id,
                comp.type_name,
                comp.category,
                inst.part,
                ", ".join(parts),
                ", ".join(ports),
            )
        )
    for link in setup.links:
        if link.kind == "cable":
            text = f"{link.src} -> {link.dst}, delay {_fmt_value(link.delay)}"
            part = ""
        else:
            length = net.lengths.get(link.id)
            text = f"{link.src} -> {link.dst}"
            if length is not None:
                text += f", length {length * 1e3:.6g} mm"
            part = getattr(link, "fiber", "") or ""
            if link.kind == "fiber" and link.connector_loss_db:
                text += f", connector loss {link.connector_loss_db:g} dB"
        rows.append(ComponentRow(link.id, f"link ({link.kind})", "link", part, text, ""))
    return rows


def build_experiment_from_setup(
    setup: Setup,
    library: Library,
    *,
    wavelength: float | None = None,
    n_max: int | None = None,
    duration: float = 1.0,
    detectors_mode: str = "setup",
    should_cancel: Callable[[], bool] | None = None,
    cache: RunCache | None = None,
    classical=None,
    limits: Limits | None = None,
    sources: Sequence[str] | None = None,
) -> SetupExperiment:
    """Photon-number experiment for the circuit drawn in ``setup``.

    Inputs are found automatically: every FockSource (N photons per pulse) and every pair source
    (thermal pair-number statistics with mean ``rate * window`` and K Schmidt modes). All
    sources must share one detection window. Detectors are the setup's single-photon detectors
    reachable from the sources: ``detectors_mode`` "setup" gives threshold detectors for SPAD and
    SNSPD, "pnr" ideal photon-number-resolving ones.
    """
    if detectors_mode not in ("setup", "pnr"):
        raise FockError("detectors_mode must be 'setup' or 'pnr'")
    limits = limits or Limits()
    found = _discover(setup, library, classical, should_cancel, cache)
    plan = _make_plan(found, wavelength, n_max, limits, sources)
    net = found.net
    check_cancel(should_cancel)
    specs = [f"{c.inst}.{c.port}" for c in plan.channels]
    reach = _reachable(net, {c.inst for c in plan.channels})
    det_ids = [
        i for i, c in net.components.items() if c.detector_model() is not None and i in reach
    ]
    if not det_ids:
        raise FockCompileError(
            "no single-photon detector (SPAD, SNSPD) is reachable from the photon sources; "
            "a photodiode reads classical power and has no photon-number output"
        )
    kinds = {
        d: (
            "pnr"
            if detectors_mode == "pnr"
            else KIND_TO_MODEL.get(net.components[d].detector_model().kind, "pnr")
        )
        for d in det_ids
    }
    compiled = compile_setup_network(
        setup,
        library,
        inputs=specs,
        detectors=det_ids,
        wavelengths=[c.wavelength for c in plan.channels],
        bases=[c.basis for c in plan.channels],
        seed_modes=[c.mode for c in plan.channels],
        species=[c.species for c in plan.channels],
        gate_window=plan.window,
        should_cancel=should_cancel,
        detector_kind=kinds,
        net=net,
    )
    mixture = plan.make(lambda c, p: 2 * c + p)
    n = compiled.circuit.n_modes
    labels = [str(m) for m in compiled.input_modes] + [""] * (n - len(compiled.input_modes))
    window = plan.window
    exp = Experiment(
        compiled.circuit,
        [0] * n,
        detectors=list(compiled.detectors),
        mode_labels=labels,
        name=setup.name,
        limits=limits,
        input_mixture=mixture,
        rate_hz=1.0 / window,
        window_s=window,
        duration_s=float(duration),
    )
    notes = list(plan.notes)
    notes.extend(compiled.warnings)
    notes.extend(f"{i.subject}: {i.message}" for i in found.issues if i.severity == "warning")
    notes.extend(mixture.notes)
    lams = sorted({round(c.wavelength * 1e9, 3) for c in plan.channels})
    notes.append(
        "photon species of "
        + ", ".join(f"{x:g} nm" for x in lams)
        + " accumulate path phase 2 pi OPL / lambda at their own wavelength; a single wavelength "
        "per species (no spectrum), perfect spatial-mode overlap where paths recombine, no "
        "timing, jitter or dead time"
    )
    notes.append(
        "pair-number statistics: thermal with K equal-weight Schmidt modes; photons of one "
        "Schmidt mode bunch, pairs in different modes are mutually distinguishable"
    )
    if plan.omitted > TAIL:
        notes.append(
            f"omitted input weight (pair numbers beyond the truncation): {plan.omitted:.3g}"
        )
    return SetupExperiment(
        exp,
        compiled,
        [str(m) for m in compiled.input_modes],
        det_ids,
        [str(m) for m in compiled.output_modes],
        plan.descriptions,
        list(dict.fromkeys(notes)),
        list_components(setup, library),
        window,
        1.0 / window,
        float(duration),
        plan.omitted,
        plan,
        setup,
        library,
        {d: float(net.components[d].detector_model().dead_time) for d in det_ids},
    )


def run_setup_experiment(
    se: SetupExperiment,
    *,
    shots: int = 0,
    seed: int | None = None,
    backend: str = "native",
    duration: float | None = None,
    should_cancel=None,
    progress=None,
) -> SetupResult:
    exp = replace(se.experiment, shots=0, backend=backend, conditions={})
    res = run_experiment(exp, should_cancel, progress)
    n_det = se.n_detectors
    n_out = 2 * n_det
    detected: dict[tuple[int, ...], float] = {}
    for s, p in res.probabilities.items():
        detected[s[:n_det]] = detected.get(s[:n_det], 0.0) + p
    detected = dict(sorted(detected.items()))
    per_mode: dict[tuple[int, ...], float] = {}
    for s, p in res.raw.probs.items():
        per_mode[s[:n_out]] = per_mode.get(s[:n_out], 0.0) + p
    per_mode = dict(sorted(per_mode.items()))
    pn: dict[tuple[int, ...], float] = {}
    for s, p in per_mode.items():
        key = tuple(s[2 * k] + s[2 * k + 1] for k in range(n_det))
        pn[key] = pn.get(key, 0.0) + p
    dur = se.duration_s if duration is None else float(duration)
    counts = expected_counts(detected, se.rate_hz, None, dur)
    messages = list(res.messages)
    status = res.status
    samples = None
    if shots > 0 and detected:
        try:
            samples = sample(detected, shots, seed, sequence=False, should_cancel=should_cancel)
        except Cancelled:
            status = "partial"
            messages.append(CANCELLED_SAMPLING)
    load: dict[str, float] = {}
    for k, d in enumerate(se.detector_ids):
        p_click = sum(p for s, p in detected.items() if s[k] > 0)
        load[d] = p_click * counts.windows_per_second * se.dead_times.get(d, 0.0)
        if load[d] > DILUTE_LOAD:
            messages.append(
                f"'{d}': the potential click rate {p_click * counts.windows_per_second:.4g}/s "
                f"times the dead time {se.dead_times[d] * 1e9:.4g} ns is {load[d]:.3g}, outside "
                f"the dilute regime (< {DILUTE_LOAD:g}); this model omits dead time, so its "
                "click rates are potential clicks, not recorded counts. Attenuate the input or "
                "use the Monte Carlo engine for recorded tag rates"
            )
    return SetupResult(
        se,
        res,
        detected,
        list(se.detector_ids),
        dict(sorted(pn.items())),
        per_mode,
        counts,
        samples,
        res.raw.omitted_weight,
        messages,
        load,
        status,
    )
