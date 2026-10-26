import math
from collections.abc import Callable
from dataclasses import dataclass, field

from sequoia.engines.cache import Cancelled, check_cancel
from sequoia.errors import SequoiaError
from sequoia.fock.analysis import count_variance
from sequoia.fock.engine import Limits
from sequoia.fock.experiment import Experiment, FockResult, run_experiment
from sequoia.fock.extract import Extraction
from sequoia.fock.from_setup import (
    ComponentRow,
    InputDescription,
    SetupExperiment,
    SetupResult,
    build_experiment_from_setup,
    extraction_experiment,
    list_components,
    list_inputs,
    run_setup_experiment,
)
from sequoia.fock.preflight import Estimate, estimate_experiment
from sequoia.library.store import Library
from sequoia.setup.model import Setup

State = tuple[int, ...]
CHUNK = 20_000
CANCELLED_VIEWS = "partial result: cancelled while preparing the tables; the distribution is kept"


@dataclass(frozen=True)
class SetupRequest:
    """One Photon counts action on the drawn setup: ``kind`` is "import", "run" or "extract"."""

    kind: str
    setup: Setup
    library: Library
    duration: float = 1.0
    n_max: int | None = None
    detectors_mode: str = "setup"
    max_photons: int = Limits().max_photons
    shots: int = 0
    seed: int | None = None
    backend: str = "native"


@dataclass(frozen=True)
class DetectorRow:
    id: str
    kind: str
    efficiency: float
    dark: float
    p_click: float
    mean: float
    rate: float
    total: float
    sigma: float


@dataclass(frozen=True)
class SetupViews:
    """Table data of a setup run, sorted and aggregated off the GUI thread. ``patterns`` rows are
    (label, p, rate, total, sigma); ``marginals`` rows (kind, name, n, p arrive, p detected)."""

    states: tuple[tuple[State, float], ...]
    detectors: tuple[DetectorRow, ...]
    patterns: tuple[tuple[str, float, float, float, float], ...]
    marginals: tuple[tuple[str, str, int, float, float | None], ...]
    photon_numbers: dict[int, float]


@dataclass(frozen=True)
class SetupOutcome:
    request: SetupRequest
    components: list[ComponentRow] = field(default_factory=list)
    inputs: list[InputDescription] = field(default_factory=list)
    se: SetupExperiment | None = None
    result: SetupResult | None = None
    extraction: tuple[Experiment, Extraction] | None = None
    estimate: Estimate | None = None
    views: SetupViews | None = None
    error: str = ""
    status: str = "complete"

    @property
    def kind(self) -> str:
        return self.request.kind

    @property
    def rejected(self) -> bool:
        return self.kind == "run" and self.estimate is not None and not self.estimate.ok


@dataclass(frozen=True)
class EditorOutcome:
    result: FockResult
    ranked: tuple[tuple[State, float], ...]
    estimate: Estimate | None
    status: str


def by_photon_number(dist: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for s, p in dist.items():
        out[sum(s)] = out.get(sum(s), 0.0) + p
    return out


def ranked(dist: dict, should_cancel=None) -> tuple[tuple[State, float], ...]:
    check_cancel(should_cancel)
    return tuple(sorted(dist.items(), key=lambda kv: -kv[1]))


def _chunks(items, should_cancel):
    for i, item in enumerate(items):
        if i % CHUNK == 0:
            check_cancel(should_cancel)
        yield item


def setup_views(res: SetupResult, should_cancel=None) -> SetupViews:
    se = res.setup_experiment
    ids = res.detector_ids
    cnt = res.counts
    models = {d: spec.model for d, spec in zip(se.detector_ids, se.compiled.detectors, strict=True)}
    detected = list(res.detected.items())
    detectors = []
    for k, d in enumerate(ids):
        p_click = mean = 0.0
        for s, p in _chunks(detected, should_cancel):
            if s[k] > 0:
                p_click += p
            mean += s[k] * p
        rate = p_click * cnt.windows_per_second
        model = models[d]
        detectors.append(
            DetectorRow(
                d,
                model.kind,
                model.efficiency,
                model.dark,
                p_click,
                mean,
                rate,
                rate * cnt.duration_s,
                math.sqrt(count_variance(p_click, cnt.n_windows)),
            )
        )
    sums: dict[str, float] = {}
    for s, p in _chunks(detected, should_cancel):
        label = " + ".join(
            f"{d}" if n == 1 else f"{d} x{n}" for d, n in zip(ids, s, strict=True) if n > 0
        )
        sums[label or "no click"] = sums.get(label or "no click", 0.0) + p
    patterns = []
    for label, p in sorted(sums.items(), key=lambda kv: -kv[1]):
        rate = p * cnt.windows_per_second
        patterns.append(
            (label, p, rate, rate * cnt.duration_s, math.sqrt(count_variance(p, cnt.n_windows)))
        )
    marginals: list[tuple[str, str, int, float, float | None]] = []
    numbers = list(res.photon_numbers.items())
    for k, d in enumerate(ids):
        arrive: dict[int, float] = {}
        for s, p in _chunks(numbers, should_cancel):
            arrive[s[k]] = arrive.get(s[k], 0.0) + p
        seen: dict[int, float] = {}
        for s, p in _chunks(detected, should_cancel):
            seen[s[k]] = seen.get(s[k], 0.0) + p
        for n in sorted(set(arrive) | set(seen)):
            marginals.append(("detector", d, n, arrive.get(n, 0.0), seen.get(n, 0.0)))
    per_mode = list(res.per_mode.items())
    for m, lab in enumerate(se.output_labels):
        dist: dict[int, float] = {}
        for s, p in _chunks(per_mode, should_cancel):
            dist[s[m]] = dist.get(s[m], 0.0) + p
        for n in sorted(dist):
            marginals.append(("mode", lab, n, dist[n], None))
    return SetupViews(
        ranked(res.detected, should_cancel),
        tuple(detectors),
        tuple(patterns),
        tuple(marginals),
        dict(sorted(by_photon_number(res.photon_numbers).items())),
    )


def rejection_text(estimate: Estimate) -> str:
    return "not run: " + "; ".join(estimate.errors) + f". Estimate: {estimate.summary()}"


def run_setup_request(
    req: SetupRequest,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SetupOutcome:
    """Import, run or extract the drawn setup. A run is refused before computing when its
    estimate exceeds the limits; cancellation after the distribution is computed gives a result
    labelled ``status='partial'``."""
    if progress is not None:
        progress(0, 0)
    try:
        components = list_components(req.setup, req.library)
    except SequoiaError as exc:
        return SetupOutcome(req, error=str(exc))
    limits = Limits(max_photons=req.max_photons)
    se = result = extraction = estimate = views = None
    inputs: list[InputDescription] = []
    error, status = "", "complete"
    try:
        se = build_experiment_from_setup(
            req.setup,
            req.library,
            n_max=req.n_max,
            duration=req.duration,
            detectors_mode=req.detectors_mode,
            should_cancel=should_cancel,
            limits=limits,
        )
        inputs = se.inputs
        estimate = estimate_experiment(se.experiment, shots=req.shots, backend=req.backend)
        if req.kind == "run" and estimate.ok:
            result = run_setup_experiment(
                se,
                shots=req.shots,
                seed=req.seed,
                backend=req.backend,
                should_cancel=should_cancel,
                progress=progress,
            )
            status = result.status
            try:
                views = setup_views(result, should_cancel if status == "complete" else None)
            except Cancelled:
                status = "partial"
        elif req.kind == "run":
            error = rejection_text(estimate)
        elif req.kind == "extract":
            extraction = extraction_experiment(se, should_cancel=should_cancel)
    except Cancelled:
        raise
    except (SequoiaError, ValueError) as exc:
        error = str(exc)
        try:
            inputs = list_inputs(
                req.setup, req.library, n_max=req.n_max, should_cancel=should_cancel, limits=limits
            )
        except (SequoiaError, ValueError):
            pass
    return SetupOutcome(
        req, components, inputs, se, result, extraction, estimate, views, error, status
    )


def run_editor_experiment(
    exp: Experiment,
    estimate: Estimate | None = None,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EditorOutcome:
    result = run_experiment(exp, should_cancel, progress)
    return EditorOutcome(result, ranked(result.probabilities), estimate, result.status)
