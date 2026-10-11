import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sequoia.engines.cache import ResourceLimitError, RunCache, check_cancel
from sequoia.engines.classical import SimulationResult
from sequoia.engines.rates import (
    FWHM_TO_SIGMA,
    ROLES,
    _Channel,
    _Engine,
    _series_survival,
    _Source,
)
from sequoia.errors import Issue
from sequoia.library.store import Library
from sequoia.setup.model import Setup

MAX_EVENTS = 60_000_000
MAX_AFTERPULSE_GENERATIONS = 8
EXCESS_PROBABILITY = 1.0 + 1e-6


@dataclass
class TimeTags:
    """Time-tag record: sorted arrival times (s) per tagger channel over ``duration`` seconds."""

    channels: dict[str, np.ndarray]
    duration: float
    seed: int | None = None
    issues: list[Issue] = field(default_factory=list)
    repetition_rates: list[float] = field(default_factory=list)
    truth: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {name: int(len(t)) for name, t in self.channels.items()}

    def rates(self) -> dict[str, float]:
        return {name: len(t) / self.duration for name, t in self.channels.items()}


@dataclass
class _PairSampler:
    profile: Any
    shift: float


@dataclass
class _SourceModel:
    inst: str
    rate: float
    repetition_rate: float | None
    patterns: list[tuple[tuple[str, ...], float]]
    double_hz: dict[str, float]
    delays: dict[str, float]
    samplers: dict[tuple[str, str], _PairSampler]


def dead_time_filter(times: np.ndarray, dead: float) -> np.ndarray:
    """Non-paralyzable dead time: drop events within ``dead`` of the last kept event.

    ``times`` must be sorted. Events with a quiet predecessor are kept without a loop; only the
    bursts shorter than the dead time are walked event by event.
    """
    n = len(times)
    if dead <= 0.0 or n < 2:
        return times
    gap = np.diff(times, prepend=-np.inf)
    starts = np.flatnonzero(gap >= dead)
    if len(starts) == n:
        return times
    keep = np.zeros(n, dtype=bool)
    keep[starts] = True
    bounds = np.append(starts, n)
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if hi - lo < 2:
            continue
        block = times[lo:hi]
        i = 0
        size = len(block)
        while i < size:
            keep[lo + i] = True
            i = int(np.searchsorted(block, block[i] + dead, side="left"))
    return times[keep]


def _jitter(rng: np.random.Generator, n: int, fwhm: float, fraction: float, tau: float):
    out = np.zeros(n)
    if fwhm > 0.0:
        out += rng.normal(0.0, fwhm * FWHM_TO_SIGMA, n)
    if fraction > 0.0 and tau > 0.0 and n:
        tail = rng.random(n) < fraction
        out[tail] += rng.exponential(tau, int(tail.sum()))
    return out


def _poisson_times(rng: np.random.Generator, rate: float, duration: float) -> np.ndarray:
    if rate <= 0.0:
        return np.empty(0)
    return rng.uniform(0.0, duration, rng.poisson(rate * duration))


class _Builder:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.tagged: dict[str, _Channel] = {}
        for channel in engine.channels.values():
            self.tagged.setdefault(channel.detector, channel)

    def delay_of(self, src: _Source, det: str) -> float:
        singles = src.singles.get(det, {})
        role = max(ROLES, key=lambda r: singles.get(r, 0.0))
        paths = src.delays.get((det, role), [])
        return max(paths, key=lambda p: p[1])[0] if paths else 0.0

    def source_model(self, src: _Source) -> _SourceModel:
        eng = self.engine
        dets = list(self.tagged)
        rate = src.rate
        single = {d: sum(src.singles[d].values()) / rate for d in dets}
        both: dict[tuple[str, str], float] = {}
        samplers: dict[tuple[str, str], _PairSampler] = {}
        same: dict[str, float] = {}
        for i, a in enumerate(dets):
            ca = self.tagged[a]
            data = eng.pair_profile(ca, ca, None, jitter=False)
            same[a] = 0.5 * sum(t.weight for t in data["terms"] if t.source == src.inst) / rate
            for b in dets[i + 1 :]:
                cb = self.tagged[b]
                data = eng.pair_profile(ca, cb, None, jitter=False)
                terms = [t for t in data["terms"] if t.source == src.inst]
                weight = sum(t.weight for t in terms)
                if weight <= 0.0:
                    continue
                both[(a, b)] = weight / rate
                strongest = max(terms, key=lambda t: t.weight)
                samplers[(a, b)] = _PairSampler(strongest.profile, eng.term_shift(strongest))
        patterns: list[tuple[tuple[str, ...], float]] = []
        for a in dets:
            partners = sum(c for (x, y), c in both.items() if a in (x, y))
            alone = single[a] - 2.0 * same[a] - partners
            p_click = max(alone, 0.0) + same[a]
            if p_click > 0.0:
                patterns.append(((a,), p_click))
        for pair, c in both.items():
            patterns.append((pair, c))
        total = sum(p for _, p in patterns)
        if total > EXCESS_PROBABILITY:
            eng.issue(
                "warning",
                "pair_probability_exceeds_one",
                f"'{src.inst}'",
                f"the detection probabilities per pair add up to {total:.3g}; they were "
                "renormalised. The two-photon description is not valid at this source setting",
            )
            patterns = [(d, p / total) for d, p in patterns]
        delays = {d: self.delay_of(src, d) for d in dets}
        rep = src.emission.source.repetition_rate
        double_hz = {d: same[d] * rate for d in dets}
        return _SourceModel(src.inst, rate, rep, patterns, double_hz, delays, samplers)


def _emission_times(
    rng: np.random.Generator,
    model: _SourceModel,
    duration: float,
    schmidt_number: float | None = None,
) -> np.ndarray:
    if not model.repetition_rate:
        return rng.uniform(0.0, duration, int(rng.poisson(model.rate * duration)))
    pulses = max(int(duration * model.repetition_rate), 1)
    mu = model.rate / model.repetition_rate
    if schmidt_number is None or schmidt_number <= 0 or not math.isfinite(schmidt_number):
        n = int(rng.poisson(model.rate * duration))
        return rng.integers(0, pulses, n) / model.repetition_rate
    k = float(schmidt_number)
    p = k / (k + mu)
    occupied = int(rng.binomial(pulses, 1.0 - p**k))
    if occupied == 0:
        return np.empty(0)
    which = rng.choice(pulses, occupied, replace=False)
    counts = rng.negative_binomial(k, p, occupied)
    while True:
        zero = counts == 0
        if not zero.any():
            break
        counts[zero] = rng.negative_binomial(k, p, int(zero.sum()))
    return np.repeat(which, counts) / model.repetition_rate


def _sample_profile(rng: np.random.Generator, sampler: _PairSampler, n: int) -> np.ndarray:
    p = sampler.profile
    u = rng.random(n)
    k = np.clip(np.searchsorted(p.cum, u, side="right") - 1, 0, len(p.mass) - 1)
    mass = p.mass[k]
    frac = np.where(mass > 0, (u - p.cum[k]) / np.where(mass > 0, mass, 1.0), 0.5)
    return p.x0 + (k + np.clip(frac, 0.0, 1.0)) * p.dtau + sampler.shift


def simulate_timetags(
    setup: Setup,
    library: Library,
    duration: float,
    *,
    seed: int | None = None,
    classical: SimulationResult | None = None,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_events: int = MAX_EVENTS,
    max_memory_bytes: float | None = None,
    schmidt_number: float | None = None,
) -> TimeTags:
    """Monte Carlo time tags for every tagger channel of ``setup``.

    Pairs are emitted as a Poisson process (CW) or per pulse (pulsed, several pairs per pulse
    possible). The detection pattern of each pair (which tagged detectors click) and the delay
    between its clicks come from the same pair amplitudes as the rate engine, so interference
    and polarization correlations are included. Detector dark counts, classical leakage,
    afterpulsing, jitter with exponential tail, detector and tagger dead times and channel
    delays are applied per event. Pair-number statistics are Poissonian unless
    ``schmidt_number`` is given: pulsed sources then emit a negative-binomial number of pairs per
    pulse with that number of modes (thermal statistics, mean unchanged). Thermal bunching of
    continuous sources on the coherence-time scale is not modelled.
    """
    if not duration > 0 or not math.isfinite(duration):
        raise ValueError("duration must be a positive number of seconds")
    engine = _Engine(setup, library, classical, should_cancel, cache, max_memory_bytes)
    engine.discover()
    engine.emit_sources()
    engine.transfer_sources()
    engine.cancel_point()
    engine.background()
    builder = _Builder(engine)
    rng = np.random.default_rng(seed)

    models = []
    for src in engine.sources:
        if src.rate > 0 and builder.tagged:
            models.append(builder.source_model(src))
    expected = 0.0
    for det_id in builder.tagged:
        expected += engine.det_data[det_id]["incident_hz"] * duration
    if expected > max_events:
        raise ResourceLimitError(
            f"about {expected:.3g} detector events would be generated (limit {max_events}); "
            "reduce the duration or the source rate"
        )

    arrivals: dict[str, list[np.ndarray]] = {d: [] for d in builder.tagged}
    total = len(models) + len(builder.tagged)
    step = 0
    for model in models:
        check_cancel(should_cancel)
        times = _emission_times(rng, model, duration, schmidt_number)
        n = len(times)
        if n and model.patterns:
            cum = np.cumsum([p for _, p in model.patterns])
            which = np.searchsorted(cum, rng.random(n), side="right")
            for index, (dets, _p) in enumerate(model.patterns):
                chosen = times[which == index]
                if len(chosen) == 0:
                    continue
                if len(dets) == 1:
                    arrivals[dets[0]].append(chosen + model.delays[dets[0]])
                else:
                    a, b = dets
                    t_b = chosen + model.delays[b]
                    arrivals[b].append(t_b)
                    arrivals[a].append(
                        t_b + _sample_profile(rng, model.samplers[(a, b)], len(chosen))
                    )
        step += 1
        if progress:
            progress(step, total)

    channels: dict[str, np.ndarray] = {}
    for det_id in builder.tagged:
        check_cancel(should_cancel)
        info = engine.det_data[det_id]
        dm = engine.detectors[det_id].model
        parts = list(arrivals[det_id])
        parts.append(_poisson_times(rng, info["dark_hz"], duration))
        parts.append(_poisson_times(rng, info["leakage_hz"], duration))
        base = np.sort(np.concatenate(parts)) if parts else np.empty(0)
        registered = dead_time_filter(base, dm.dead_time)
        if dm.afterpulse_probability > 0.0 and dm.afterpulse_tau > 0.0 and len(registered):
            after = np.empty(0)
            parents = np.empty(0)
            for _ in range(MAX_AFTERPULSE_GENERATIONS):
                fresh = registered[~np.isin(registered, parents)]
                if len(fresh) == 0:
                    break
                parents = np.concatenate([parents, fresh])
                spawn = fresh[rng.random(len(fresh)) < dm.afterpulse_probability]
                if len(spawn) == 0:
                    break
                after = np.concatenate(
                    [after, spawn + rng.exponential(dm.afterpulse_tau, len(spawn))]
                )
                registered = dead_time_filter(np.sort(np.concatenate([base, after])), dm.dead_time)
        registered = registered[registered < duration] if len(registered) else registered
        t = registered + _jitter(
            rng, len(registered), dm.jitter_fwhm, dm.jitter_tail_fraction, dm.jitter_tail_tau
        )
        group = [c for c in engine.channels.values() if c.detector == det_id]
        for ch in group:
            tagged = t + _jitter(rng, len(t), ch.tagger_model.jitter_fwhm, 0.0, 0.0)
            tagged = dead_time_filter(np.sort(tagged), ch.tagger_model.dead_time)
            channels[ch.name] = tagged + ch.delay
        step += 1
        if progress:
            progress(step, total)
    double: dict[str, float] = {}
    for m in models:
        for det_id, hz in m.double_hz.items():
            double[det_id] = double.get(det_id, 0.0) + hz
    truth = {
        "double_click_hz": double,
        "sources": {m.inst: {"rate_hz": m.rate, "patterns": m.patterns} for m in models},
        "detectors": engine.det_data,
    }
    return TimeTags(
        channels,
        duration,
        seed,
        list(engine.issues),
        [m.repetition_rate for m in models if m.repetition_rate],
        truth,
    )


@dataclass(frozen=True)
class Comparison:
    name: str
    quantity: str
    monte_carlo: float
    sigma: float
    expected: float

    @property
    def z(self) -> float:
        if self.sigma <= 0:
            return 0.0 if abs(self.monte_carlo - self.expected) < 1e-12 else math.inf
        return (self.monte_carlo - self.expected) / self.sigma

    @property
    def ok(self) -> bool:
        return abs(self.z) <= 3.0


def compare_with_rates(
    tags: TimeTags, analyses: dict[str, dict[str, Any]], rates: Any
) -> list[Comparison]:
    """Monte Carlo results against the rate engine, as (value, sigma, expected) rows.

    Singles and coincidences carry Poisson errors from the recorded counts. The engine's singles
    are photon detection rates; a threshold detector clicks once when both photons of a pair
    reach it, so the expected click rate subtracts the engine's own double-detection rate.
    The heralded triple rate is compared with the Poisson (uncorrelated multi-pair) expectation
    built from the engine's own coincidence and singles rates, since the engine's thermal
    bunching terms are not part of the Monte Carlo.
    """
    rows: list[Comparison] = []
    double = tags.truth.get("double_click_hz", {})
    for name, ch in rates.channels.items():
        if name in tags.channels:
            n = len(tags.channels[name])
            expected = ch["singles_hz"]
            lost = double.get(ch["detector_instance"], 0.0)
            if lost > 0:
                incident = ch["incident_hz"] - lost
                expected = incident * _series_survival(
                    incident, ch["dead_time_detector_s"], ch["dead_time_tagger_s"]
                )
            rows.append(
                Comparison(
                    name,
                    "singles_hz",
                    n / tags.duration,
                    math.sqrt(max(n, 1)) / tags.duration,
                    expected,
                )
            )
    for name, mc in analyses.items():
        ref = rates.analyses.get(name)
        if ref is None:
            continue
        if mc["type"] == "coincidences":
            rows.append(
                Comparison(name, "true_hz", mc["true_hz"], mc["true_sigma_hz"], ref["true_hz"])
            )
            n = mc["counts"]
            rows.append(
                Comparison(
                    name,
                    "total_hz",
                    mc["total_hz"],
                    math.sqrt(max(n, 1)) / tags.duration,
                    ref["total_hz"],
                )
            )
        elif mc["type"] == "heralded_g2":
            w = ref["window_s"]
            s_a = rates.channels[ref["a"]]["singles_hz"]
            s_b = rates.channels[ref["b"]]["singles_hz"]
            expected = (
                ref["true_ha_hz"] * s_b * w
                + ref["true_hb_hz"] * s_a * w
                + ref["herald_singles_hz"] * s_a * s_b * w * w
            )
            rows.append(
                Comparison(name, "triples_hz", mc["triples_hz"], mc["triples_sigma_hz"], expected)
            )
    return rows
