import math
from dataclasses import dataclass, field
from typing import Literal

from sequoia.errors import SequoiaError

Model = Literal["fixed_windows", "poisson"]
Split = Literal["per_setting", "total"]


class PlanningError(SequoiaError):
    pass


@dataclass(frozen=True)
class AcquisitionPlan:
    """Expected useful events and anticipated counting precision for one acquisition model.

    ``time_per_setting_s`` is the acquisition duration of one setting; ``total_time_s`` covers
    all ``settings`` and ``repeats``. ``sigma`` and ``relative_precision`` are counting
    statistics only: drift, calibration and detector correlations are not included."""

    quantity: str
    model: str
    time_per_setting_s: float
    settings: int
    repeats: int
    total_time_s: float
    accepted_per_setting: float
    events_per_setting: float
    value: float
    sigma: float | None
    relative_precision: float | None
    success_probability: float = 1.0
    notes: list[str] = field(default_factory=list)


def _positive(name: str, value: float | None, allow_zero: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanningError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise PlanningError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        )
    return v


def _times(time_s, settings: int, repeats: int, split: Split) -> tuple[float, float]:
    if int(settings) != settings or settings < 1 or int(repeats) != repeats or repeats < 1:
        raise PlanningError("settings and repeats must be positive integers")
    if split not in ("per_setting", "total"):
        raise PlanningError("split must be 'per_setting' or 'total'")
    t = _positive("time_s", time_s)
    per = t if split == "per_setting" else t / (settings * repeats)
    return per, per * settings * repeats


def _one_target(time_s, target_relative, target_events) -> None:
    given = [x is not None for x in (time_s, target_relative, target_events)]
    if sum(given) != 1:
        raise PlanningError("give exactly one of time_s, target_relative, target_events")


def plan_probability(
    probability: float,
    windows_per_second: float,
    *,
    success_probability: float = 1.0,
    model: Model = "fixed_windows",
    time_s: float | None = None,
    target_relative: float | None = None,
    target_events: float | None = None,
    settings: int = 1,
    repeats: int = 1,
    split: Split = "per_setting",
    quantity: str = "probability per window",
) -> AcquisitionPlan:
    """Plan the estimate of a probability ``p`` per detection window (per pulse), conditioned
    on a herald with ``success_probability`` per window when that is below 1.

    Accepted windows N = f T p_s. fixed_windows: k ~ Binomial(N, p), sigma(p) = sqrt(p(1-p)/N);
    poisson: sigma(p) = sqrt(p/N). The accepted count is taken at its expectation."""
    _one_target(time_s, target_relative, target_events)
    p = _positive("probability", probability, allow_zero=True)
    if p > 1:
        raise PlanningError("probability must not exceed 1")
    f = _positive("windows_per_second", windows_per_second)
    ps = _positive("success_probability", success_probability)
    if ps > 1:
        raise PlanningError("success_probability must not exceed 1")
    if model not in ("fixed_windows", "poisson"):
        raise PlanningError("model must be 'fixed_windows' or 'poisson'")
    spread = p * (1.0 - p) if model == "fixed_windows" else p
    notes = []
    if time_s is None:
        if p == 0:
            raise PlanningError("an outcome of probability 0 never accumulates events")
        if target_events is not None:
            accepted = _positive("target_events", target_events) / p
        else:
            r = _positive("target_relative", target_relative)
            if spread == 0:
                raise PlanningError(
                    "p = 1 has no binomial spread: any duration gives zero counting uncertainty"
                )
            accepted = spread / (p * p * r * r)
        time_s = accepted / (f * ps)
        split = "per_setting"
    per, total = _times(time_s, settings, repeats, split)
    accepted = f * per * ps
    sigma = math.sqrt(spread / accepted) if accepted > 0 else None
    rel = sigma / p if sigma is not None and p > 0 else None
    if ps < 1:
        notes.append(f"conditioned on a herald with success probability {ps:.4g} per window")
    if accepted * p < 10:
        notes.append("fewer than 10 expected events per setting: the Gaussian error is unreliable")
    return AcquisitionPlan(
        quantity,
        model,
        per,
        int(settings),
        int(repeats),
        total,
        accepted,
        accepted * p,
        p,
        sigma,
        rel,
        ps,
        notes,
    )


def plan_rate(
    rate_hz: float,
    *,
    success_probability: float = 1.0,
    time_s: float | None = None,
    target_relative: float | None = None,
    target_events: float | None = None,
    settings: int = 1,
    repeats: int = 1,
    split: Split = "per_setting",
    quantity: str = "event rate [Hz]",
) -> AcquisitionPlan:
    """Poisson counting of useful events at ``rate_hz * success_probability`` (e.g. herald
    rate times heralding success). Relative precision 1/sqrt(events)."""
    _one_target(time_s, target_relative, target_events)
    rate = _positive("rate_hz", rate_hz, allow_zero=True)
    ps = _positive("success_probability", success_probability)
    if ps > 1:
        raise PlanningError("success_probability must not exceed 1")
    useful = rate * ps
    if time_s is None:
        if useful == 0:
            raise PlanningError("the useful event rate is zero")
        if target_events is not None:
            events = _positive("target_events", target_events)
        else:
            events = 1.0 / _positive("target_relative", target_relative) ** 2
        time_s = events / useful
        split = "per_setting"
    per, total = _times(time_s, settings, repeats, split)
    events = useful * per
    sigma = math.sqrt(events) / per if events > 0 else None
    notes = [] if events >= 10 else ["fewer than 10 expected events per setting"]
    if ps < 1:
        notes.append(f"useful fraction (success probability) {ps:.4g}")
    return AcquisitionPlan(
        quantity,
        "poisson",
        per,
        int(settings),
        int(repeats),
        total,
        rate * per,
        events,
        useful,
        sigma,
        1.0 / math.sqrt(events) if events > 0 else None,
        ps,
        notes,
    )


@dataclass(frozen=True)
class ChshPlan:
    time_per_setting_s: float
    settings: int
    repeats: int
    total_time_s: float
    s: float
    s_sigma: float | None
    significance: float | None
    counts_per_setting: tuple[float, float, float, float]
    notes: list[str] = field(default_factory=list)


def plan_chsh(
    result,
    *,
    time_s: float | None = None,
    target_sigma: float | None = None,
    target_significance: float | None = None,
    repeats: int = 1,
    split: Split = "per_setting",
) -> ChshPlan:
    """Scale a CHSH forecast to another acquisition duration. The counting sigma of S falls as
    1/sqrt(T) per setting pair; ``split='total'`` divides ``time_s`` among the four setting
    pairs (and repeats). ``target_significance`` is (|S| - 2) / sigma."""
    _one_target(time_s, target_sigma, target_significance)
    t0 = result.integration_time
    s0 = result.s_sigma
    if s0 is None or not s0 > 0:
        raise PlanningError("the forecast has no counting uncertainty to scale")
    if time_s is None:
        if target_significance is not None:
            excess = abs(result.s) - 2.0
            if not excess > 0:
                raise PlanningError(
                    f"|S| = {abs(result.s):.4f} does not exceed 2: no duration gives a violation"
                )
            target_sigma = excess / _positive("target_significance", target_significance)
        time_s = t0 * (s0 / _positive("target_sigma", target_sigma)) ** 2
        split = "per_setting"
    per, total = _times(time_s, 4, repeats, split)
    sigma = s0 * math.sqrt(t0 / per)
    rates = result.total_rates if result.total_rates is not None else result.rates
    counts = tuple(float(rates[i, j].sum()) * per for i in range(2) for j in range(2))
    z = (abs(result.s) - 2.0) / sigma
    notes = [
        "time per setting pair; the four pairs are measured one after another",
        "counting statistics only: drift and systematics are not included",
    ]
    if min(counts) < 10:
        notes.append("fewer than 10 coincidences in at least one setting pair")
    return ChshPlan(per, 4, int(repeats), total, result.s, sigma, z, counts, notes)


def format_plan(plan: AcquisitionPlan | ChshPlan) -> str:
    if isinstance(plan, ChshPlan):
        text = (
            f"{plan.time_per_setting_s:.4g} s per setting pair x 4 pairs x {plan.repeats} "
            f"repeat(s) = {plan.total_time_s:.4g} s total; sigma(S) {plan.s_sigma:.4g}, "
            f"(|S| - 2)/sigma {plan.significance:.3g}; raw coincidences per pair "
            + ", ".join(f"{c:.4g}" for c in plan.counts_per_setting)
        )
    else:
        rel = "undefined" if plan.relative_precision is None else f"{plan.relative_precision:.3g}"
        text = (
            f"{plan.time_per_setting_s:.4g} s per setting x {plan.settings} setting(s) x "
            f"{plan.repeats} repeat(s) = {plan.total_time_s:.4g} s total; "
            f"{plan.events_per_setting:.4g} expected events per setting; relative precision "
            f"of {plan.quantity}: {rel} ({plan.model})"
        )
    return text + "".join(f"; {n}" for n in plan.notes)
