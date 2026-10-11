import math
from collections.abc import Callable
from dataclasses import dataclass

from sequoia.analysis.sweep import evaluate, metrics_of
from sequoia.engines.cache import RunCache, check_cancel
from sequoia.engines.classical import SimulationResult
from sequoia.engines.outcome import SimulationOutcome
from sequoia.library.store import Library
from sequoia.setup.io import setup_from_dict, setup_to_dict
from sequoia.setup.model import Setup

IDEAL_PARAMS: dict[str, dict[str, object]] = {
    "Lens": {"transmission": 1.0},
    "Objective": {"transmission": 1.0},
    "Mirror": {"reflectance": 1.0},
    "BeamSplitter": {"absorption": 0.0},
    "Polarizer": {"transmission": 1.0},
    "WavePlate": {"transmission": 1.0},
    "PBS": {
        "transmittance_p": 1.0,
        "transmittance_s": 0.0,
        "reflectance_s": 1.0,
        "reflectance_p": 0.0,
    },
    "NDFilter": {"optical_density": 0.0},
    "FiberCoupler": {"transmission": 1.0},
    "FiberBeamSplitter": {"transmission": 1.0},
    "SPAD": {"efficiency": 1.0, "dark_count_rate": "0 Hz", "afterpulse_probability": 0.0},
    "SNSPD": {"efficiency": 1.0, "dark_count_rate": "0 Hz", "afterpulse_probability": 0.0},
    "Photodiode": {"dark_current": "0 A"},
}


@dataclass(frozen=True)
class BudgetStep:
    element: str
    kind: str
    power_in: float
    power_out: float

    @property
    def transmission(self) -> float:
        return self.power_out / self.power_in if self.power_in > 0 else math.nan


def upstream_links(result: SimulationResult, link_id: str) -> list[str]:
    """Links from a source to ``link_id`` along the strongest feeding branch of each component."""
    chain = [link_id]
    seen = {link_id}
    while True:
        state = result.links[chain[-1]]
        inst = state.src.split(".")[0]
        feeders = [
            s
            for s in result.links.values()
            if s.kind != "cable" and s.dst.split(".")[0] == inst and s.id not in seen
        ]
        if not feeders:
            break
        best = max(feeders, key=lambda s: s.end.power if s.end is not None else 0.0)
        chain.append(best.id)
        seen.add(best.id)
    return chain[::-1]


def power_budget(result: SimulationResult, link_id: str) -> list[BudgetStep]:
    """Classical power along the chain feeding ``link_id``: components then link losses."""
    steps: list[BudgetStep] = []
    chain = upstream_links(result, link_id)
    previous_end: float | None = None
    for lid in chain:
        state = result.links[lid]
        start = state.start.power if state.start is not None else 0.0
        end = state.end.power if state.end is not None else 0.0
        element = state.src.split(".")[0]
        if previous_end is None:
            steps.append(BudgetStep(element, "source", start, start))
        else:
            steps.append(BudgetStep(element, "component", previous_end, start))
        steps.append(BudgetStep(lid, "link", start, end))
        previous_end = end
    return steps


def photon_budget(outcome: SimulationOutcome, link_id: str) -> list[BudgetStep]:
    """Pair-photon flux (photons/s) at the end of each link feeding ``link_id``."""
    if outcome.classical is None or outcome.rates is None:
        return []
    flux = {lid: float(v.get("photons_hz", 0.0)) for lid, v in outcome.rates.links.items()}
    chain = [lid for lid in upstream_links(outcome.classical, link_id) if flux.get(lid, 0.0) > 0]
    steps: list[BudgetStep] = []
    for position, lid in enumerate(chain):
        previous = flux[chain[position - 1]] if position else flux[lid]
        steps.append(BudgetStep(lid, "link", previous, flux[lid]))
    return steps


@dataclass(frozen=True)
class IdealGain:
    instance: str
    part_type: str
    value: float
    gain: float


@dataclass
class IdealBudget:
    metric: str
    baseline: float
    all_ideal: float
    entries: list[IdealGain]


def idealizable(setup: Setup, library: Library) -> dict[str, dict[str, object]]:
    out = {}
    for inst in setup.instances:
        part_type = library.load_part(inst.part).type
        if part_type in IDEAL_PARAMS:
            out[inst.id] = IDEAL_PARAMS[part_type]
    return out


def _ideal_setup(
    setup: Setup, targets: dict[str, dict[str, object]], connectors: bool = False
) -> Setup:
    raw = setup_to_dict(setup)
    for inst in raw["instances"]:
        extra = targets.get(inst["id"])
        if extra:
            inst.setdefault("overrides", {}).update(extra)
    if connectors:
        for link in raw.get("links", []):
            if link["kind"] == "fiber":
                link["connector_loss_db"] = 0.0
    return setup_from_dict(raw, source="ideal budget")


def ideal_budget(
    setup: Setup,
    library: Library,
    metric: str,
    *,
    cache: RunCache | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> IdealBudget:
    """How much ``metric`` improves if each part alone, or all together, were ideal."""
    cache = cache if cache is not None else RunCache()

    def value(candidate: Setup) -> float:
        outcome: SimulationOutcome = evaluate(
            candidate, library, cache=cache, should_cancel=should_cancel
        )
        if outcome.classical is None or outcome.has_errors:
            return math.nan
        return metrics_of(outcome).get(metric, math.nan)

    targets = idealizable(setup, library)
    baseline = value(setup)
    entries = []
    for inst_id, params in targets.items():
        check_cancel(should_cancel)
        v = value(_ideal_setup(setup, {inst_id: params}))
        part_type = library.load_part(next(i.part for i in setup.instances if i.id == inst_id)).type
        entries.append(IdealGain(inst_id, part_type, v, v - baseline))
    check_cancel(should_cancel)
    everything = value(_ideal_setup(setup, targets, connectors=True))
    entries.sort(key=lambda e: -(e.gain if math.isfinite(e.gain) else -math.inf))
    return IdealBudget(metric, baseline, everything, entries)


__all__ = [
    "BudgetStep",
    "IdealBudget",
    "IdealGain",
    "idealizable",
    "ideal_budget",
    "photon_budget",
    "power_budget",
    "upstream_links",
]
