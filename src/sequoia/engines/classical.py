import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from sequoia.components.base import Component, Context, get_component
from sequoia.data import ExtrapolationWarning
from sequoia.engines.cache import (
    Cancelled,
    RunCache,
    check_cancel,
    record_warnings,
    run_library,
)
from sequoia.errors import Issue
from sequoia.library.store import Library
from sequoia.light.state import Beam, LinkLight, SourceLight, combine
from sequoia.setup.model import FiberLink, FreeSpaceLink, Link, Setup
from sequoia.setup.resolve import instantiate
from sequoia.setup.validate import validate_setup

LOW_COUPLING = 0.5
NEGLIGIBLE_POWER_FRACTION = 1e-6
__all__ = ["Cancelled"]


@dataclass
class LinkState:
    id: str
    kind: str
    src: str
    dst: str
    length: float | None
    start: LinkLight | None = None
    end: LinkLight | None = None
    info: dict[str, float] = field(default_factory=dict)
    start_beams: list[Beam] = field(default_factory=list)


@dataclass
class SimulationResult:
    links: dict[str, LinkState]
    readouts: dict[str, dict[str, float]]
    probes: dict[str, dict[str, Any]]
    issues: list[Issue]
    sources: dict[str, SourceLight]
    network: "Network | None" = field(default=None, repr=False, compare=False)


def _canvas_length(setup: Setup, link: FreeSpaceLink) -> float:
    positions = {i.id: i.position for i in setup.instances}
    a = positions[link.src.split(".")[0]]
    b = positions[link.dst.split(".")[0]]
    return math.dist(a, b) * setup.grid_cell.si


def _free_length(setup: Setup, link: FreeSpaceLink) -> float:
    if link.length is not None:
        return link.length.si
    return _canvas_length(setup, link)


@dataclass
class Network:
    setup: Setup
    library: Library
    components: dict[str, Component]
    out_link: dict[str, Link]
    order: list[str]
    lengths: dict[str, float | None]


def build_network(setup: Setup, library: Library) -> Network:
    components = {inst.id: instantiate(library, inst) for inst in setup.instances}
    out_link = {link.src: link for link in setup.links if link.kind != "cable"}
    graph = nx.DiGraph()
    graph.add_nodes_from(components)
    for link in setup.links:
        graph.add_edge(link.src.split(".")[0], link.dst.split(".")[0])
    lengths = {
        link.id: _free_length(setup, link)
        if link.kind == "free_space"
        else (link.length.si if link.kind == "fiber" else None)
        for link in setup.links
    }
    return Network(setup, library, components, out_link, list(nx.topological_sort(graph)), lengths)


Emitter = Callable[[str, Component, Context], list[tuple[str, Beam]]]


def propagate_network(
    net: Network,
    ctx: Context,
    emit: Emitter,
    *,
    states: dict[str, LinkState] | None = None,
    coupling_note: str = "",
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict[tuple[str, str], list[Beam]], dict[str, list[Beam]]]:
    """Push seed beams through the feed-forward network in topological order.

    ``emit(instance_id, component, ctx)`` returns the beams a component injects at its own
    output ports. Returns the beams arriving at every ``(instance, port)`` and at the end of
    every link. ``states`` (optional) records link start beams and fiber info.
    """
    library = net.library
    arrivals: dict[tuple[str, str], list[Beam]] = {}
    arrived_beams: dict[str, list[Beam]] = {link.id: [] for link in net.setup.links}
    fibers: dict[str, Any] = {}

    def deliver(inst_id: str, port: str, beam: Beam) -> None:
        link = net.out_link.get(f"{inst_id}.{port}")
        if link is None:
            return
        length = net.lengths[link.id]
        if states is not None:
            states[link.id].start_beams.append(beam)
        if isinstance(link, FreeSpaceLink):
            end = beam.through(link.id, mode=beam.mode.propagate(length), opl=length)
        else:
            assert isinstance(link, FiberLink)
            if link.id not in fibers:
                part = library.load_part(link.fiber)
                fibers[link.id] = get_component(part.type)(part.params)
            previous = ctx.subject
            ctx.subject = f"link '{link.id}'"
            end, info = fibers[link.id].transmit(beam, link, ctx, link.id)
            ctx.subject = previous
            if states is not None:
                states[link.id].info = info
            if info["coupling_efficiency"] < LOW_COUPLING:
                previous = ctx.subject
                ctx.subject = f"link '{link.id}'"
                ctx.warn(
                    "low_coupling",
                    f"only {100 * info['coupling_efficiency']:.1f}% of the {coupling_note}light "
                    "couples into the fiber",
                )
                ctx.subject = previous
        arrived_beams[link.id].append(end)
        dst_inst, dst_port = link.dst.split(".")
        arrivals.setdefault((dst_inst, dst_port), []).append(end)

    for inst_id in net.order:
        check_cancel(should_cancel)
        comp = net.components[inst_id]
        ctx.subject = f"'{inst_id}'"
        ctx.instance_id = inst_id
        for port, beam in emit(inst_id, comp, ctx):
            deliver(inst_id, port, beam)
        for port in [p.name for p in comp.ports()]:
            for beam in arrivals.get((inst_id, port), []):
                outputs = comp.apply_quality(beam, comp.propagate(port, beam, ctx), ctx)
                for out_port, out_beam in outputs:
                    deliver(inst_id, out_port, out_beam)
    return arrivals, arrived_beams


def _drop_negligible_multimode(
    issues: list[Issue], states: dict[str, LinkState], total_power: float, fraction: float
) -> list[Issue]:
    kept = []
    for issue in issues:
        if issue.code == "fiber_multimode" and total_power > 0:
            match = re.search(r"link '([^']+)'", issue.subject + " " + issue.message)
            state = states.get(match.group(1)) if match else None
            if state is not None and state.start is not None:
                if state.start.power < fraction * total_power:
                    continue
        kept.append(issue)
    return kept


def simulate(
    setup: Setup,
    library: Library,
    *,
    should_cancel: Callable[[], bool] | None = None,
    cache: RunCache | None = None,
    negligible_power_fraction: float = NEGLIGIBLE_POWER_FRACTION,
    min_power_for_warnings: float = 0.0,
) -> SimulationResult:
    """Classical (power, polarization, Gaussian mode) propagation of every source.

    ``fiber_multimode`` diagnostics are dropped for links whose incident power is below
    ``negligible_power_fraction`` of the total source power, and components skip such warnings
    for beams below the absolute ``min_power_for_warnings`` (W). ``should_cancel`` is polled
    between stages and raises ``Cancelled``; ``cache`` reuses parsed parts across runs.
    """
    library = run_library(library, cache)
    check_cancel(should_cancel)
    issues = list(validate_setup(setup, library))
    check_cancel(should_cancel)
    net = build_network(setup, library)
    components = net.components
    states = {
        link.id: LinkState(link.id, link.kind, link.src, link.dst, net.lengths[link.id])
        for link in setup.links
    }
    sources: dict[str, SourceLight] = {}
    ctx = Context(issues=issues, sources=sources, min_power_for_warnings=min_power_for_warnings)

    def emit(inst_id: str, comp: Component, ctx: Context) -> list[tuple[str, Beam]]:
        emission = comp.emit(ctx)
        if emission is None:
            return []
        sources[inst_id] = emission.source
        return list(emission.beams)

    with record_warnings() as caught:
        arrivals, arrived_beams = propagate_network(
            net, ctx, emit, states=states, should_cancel=should_cancel
        )

        for link_id, state in states.items():
            state.start = combine(state.start_beams, sources)
            state.end = combine(arrived_beams[link_id], sources)

        check_cancel(should_cancel)
        readouts: dict[str, dict[str, float]] = {}
        for inst_id, comp in components.items():
            ctx.subject = f"'{inst_id}'"
            lights = {}
            for port in [p.name for p in comp.ports()]:
                light = combine(arrivals.get((inst_id, port), []), sources)
                if light is not None:
                    lights[port] = light
            values = comp.detect(lights, ctx)
            if values:
                readouts[inst_id] = values

        check_cancel(should_cancel)
        probe_results: dict[str, dict[str, Any]] = {}
        for probe in setup.probes:
            state = states[probe.link]
            ctx.subject = f"probe '{probe.id}'"
            comp = instantiate(library, probe)
            light = state.end
            position = comp.settings.position
            if position is not None and state.kind == "free_space":
                moved = [
                    b.through(probe.link, mode=b.mode.propagate(position.si), opl=position.si)
                    for b in state.start_beams
                ]
                light = combine(moved, sources)
            if light is None:
                probe_results[probe.id] = {"power_W": 0.0}
            else:
                probe_results[probe.id] = comp.measure(light)

    seen = set()
    for w in caught:
        if issubclass(w.category, ExtrapolationWarning) and str(w.message) not in seen:
            seen.add(str(w.message))
            issues.append(Issue("warning", "curve_extrapolation", "data", str(w.message)))
    unique = list(dict.fromkeys(issues))
    total_power = sum(src.power for src in sources.values())
    unique = _drop_negligible_multimode(unique, states, total_power, negligible_power_fraction)
    issues[:] = unique
    return SimulationResult(states, readouts, probe_results, issues, sources, net)
