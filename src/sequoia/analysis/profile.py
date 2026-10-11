from dataclasses import dataclass, field

import numpy as np

from sequoia.analysis.budget import upstream_links
from sequoia.engines.classical import SimulationResult


@dataclass
class Caustic:
    z: np.ndarray
    w_x: np.ndarray
    w_y: np.ndarray
    markers: list[tuple[float, str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


def caustic(result: SimulationResult, link_id: str, points_per_link: int = 40) -> Caustic:
    """Beam radius along the free-space stretch that ends at the end of ``link_id``.

    The path is the strongest feeding branch back to the source; a fiber link restarts the
    axis, so only the free-space run after the last fiber is shown.
    """
    chain = upstream_links(result, link_id)
    start = 0
    for index, lid in enumerate(chain):
        if result.links[lid].kind != "free_space":
            start = index + 1
    chain = [lid for lid in chain[start:] if result.links[lid].kind == "free_space"]
    z_all: list[np.ndarray] = []
    wx_all: list[np.ndarray] = []
    wy_all: list[np.ndarray] = []
    markers: list[tuple[float, str]] = []
    origin = 0.0
    for lid in chain:
        state = result.links[lid]
        if not state.start_beams or state.length is None:
            continue
        beam = max(state.start_beams, key=lambda b: float(np.sum(np.abs(b.jones) ** 2)))
        distances = np.linspace(0.0, state.length, points_per_link)
        modes = [beam.mode.propagate(float(d)) for d in distances]
        z_all.append(origin + distances)
        wx_all.append(np.array([m.w_x for m in modes]))
        wy_all.append(np.array([m.w_y for m in modes]))
        markers.append((origin, state.src.split(".")[0]))
        origin += state.length
    if not z_all:
        empty = np.array([])
        return Caustic(empty, empty, empty, [], [])
    last = result.links[chain[-1]]
    markers.append((origin, last.dst.split(".")[0]))
    return Caustic(
        np.concatenate(z_all), np.concatenate(wx_all), np.concatenate(wy_all), markers, chain
    )
