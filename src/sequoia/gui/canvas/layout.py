import itertools
import math
from dataclasses import dataclass, field

from sequoia.gui.canvas.items import CELL, body_size
from sequoia.gui.document import SetupDocument

LAYER_SPACING = 3.0
LAYER_GAP = 1.5
MEMBER_GAP = 1.0
BAND_GAP = 2.5
MAX_BANDS = 3
MAX_FOLD_LAYERS = 40
DEFAULT_VIEWPORT = (850.0, 600.0)


@dataclass
class LayoutPlan:
    positions: dict[str, list[float]] = field(default_factory=dict)
    rotations: dict[str, float] = field(default_factory=dict)
    bands: int = 1
    width: float = 0.0
    height: float = 0.0

    def fit_zoom(self, viewport: tuple[float, float] = DEFAULT_VIEWPORT) -> float:
        vw, vh = viewport
        w = (self.width + 2.0) * CELL
        h = (self.height + 2.0) * CELL
        return min(vw / w, vh / h)


def _snap(value: float) -> float:
    return round(value * 2.0) / 2.0


def _layers(document: SetupDocument) -> list[list[str]]:
    setup = document.setup
    ids = [i.id for i in setup.instances]
    preds: dict[str, set[str]] = {i: set() for i in ids}
    for link in setup.links:
        a, b = link.src.split(".")[0], link.dst.split(".")[0]
        if a in preds and b in preds and a != b:
            preds[b].add(a)
    layer: dict[str, int] = {}

    def depth(node: str, stack: tuple = ()) -> int:
        if node in layer:
            return layer[node]
        if node in stack:
            return 0
        value = 1 + max((depth(p, (*stack, node)) for p in preds[node]), default=-1)
        layer[node] = value
        return value

    for i in ids:
        depth(i)
    columns: dict[int, list[str]] = {}
    for i in ids:
        columns.setdefault(layer[i], []).append(i)
    row: dict[str, float] = {}
    out: list[list[str]] = []
    for index in sorted(columns):
        members = columns[index]
        if index > 0:
            members.sort(
                key=lambda n: (
                    sum(row[p] for p in preds[n] if p in row) / max(1, len(preds[n])),
                    ids.index(n),
                )
            )
        for k, node in enumerate(members):
            row[node] = float(k)
        out.append(members)
    return out


def _extent(document: SetupDocument, instance) -> tuple[float, float]:
    label = instance.part.rsplit("/", 1)[-1]
    w, h = body_size(instance.id, label, document.ports(instance.id))
    w, h = w / CELL, h / CELL
    if round(instance.rotation) % 180 == 90:
        w, h = h, w
    return w + 0.5, h + 1.0


def _place(layers, sizes, cuts):
    n = len(layers)
    bounds = [0, *cuts, n]
    bands = [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]
    widths = [max(sizes[m][0] for m in layer) for layer in layers]
    stacks = [sum(sizes[m][1] for m in layer) + MEMBER_GAP * (len(layer) - 1) for layer in layers]
    positions: dict[str, list[float]] = {}
    y_top = 0.0
    edge = 0.0
    min_x, max_x = math.inf, -math.inf
    first_cy = 0.0
    for r, band in enumerate(bands):
        band_h = max(stacks[i] for i in band)
        cy = y_top + band_h / 2.0
        if r == 0:
            first_cy = cy
        forward = r % 2 == 0
        x = edge
        cx = x
        w = 0.0
        for i in band:
            w = widths[i]
            cx = x + w / 2.0 if forward else x - w / 2.0
            y = cy - stacks[i] / 2.0
            for m in layers[i]:
                h = sizes[m][1]
                positions[m] = [cx, y + h / 2.0]
                y += h + MEMBER_GAP
            min_x = min(min_x, cx - w / 2.0)
            max_x = max(max_x, cx + w / 2.0)
            x = x + w + LAYER_GAP if forward else x - w - LAYER_GAP
        edge = cx + w / 2.0 if forward else cx - w / 2.0
        y_top += band_h + BAND_GAP
    return positions, max_x - min_x, y_top - BAND_GAP, first_cy


def _zoom(width: float, height: float, viewport) -> float:
    return min(
        1.0,
        viewport[0] / ((width + 2.0) * CELL),
        viewport[1] / ((height + 2.0) * CELL),
    )


def plan_layout(
    document: SetupDocument, viewport: tuple[float, float] = DEFAULT_VIEWPORT
) -> LayoutPlan:
    setup = document.setup
    if not setup.instances:
        return LayoutPlan()
    layers = _layers(document)
    by_id = {i.id: i for i in setup.instances}
    sizes = {i.id: _extent(document, i) for i in setup.instances}
    n = len(layers)
    single = _place(layers, sizes, ())
    best_cuts: tuple = ()
    best = single
    best_zoom = _zoom(single[1], single[2], viewport)
    single_zoom = best_zoom
    if n <= MAX_FOLD_LAYERS and n > 1:
        for count in range(2, min(MAX_BANDS, n) + 1):
            for cuts in itertools.combinations(range(1, n), count - 1):
                placed = _place(layers, sizes, cuts)
                zoom = _zoom(placed[1], placed[2], viewport)
                if zoom > best_zoom * 1.001:
                    best, best_cuts, best_zoom = placed, cuts, zoom
    if best_cuts and best_zoom < single_zoom * 1.15:
        best, best_cuts, best_zoom = single, (), single_zoom
    positions, width, height, first_cy = best
    ox = positions[layers[0][0]][0]
    snapped = {m: [_snap(p[0] - ox), _snap(p[1] - first_cy)] for m, p in positions.items()}
    rotations: dict[str, float] = {}
    bands = len(best_cuts) + 1
    if bands > 1:
        bounds = [0, *best_cuts, n]
        for r in range(bands):
            want_left = r % 2 == 1
            for i in range(bounds[r], bounds[r + 1]):
                for m in layers[i]:
                    rotation = by_id[m].rotation
                    if (round(rotation) % 360 == 180) != want_left:
                        rotations[m] = (rotation + 180.0) % 360.0
    return LayoutPlan(snapped, rotations, bands, width, height)


def auto_layout(
    document: SetupDocument, viewport: tuple[float, float] = DEFAULT_VIEWPORT
) -> dict[str, list[float]]:
    return plan_layout(document, viewport).positions
