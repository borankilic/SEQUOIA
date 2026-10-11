from sequoia.gui.document import SetupDocument

LAYER_SPACING = 7.0
ROW_SPACING = 3.0


def auto_layout(document: SetupDocument) -> dict[str, list[float]]:
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
    positions: dict[str, list[float]] = {}
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
            row[node] = (k - (len(members) - 1) / 2) * ROW_SPACING
        for node in members:
            positions[node] = [index * LAYER_SPACING, row[node]]
    return positions
