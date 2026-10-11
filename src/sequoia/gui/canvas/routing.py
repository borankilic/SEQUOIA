from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from PySide6.QtGui import QPainterPath

Point = tuple[float, float]
Rect = tuple[float, float, float, float]

_EPS = 1e-7
_TOL = 1e-6
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_STRICT_COS = math.cos(math.radians(2.0))
_LOOSE_COS = math.cos(math.radians(20.0))
_MAX_CELLS = 200


def path_length(points: Sequence[Point]) -> float:
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


def segment_hits_rect(p: Point, q: Point, rect: Rect, margin: float = 0.0) -> bool:
    left = rect[0] - margin + _EPS
    top = rect[1] - margin + _EPS
    right = rect[2] + margin - _EPS
    bottom = rect[3] + margin - _EPS
    if left >= right or top >= bottom:
        return False
    lo, hi = 0.0, 1.0
    for a, d, mn, mx in (
        (p[0], q[0] - p[0], left, right),
        (p[1], q[1] - p[1], top, bottom),
    ):
        if abs(d) < 1e-12:
            if not (mn < a < mx):
                return False
            continue
        t1 = (mn - a) / d
        t2 = (mx - a) / d
        if t1 > t2:
            t1, t2 = t2, t1
        lo = max(lo, t1)
        hi = min(hi, t2)
        if lo >= hi - 1e-12:
            return False
    return lo < hi - 1e-12


def rounded_path(points: Sequence[Point], radius: float = 10.0) -> QPainterPath:
    path = QPainterPath()
    n = len(points)
    if n == 0:
        return path
    path.moveTo(points[0][0], points[0][1])
    for i in range(1, n - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        abx, aby = b[0] - a[0], b[1] - a[1]
        bcx, bcy = c[0] - b[0], c[1] - b[1]
        la = math.hypot(abx, aby)
        lc = math.hypot(bcx, bcy)
        r = min(radius, la / 2.0, lc / 2.0)
        cross = abx * bcy - aby * bcx
        if r <= 1e-9 or abs(cross) <= 1e-9 * la * lc:
            path.lineTo(b[0], b[1])
            continue
        path.lineTo(b[0] - abx / la * r, b[1] - aby / la * r)
        path.quadTo(b[0], b[1], b[0] + bcx / lc * r, b[1] + bcy / lc * r)
    if n > 1:
        path.lineTo(points[-1][0], points[-1][1])
    return path


def _norm_rect(r: Sequence[float]) -> Rect:
    return (
        float(min(r[0], r[2])),
        float(min(r[1], r[3])),
        float(max(r[0], r[2])),
        float(max(r[1], r[3])),
    )


def _cheb(p: Point, r: Rect) -> float:
    return max(r[0] - p[0], 0.0, p[0] - r[2], r[1] - p[1], 0.0, p[1] - r[3])


def _inside(p: Point, r: Rect) -> bool:
    return r[0] < p[0] < r[2] and r[1] < p[1] < r[3]


def _unit(v: Point, fallback: Point) -> Point:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        v = fallback
        n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)


def _snap(v: Point, fallback: Point) -> Point:
    x, y = v
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        x, y = fallback
    if abs(x) >= abs(y):
        return (1.0 if x >= 0 else -1.0, 0.0)
    return (0.0, 1.0 if y >= 0 else -1.0)


def _simplify(points: Sequence[Point]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < 1e-9:
            continue
        while len(out) >= 2:
            a, b = out[-2], out[-1]
            ux, uy = b[0] - a[0], b[1] - a[1]
            vx, vy = p[0] - b[0], p[1] - b[1]
            if abs(ux * vy - uy * vx) <= 1e-9 * math.hypot(ux, uy) * math.hypot(vx, vy) + 1e-12:
                out.pop()
            else:
                break
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < 1e-9:
            continue
        out.append(p)
    return out


def _connect(p: Point, d0: Point, q: Point, d1: Point, off: float) -> list[Point]:
    ax = 0 if d0[0] else 1
    bx = 0 if d1[0] else 1
    if ax != bx:
        return [(q[0], p[1])] if ax == 0 else [(p[0], q[1])]
    c = 1 - ax

    def pt(u: float, v: float) -> Point:
        return (u, v) if ax == 0 else (v, u)

    s0, s1 = d0[ax], d1[ax]
    a, b = p[ax], q[ax]
    if s0 == s1:
        if (b - a) * s0 > 0:
            m = (a + b) / 2.0
            return [pt(m, p[c]), pt(m, q[c])]
        m0 = a + s0 * off
        m1 = b - s0 * off
        mid = (p[c] + q[c]) / 2.0
        if abs(p[c] - q[c]) < 1e-9:
            mid = p[c] + 2.0 * off
        return [pt(m0, p[c]), pt(m0, mid), pt(m1, mid), pt(m1, q[c])]
    m = (max(a, b) if s0 > 0 else min(a, b)) + s0 * off
    return [pt(m, p[c]), pt(m, q[c])]


def _fallback(s: Point, sd: Point, e: Point, ed: Point, off: float) -> list[Point]:
    s1 = (s[0] + sd[0] * off, s[1] + sd[1] * off)
    e1 = (e[0] + ed[0] * off, e[1] + ed[1] * off)
    mid = _connect(s1, sd, e1, (-ed[0], -ed[1]), off)
    return _simplify([s, s1, *mid, e1, e])


def _poly_clear(
    pts: Sequence[Point], rects: Sequence[Rect], margins: Sequence[float], own: set[int]
) -> bool:
    last = len(pts) - 2
    for k in range(last + 1):
        p, q = pts[k], pts[k + 1]
        end_seg = k == 0 or k == last
        for idx, r in enumerate(rects):
            m = 0.0 if (end_seg and idx in own) else margins[idx]
            if segment_hits_rect(p, q, r, m):
                return False
    return True


def _coords(prio: Sequence[float], lo: float, hi: float, g: float) -> list[float]:
    out: list[float] = []

    def add(v: float, tol: float) -> None:
        k = bisect_left(out, v)
        if k < len(out) and out[k] - v <= tol:
            return
        if k > 0 and v - out[k - 1] <= tol:
            return
        out.insert(k, v)

    for v in prio:
        add(v, _TOL)
    k0 = math.ceil(lo / g)
    k1 = math.floor(hi / g)
    for k in range(k0, k1 + 1):
        add(k * g, 0.25 * g)
    return out


def _index(vals: list[float], v: float) -> int:
    return bisect_left(vals, v - _TOL)


def _search(
    xs: list[float],
    ys: list[float],
    hb: bytearray,
    vb: bytearray,
    src: tuple[int, int],
    d0: int,
    dst: tuple[int, int],
    dreq: int,
    bend: float,
    hug: float,
    max_nodes: int,
) -> list[tuple[int, int]] | None:
    nx, ny = len(xs), len(ys)
    sn = src[1] * nx + src[0]
    gn = dst[1] * nx + dst[0]
    gx, gy = xs[dst[0]], ys[dst[1]]
    inf = math.inf

    def final_cost(d: int) -> float:
        if d == dreq:
            return 0.0
        if d == dreq ^ 1:
            return inf
        return bend

    st0 = sn * 4 + d0
    best: dict[int, float] = {st0: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, int, float, int]] = []
    cnt = 0
    goal_g = inf
    goal_parent = -1
    if sn == gn:
        fc = final_cost(d0)
        if fc < inf:
            goal_g = fc
            goal_parent = st0
            heap.append((fc, cnt, fc, -1))
            cnt += 1
    heap.append((abs(xs[src[0]] - gx) + abs(ys[src[1]] - gy), cnt, 0.0, st0))
    cnt += 1
    heapq.heapify(heap)
    pops = 0
    while heap:
        _, _, g, st = heapq.heappop(heap)
        if st == -1:
            break
        if g > best.get(st, inf):
            continue
        pops += 1
        if pops > max_nodes:
            return None
        n, d = divmod(st, 4)
        j, i = divmod(n, nx)
        for nd in range(4):
            if nd == d ^ 1:
                continue
            if nd == 0:
                if i + 1 >= nx:
                    continue
                flag = hb[n]
                ln = xs[i + 1] - xs[i]
                nn = n + 1
            elif nd == 1:
                if i < 1:
                    continue
                flag = hb[n - 1]
                ln = xs[i] - xs[i - 1]
                nn = n - 1
            elif nd == 2:
                if j + 1 >= ny:
                    continue
                flag = vb[n]
                ln = ys[j + 1] - ys[j]
                nn = n + nx
            else:
                if j < 1:
                    continue
                flag = vb[n - nx]
                ln = ys[j] - ys[j - 1]
                nn = n - nx
            if flag == 1:
                continue
            g2 = g + (ln * hug if flag == 2 else ln) + (bend if nd != d else 0.0)
            nst = nn * 4 + nd
            if g2 < best.get(nst, inf):
                best[nst] = g2
                parent[nst] = st
                nj, ni = divmod(nn, nx)
                h = abs(xs[ni] - gx) + abs(ys[nj] - gy)
                heapq.heappush(heap, (g2 + h, cnt, g2, nst))
                cnt += 1
                if nn == gn:
                    gt = g2 + final_cost(nd)
                    if gt < goal_g:
                        goal_g = gt
                        goal_parent = nst
                        heapq.heappush(heap, (gt, cnt, gt, -1))
                        cnt += 1
    if goal_parent < 0 or goal_g == inf:
        return None
    nodes: list[tuple[int, int]] = []
    st = goal_parent
    while True:
        n = st // 4
        nodes.append((n % nx, n // nx))
        if st == st0:
            break
        st = parent[st]
    nodes.reverse()
    return nodes


def _route(
    start: Point,
    start_dir: Point,
    end: Point,
    end_dir: Point,
    obstacles: Sequence[Rect],
    clearance: float,
    stub: float,
    grid: float,
    prefer_straight: bool,
    bend_penalty: float | None,
    max_nodes: int,
) -> tuple[list[Point], bool]:
    s = (float(start[0]), float(start[1]))
    e = (float(end[0]), float(end[1]))
    if math.hypot(e[0] - s[0], e[1] - s[1]) < 1e-9:
        return [s, e], True
    clearance = max(0.0, float(clearance))
    grid = max(float(grid), 1e-3)
    rects = [_norm_rect(r) for r in obstacles]
    toward = (e[0] - s[0], e[1] - s[1])
    sd = _snap(start_dir, toward)
    ed = _snap(end_dir, (-toward[0], -toward[1]))
    own = {k for k, r in enumerate(rects) if _cheb(s, r) <= 1.0 or _cheb(e, r) <= 1.0}
    off = max(float(stub), clearance, 1e-6)
    fb = _fallback(s, sd, e, ed, off)

    length = math.hypot(*toward)
    ux, uy = toward[0] / length, toward[1] / length
    su = _unit(start_dir, toward)
    eu = _unit(end_dir, (-toward[0], -toward[1]))
    cs = su[0] * ux + su[1] * uy
    ce = -(eu[0] * ux + eu[1] * uy)
    if min(cs, ce) >= (_LOOSE_COS if prefer_straight else _STRICT_COS) and not any(
        segment_hits_rect(s, e, r, 0.0 if k in own else clearance) for k, r in enumerate(rects)
    ):
        return [s, e], True

    s1 = (s[0] + sd[0] * off, s[1] + sd[1] * off)
    e1 = (e[0] + ed[0] * off, e[1] + ed[1] * off)
    margins = [clearance] * len(rects)
    for k, r in enumerate(rects):
        for p in (s1, e1):
            if _inside(p, r):
                return fb, False
            margins[k] = min(margins[k], _cheb(p, r))
    for r in rects:
        if segment_hits_rect(s, s1, r) or segment_hits_rect(e, e1, r):
            return fb, False

    infl = [(r[0] - m, r[1] - m, r[2] + m, r[3] + m) for r, m in zip(rects, margins, strict=True)]
    x0 = min(s[0], e[0], s1[0], e1[0])
    x1 = max(s[0], e[0], s1[0], e1[0])
    y0 = min(s[1], e[1], s1[1], e1[1])
    y1 = max(s[1], e[1], s1[1], e1[1])
    used = [False] * len(rects)
    changed = True
    while changed:
        changed = False
        for k, r in enumerate(infl):
            if not used[k] and r[0] < x1 and r[2] > x0 and r[1] < y1 and r[3] > y0:
                used[k] = True
                changed = True
                x0, x1 = min(x0, r[0]), max(x1, r[2])
                y0, y1 = min(y0, r[1]), max(y1, r[3])
    pad = clearance + grid
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    g = max(grid, (x1 - x0) / _MAX_CELLS, (y1 - y0) / _MAX_CELLS)
    rel = [k for k in range(len(rects)) if used[k]]
    xs = _coords(
        [s1[0], e1[0], s[0], e[0], x0, x1] + [v for k in rel for v in (infl[k][0], infl[k][2])],
        x0,
        x1,
        g,
    )
    ys = _coords(
        [s1[1], e1[1], s[1], e[1], y0, y1] + [v for k in rel for v in (infl[k][1], infl[k][3])],
        y0,
        y1,
        g,
    )
    nx, ny = len(xs), len(ys)
    hb = bytearray(nx * ny)
    vb = bytearray(nx * ny)
    spans = []
    for k in rel:
        left, top, right, bottom = infl[k]
        i0, i1 = _index(xs, left), bisect_right(xs, right + _TOL) - 1
        j0, j1 = _index(ys, top), bisect_right(ys, bottom + _TOL) - 1
        spans.append((i0, i1, j0, j1))
        for j in range(j0 + 1, j1):
            row = j * nx
            for i in range(i0, i1):
                hb[row + i] = 1
        for j in range(j0, j1):
            row = j * nx
            for i in range(i0 + 1, i1):
                vb[row + i] = 1
    for i0, i1, j0, j1 in spans:
        for j in (j0, j1):
            row = j * nx
            for i in range(i0, i1):
                if hb[row + i] == 0:
                    hb[row + i] = 2
        for i in (i0, i1):
            for j in range(j0, j1):
                if vb[j * nx + i] == 0:
                    vb[j * nx + i] = 2

    dirs = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}
    d0 = dirs[(int(sd[0]), int(sd[1]))]
    dreq = dirs[(int(-ed[0]), int(-ed[1]))]
    bend = 4.0 * grid if bend_penalty is None else bend_penalty
    nodes = _search(
        xs,
        ys,
        hb,
        vb,
        (_index(xs, s1[0]), _index(ys, s1[1])),
        d0,
        (_index(xs, e1[0]), _index(ys, e1[1])),
        dreq,
        bend,
        1.15,
        max_nodes,
    )
    if nodes is None:
        return fb, False
    pts = _simplify([s, *[(xs[i], ys[j]) for i, j in nodes], e])
    rel_rects = [rects[k] for k in rel]
    rel_margins = [margins[k] for k in rel]
    rel_own = {n for n, k in enumerate(rel) if k in own}

    def clear(p: Sequence[Point]) -> bool:
        return _poly_clear(p, rel_rects, rel_margins, rel_own)

    pts = _centre(pts, clear, off, g)
    if not clear(pts):
        return fb, False
    return pts, True


def _centre(pts: list[Point], clear, off: float, g: float) -> list[Point]:
    n = len(pts)
    for i in range(1, n - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        if p1[0] == p2[0]:
            ax = 0
        elif p1[1] == p2[1]:
            ax = 1
        else:
            continue
        a, b, cur = p0[ax], p3[ax], p1[ax]
        lo, hi = min(a, b), max(a, b)
        if not (lo + _TOL < cur < hi - _TOL):
            continue
        new = (a + b) / 2.0
        snapped = round(new / g) * g
        if lo + _TOL < snapped < hi - _TOL:
            new = snapped
        if abs(new - cur) < _TOL:
            continue
        if i == 1 and abs(new - pts[0][ax]) < off - _TOL:
            continue
        if i + 2 == n - 1 and abs(new - pts[-1][ax]) < off - _TOL:
            continue
        q1 = (new, p1[1]) if ax == 0 else (p1[0], new)
        q2 = (new, p2[1]) if ax == 0 else (p2[0], new)
        cand = [*pts[:i], q1, q2, *pts[i + 2 :]]
        if clear(cand):
            pts = cand
    return _simplify(pts)


def route_link_ex(
    start: Point,
    start_dir: Point,
    end: Point,
    end_dir: Point,
    obstacles: Sequence[Rect],
    *,
    clearance: float = 12.0,
    stub: float = 18.0,
    grid: float = 12.5,
    prefer_straight: bool = True,
    bend_penalty: float | None = None,
    max_nodes: int = 20000,
) -> tuple[list[Point], bool]:
    try:
        return _route(
            start,
            start_dir,
            end,
            end_dir,
            obstacles,
            clearance,
            stub,
            grid,
            prefer_straight,
            bend_penalty,
            max_nodes,
        )
    except Exception:
        s = (float(start[0]), float(start[1]))
        e = (float(end[0]), float(end[1]))
        return [s, e], False


def route_link(
    start: Point,
    start_dir: Point,
    end: Point,
    end_dir: Point,
    obstacles: Sequence[Rect],
    *,
    clearance: float = 12.0,
    stub: float = 18.0,
    grid: float = 12.5,
    prefer_straight: bool = True,
) -> list[Point]:
    return route_link_ex(
        start,
        start_dir,
        end,
        end_dir,
        obstacles,
        clearance=clearance,
        stub=stub,
        grid=grid,
        prefer_straight=prefer_straight,
    )[0]
