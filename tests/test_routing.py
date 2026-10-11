import math
import random
import time

from sequoia.gui.canvas.routing import (
    path_length,
    rounded_path,
    route_link,
    route_link_ex,
    segment_hits_rect,
)

CL = 12.0


def own_box(p, d, size=50.0):
    cx, cy = p[0] - d[0] * size / 2, p[1] - d[1] * size / 2
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def clear(pts, rects, own=(), clearance=CL):
    last = len(pts) - 2
    for k in range(last + 1):
        for idx, r in enumerate(rects):
            m = 0.0 if (idx in own and k in (0, last)) else clearance
            if segment_hits_rect(pts[k], pts[k + 1], r, m):
                return False
    return True


def scene(s, sd, e, ed, others=()):
    return [own_box(s, sd), own_box(e, ed), *others]


def test_unobstructed_straight():
    s, e = (0.0, 0.0), (300.0, 0.0)
    pts = route_link(s, (1, 0), e, (-1, 0), scene(s, (1, 0), e, (-1, 0)))
    assert pts == [s, e]


def test_obstacle_in_middle():
    s, e = (0.0, 0.0), (300.0, 0.0)
    obs = scene(s, (1, 0), e, (-1, 0), [(130, -30, 170, 30)])
    pts, found = route_link_ex(s, (1, 0), e, (-1, 0), obs)
    assert found
    assert pts[0] == s and pts[-1] == e
    assert len(pts) - 2 <= 4
    assert pts[1][1] == s[1] and pts[1][0] > s[0]
    assert pts[-2][1] == e[1] and pts[-2][0] < e[0]
    assert clear(pts, obs, own=(0, 1))


def test_corridor():
    s, e = (0.0, 0.0), (300.0, 60.0)
    walls = [(100, -200, 140, -15), (100, 15, 140, 200)]
    obs = scene(s, (1, 0), e, (-1, 0), walls)
    pts, found = route_link_ex(s, (1, 0), e, (-1, 0), obs)
    assert found and len(pts) - 2 <= 4
    assert clear(pts, obs, own=(0, 1))
    crossing = [
        (p, q)
        for p, q in zip(pts, pts[1:], strict=False)
        if min(p[0], q[0]) <= 120 <= max(p[0], q[0])
    ]
    assert any(abs(p[1]) < 15 and abs(q[1]) < 15 for p, q in crossing)


def test_fully_blocked_falls_back():
    s, e = (0.0, 0.0), (300.0, 0.0)
    ring = [(230, -60, 370, -40), (230, 40, 370, 60), (230, -60, 250, 60), (360, -60, 370, 60)]
    obs = scene(s, (1, 0), e, (-1, 0), ring)
    pts, found = route_link_ex(s, (1, 0), e, (-1, 0), obs)
    assert not found
    assert pts[0] == s and pts[-1] == e
    obs2 = [(-100, -100, 400, 100)]
    pts2, found2 = route_link_ex(s, (1, 0), e, (-1, 0), obs2)
    assert not found2 and pts2[0] == s and pts2[-1] == e


def test_start_equals_end():
    p = (10.0, 10.0)
    pts, found = route_link_ex(p, (1, 0), p, (-1, 0), [])
    assert found and pts[0] == p and pts[-1] == p


def test_facing_offset_is_z_with_centred_jog():
    s, e = (0.0, 0.0), (200.0, 100.0)
    obs = scene(s, (1, 0), e, (-1, 0))
    pts = route_link(s, (1, 0), e, (-1, 0), obs)
    assert len(pts) == 4
    assert pts[1][0] == pts[2][0] == 100.0


def test_facing_away():
    s, e = (0.0, 0.0), (200.0, 0.0)
    obs = scene(s, (-1, 0), e, (1, 0))
    pts, found = route_link_ex(s, (-1, 0), e, (1, 0), obs)
    assert found
    assert pts[0] == s and pts[-1] == e
    assert pts[1][0] < s[0] and pts[1][1] == s[1]
    assert clear(pts, obs, own=(0, 1))


def test_perpendicular_is_single_bend():
    s, e = (0.0, 0.0), (100.0, 100.0)
    obs = scene(s, (1, 0), e, (0, -1))
    pts = route_link(s, (1, 0), e, (0, -1), obs)
    assert pts == [s, (100.0, 0.0), e]


def test_rounded_path_bounds():
    pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (160.0, 60.0)]
    path = rounded_path(pts, 10.0)
    assert path.elementAt(0).x == 0.0 and path.elementAt(0).y == 0.0
    last = path.elementAt(path.elementCount() - 1)
    assert (last.x, last.y) == (160.0, 60.0)
    br = path.boundingRect()
    assert br.left() >= 0 and br.top() >= 0 and br.right() <= 160 and br.bottom() <= 60
    short = rounded_path([(0, 0), (4, 0), (4, 4)], 10.0)
    assert short.boundingRect().right() <= 4


def test_deterministic():
    s, e = (0.0, 0.0), (300.0, 0.0)
    obs = scene(s, (1, 0), e, (-1, 0), [(130, -30, 170, 30), (60, -80, 90, 20)])
    a = route_link(s, (1, 0), e, (-1, 0), obs)
    assert all(route_link(s, (1, 0), e, (-1, 0), obs) == a for _ in range(5))


def test_path_length():
    assert path_length([(0, 0), (3, 4), (3, 10)]) == 11.0
    assert path_length([]) == 0.0


def test_random_routes_never_cross():
    rng = random.Random(1234)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    checked = 0
    for _ in range(200):
        s = (rng.randrange(0, 33) * 12.5, rng.randrange(0, 33) * 12.5)
        e = (rng.randrange(0, 33) * 12.5, rng.randrange(0, 33) * 12.5)
        sd, ed = rng.choice(dirs), rng.choice(dirs)
        others = []
        for _k in range(rng.randrange(0, 9)):
            x, y = rng.uniform(-50, 450), rng.uniform(-50, 450)
            others.append((x, y, x + rng.uniform(25, 125), y + rng.uniform(25, 125)))
        obs = scene(s, sd, e, ed, others)
        s1 = (s[0] + sd[0] * 18, s[1] + sd[1] * 18)
        e1 = (e[0] + ed[0] * 18, e[1] + ed[1] * 18)
        a, b = obs[0], obs[1]
        if a[0] < b[2] + 2 * CL and b[0] < a[2] + 2 * CL and a[1] < b[3] + 2 * CL:
            if b[1] < a[3] + 2 * CL:
                continue
        if any(segment_hits_rect(p, p, r, CL) for p in (s1, e1) for r in obs):
            continue
        if any(segment_hits_rect(s, s1, r, CL) or segment_hits_rect(e, e1, r, CL) for r in others):
            continue
        pts, found = route_link_ex(s, sd, e, ed, obs)
        assert pts[0] == s and pts[-1] == e
        if not found:
            continue
        checked += 1
        assert clear(pts, obs, own=(0, 1))
        for p, q in zip(pts, pts[1:], strict=False):
            assert math.hypot(q[0] - p[0], q[1] - p[1]) > 0
        if len(pts) > 2:
            for p, q in zip(pts, pts[1:], strict=False):
                assert p[0] == q[0] or p[1] == q[1]
    assert checked > 60


def test_performance():
    rng = random.Random(7)
    obs = []
    for _ in range(30):
        x, y = rng.uniform(0, 900), rng.uniform(0, 600)
        obs.append((x, y, x + 40, y + 40))
    links = []
    for _ in range(20):
        s = (rng.randrange(0, 72) * 12.5, rng.randrange(0, 48) * 12.5)
        e = (rng.randrange(0, 72) * 12.5, rng.randrange(0, 48) * 12.5)
        links.append((s, e))
    t0 = time.perf_counter()
    for s, e in links:
        route_link_ex(s, (1, 0), e, (-1, 0), obs)
    assert (time.perf_counter() - t0) / len(links) < 0.05
