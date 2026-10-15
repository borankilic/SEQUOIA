import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

DEFAULT_ACCENT = QColor("#3b7dd8")
DEFAULT_COLOR = QColor("#3a3f47")


def _alpha(c: QColor, a: int) -> QColor:
    out = QColor(c)
    out.setAlpha(a)
    return out


def _hue_shift(c: QColor, deg: float) -> QColor:
    h, s, v, a = c.getHsvF()
    h = 0.0 if h < 0 else h
    return QColor.fromHsvF((h + deg / 360.0) % 1.0, max(s, 0.55), v, a)


class _Ctx:
    def __init__(
        self, p: QPainter, color: QColor, accent: QColor, fiber: bool, px: float, s: float
    ):
        self.p = p
        self.color = color
        self.accent = accent
        self.fiber = fiber
        self.w = px / s
        self.detail = px >= 1.2
        self.alt = _hue_shift(accent, 150)

    def pen(self, c: QColor | None = None, k: float = 1.0, dash: bool = False) -> QPen:
        pen = QPen(c or self.color, self.w * k)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if dash:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([2.5, 2.5])
        return pen

    def stroke(self, path: QPainterPath, c: QColor | None = None, k: float = 1.0, dash=False):
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        self.p.setPen(self.pen(c, k, dash))
        self.p.drawPath(path)

    def fill(self, path: QPainterPath, fill: QColor | None, c: QColor | None = None, k=1.0):
        self.p.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)
        self.p.setPen(self.pen(c, k))
        self.p.drawPath(path)

    def line(self, x1, y1, x2, y2, c: QColor | None = None, k: float = 1.0, dash=False):
        path = QPainterPath(QPointF(x1, y1))
        path.lineTo(x2, y2)
        self.stroke(path, c, k, dash)

    def poly(self, pts, fill: QColor | None = None, c: QColor | None = None, close=True, k=1.0):
        path = QPainterPath(QPointF(*pts[0]))
        for x, y in pts[1:]:
            path.lineTo(x, y)
        if close:
            path.closeSubpath()
        if fill is None and not close:
            self.stroke(path, c, k)
        else:
            self.fill(path, fill, c, k)

    def rect(self, x1, y1, x2, y2, fill: QColor | None = None, c: QColor | None = None, r=0.0):
        path = QPainterPath()
        if r:
            path.addRoundedRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)), r, r)
        else:
            path.addRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
        self.fill(path, fill, c)

    def ellipse(self, cx, cy, rx, ry, fill: QColor | None = None, c: QColor | None = None, k=1.0):
        path = QPainterPath()
        path.addEllipse(QPointF(cx, cy), rx, ry)
        self.fill(path, fill, c, k)

    def dot(self, x, y, r, c: QColor | None = None):
        c = c or self.color
        self.p.setPen(Qt.PenStyle.NoPen)
        self.p.setBrush(c)
        self.p.drawEllipse(QPointF(x, y), r, r)

    def head(self, x2, y2, ang, c: QColor, size=0.15):
        pts = []
        for da in (math.radians(155), -math.radians(155)):
            pts.append(QPointF(x2 + size * math.cos(ang + da), y2 + size * math.sin(ang + da)))
        path = QPainterPath(QPointF(x2, y2))
        path.lineTo(pts[0])
        path.lineTo(pts[1])
        path.closeSubpath()
        self.p.setPen(self.pen(c, 0.6))
        self.p.setBrush(c)
        self.p.drawPath(path)

    def arrow(self, x1, y1, x2, y2, c: QColor | None = None, k=1.0, size=0.15):
        c = c or self.accent
        ang = math.atan2(y2 - y1, x2 - x1)
        self.line(x1, y1, x2 - size * 0.6 * math.cos(ang), y2 - size * 0.6 * math.sin(ang), c, k)
        self.head(x2, y2, ang, c, size)

    def beam(self, x1, y1, x2, y2, c: QColor | None = None, arrow=True):
        if self.fiber:
            self.line(x1, y1, x2, y2, self.accent, 1.8)
        elif arrow:
            self.arrow(x1, y1, x2, y2, c)
        else:
            self.line(x1, y1, x2, y2, c or self.accent)

    def wave(self, x1, x2, y, amp=0.1, cycles=2.5, c: QColor | None = None, head=True):
        c = c or self.accent
        n = 40
        end = x2 - (0.12 if head else 0.0)
        path = QPainterPath(QPointF(x1, y))
        for i in range(1, n + 1):
            t = i / n
            path.lineTo(x1 + (end - x1) * t, y + amp * math.sin(2 * math.pi * cycles * t))
        self.stroke(path, c)
        if head:
            self.head(x2, y, 0.0, c, 0.15)

    def lens_arc(self, x, h, bulge, c: QColor | None = None):
        path = QPainterPath(QPointF(x, -h))
        path.quadTo(x + 2 * bulge, 0, x, h)
        self.stroke(path, c)

    def biconvex(self, cx, h, b, fill: QColor | None = None):
        path = QPainterPath(QPointF(cx, -h))
        path.quadTo(cx + 2 * b, 0, cx, h)
        path.quadTo(cx - 2 * b, 0, cx, -h)
        self.fill(path, fill)

    def axis(self):
        self.line(-0.95, 0, 0.95, 0, _alpha(self.color, 70), 0.7, dash=True)


_DRAWERS = {}


def _icon(*names):
    def deco(fn):
        for n in names:
            _DRAWERS[n] = fn
        return fn

    return deco


@_icon("Laser")
def _laser(k: _Ctx):
    body = _alpha(k.color, 38)
    k.rect(-0.9, -0.34, 0.2, 0.34, body, r=0.07)
    k.rect(0.2, -0.12, 0.34, 0.12, body)
    k.line(-0.72, -0.34, -0.72, 0.34, _alpha(k.color, 120), 0.7)
    if k.detail:
        k.wave(-0.55, 0.05, 0.0, 0.13, 1.5, k.accent, head=False)
    cx = 0.34
    for ang in (-50, -25, 25, 50):
        a = math.radians(ang)
        k.line(
            cx + 0.06 * math.cos(a),
            0.18 * math.sin(a) * 0.0 + 0.06 * math.sin(a),
            cx + 0.26 * math.cos(a),
            0.26 * math.sin(a),
            k.accent,
            0.9,
        )
    k.arrow(0.34, 0, 0.95, 0, k.accent, 1.3)


@_icon("FockSource")
def _fock(k: _Ctx):
    body = _alpha(k.color, 38)
    k.rect(-0.9, -0.3, -0.1, 0.3, body, r=0.07)
    for i, x in enumerate((-0.72, -0.5, -0.28)):
        h = 0.18 if i != 1 else 0.24
        k.line(x, h, x, -h, k.accent, 1.4)
    k.ellipse(0.25, 0, 0.14, 0.14, _alpha(k.accent, 90), k.accent)
    k.arrow(-0.1, 0, 0.95, 0, k.accent, 1.1)


@_icon("PhaseShifter")
def _phase_shifter(k: _Ctx):
    k.axis()
    k.rect(-0.18, -0.55, 0.18, 0.55, _alpha(k.accent, 45))
    k.line(-0.95, 0.2, -0.18, 0.2, k.accent, 1.0)
    path = QPainterPath(QPointF(0.18, -0.2))
    path.lineTo(0.95, -0.2)
    k.stroke(path, k.accent, 1.0)
    k.arrow(0.05, 0.4, 0.05, -0.4, _alpha(k.color, 150), 0.8, 0.1)


@_icon("SPDCCrystal")
def _spdc(k: _Ctx):
    k.beam(-0.95, 0, -0.55, 0)
    k.rect(-0.55, -0.42, 0.55, 0.42, _alpha(k.color, 30))
    n = 6 if k.detail else 4
    w = 1.1 / n
    for i in range(n):
        if i % 2 == 0:
            k.rect(-0.55 + i * w, -0.42, -0.55 + (i + 1) * w, 0.42, _alpha(k.accent, 80))
    k.beam(0.55, 0, 0.95, 0, k.alt)


@_icon("EntangledPairSource")
def _pair(k: _Ctx):
    k.rect(-0.9, -0.3, -0.1, 0.3, _alpha(k.color, 30))
    n = 5 if k.detail else 3
    w = 0.8 / n
    for i in range(n):
        if i % 2 == 0:
            k.rect(-0.9 + i * w, -0.3, -0.9 + (i + 1) * w, 0.3, _alpha(k.accent, 80))
    k.arrow(-0.1, -0.08, 0.9, -0.55, k.accent)
    k.arrow(-0.1, 0.08, 0.9, 0.55, k.alt)
    if k.detail:
        path = QPainterPath(QPointF(0.62, -0.42))
        path.quadTo(0.82, 0.0, 0.62, 0.42)
        k.stroke(path, _alpha(k.color, 160), 0.8, dash=True)


@_icon("Lens")
def _lens(k: _Ctx):
    k.axis()
    k.biconvex(0, 0.8, 0.26, _alpha(k.accent, 50))


@_icon("Objective")
def _objective(k: _Ctx):
    k.poly(
        [(-0.85, -0.6), (0.0, -0.6), (0.8, -0.22), (0.8, 0.22), (0.0, 0.6), (-0.85, 0.6)],
        _alpha(k.color, 30),
    )
    k.line(0.0, -0.6, 0.0, 0.6, _alpha(k.color, 110), 0.7)
    k.lens_arc(-0.55, 0.46, 0.1)
    k.lens_arc(-0.25, 0.42, 0.1)
    if k.detail:
        k.lens_arc(0.28, 0.3, -0.1)
    k.line(0.8, -0.22, 0.95, -0.22)
    k.line(0.8, 0.22, 0.95, 0.22)


def _slab(k: _Ctx, half: float, two_tone: bool):
    t = 0.09
    u = 1 / math.sqrt(2)
    d = (u, -u)
    n = (u, u)
    cx, cy = 0.0, 0.0
    a = (cx - d[0] * half, cy - d[1] * half)
    b = (cx + d[0] * half, cy + d[1] * half)
    corners = [
        (a[0] - n[0] * t, a[1] - n[1] * t),
        (b[0] - n[0] * t, b[1] - n[1] * t),
        (b[0] + n[0] * t, b[1] + n[1] * t),
        (a[0] + n[0] * t, a[1] + n[1] * t),
    ]
    if two_tone:
        mid_t = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        mid_b = ((corners[3][0] + corners[2][0]) / 2, (corners[3][1] + corners[2][1]) / 2)
        k.poly([corners[0], mid_t, mid_b, corners[3]], _alpha(k.accent, 150))
        k.poly([mid_t, corners[1], corners[2], mid_b], _alpha(k.alt, 150))
    else:
        k.poly(corners, _alpha(k.color, 150))
    if k.detail:
        for i in range(1, 7):
            f = i / 7
            px = corners[3][0] + (corners[2][0] - corners[3][0]) * f
            py = corners[3][1] + (corners[2][1] - corners[3][1]) * f
            k.line(px, py, px + 0.13, py + 0.2, _alpha(k.color, 170), 0.8)


@_icon("Mirror")
def _mirror(k: _Ctx):
    k.arrow(-0.95, 0, -0.06, 0, k.accent)
    k.arrow(0.0, -0.06, 0.0, -0.95, k.accent)
    _slab(k, 0.72, False)


@_icon("DichroicMirror")
def _dichroic(k: _Ctx):
    k.arrow(-0.95, 0, -0.06, 0, k.accent)
    k.arrow(0.0, -0.06, 0.0, -0.95, k.accent)
    k.arrow(0.1, 0, 0.95, 0, k.alt)
    _slab(k, 0.72, True)


@_icon("BeamSplitter")
def _bs(k: _Ctx):
    h = 0.48
    k.beam(-0.95, 0, -h, 0)
    k.beam(h, 0, 0.95, 0)
    k.beam(0, -h, 0, -0.95)
    k.rect(-h, -h, h, h, _alpha(k.color, 28))
    k.line(-h, h, h, -h, k.accent, 1.4)


@_icon("PBS")
def _pbs(k: _Ctx):
    h = 0.42
    k.beam(-0.95, 0, -h, 0, arrow=False)
    k.rect(-h, -h, h, h, _alpha(k.color, 28))
    k.line(-h, h, h, -h, k.accent, 1.4)
    k.line(h, 0, 0.62, 0, k.accent)
    k.line(0, -h, 0, -0.62, k.accent)
    k.line(0.62, 0, 0.95, 0, k.accent, 0.6)
    k.line(0, -0.62, 0, -0.95, k.accent, 0.6)
    k.dot(0, -0.8, 0.07)
    k.ellipse(0, -0.8, 0.13, 0.13, None, k.color, 0.8)
    k.line(0.8, -0.14, 0.8, 0.14, k.color)
    k.head(0.8, -0.14, -math.pi / 2, k.color, 0.1)
    k.head(0.8, 0.14, math.pi / 2, k.color, 0.1)


@_icon("FiberBeamSplitter")
def _fbs(k: _Ctx):
    fc = k.accent
    for sy in (-1, 1):
        path = QPainterPath(QPointF(-0.95, sy * 0.5))
        path.cubicTo(-0.5, sy * 0.5, -0.45, sy * 0.1, -0.12, sy * 0.1)
        path.lineTo(0.12, sy * 0.1)
        path.cubicTo(0.45, sy * 0.1, 0.5, sy * 0.5, 0.95, sy * 0.5)
        k.stroke(path, fc, 1.7)
    k.rect(-0.2, -0.22, 0.2, 0.22, _alpha(k.color, 40), k.color, r=0.12)
    if k.detail:
        k.dot(-0.95, -0.5, 0.05, fc)
        k.dot(-0.95, 0.5, 0.05, fc)
        k.dot(0.95, -0.5, 0.05, fc)
        k.dot(0.95, 0.5, 0.05, fc)


@_icon("WavePlate")
def _waveplate(k: _Ctx):
    k.axis()
    k.rect(-0.2, -0.6, 0.2, 0.6, _alpha(k.accent, 45))
    if k.detail:
        for i in range(-2, 3):
            y = i * 0.2
            k.line(-0.2, y + 0.2, 0.2, y - 0.2, _alpha(k.color, 150), 0.7)
    k.arrow(0.0, 0.0, 0.5, -0.5, k.accent, 1.2, 0.12)
    k.arrow(0.0, 0.0, -0.5, 0.5, k.accent, 1.2, 0.12)


@_icon("Polarizer")
def _polarizer(k: _Ctx):
    k.axis()
    k.rect(-0.14, -0.7, 0.14, 0.7, _alpha(k.color, 30))
    n = 7 if k.detail else 5
    for i in range(n):
        y = -0.55 + 1.1 * i / (n - 1)
        k.line(-0.14, y, 0.14, y, k.accent, 0.9)


@_icon("FiberPaddleController")
def _paddles(k: _Ctx):
    for x in (-0.5, 0.0, 0.5):
        k.ellipse(x, 0, 0.17, 0.52, _alpha(k.color, 25), k.color)
    path = QPainterPath(QPointF(-0.95, 0.0))
    path.lineTo(-0.67, 0.0)
    for x in (-0.5, 0.0, 0.5):
        path.cubicTo(x - 0.17, -0.7, x + 0.17, -0.7, x + 0.17, 0.0)
        path.cubicTo(x + 0.17, 0.4, x + 0.1, 0.4, x + 0.17 + 0.1, 0.0)
    path.lineTo(0.95, 0.0)
    k.stroke(path, k.accent, 1.5)


@_icon("SpectralFilter")
def _spectral(k: _Ctx):
    k.axis()
    k.rect(-0.62, -0.5, 0.62, 0.5, _alpha(k.color, 25), r=0.05)
    path = QPainterPath(QPointF(-0.48, 0.3))
    path.cubicTo(-0.18, 0.3, -0.25, -0.32, 0.0, -0.32)
    path.cubicTo(0.25, -0.32, 0.18, 0.3, 0.48, 0.3)
    k.stroke(path, k.accent, 1.4)
    k.line(-0.48, 0.3, 0.48, 0.3, _alpha(k.color, 130), 0.7)


@_icon("NDFilter")
def _nd(k: _Ctx):
    k.axis()
    g = QLinearGradient(-0.3, 0, 0.3, 0)
    g.setColorAt(0.0, _alpha(k.color, 15))
    g.setColorAt(1.0, _alpha(k.color, 230))
    k.p.setBrush(QBrush(g))
    k.p.setPen(k.pen())
    path = QPainterPath()
    path.addRoundedRect(QRectF(QPointF(-0.3, -0.6), QPointF(0.3, 0.6)), 0.05, 0.05)
    k.p.drawPath(path)


@_icon("Iris")
def _iris(k: _Ctx):
    k.arrow(-0.95, 0, 0.95, 0, _alpha(k.accent, 200), 0.9)
    k.poly(
        [(-0.1, -0.85), (0.1, -0.85), (0.1, -0.3), (0.0, -0.18), (-0.1, -0.3)], _alpha(k.color, 90)
    )
    k.poly([(-0.1, 0.85), (0.1, 0.85), (0.1, 0.3), (0.0, 0.18), (-0.1, 0.3)], _alpha(k.color, 90))


@_icon("BeamDump")
def _dump(k: _Ctx):
    k.rect(-0.05, -0.65, 0.8, 0.65, _alpha(k.color, 60))
    k.poly([(-0.05, -0.42), (0.58, 0.0), (-0.05, 0.42)], _alpha(k.color, 190))
    k.arrow(-0.95, 0, 0.2, 0, k.accent)
    if k.detail:
        k.poly([(0.2, 0), (0.05, -0.14), (0.3, -0.07)], None, k.accent, close=False, k=0.7)


@_icon("OpticalFiber")
def _fiber(k: _Ctx):
    k.rect(-0.9, -0.16, -0.62, 0.16, _alpha(k.color, 60))
    k.rect(0.62, -0.16, 0.9, 0.16, _alpha(k.color, 60))
    path = QPainterPath(QPointF(-0.62, 0.0))
    path.cubicTo(-0.2, 0.0, -0.1, -0.8, 0.15, -0.55)
    path.cubicTo(0.35, -0.3, 0.0, 0.1, -0.2, -0.1)
    path.cubicTo(0.0, 0.1, 0.3, 0.0, 0.62, 0.0)
    k.stroke(path, k.accent, 1.7)


@_icon("FiberCoupler")
def _coupler(k: _Ctx):
    k.arrow(-0.95, 0, -0.52, 0, k.accent, 0.9, 0.1)
    k.biconvex(-0.4, 0.5, 0.12, _alpha(k.accent, 50))
    k.line(-0.35, -0.42, 0.3, -0.07, k.accent, 0.9)
    k.line(-0.35, 0.42, 0.3, 0.07, k.accent, 0.9)
    k.rect(0.3, -0.16, 0.5, 0.16, _alpha(k.color, 60))
    k.line(0.5, 0, 0.95, 0, k.accent, 1.8)


@_icon("Photodiode")
def _photodiode(k: _Ctx):
    k.ellipse(0.1, 0.1, 0.58, 0.58, _alpha(k.color, 25))
    k.poly([(-0.15, -0.18), (-0.15, 0.38), (0.25, 0.1)], _alpha(k.color, 80))
    k.line(0.28, -0.12, 0.28, 0.32)
    k.line(-0.45, 0.1, -0.15, 0.1)
    k.line(0.28, 0.1, 0.68, 0.1)
    k.arrow(-0.95, -0.7, -0.25, -0.25, k.accent, 1.0, 0.13)
    k.arrow(-0.7, -0.88, 0.0, -0.43, k.accent, 1.0, 0.13)


@_icon("SPAD")
def _spad(k: _Ctx):
    k.wave(-0.95, -0.3, 0.0, 0.1, 1.5)
    k.rect(-0.3, -0.5, 0.7, 0.5, _alpha(k.color, 30), r=0.06)
    path = QPainterPath(QPointF(-0.18, 0.28))
    path.lineTo(0.05, 0.28)
    path.lineTo(0.12, -0.32)
    path.cubicTo(0.2, 0.05, 0.3, 0.26, 0.55, 0.28)
    k.stroke(path, k.accent, 1.4)
    if k.detail:
        k.line(-0.18, 0.28, 0.55, 0.28, _alpha(k.color, 120), 0.6)


@_icon("SNSPD")
def _snspd(k: _Ctx):
    k.wave(-0.95, -0.55, 0.0, 0.09, 1.2)
    k.rect(-0.55, -0.55, 0.65, 0.55, _alpha(k.color, 25), r=0.05)
    n = 5 if k.detail else 4
    x1, x2 = -0.38, 0.48
    ys = [-0.36 + 0.72 * i / (n - 1) for i in range(n)]
    path = QPainterPath(QPointF(x1, ys[0]))
    for i, y in enumerate(ys):
        if i:
            path.lineTo(x1 if i % 2 == 0 else x2, ys[i - 1])
            path.lineTo(x1 if i % 2 == 0 else x2, y)
        path.lineTo(x2 if i % 2 == 0 else x1, y)
    k.stroke(path, k.accent, 1.5)


@_icon("TimeTagger")
def _tagger(k: _Ctx):
    k.ellipse(0.25, 0, 0.6, 0.6, _alpha(k.color, 25))
    k.line(-0.45, -0.62, -0.45, 0.62, k.color, 1.1)
    for y in (-0.5, -0.17, 0.17, 0.5):
        k.line(-0.95, y, -0.45, y, k.accent, 1.1)
    n = 12 if k.detail else 4
    for i in range(n):
        a = 2 * math.pi * i / n
        r0 = 0.46 if k.detail else 0.44
        k.line(
            0.25 + r0 * math.cos(a),
            r0 * math.sin(a),
            0.25 + 0.56 * math.cos(a),
            0.56 * math.sin(a),
            k.color,
            0.9,
        )
    k.line(0.25, 0, 0.25, -0.34, k.accent, 1.3)
    k.line(0.25, 0, 0.45, 0.14, k.accent, 1.3)
    k.dot(0.25, 0, 0.04)


@_icon("PowerMeter")
def _power(k: _Ctx):
    k.arrow(-0.95, 0, -0.72, 0, k.accent, 1.0, 0.1)
    k.rect(-0.72, -0.6, 0.85, 0.6, _alpha(k.color, 25), r=0.1)
    cx, cy, r = 0.065, 0.3, 0.55
    arc = QPainterPath()
    arc.arcMoveTo(QRectF(cx - r, cy - r, 2 * r, 2 * r), 150)
    arc.arcTo(QRectF(cx - r, cy - r, 2 * r, 2 * r), 150, -120)
    k.stroke(arc, k.color, 1.0)
    n = 7 if k.detail else 4
    for i in range(n):
        a = math.radians(150 - 120 * i / (n - 1))
        k.line(
            cx + 0.45 * math.cos(a),
            cy - 0.45 * math.sin(a),
            cx + 0.55 * math.cos(a),
            cy - 0.55 * math.sin(a),
            k.color,
            0.9,
        )
    a = math.radians(62)
    k.line(cx, cy, cx + 0.5 * math.cos(a), cy - 0.5 * math.sin(a), k.accent, 1.4)
    k.dot(cx, cy, 0.06)


@_icon("BeamProfiler")
def _profiler(k: _Ctx):
    k.arrow(-0.95, 0, -0.55, 0, k.accent, 1.0, 0.1)
    k.rect(-0.55, -0.55, 0.65, 0.55, _alpha(k.color, 25))
    cx = 0.05
    if k.detail:
        for i in range(1, 4):
            f = -0.55 + 1.2 * i / 4
            k.line(f, -0.55, f, 0.55, _alpha(k.color, 60), 0.5)
            k.line(-0.55, f * 0.917, 0.65, f * 0.917, _alpha(k.color, 60), 0.5)
    for r, a in ((0.42, 40), (0.29, 90), (0.16, 170)):
        k.ellipse(cx, 0, r, r * 0.8, _alpha(k.accent, a), k.accent, 0.7)
    k.rect(0.65, -0.18, 0.85, 0.18, _alpha(k.color, 60))


@_icon("Polarimeter")
def _polarimeter(k: _Ctx):
    k.arrow(-0.95, 0, -0.58, 0, k.accent, 1.0, 0.1)
    cx, r = 0.2, 0.62
    k.ellipse(cx, 0, r, r, _alpha(k.color, 25))
    k.ellipse(cx, 0, r, r * 0.3, None, k.accent, 0.9)
    path = QPainterPath()
    path.addEllipse(QPointF(cx, 0), r * 0.3, r)
    k.stroke(path, _alpha(k.color, 150), 0.8)
    k.line(cx, -r, cx, r, _alpha(k.color, 90), 0.5)
    k.dot(cx + 0.28, -0.35, 0.07, k.accent)


@_icon("Spectrometer")
def _spectrometer(k: _Ctx):
    k.arrow(-0.95, 0.05, -0.22, 0.05, k.accent, 1.0, 0.1)
    k.poly([(-0.4, 0.4), (0.4, 0.4), (0.0, -0.3)], _alpha(k.color, 35))
    k.line(-0.2, 0.05, 0.2, 0.05, _alpha(k.accent, 200), 0.9)
    n = 5 if k.detail else 3
    for i in range(n):
        f = i / (n - 1)
        col = _hue_shift(k.accent, -60 + 200 * f)
        k.line(0.2, 0.05, 0.95, 0.0 + 0.05 + 0.1 + 0.5 * f - 0.2, col, 1.0)


@_icon("_generic")
def _generic(k: _Ctx):
    k.p.setBrush(_alpha(k.color, 25))
    k.p.setPen(k.pen(dash=True))
    path = QPainterPath()
    path.addRoundedRect(QRectF(QPointF(-0.7, -0.7), QPointF(0.7, 0.7)), 0.15, 0.15)
    k.p.drawPath(path)
    q = QPainterPath()
    q.arcMoveTo(QRectF(-0.22, -0.46, 0.44, 0.4), 160)
    q.arcTo(QRectF(-0.22, -0.46, 0.44, 0.4), 160, -250)
    q.lineTo(0.0, 0.14)
    k.stroke(q, k.accent, 1.6)
    k.dot(0.0, 0.36, 0.06, k.accent)


ICON_TYPES: tuple[str, ...] = tuple(n for n in _DRAWERS if not n.startswith("_"))


def paint_icon(
    painter: QPainter,
    type_name: str,
    rect: QRectF,
    *,
    color: QColor,
    accent: QColor | None = None,
    fiber: bool = False,
) -> None:
    size = min(rect.width(), rect.height())
    if size <= 0:
        return
    s = size * 0.42
    px = min(2.4, max(1.0, size / 34.0))
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(rect.center())
        painter.scale(s, s)
        k = _Ctx(painter, QColor(color), QColor(accent or DEFAULT_ACCENT), fiber, px, s)
        _DRAWERS.get(type_name, _generic)(k)
    finally:
        painter.restore()


def component_pixmap(type_name: str, size: int = 48, *, color: QColor | None = None) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    try:
        paint_icon(p, type_name, QRectF(0, 0, size, size), color=color or DEFAULT_COLOR)
    finally:
        p.end()
    return pm
