import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
)
from PySide6.QtWidgets import QApplication, QGraphicsItem

from sequoia.components.base import PortDirection, PortKind

CELL = 40.0
SNAP_CELLS = 0.5
PORT_RADIUS = 4.0
PORT_HIT = 9.0

KIND_COLORS = {
    PortKind.FREE_SPACE: QColor("#e67e22"),
    PortKind.FIBER: QColor("#2a6fdb"),
    PortKind.ELECTRICAL: QColor("#8e44ad"),
}
LINK_COLORS = {
    "free_space": KIND_COLORS[PortKind.FREE_SPACE],
    "fiber": KIND_COLORS[PortKind.FIBER],
    "cable": KIND_COLORS[PortKind.ELECTRICAL],
}
SEVERITY_COLORS = {
    "error": QColor("#d62728"),
    "warning": QColor("#e6a700"),
    "info": QColor("#1f77b4"),
}
VALID_COLOR = QColor("#2ecc71")
INVALID_COLOR = QColor("#e74c3c")


_FONT_SPECS = {
    "type": (9, False, False),
    "id": (11, True, False),
    "port": (8, False, False),
    "label": (10, False, False),
    "label_italic": (10, False, True),
}
_FONTS: dict = {}


MIN_LABEL_PIXELS = 6.0
HIDE_ALL_TEXT_BELOW = 0.3


def font_scale(name: str, zoom: float) -> float:
    return 1.0


def label_visible(name: str, zoom: float) -> bool:
    return _FONT_SPECS[name][0] * zoom >= MIN_LABEL_PIXELS


def font(name: str, zoom: float = 1.0) -> QFont:
    pixels, bold, italic = _FONT_SPECS[name]
    key = (name, pixels)
    if key not in _FONTS:
        f = QFont(QApplication.font())
        f.setPixelSize(pixels)
        f.setBold(bold)
        f.setItalic(italic)
        _FONTS[key] = f
    return _FONTS[key]


def body_size(instance_id: str, type_label: str, ports) -> tuple[float, float]:
    ins = [p for p in ports if p.direction == PortDirection.IN]
    outs = [p for p in ports if p.direction == PortDirection.OUT]
    n = max(len(ins), len(outs), 1)
    height = max(2.0, 0.5 * (n + 1)) * CELL
    fm = QFontMetricsF(font("port"))
    side = max([fm.horizontalAdvance(p.name) for p in ports if p.name not in ("in", "out")] or [0])
    need = 2 * side + 36
    return max(2, math.ceil(need / CELL)) * CELL, height


def snap_value(value: float, step: float) -> float:
    return round(value / step) * step


def marker_path(kind: PortKind, radius: float = PORT_RADIUS) -> QPainterPath:
    path = QPainterPath()
    r = radius
    if kind == PortKind.ELECTRICAL:
        path.addRect(QRectF(-r, -r, 2 * r, 2 * r))
    elif kind == PortKind.FIBER:
        r *= 1.3
        path.moveTo(0, -r)
        path.lineTo(r, 0)
        path.lineTo(0, r)
        path.lineTo(-r, 0)
        path.closeSubpath()
    else:
        path.addEllipse(QPointF(0, 0), r, r)
    return path


class ComponentItem(QGraphicsItem):
    def __init__(self, canvas, instance, ports, valid: bool) -> None:
        super().__init__()
        self.canvas = canvas
        self.instance_id = instance.id
        self.part = instance.part
        self.rotation_deg = 0.0
        self.flipped = False
        self.ports = ()
        self.valid = True
        self._port_states: dict[str, str] = {}
        self._local: dict[str, tuple[float, float, float, float]] = {}
        self._w = 2 * CELL
        self._h = 2 * CELL
        self.type_name: str | None = None
        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.update_from(instance, ports, valid)

    @property
    def type_label(self) -> str:
        return self.part.rsplit("/", 1)[-1]

    def update_from(self, instance, ports, valid: bool) -> None:
        self.prepareGeometryChange()
        self.part = instance.part
        self.rotation_deg = float(instance.rotation)
        self.flipped = bool(instance.flipped)
        self.ports = tuple(ports)
        self.valid = valid
        document = getattr(self.canvas, "document", None)
        component = document.component(instance.id) if document is not None else None
        self.type_name = type(component).__name__ if component is not None else None
        self._layout()
        target = QPointF(instance.position[0] * CELL, instance.position[1] * CELL)
        if self.pos() != target:
            self.setPos(target)
        self.update()

    @staticmethod
    def _shows_label(name: str) -> bool:
        return name not in ("in", "out")

    def _layout(self) -> None:
        ins = [p for p in self.ports if p.direction == PortDirection.IN]
        outs = [p for p in self.ports if p.direction == PortDirection.OUT]
        self._w, self._h = body_size(self.instance_id, self.type_label, self.ports)
        self._local = {}
        for group, x, inward in ((ins, -self._w / 2, 1.0), (outs, self._w / 2, -1.0)):
            for i, port in enumerate(group):
                y = -self._h / 2 + self._h * (i + 1) / (len(group) + 1)
                self._local[port.name] = (x, y, inward, 0.0)

    def _matrix(self) -> tuple[float, float, float, float]:
        a = math.radians(-self.rotation_deg)
        c, s = math.cos(a), math.sin(a)
        if self.rotation_deg % 90 == 0:
            c, s = round(c), round(s)
        fx = -1.0 if self.flipped else 1.0
        return c * fx, s * fx, -s, c

    def _map(self, x: float, y: float) -> QPointF:
        m11, m12, m21, m22 = self._matrix()
        return QPointF(m11 * x + m21 * y, m12 * x + m22 * y)

    def port_offset(self, name: str) -> QPointF | None:
        entry = self._local.get(name)
        return None if entry is None else self._map(entry[0], entry[1])

    def port_inward(self, name: str) -> QPointF | None:
        entry = self._local.get(name)
        return None if entry is None else self._map(entry[2], 0.0)

    def port_scene_position(self, name: str) -> QPointF | None:
        offset = self.port_offset(name)
        return None if offset is None else self.pos() + offset

    def port_at(self, scene_pos: QPointF, radius: float = PORT_HIT) -> str | None:
        best, best_d = None, radius
        for name in self._local:
            d = math.dist(
                (scene_pos.x(), scene_pos.y()),
                (self.port_scene_position(name).x(), self.port_scene_position(name).y()),
            )
            if d <= best_d:
                best, best_d = name, d
        return best

    def port_spec(self, name: str):
        return next((p for p in self.ports if p.name == name), None)

    def set_port_state(self, name: str | None, state: str | None) -> None:
        if name is None:
            self._port_states.clear()
        elif state is None:
            self._port_states.pop(name, None)
        else:
            self._port_states[name] = state
        self.update()

    def _body_path(self) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(QRectF(-self._w / 2, -self._h / 2, self._w, self._h), 6, 6)
        m11, m12, m21, m22 = self._matrix()
        return QTransform(m11, m12, m21, m22, 0, 0).map(path)

    LABEL_HEIGHT = 30.0

    def _label_rect(self) -> QRectF:
        body = self._body_path().boundingRect()
        width = max(body.width(), 2.4 * CELL)
        return QRectF(body.center().x() - width / 2, body.bottom() + 2, width, self.LABEL_HEIGHT)

    def boundingRect(self) -> QRectF:
        rect = self._body_path().boundingRect().adjusted(-12, -12, 12, 12)
        return rect.united(self._label_rect().adjusted(0, 0, 0, 4))

    def shape(self) -> QPainterPath:
        path = self._body_path()
        for name in self._local:
            p = self.port_offset(name)
            path.addEllipse(p, PORT_RADIUS + 1, PORT_RADIUS + 1)
        return path

    def hoverMoveEvent(self, event) -> None:
        name = self.port_at(event.scenePos())
        if name is None:
            self.setToolTip(f"{self.instance_id} ({self.part})")
        else:
            spec = self.port_spec(name)
            self.setToolTip(f"{self.instance_id}.{name} ({spec.kind}, {spec.direction})")
        super().hoverMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.canvas.item_activated.emit(self.instance_id)

    def itemChange(self, change, value):
        canvas = self.canvas
        if change == QGraphicsItem.ItemPositionChange and canvas is not None and canvas.snap:
            step = SNAP_CELLS * CELL
            return QPointF(snap_value(value.x(), step), snap_value(value.y(), step))
        if change == QGraphicsItem.ItemPositionHasChanged and canvas is not None:
            canvas.item_moved(self)
        return super().itemChange(change, value)

    def _text_angle(self) -> float:
        ang = -(self.rotation_deg % 360.0)
        while ang < -90:
            ang += 180
        while ang >= 90:
            ang -= 180
        return ang

    def _paint_icon(self, painter: QPainter, color: QColor, gutter: float) -> None:
        try:
            from sequoia.gui.canvas.icons import ICON_TYPES, paint_icon
        except ImportError:
            return
        name = self.type_name
        if name is None or name not in ICON_TYPES or self.canvas.zoom < 0.12:
            return
        m11, m12, m21, m22 = self._matrix()
        inset = max(gutter + 3.0, 4.0)
        rect = QRectF(-self._w / 2 + inset, -self._h / 2 + 3, self._w - 2 * inset, self._h - 6)
        painter.save()
        painter.setTransform(QTransform(m11, m12, m21, m22, 0, 0), True)
        paint_icon(painter, name, rect, color=color, fiber=name.startswith("Fiber"))
        painter.restore()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pal = QApplication.palette()
        painter.setRenderHint(QPainter.Antialiasing)
        severity = self.canvas.highlights.get(self.instance_id)
        body = self._body_path()
        text_color = pal.windowText().color()
        edge = QColor(text_color)
        edge.setAlpha(170)
        pen = QPen(edge, 1.3)
        if not self.valid:
            pen = QPen(SEVERITY_COLORS["error"], 1.6, Qt.DashLine)
        if severity in SEVERITY_COLORS:
            pen = QPen(SEVERITY_COLORS[severity], 3.0)
        if self.isSelected():
            glow = QColor(pal.highlight().color())
            glow.setAlpha(110)
            painter.setPen(QPen(glow, 8))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(body)
            if severity not in SEVERITY_COLORS:
                pen = QPen(pal.highlight().color(), 2.2)
        painter.setPen(pen)
        painter.setBrush(QBrush(pal.base().color().darker(103)))
        painter.drawPath(body)

        zoom = self.canvas.zoom
        f_port = font("port")
        fm_p = QFontMetricsF(f_port)
        show_ports = zoom >= HIDE_ALL_TEXT_BELOW and label_visible("port", zoom)
        shown = [
            p.name
            for p in self.ports
            if show_ports
            and p.name in self._local
            and self._shows_label(p.name)
            and fm_p.horizontalAdvance(p.name) <= self._w / 2 - 6
        ]
        side = max([fm_p.horizontalAdvance(n) for n in shown] or [0.0])
        gray = QColor(text_color)
        gray.setAlpha(150)
        self._paint_icon(painter, text_color, side)
        if zoom >= HIDE_ALL_TEXT_BELOW and label_visible("id", zoom):
            rect = self._label_rect()
            painter.setFont(font("id"))
            painter.setPen(text_color)
            fm_i = QFontMetricsF(font("id"))
            painter.drawText(
                QRectF(rect.left(), rect.top(), rect.width(), fm_i.height()),
                Qt.AlignHCenter | Qt.AlignTop,
                fm_i.elidedText(self.instance_id, Qt.ElideRight, rect.width()),
            )
            if label_visible("type", zoom) and zoom >= 0.9:
                fm_t = QFontMetricsF(font("type"))
                painter.setFont(font("type"))
                painter.setPen(gray)
                painter.drawText(
                    QRectF(rect.left(), rect.top() + fm_i.height(), rect.width(), fm_t.height()),
                    Qt.AlignHCenter | Qt.AlignTop,
                    fm_t.elidedText(self.type_label, Qt.ElideRight, rect.width()),
                )

        ang = self._text_angle()
        t = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
        for port in self.ports:
            name = port.name
            if name not in self._local:
                continue
            p = self.port_offset(name)
            state = self._port_states.get(name)
            color = KIND_COLORS[port.kind]
            if state == "valid":
                color = VALID_COLOR
            elif state == "invalid":
                color = INVALID_COLOR
            painter.save()
            painter.translate(p)
            if state:
                painter.setPen(Qt.NoPen)
                halo = QColor(color)
                halo.setAlpha(90)
                painter.setBrush(halo)
                painter.drawEllipse(QPointF(0, 0), PORT_RADIUS + 5, PORT_RADIUS + 5)
            painter.setPen(QPen(text_color, 1.0))
            painter.setBrush(color)
            painter.drawPath(marker_path(port.kind))
            painter.restore()
            if name in shown:
                d = self.port_inward(name)
                painter.save()
                painter.translate(p + d * 7)
                painter.rotate(ang)
                painter.setFont(f_port)
                painter.setPen(gray)
                tw = fm_p.horizontalAdvance(name)
                forward = t[0] * d.x() + t[1] * d.y() >= 0
                x0 = 0.0 if forward else -tw
                painter.drawText(
                    QRectF(x0, -fm_p.height() / 2, tw, fm_p.height()), Qt.AlignCenter, name
                )
                painter.restore()


class ProbeItem(QGraphicsItem):
    def __init__(self, canvas, probe_id: str, part: str, link_id: str) -> None:
        super().__init__()
        self.canvas = canvas
        self.probe_id = probe_id
        self.part = part
        self.link_id = link_id
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setZValue(3)
        self.setToolTip(f"{probe_id} ({part}) on {link_id}")

    def boundingRect(self) -> QRectF:
        f = font("port", self.canvas.zoom)
        w = QFontMetricsF(f).horizontalAdvance(self.probe_id)
        return QRectF(
            -9,
            -14 - QFontMetricsF(f).height() - 4,
            max(18.0, w + 8),
            18 + QFontMetricsF(f).height() + 14,
        )

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def mouseDoubleClickEvent(self, event) -> None:
        self.canvas.item_activated.emit(self.probe_id)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pal = QApplication.palette()
        painter.setRenderHint(QPainter.Antialiasing)
        severity = self.canvas.highlights.get(self.probe_id)
        color = SEVERITY_COLORS.get(severity, pal.windowText().color())
        if self.isSelected():
            color = pal.highlight().color()
        tri = QPainterPath()
        tri.moveTo(0, 0)
        tri.lineTo(-5, -10)
        tri.lineTo(5, -10)
        tri.closeSubpath()
        painter.setPen(QPen(pal.windowText().color(), 1))
        painter.setBrush(color)
        painter.drawPath(tri)
        painter.setFont(font("port", self.canvas.zoom))
        painter.setPen(pal.windowText().color())
        painter.drawText(QPointF(-8, -14), self.probe_id)
