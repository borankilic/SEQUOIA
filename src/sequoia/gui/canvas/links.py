import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QApplication, QGraphicsItem

from sequoia.gui.canvas.items import (
    CELL,
    HIDE_ALL_TEXT_BELOW,
    LINK_COLORS,
    SEVERITY_COLORS,
    font,
    label_visible,
)
from sequoia.gui.canvas.routing import rounded_path, route_link


def format_scalar(scalar) -> str:
    return f"{scalar.value:g} {scalar.unit}"


def format_metres(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3g} m"
    if value >= 1e-3:
        return f"{value * 1e3:.3g} mm"
    return f"{value * 1e6:.3g} um"


def edit_field(link) -> str:
    return "delay" if link.kind == "cable" else "length"


def canvas_length(document, link) -> float | None:
    positions = {i.id: i.position for i in document.setup.instances}
    a = positions.get(link.src.split(".")[0])
    b = positions.get(link.dst.split(".")[0])
    if a is None or b is None:
        return None
    return math.dist(a, b) * document.setup.grid_cell.si


def label_text(document, link) -> tuple[str, bool]:
    if link.kind == "cable":
        return format_scalar(link.delay), False
    if link.kind == "free_space" and link.lock_to_canvas:
        length = canvas_length(document, link)
        return ("?" if length is None else format_metres(length)), True
    return format_scalar(link.length), False


class LinkLabel(QGraphicsItem):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.text = ""
        self.italic = False
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(1)

    def set_text(self, text: str, italic: bool) -> None:
        if text != self.text or italic != self.italic:
            self.prepareGeometryChange()
            self.text, self.italic = text, italic
            self.update()

    def _font(self):
        zoom = getattr(self.parentItem().canvas, "zoom", 1.0)
        return font("label_italic" if self.italic else "label", zoom)

    def boundingRect(self) -> QRectF:
        fm = QFontMetricsF(self._font())
        w = fm.horizontalAdvance(self.text) + 10
        h = fm.height() + 2
        return QRectF(-w / 2, -h / 2, w, h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        zoom = getattr(self.parentItem().canvas, "zoom", 1.0)
        if zoom < HIDE_ALL_TEXT_BELOW or not label_visible("label", zoom):
            return
        pal = QApplication.palette()
        painter.setRenderHint(QPainter.Antialiasing)
        bg = QColor(pal.base().color())
        bg.setAlpha(225)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        r = self.boundingRect()
        painter.drawRoundedRect(r, 4, 4)
        painter.setFont(self._font())
        painter.setPen(pal.windowText().color())
        painter.drawText(r, Qt.AlignCenter, self.text)


class LinkItem(QGraphicsItem):
    def __init__(self, canvas, link) -> None:
        super().__init__()
        self.canvas = canvas
        self.link_id = link.id
        self.link = link
        self.path = QPainterPath()
        self.label = LinkLabel(self)
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self.update_link(link)

    def update_link(self, link) -> None:
        self.link = link
        self.update_geometry()

    def _anchors(self):
        src_inst, src_port = self.link.src.split(".")
        dst_inst, dst_port = self.link.dst.split(".")
        return self.canvas.port_anchor(src_inst, src_port), self.canvas.port_anchor(
            dst_inst, dst_port
        )

    def update_geometry(self) -> None:
        self.prepareGeometryChange()
        a, b = self._anchors()
        path = QPainterPath()
        self.route: list[tuple[float, float]] = []
        if a is not None and b is not None:
            (p0, d0), (p1, d1) = a, b
            obstacles = self.canvas.obstacle_rects()
            points = route_link(
                (p0.x(), p0.y()),
                (d0.x(), d0.y()),
                (p1.x(), p1.y()),
                (d1.x(), d1.y()),
                obstacles,
                clearance=10.0,
                stub=14.0,
                grid=CELL / 4,
            )
            self.route = points
            path = rounded_path(points, 14.0 if self.link.kind == "fiber" else 8.0)
        self.path = path
        text, italic = label_text(self.canvas.document, self.link)
        self.label.set_text(text, italic)
        if path.isEmpty():
            self.label.hide()
        else:
            self.label.show()
            self.label.setPos(path.pointAtPercent(0.5))
        self.update()

    def point_at(self, t: float) -> QPointF:
        return self.path.pointAtPercent(t) if not self.path.isEmpty() else QPointF()

    def _label_rect(self) -> QRectF:
        return self.label.boundingRect().translated(self.label.pos())

    def shape(self) -> QPainterPath:
        if self.path.isEmpty():
            return QPainterPath()
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        out = stroker.createStroke(self.path)
        out.addRect(self._label_rect())
        return out

    def boundingRect(self) -> QRectF:
        return self.shape().boundingRect().adjusted(-6, -6, 6, 6)

    def hoverEnterEvent(self, event) -> None:
        self.canvas.show_link_card(self.link_id, event.screenPos())

    def hoverMoveEvent(self, event) -> None:
        self.canvas.show_link_card(self.link_id, event.screenPos())

    def hoverLeaveEvent(self, event) -> None:
        self.canvas.hide_link_card()

    def mouseDoubleClickEvent(self, event) -> None:
        self.canvas.begin_edit_link(self.link_id)
        self.canvas.item_activated.emit(self.link_id)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.path.isEmpty():
            return
        painter.setRenderHint(QPainter.Antialiasing)
        pal = QApplication.palette()
        color = LINK_COLORS[self.link.kind]
        info = self.canvas.link_infos.get(self.link_id)
        if info is not None and self.link.kind != "cable" and info.color is not None:
            if info.power_w > 0 or (info.photons_hz or 0) > 0:
                color = info.color
        severity = self.canvas.highlights.get(self.link_id)
        painter.setBrush(Qt.NoBrush)
        if severity in SEVERITY_COLORS:
            glow = QColor(SEVERITY_COLORS[severity])
            glow.setAlpha(120)
            painter.setPen(QPen(glow, 8))
            painter.drawPath(self.path)
        if self.isSelected():
            glow = QColor(pal.highlight().color())
            glow.setAlpha(130)
            painter.setPen(QPen(glow, 8))
            painter.drawPath(self.path)
        if self.link.kind == "cable":
            pen = QPen(color, 1.2, Qt.DashLine)
        else:
            pen = QPen(color, 2.0)
        painter.setPen(pen)
        painter.drawPath(self.path)
