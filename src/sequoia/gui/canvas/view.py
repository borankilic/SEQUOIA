from PySide6.QtCore import QLineF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsView

from sequoia.gui.canvas.items import CELL
from sequoia.gui.canvas.scene import CanvasScene

PART_MIME = "application/x-sequoia-part"
MIN_ZOOM, MAX_ZOOM = 0.1, 4.0


class CanvasView(QGraphicsView):
    status_message = Signal(str)

    def __init__(self, scene: CanvasScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        self._space = False
        self._pan_last = None
        scene.status_message.connect(self.status_message)

    def set_snap(self, enabled: bool) -> None:
        self.scene().set_snap(enabled)

    @property
    def zoom_level(self) -> float:
        return self.transform().m11()

    def zoom_by(self, factor: float) -> None:
        target = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom_level * factor))
        factor = target / self.zoom_level
        self.scale(factor, factor)

    def wheelEvent(self, event) -> None:
        dy = event.angleDelta().y()
        if dy:
            self.zoom_by(1.0015**dy)
        event.accept()

    def fit_to_content(self) -> None:
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            self.resetTransform()
            self.centerOn(0, 0)
            return
        self.fitInView(rect.adjusted(-CELL, -CELL, CELL, CELL), Qt.KeepAspectRatio)
        if self.zoom_level > 2.0:
            self.zoom_by(2.0 / self.zoom_level)

    def view_center_cells(self) -> tuple[float, float]:
        c = self.mapToScene(self.viewport().rect().center())
        return c.x() / CELL, c.y() / CELL

    def drawBackground(self, painter: QPainter, rect) -> None:
        base = self.palette().base().color()
        painter.fillRect(rect, base)
        dark = base.lightness() < 128
        minor = QColor(255, 255, 255, 18) if dark else QColor(0, 0, 0, 20)
        major = QColor(255, 255, 255, 40) if dark else QColor(0, 0, 0, 45)
        left, right = int(rect.left() // CELL) - 1, int(rect.right() // CELL) + 1
        top, bottom = int(rect.top() // CELL) - 1, int(rect.bottom() // CELL) + 1
        minor_lines, major_lines = [], []
        draw_minor = self.zoom_level * CELL >= 8
        for i in range(left, right + 1):
            line = QLineF(i * CELL, rect.top(), i * CELL, rect.bottom())
            (major_lines if i % 5 == 0 else minor_lines).append(line)
        for j in range(top, bottom + 1):
            line = QLineF(rect.left(), j * CELL, rect.right(), j * CELL)
            (major_lines if j % 5 == 0 else minor_lines).append(line)
        if draw_minor:
            pen = QPen(minor, 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLines(minor_lines)
        pen = QPen(major, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLines(major_lines)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space):
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_last is not None:
            pos = event.position().toPoint()
            delta = pos - self._pan_last
            self._pan_last = pos
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_last is not None:
            self._pan_last = None
            if self._space:
                self.viewport().setCursor(Qt.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(PART_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(PART_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        if not mime.hasFormat(PART_MIME):
            super().dropEvent(event)
            return
        ref = bytes(mime.data(PART_MIME)).decode("utf-8").strip()
        point = self.mapToScene(event.position().toPoint())
        self.scene().add_part(ref, (point.x() / CELL, point.y() / CELL))
        event.acceptProposedAction()
        self.setFocus()
