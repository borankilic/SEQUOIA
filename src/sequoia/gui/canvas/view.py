import math

from PySide6.QtCore import QEvent, QLineF, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsView, QToolButton

from sequoia.gui.canvas.items import CELL
from sequoia.gui.canvas.scene import CanvasScene

PART_MIME = "application/x-sequoia-part"
MIN_ZOOM, MAX_ZOOM = 0.1, 4.0
MAX_GRID_LINES = 300


class CanvasView(QGraphicsView):
    status_message = Signal(str)

    def __init__(self, scene: CanvasScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        self._space = False
        self._pan_last = None
        self._press_pos: QPoint | None = None
        self._panning_empty = False
        self._build_overlay()
        self._bound_scene: CanvasScene | None = None
        self._bind_scene(scene)

    def _bind_scene(self, scene) -> None:
        if self._bound_scene is not None:
            try:
                self._bound_scene.status_message.disconnect(self.status_message)
            except (RuntimeError, TypeError):
                pass
        self._bound_scene = scene
        if scene is not None:
            scene.status_message.connect(self.status_message)
            scene.set_zoom(self.zoom_level)

    def setScene(self, scene) -> None:
        super().setScene(scene)
        self._bind_scene(scene)

    def _notify_zoom(self) -> None:
        scene = self.scene()
        if scene is not None:
            scene.set_zoom(self.zoom_level)

    def set_snap(self, enabled: bool) -> None:
        self.scene().set_snap(enabled)

    @property
    def zoom_level(self) -> float:
        return self.transform().m11()

    def zoom_by(self, factor: float) -> None:
        target = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom_level * factor))
        factor = target / self.zoom_level
        self.scale(factor, factor)
        self._notify_zoom()

    def _build_overlay(self) -> None:
        self.overlay = []
        for text, tip, slot in (
            ("+", "Zoom in", lambda: self.zoom_by(1.25)),
            ("\u2212", "Zoom out", lambda: self.zoom_by(0.8)),
            ("\u2922", "Fit everything in view", self.fit_to_content),
        ):
            button = QToolButton(self.viewport())
            button.setText(text)
            button.setToolTip(tip)
            button.setAutoRaise(False)
            button.setFixedSize(30, 30)
            button.setStyleSheet(
                "QToolButton{background:rgba(255,255,255,0.92);color:#27303a;"
                "border:1px solid #c5ccd3;border-radius:6px;font-size:16px;}"
                "QToolButton:hover{background:#ffffff;border-color:#2f6fed;}"
            )
            button.clicked.connect(slot)
            self.overlay.append(button)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        h = self.viewport().height()
        for i, button in enumerate(self.overlay):
            button.move(10, h - 10 - 30 * (len(self.overlay) - i) - 4 * (len(self.overlay) - i - 1))
            button.raise_()

    def pan_by(self, dx: float, dy: float) -> None:
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(int(h.value() - dx))
        v.setValue(int(v.value() - dy))

    def wheelEvent(self, event) -> None:
        pixel = event.pixelDelta()
        ctrl = bool(event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))
        trackpad = (not pixel.isNull()) or event.phase() != Qt.NoScrollPhase
        if ctrl:
            dy = pixel.y() * 2 if not pixel.isNull() else event.angleDelta().y()
            if dy:
                self.zoom_by(1.0015**dy)
        elif trackpad:
            delta = pixel if not pixel.isNull() else event.angleDelta()
            self.pan_by(delta.x(), delta.y())
        else:
            dy = event.angleDelta().y()
            if dy:
                self.zoom_by(1.0015**dy)
        event.accept()

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            self.zoom_by(1.0 + event.value())
            event.accept()
            return True
        return super().viewportEvent(event)

    def fit_to_content(self) -> None:
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            self.resetTransform()
            self.centerOn(0, 0)
            self._notify_zoom()
            return
        padded = rect.adjusted(-CELL, -CELL, CELL, CELL)
        self.fitInView(padded, Qt.KeepAspectRatio)
        if self.zoom_level > 2.0:
            self.zoom_by(2.0 / self.zoom_level)
        elif self.zoom_level < MIN_ZOOM:
            self.resetTransform()
            self.scale(MIN_ZOOM, MIN_ZOOM)
            self.centerOn(padded.center())
        self._notify_zoom()

    def view_center_cells(self) -> tuple[float, float]:
        c = self.mapToScene(self.viewport().rect().center())
        return c.x() / CELL, c.y() / CELL

    def drawBackground(self, painter: QPainter, rect) -> None:
        base = self.palette().base().color()
        painter.fillRect(rect, base)
        dark = base.lightness() < 128
        minor = QColor(255, 255, 255, 18) if dark else QColor(0, 0, 0, 20)
        major = QColor(255, 255, 255, 40) if dark else QColor(0, 0, 0, 45)
        if not all(map(math.isfinite, (rect.left(), rect.right(), rect.top(), rect.bottom()))):
            return
        left, right = int(rect.left() // CELL) - 1, int(rect.right() // CELL) + 1
        top, bottom = int(rect.top() // CELL) - 1, int(rect.bottom() // CELL) + 1
        span = max(right - left, bottom - top)
        step = 1
        while span // step > MAX_GRID_LINES:
            step *= 5
        draw_minor = step == 1 and self.zoom_level * CELL >= 8
        minor_lines, major_lines = [], []
        for i in range(left - left % step, right + 1, step):
            line = QLineF(i * CELL, rect.top(), i * CELL, rect.bottom())
            (major_lines if i % 5 == 0 else minor_lines).append(line)
        for j in range(top - top % step, bottom + 1, step):
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
        pos = event.position().toPoint()
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space):
            self._pan_last = pos
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if (
            event.button() == Qt.LeftButton
            and not shift
            and self.itemAt(pos) is None
            and self.scene() is not None
            and self.scene().link_editor is None
        ):
            self._press_pos = pos
            self._panning_empty = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._press_pos is not None:
            if not self._panning_empty and (pos - self._press_pos).manhattanLength() > 3:
                self._panning_empty = True
                self.viewport().setCursor(Qt.ClosedHandCursor)
                self._pan_last = self._press_pos
            if self._panning_empty:
                delta = pos - self._pan_last
                self._pan_last = pos
                self.pan_by(delta.x(), delta.y())
            event.accept()
            return
        if self._pan_last is not None:
            delta = pos - self._pan_last
            self._pan_last = pos
            self.pan_by(delta.x(), delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._press_pos is not None:
            if not self._panning_empty and self.scene() is not None:
                self.scene().clearSelection()
            self._press_pos = None
            self._panning_empty = False
            self._pan_last = None
            self.viewport().unsetCursor()
            event.accept()
            return
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
        self.scene().add_part(ref, (point.x() / CELL, point.y() / CELL), point)
        event.acceptProposedAction()
        self.setFocus()
