from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sequoia.setup.io import load_setup
from sequoia.setup.model import Setup

BOX_W = 2.0
BOX_H = 1.5
THUMB_SIZE = QSize(240, 140)
COLUMNS = 3
MAX_RECENT_SHOWN = 8
ORDER = [
    "laser_power_meter",
    "malus_law",
    "beam_expander",
    "mach_zehnder",
    "laser_fiber_coupling",
    "spdc_car",
    "hbt_heralded",
    "hom",
    "bell_test",
]
CATEGORY_COLORS = {
    "sources": "#d9822b",
    "nonlinear": "#c0507a",
    "detectors": "#3a7ca5",
    "electronics": "#6b7280",
    "fiber": "#2f9e8f",
    "polarization": "#7c5cbf",
    "filters": "#8a9a3b",
    "probes": "#5b8c5a",
}
DEFAULT_COLOR = "#7a8794"
LINK_COLORS = {"free_space": "#d9822b", "fiber": "#2f9e8f", "cable": "#8b95a1"}


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    path: Path
    tags: tuple[str, ...] = field(default_factory=tuple)


def default_templates_root() -> Path:
    packaged = Path(__file__).resolve().parents[3] / "templates"
    return packaged if packaged.is_dir() else Path("templates")


def _tags(setup: Setup) -> tuple[str, ...]:
    parts = [instance.part for instance in setup.instances]
    categories = {part.split("/", 1)[0] for part in parts}
    tags = []
    if "nonlinear" in categories:
        tags.append("photon pairs")
    if any(("spad" in part or "snspd" in part) for part in parts):
        tags.append("single photons")
    if "electronics" in categories:
        tags.append("time tagging")
    if "polarization" in categories:
        tags.append("polarization")
    if not tags:
        tags.append("classical")
    if any(link.kind == "fiber" for link in setup.links):
        tags.append("fiber")
    return tuple(tags)


def discover_templates(root: Path) -> list[Template]:
    root = Path(root)
    if not root.is_dir():
        return []
    templates = []
    for path in sorted(list(root.glob("*.yaml")) + list(root.glob("*.yml"))):
        try:
            setup = load_setup(path)
        except Exception:
            continue
        templates.append(
            Template(setup.name or path.stem, setup.description.strip(), path, _tags(setup))
        )

    def key(template: Template):
        stem = template.path.stem
        return (ORDER.index(stem) if stem in ORDER else len(ORDER), stem)

    return sorted(templates, key=key)


def box_size(rotation: float) -> tuple[float, float]:
    if round(rotation) % 180 == 90:
        return BOX_H, BOX_W
    return BOX_W, BOX_H


def render_thumbnail(setup: Setup, size: QSize = THUMB_SIZE, ratio: float = 2.0) -> QPixmap:
    pixmap = QPixmap(int(size.width() * ratio), int(size.height() * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QColor("#f7f8fa"))
    if not setup.instances:
        return pixmap
    boxes = {}
    for instance in setup.instances:
        w, h = box_size(instance.rotation)
        x, y = instance.position
        boxes[instance.id] = (instance, QRectF(x - w / 2, y - h / 2, w, h))
    left = min(r.left() for _, r in boxes.values())
    right = max(r.right() for _, r in boxes.values())
    top = min(r.top() for _, r in boxes.values())
    bottom = max(r.bottom() for _, r in boxes.values())
    pad = 14.0
    avail_w = size.width() - 2 * pad
    avail_h = size.height() - 2 * pad
    scale = min(avail_w / (right - left), avail_h / (bottom - top), 14.0)
    off_x = pad + (avail_w - (right - left) * scale) / 2 - left * scale
    off_y = pad + (avail_h - (bottom - top) * scale) / 2 - top * scale

    def point(x: float, y: float) -> QPointF:
        return QPointF(off_x + x * scale, off_y + y * scale)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for link in setup.links:
        src = boxes.get(link.src.split(".")[0])
        dst = boxes.get(link.dst.split(".")[0])
        if src is None or dst is None:
            continue
        pen = QPen(QColor(LINK_COLORS.get(link.kind, "#8b95a1")))
        pen.setWidthF(1.4)
        if link.kind == "cable":
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(point(*src[0].position), point(*dst[0].position))
    for instance, rect in boxes.values():
        color = QColor(CATEGORY_COLORS.get(instance.part.split("/", 1)[0], DEFAULT_COLOR))
        target = QRectF(
            off_x + rect.left() * scale,
            off_y + rect.top() * scale,
            rect.width() * scale,
            rect.height() * scale,
        )
        painter.setPen(QPen(color.darker(130), 1.0))
        painter.setBrush(color.lighter(125))
        painter.drawRoundedRect(target, 2.5, 2.5)
    painter.end()
    return pixmap


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 == len(text) or text[i + 1] == " "):
            text = text[: i + 1]
            break
    if len(text) > 110:
        text = text[:107].rsplit(" ", 1)[0] + "..."
    return text


class TemplateCard(QFrame):
    clicked = Signal(str)

    def __init__(self, template: Template, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.template = template
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(THUMB_SIZE.width() + 24)
        self.setToolTip(template.description or template.name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(THUMB_SIZE)
        self.thumbnail.setObjectName("thumb")
        try:
            self.thumbnail.setPixmap(render_thumbnail(load_setup(template.path)))
        except Exception:
            pass
        title = QLabel(template.name)
        font = title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        title.setFont(font)
        title.setWordWrap(True)
        desc = QLabel(first_sentence(template.description))
        desc.setWordWrap(True)
        desc.setObjectName("muted")
        desc.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        tags = QLabel("  ·  ".join(template.tags[:2]))
        tags.setObjectName("tags")
        layout.addWidget(self.thumbnail)
        layout.addWidget(title)
        layout.addWidget(desc, 1)
        layout.addWidget(tags)
        for child in (self.thumbnail, title, desc, tags):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit(str(self.template.path))
        super().mouseReleaseEvent(event)


class StartScreen(QWidget):
    new_requested = Signal()
    open_requested = Signal()
    path_requested = Signal(str)
    template_requested = Signal(str)

    def __init__(
        self,
        templates_root: Path | None = None,
        settings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SEQUOIA")
        self.settings = settings
        if self.settings is None:
            from sequoia.gui.main_window import make_settings

            self.settings = make_settings()
        root = templates_root if templates_root is not None else default_templates_root()
        self.templates = discover_templates(root)
        self.cards: list[TemplateCard] = []
        self._build()
        self.resize(1200, 720)
        self.setStyleSheet(STYLE)

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(300)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(24, 28, 24, 24)
        side_layout.setSpacing(10)
        title = QLabel("SEQUOIA")
        title.setObjectName("title")
        subtitle = QLabel("Quantum-optics lab simulator")
        subtitle.setObjectName("muted")
        self.new_button = QPushButton("New project")
        self.new_button.setObjectName("primary")
        self.open_button = QPushButton("Open project...")
        self.new_button.clicked.connect(self.new_requested)
        self.open_button.clicked.connect(self.open_requested)
        recent_label = QLabel("Recent projects")
        recent_label.setObjectName("section")
        self.recent_list = QListWidget()
        self.recent_list.setFrameShape(QFrame.Shape.NoFrame)
        self.recent_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.recent_list.itemActivated.connect(self._recent_activated)
        self.recent_list.itemDoubleClicked.connect(self._recent_activated)
        self.recent_empty = QLabel("No recent projects yet.")
        self.recent_empty.setObjectName("muted")
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(18)
        side_layout.addWidget(self.new_button)
        side_layout.addWidget(self.open_button)
        side_layout.addSpacing(18)
        side_layout.addWidget(recent_label)
        side_layout.addWidget(self.recent_empty)
        side_layout.addWidget(self.recent_list, 1)
        self.refresh_recent()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(28, 28, 28, 0)
        heading = QLabel("Start from a template")
        heading.setObjectName("heading")
        hint = QLabel(
            "A template opens as a new untitled project; the shipped file is never overwritten."
        )
        hint.setObjectName("muted")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 8, 8, 24)
        grid.setSpacing(16)
        for index, template in enumerate(self.templates):
            card = TemplateCard(template)
            card.clicked.connect(self.template_requested)
            self.cards.append(card)
            grid.addWidget(card, index // COLUMNS, index % COLUMNS)
        grid.setColumnStretch(COLUMNS, 1)
        grid.setRowStretch(len(self.templates) // COLUMNS + 1, 1)
        if not self.templates:
            grid.addWidget(QLabel("No templates found."), 0, 0)
        scroll.setWidget(holder)
        main_layout.addWidget(heading)
        main_layout.addWidget(hint)
        main_layout.addWidget(scroll, 1)

        outer.addWidget(side)
        outer.addWidget(main, 1)

    def recent_paths(self) -> list[str]:
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [str(p) for p in (value or [])]

    def refresh_recent(self) -> None:
        self.recent_list.clear()
        for path in self.recent_paths():
            if not Path(path).is_file():
                continue
            if self.recent_list.count() >= MAX_RECENT_SHOWN:
                break
            p = Path(path)
            item = QListWidgetItem(f"{p.stem}\n{p.parent.name}/{p.name}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.recent_list.addItem(item)
        empty = self.recent_list.count() == 0
        self.recent_empty.setVisible(empty)
        self.recent_list.setVisible(not empty)

    def _recent_activated(self, item: QListWidgetItem) -> None:
        self.path_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))


STYLE = """
QWidget { font-size: 13px; }
QFrame#side { background: palette(alternate-base); border-right: 1px solid palette(mid); }
QLabel#title { font-size: 26px; font-weight: 700; letter-spacing: 3px; }
QLabel#heading { font-size: 20px; font-weight: 600; }
QLabel#section { font-weight: 600; }
QLabel#muted { color: palette(placeholder-text); }
QLabel#tags { color: #3a7ca5; font-size: 11px; }
QLabel#thumb { background: #f7f8fa; border: 1px solid palette(mid); border-radius: 6px; }
QFrame#card { background: palette(base); border: 1px solid palette(mid); border-radius: 10px; }
QFrame#card:hover { border: 1px solid #3a7ca5; }
QPushButton { padding: 9px 14px; border-radius: 6px; border: 1px solid palette(mid);
  background: palette(button); text-align: left; }
QPushButton:hover { border-color: #3a7ca5; }
QPushButton#primary { background: #2f6f9f; color: white; border: 1px solid #2f6f9f;
  font-weight: 600; }
QPushButton#primary:hover { background: #3a7ca5; }
QListWidget { background: transparent; outline: 0; }
QListWidget::item { padding: 7px 8px; border-radius: 6px; }
QListWidget::item:hover { background: palette(midlight); }
QListWidget::item:selected { background: #3a7ca5; color: white; }
"""
