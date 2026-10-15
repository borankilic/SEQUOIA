import os
from pathlib import Path

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from sequoia.gui.canvas.icons import ICON_TYPES, component_pixmap, paint_icon

SIZES = (24, 48, 200)
EXPECTED = {
    "Laser", "SPDCCrystal", "EntangledPairSource", "Lens", "Objective", "Mirror", "BeamSplitter",
    "DichroicMirror", "Iris", "BeamDump", "WavePlate", "Polarizer", "PBS", "FiberPaddleController",
    "SpectralFilter", "NDFilter", "OpticalFiber", "FiberCoupler", "FiberBeamSplitter",
    "Photodiode", "SPAD", "SNSPD", "TimeTagger", "PowerMeter", "BeamProfiler", "Polarimeter",
    "Spectrometer", "FockSource", "PhaseShifter",
}  # fmt: skip


def render(name, size, bg="white", color="#222222", **kw):
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(QColor(bg))
    p = QPainter(img)
    paint_icon(p, name, QRectF(0, 0, size, size), color=QColor(color), **kw)
    p.end()
    return img


def non_bg(img):
    bg = img.pixel(0, 0)
    return sum(img.pixel(x, y) != bg for x in range(img.width()) for y in range(img.height()))


def test_all_types_present():
    assert EXPECTED <= set(ICON_TYPES)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", [*ICON_TYPES, "NoSuchComponent"])
def test_renders_not_blank(qapp, name, size):
    img = render(name, size)
    assert non_bg(img) > size


def test_stays_inside_rect(qapp):
    for name in ICON_TYPES:
        img = render(name, 200)
        for x in range(200):
            for y in (0, 199):
                assert img.pixel(x, y) == img.pixel(0, 0), name
        for y in range(200):
            for x in (0, 199):
                assert img.pixel(x, y) == img.pixel(0, 0), name


def test_beamsplitters_differ(qapp):
    for size in SIZES:
        assert render("BeamSplitter", size) != render("FiberBeamSplitter", size)


def test_fiber_flag_changes_output(qapp):
    assert render("BeamSplitter", 48) != render("BeamSplitter", 48, fiber=True)


def test_painter_state_balanced(qapp):
    img = QImage(64, 64, QImage.Format.Format_ARGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    p.translate(3, 5)
    before = (p.transform(), p.pen(), p.brush(), p.renderHints(), p.opacity())
    for name in ICON_TYPES:
        paint_icon(p, name, QRectF(0, 0, 48, 48), color=QColor("black"))
    after = (p.transform(), p.pen(), p.brush(), p.renderHints(), p.opacity())
    assert p.isActive()
    p.end()
    assert before == after


def test_component_pixmap(qapp):
    pm = component_pixmap("Laser", 32)
    assert pm.width() == 32 and pm.height() == 32
    assert non_bg(pm.toImage()) > 0
    assert component_pixmap("Laser", color=QColor("red")).width() == 48


def test_contact_sheet(qapp):
    target = os.environ.get("SEQUOIA_ICON_SHEET")
    if not target:
        pytest.skip("SEQUOIA_ICON_SHEET not set")
    names = [*ICON_TYPES, "NoSuchComponent"]
    pad = 10
    cols = len(SIZES)
    block_w = sum(SIZES) + pad * (cols + 1)
    row_h = [max(SIZES) + pad] * len(names)
    width = block_w * 2
    height = sum(row_h) + pad
    sheet = QImage(width, height, QImage.Format.Format_ARGB32)
    p = QPainter(sheet)
    for half, (bg, fg) in enumerate((("#ffffff", "#2b2f36"), ("#1e2127", "#d8dde6"))):
        p.fillRect(half * block_w, 0, block_w, height, QColor(bg))
        y = pad
        for name in names:
            x = half * block_w + pad
            for size in SIZES:
                paint_icon(
                    p,
                    name,
                    QRectF(x, y + (max(SIZES) - size) / 2, size, size),
                    color=QColor(fg),
                )
                x += size + pad
            y += max(SIZES) + pad
    p.end()
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    assert sheet.save(target)
