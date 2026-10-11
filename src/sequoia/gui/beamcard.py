import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

from sequoia.engines.outcome import SimulationOutcome
from sequoia.gui.colors import wavelength_color
from sequoia.gui.results import eng, plain


@dataclass(frozen=True)
class LinkInfo:
    link_id: str
    kind: str
    power_w: float = 0.0
    photons_hz: float | None = None
    wavelength_m: float | None = None
    w_x: float | None = None
    w_y: float | None = None
    radius_x: float | None = None
    waist_x: float | None = None
    past_waist_x: float | None = None
    divergence_x: float | None = None
    dop: float | None = None
    azimuth: float | None = None
    ellipticity: float | None = None
    coupling: float | None = None
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    spectrum: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    stokes: np.ndarray = field(default_factory=lambda: np.zeros(4), compare=False)

    @property
    def color(self) -> QColor | None:
        if self.wavelength_m is None:
            return None
        return wavelength_color(self.wavelength_m)

    def lines(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        if self.kind == "cable":
            return rows
        rows.append(("power", eng(self.power_w, "W")))
        if self.wavelength_m is not None:
            rows.append(("wavelength", f"{self.wavelength_m * 1e9:.1f} nm"))
        if self.photons_hz is not None:
            rows.append(("photon flux", eng(self.photons_hz, "1/s")))
        if self.w_x is not None:
            rows.append(("w_x / w_y", f"{eng(self.w_x, 'm', 3)} / {eng(self.w_y, 'm', 3)}"))
            if self.radius_x is not None and math.isfinite(self.radius_x):
                rows.append(("R_x", eng(self.radius_x, "m", 3)))
            if self.waist_x is not None:
                rows.append(("waist_x", eng(self.waist_x, "m", 3)))
            if self.past_waist_x is not None:
                rows.append(("past waist", eng(self.past_waist_x, "m", 3)))
            if self.divergence_x is not None:
                rows.append(("divergence", eng(self.divergence_x, "rad", 3)))
        if self.dop is not None:
            rows.append(("DOP", f"{self.dop:.3f}"))
            if self.azimuth is not None and self.ellipticity is not None:
                az, ell = math.degrees(self.azimuth), math.degrees(self.ellipticity)
                rows.append(("ellipse", f"az {az:.1f}°, ell {ell:.1f}°"))
        if self.coupling is not None:
            rows.append(("coupling", f"{100 * self.coupling:.3g} %"))
        return rows


def build_link_infos(outcome: SimulationOutcome | None) -> dict[str, LinkInfo]:
    infos: dict[str, LinkInfo] = {}
    if outcome is None or outcome.classical is None:
        return infos
    rates = outcome.rates
    pair_wavelength = None
    if rates is not None:
        for source in rates.sources.values():
            pair_wavelength = source.get("signal_wavelength_m") or pair_wavelength
    for link_id, state in outcome.classical.links.items():
        light = state.end
        flux = None
        if rates is not None and link_id in rates.links:
            flux = float(rates.links[link_id].get("photons_hz", 0.0))
        if light is None or getattr(light, "mode", None) is None:
            infos[link_id] = LinkInfo(link_id, state.kind, photons_hz=flux)
            continue
        mode = light.mode
        wavelength = light.mean_wavelength
        if flux and light.power < 1e-9 and pair_wavelength:
            wavelength = pair_wavelength
        infos[link_id] = LinkInfo(
            link_id=link_id,
            kind=state.kind,
            power_w=light.power,
            photons_hz=flux,
            wavelength_m=wavelength,
            w_x=mode.w_x,
            w_y=mode.w_y,
            radius_x=mode.radius_x,
            waist_x=mode.waist_x,
            past_waist_x=mode.distance_past_waist_x,
            divergence_x=mode.divergence_x,
            dop=light.dop,
            azimuth=light.azimuth,
            ellipticity=light.ellipticity,
            coupling=state.info.get("coupling_efficiency"),
            wavelengths=np.asarray(light.wavelengths, dtype=float),
            spectrum=np.asarray(light.power_spectrum, dtype=float),
            stokes=np.asarray(light.stokes, dtype=float),
        )
    return infos


class _Spark(QWidget):
    def __init__(self, info: LinkInfo, parent=None) -> None:
        super().__init__(parent)
        self.info = info
        self.setFixedHeight(34)
        self.setMinimumWidth(190)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        p.setPen(QPen(QApplication.palette().mid().color(), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        x, y = self.info.wavelengths, self.info.spectrum
        if len(x) < 2 or float(np.max(y)) <= 0:
            p.setPen(QApplication.palette().placeholderText().color())
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "single line")
            return
        order = np.argsort(x)
        x, y = x[order], y[order]
        span = float(x[-1] - x[0]) or 1.0
        path = QPainterPath()
        for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
            pt = QPointF(
                r.left() + r.width() * (xi - x[0]) / span,
                r.bottom() - r.height() * float(yi) / float(np.max(y)),
            )
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        color = self.info.color or QApplication.palette().text().color()
        p.setPen(QPen(color, 1.6))
        p.drawPath(path)
        p.setPen(QApplication.palette().text().color())
        small = p.font()
        small.setPointSizeF(max(6.0, small.pointSizeF() - 2))
        p.setFont(small)
        p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"{x[0] * 1e9:.1f}")
        p.drawText(
            r, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"{x[-1] * 1e9:.1f} nm"
        )


class BeamCard(QFrame):
    """Hover card with the beam state of one link."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(3)
        self._body: list[QWidget] = []

    def show_info(self, info: LinkInfo, global_pos) -> None:
        for widget in self._body:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        self._body = []
        title = QLabel(f"<b>{info.link_id}</b> ({info.kind.replace('_', ' ')})")
        rows = info.lines()
        html = "<table>" + "".join(
            f"<tr><td style='color:gray'>{k}&nbsp;&nbsp;</td><td>{v}</td></tr>" for k, v in rows
        )
        html += "</table>" if rows else "</table>"
        body = QLabel(html if rows else "<i>no data</i>")
        widgets = [title, body]
        if info.kind != "cable" and len(info.spectrum):
            widgets.append(_Spark(info))
        for w in widgets:
            self._layout.addWidget(w)
        self._body = widgets
        self.adjustSize()
        self.move(global_pos.x() + 14, global_pos.y() + 14)
        self.show()


__all__ = ["BeamCard", "LinkInfo", "build_link_infos", "plain"]
