import math
import re
from typing import Any

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sequoia.gui.simulation import SimulationOutcome

_PREFIXES = {
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
}
_QUOTED = re.compile(r"'([^']+)'")
SEVERITY_COLOR = {"error": QColor("#d32f2f"), "warning": QColor("#c77700")}


def eng(value: Any, unit: str = "", digits: int = 4) -> str:
    """Engineering notation with an SI prefix: ``eng(1.5e-3, 'W') == '1.5 mW'``."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return ("-" if x < 0 else "") + "inf" + (f" {unit}" if unit else "")
    if x == 0:
        return f"0 {unit}".strip()
    exponent = int(math.floor(math.log10(abs(x)) / 3.0) * 3)
    exponent = max(-15, min(12, exponent))
    scaled = x / 10.0**exponent
    text = f"{scaled:.{digits}g}"
    return f"{text} {_PREFIXES[exponent]}{unit}".strip()


def plain(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}g}"


def percent(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.3g} %"
    except (TypeError, ValueError):
        return str(value)


def ids_for_issue(subject: str) -> tuple[str, ...]:
    """Instance/link/probe ids named by an issue subject.

    Handles ``'bs1'``, ``link 'l3'``, ``port 'det.out'`` (the instance part), ``probe 'pm'`` and
    ``instance 'x'``; subjects such as ``data`` or ``setup`` name nothing and give ``()``.
    """
    ids: list[str] = []
    for token in _QUOTED.findall(subject):
        ident = token.split(".")[0]
        if ident and ident not in ids:
            ids.append(ident)
    return tuple(ids)


def _NumItem(text: str, key: float | None = None, tooltip: str | None = None):
    item = QTableWidgetItem(text)
    if key is not None:
        item.setData(Qt.ItemDataRole.UserRole + 1, float(key))
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _sort_by_column(table: QTableWidget, column: int) -> None:
    order = Qt.SortOrder.AscendingOrder
    if getattr(table, "_sort_column", None) == column and getattr(table, "_sort_asc", True):
        order = Qt.SortOrder.DescendingOrder
    table._sort_column = column
    table._sort_asc = order == Qt.SortOrder.AscendingOrder
    rows = [
        [table.takeItem(r, c) for c in range(table.columnCount())] for r in range(table.rowCount())
    ]

    def key(row):
        item = row[column]
        value = item.data(Qt.ItemDataRole.UserRole + 1)
        return (0, value, "") if value is not None else (1, 0.0, item.text().lower())

    rows.sort(key=key, reverse=not table._sort_asc)
    for r, row in enumerate(rows):
        for c, item in enumerate(row):
            table.setItem(r, c, item)
    table.horizontalHeader().setSortIndicator(column, order)


def _severity_icon(severity: str) -> QIcon:
    size = 16
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = SEVERITY_COLOR[severity]
    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    if severity == "error":
        painter.drawEllipse(1, 1, size - 2, size - 2)
        painter.setPen(QColor("white"))
        painter.drawLine(5, 5, size - 6, size - 6)
        painter.drawLine(size - 6, 5, 5, size - 6)
    else:
        painter.drawPolygon(
            QPolygonF([QPointF(size / 2, 1), QPointF(size - 1, size - 2), QPointF(1, size - 2)])
        )
        painter.setPen(QColor("white"))
        painter.drawLine(size // 2, 6, size // 2, 10)
        painter.drawPoint(size // 2, 12)
    painter.end()
    return QIcon(pixmap)


def _make_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(22)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    table.setShowGrid(False)
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setSortIndicatorShown(True)
    header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    header.sectionClicked.connect(lambda column, t=table: _sort_by_column(t, column))
    return table


def _fill(table: QTableWidget, rows: list[list[_NumItem]]) -> None:
    table.setUpdatesEnabled(False)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, item in enumerate(row):
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(r, c, item)
    table.setUpdatesEnabled(True)
    for c in range(table.columnCount() - 1):
        table.resizeColumnToContents(c)


def _tree(headers: list[str]) -> QTreeWidget:
    tree = QTreeWidget()
    tree.setHeaderLabels(headers)
    tree.setAlternatingRowColors(True)
    tree.setUniformRowHeights(True)
    tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return tree


def _node(parent, *texts: str, bold: bool = False) -> QTreeWidgetItem:
    item = QTreeWidgetItem(parent, [str(t) for t in texts])
    if bold:
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
    return item


_ANALYSIS_FIELDS: dict[str, list[tuple[str, str, Any]]] = {
    "coincidences": [
        ("window_s", "window", lambda v: eng(v, "s")),
        ("center_s", "centre", lambda v: eng(v, "s")),
        ("peak_s", "peak", lambda v: eng(v, "s")),
        ("window_fraction", "window fraction", plain),
        ("true_hz", "true coincidences", lambda v: eng(v, "Hz")),
        ("accidental_hz", "accidentals", lambda v: eng(v, "Hz")),
        ("car", "CAR (true / accidental)", plain),
        ("car_total", "CAR ((true + acc) / acc)", plain),
        ("coincidence_rate_hz", "all-time coincidences", lambda v: eng(v, "Hz")),
        ("heralding_a_given_b", "heralding efficiency (first | second)", percent),
        ("heralding_b_given_a", "heralding efficiency (second | first)", percent),
    ],
    "heralded_g2": [
        ("window_s", "window", lambda v: eng(v, "s")),
        ("rate_ha_hz", "herald x a", lambda v: eng(v, "Hz")),
        ("rate_hb_hz", "herald x b", lambda v: eng(v, "Hz")),
        ("triples_hz", "triple coincidences", lambda v: eng(v, "Hz")),
        ("g2_heralded", "heralded g2(0)", plain),
        ("multimode_bunching", "multimode bunching", plain),
        ("cw_bunching", "CW bunching", plain),
        ("herald_saturation", "herald saturation factor", plain),
    ],
    "g2": [
        ("bin_width_s", "bin width", lambda v: eng(v, "s")),
        ("center_s", "centre", lambda v: eng(v, "s")),
        ("g2_peak", "g2 peak", plain),
        ("g2_at_center", "g2 at centre", plain),
        ("g2_per_pulse", "g2 per pulse", plain),
    ],
}


class ResultsDock(QWidget):
    issue_activated = Signal(str)

    SPINNER = "|/-\\"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._outcome: SimulationOutcome | None = None
        self._busy = False
        self._spin = 0
        self._failure: str | None = None
        self._spinner = QTimer(self)
        self._spinner.setInterval(120)
        self._spinner.timeout.connect(self._tick)

        self.status = QLabel("No results yet")
        self.status.setContentsMargins(6, 3, 6, 3)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tabs = QTabWidget()

        self.diagnostics = _make_table(["", "Code", "Subject", "Message"])
        self.diagnostics.itemSelectionChanged.connect(self._issue_selected)
        self.diagnostics.cellDoubleClicked.connect(self._issue_double_clicked)

        self.links = _make_table(
            ["Link", "Kind", "From → To", "P [mW]", "w_x [mm]", "w_y [mm]", "R [m]", "DOP"]
        )

        self.readouts = _tree(["Name", "Value"])

        self.channels = _make_table(
            ["Channel", "Detector", "Singles", "Pair singles", "Background", "Dead time", "Delay"]
        )
        self.analyses = _tree(["Analysis", "Value"])
        self.counts = QSplitter(Qt.Orientation.Vertical)
        self.counts.addWidget(self.channels)
        self.counts.addWidget(self.analyses)

        self.tabs.addTab(self.diagnostics, "Diagnostics")
        self.tabs.addTab(self.links, "Links")
        self.tabs.addTab(self.readouts, "Readouts && probes")
        self.tabs.addTab(self.counts, "Counts")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.status)
        layout.addWidget(self.tabs, 1)
        self.resize(720, 360)
        self._refresh_status()

    def connect_controller(self, controller) -> None:
        controller.busy_changed.connect(self.set_busy)
        controller.finished.connect(self.set_outcome)
        controller.failed.connect(self.set_failed)

    @property
    def outcome(self) -> SimulationOutcome | None:
        return self._outcome

    def clear(self) -> None:
        self._outcome = None
        self._failure = None
        _fill(self.diagnostics, [])
        _fill(self.links, [])
        _fill(self.channels, [])
        self.readouts.clear()
        self.analyses.clear()
        self.tabs.setTabText(0, "Diagnostics")
        self._refresh_status()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._busy:
            self._spinner.start()
        else:
            self._spinner.stop()
        self._refresh_status()

    def set_failed(self, message: str) -> None:
        self._failure = message
        self._refresh_status()

    def set_outcome(self, outcome: SimulationOutcome) -> None:
        self._outcome = outcome
        self._failure = None
        self._fill_diagnostics(outcome)
        self._fill_links(outcome)
        self._fill_readouts(outcome)
        self._fill_counts(outcome)
        self._refresh_status()

    def _tick(self) -> None:
        self._spin = (self._spin + 1) % len(self.SPINNER)
        self._refresh_status()

    def _refresh_status(self) -> None:
        parts = []
        if self._busy:
            parts.append(f"Running {self.SPINNER[self._spin]}")
        if self._failure is not None:
            parts.append(f"Failed: {self._failure}")
        outcome = self._outcome
        if outcome is not None:
            errors = sum(i.severity == "error" for i in outcome.issues)
            warnings = sum(i.severity == "warning" for i in outcome.issues)
            state = "Invalid setup" if outcome.classical is None else "Done"
            parts.append(
                f"{state} in {eng(outcome.elapsed, 's', 3)} | "
                f"{errors} error{'s' * (errors != 1)}, {warnings} warning{'s' * (warnings != 1)}"
            )
        elif self._failure is None and not self._busy:
            parts.append("No results yet")
        self.status.setText("  -  ".join(parts))

    def _fill_diagnostics(self, outcome: SimulationOutcome) -> None:
        order = {"error": 0, "warning": 1}
        issues = sorted(outcome.issues, key=lambda i: order.get(i.severity, 2))
        rows = []
        for issue in issues:
            icon_item = _NumItem("", order.get(issue.severity, 2), issue.severity)
            icon_item.setIcon(
                _severity_icon(issue.severity if issue.severity in SEVERITY_COLOR else "warning")
            )
            color = QBrush(SEVERITY_COLOR.get(issue.severity, QColor("gray")))
            code = _NumItem(issue.code)
            code.setForeground(color)
            subject = _NumItem(issue.subject)
            subject.setData(Qt.ItemDataRole.UserRole, issue.subject)
            message = _NumItem(issue.message.replace("\n", " "), tooltip=issue.message)
            rows.append([icon_item, code, subject, message])
        _fill(self.diagnostics, rows)
        self.diagnostics.setColumnWidth(0, 28)
        n = len(issues)
        self.tabs.setTabText(0, f"Diagnostics ({n})" if n else "Diagnostics")

    def _issue_subject(self, row: int) -> str | None:
        item = self.diagnostics.item(row, 2)
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _issue_selected(self) -> None:
        rows = {i.row() for i in self.diagnostics.selectedIndexes()}
        if len(rows) == 1:
            subject = self._issue_subject(rows.pop())
            if subject is not None:
                self.issue_activated.emit(subject)

    def _issue_double_clicked(self, row: int, _column: int) -> None:
        subject = self._issue_subject(row)
        if subject is not None:
            self.issue_activated.emit(subject)

    def _fill_links(self, outcome: SimulationOutcome) -> None:
        classical = outcome.classical
        rates = outcome.rates
        headers = ["Link", "Kind", "From → To", "P [mW]", "w_x [mm]", "w_y [mm]", "R [m]", "DOP"]
        has_coupling = classical is not None and any(
            "coupling_efficiency" in s.info for s in classical.links.values()
        )
        if has_coupling:
            headers.append("Coupling")
        if rates is not None:
            headers += ["Photons [1/s]", "Photon coupling"]
        self.links.setColumnCount(len(headers))
        self.links.setHorizontalHeaderLabels(headers)
        rows = []
        if classical is not None:
            for state in classical.links.values():
                light = state.end
                row = [
                    _NumItem(state.id),
                    _NumItem(state.kind),
                    _NumItem(f"{state.src} → {state.dst}"),
                ]
                if light is None:
                    row += [_NumItem("no light", -1.0)] + [_NumItem("") for _ in range(4)]
                else:
                    m = light.mode
                    row += [
                        _NumItem(plain(light.power * 1e3), light.power * 1e3),
                        _NumItem(plain(m.w_x * 1e3), m.w_x * 1e3),
                        _NumItem(plain(m.w_y * 1e3), m.w_y * 1e3),
                        _NumItem(plain(m.radius_x), m.radius_x),
                        _NumItem(f"{light.dop:.3f}", light.dop),
                    ]
                if has_coupling:
                    eta = state.info.get("coupling_efficiency")
                    row.append(_NumItem("" if eta is None else percent(eta), eta))
                if rates is not None:
                    entry = rates.links.get(state.id)
                    if entry is None:
                        row += [_NumItem(""), _NumItem("")]
                    else:
                        couplings = ", ".join(
                            f"{role} {percent(v)}"
                            for role, v in entry.get("coupling_efficiency", {}).items()
                        )
                        row += [
                            _NumItem(eng(entry["photons_hz"]), entry["photons_hz"]),
                            _NumItem(couplings),
                        ]
                rows.append(row)
        _fill(self.links, rows)

    def _fill_readouts(self, outcome: SimulationOutcome) -> None:
        tree = self.readouts
        tree.setUpdatesEnabled(False)
        tree.clear()
        classical = outcome.classical
        if classical is not None:
            if classical.readouts:
                group = _node(tree, "Readouts", bold=True)
                for inst_id, values in classical.readouts.items():
                    inst = _node(group, inst_id)
                    for key, value in values.items():
                        _node(inst, key, _format_value(key, value))
            if classical.probes:
                group = _node(tree, "Probes", bold=True)
                for probe_id, values in classical.probes.items():
                    probe = _node(group, probe_id)
                    for key, value in values.items():
                        if (
                            isinstance(value, list | tuple)
                            or hasattr(value, "__len__")
                            and not isinstance(value, str)
                        ):
                            _node(probe, key, f"[{len(value)} values]")
                        else:
                            _node(probe, key, _format_value(key, value))
        tree.expandAll()
        tree.resizeColumnToContents(0)
        tree.setUpdatesEnabled(True)

    def _fill_counts(self, outcome: SimulationOutcome) -> None:
        rates = outcome.rates
        rows = []
        if rates is not None:
            by_det = {c["detector_instance"]: n for n, c in rates.channels.items()}
            for det_id, d in rates.detectors.items():
                name = by_det.get(det_id, "")
                delay = rates.channels[name]["delay_s"] if name in rates.channels else None
                rows.append(
                    [
                        _NumItem(name or "-"),
                        _NumItem(det_id),
                        _NumItem(eng(d["singles_hz"], "Hz"), d["singles_hz"]),
                        _NumItem(eng(d["pair_singles_hz"], "Hz"), d["pair_singles_hz"]),
                        _NumItem(eng(d["background_hz"], "Hz"), d["background_hz"]),
                        _NumItem(eng(d["dead_time_s"], "s"), d["dead_time_s"]),
                        _NumItem("" if delay is None else eng(delay, "s"), delay),
                    ]
                )
        _fill(self.channels, rows)

        tree = self.analyses
        tree.setUpdatesEnabled(False)
        tree.clear()
        if rates is not None:
            if rates.sources:
                group = _node(tree, "Pair sources", bold=True)
                for inst_id, s in rates.sources.items():
                    kind = f"pulsed, {eng(s['repetition_rate_hz'], 'Hz')}" if s["pulsed"] else "CW"
                    item = _node(group, inst_id, f"{eng(s['rate_hz'], 'pairs/s')} ({kind})")
                    _node(item, "bandwidth", eng(s["dnu_hz"], "Hz"))
                    for role in ("signal", "idler"):
                        wl = s.get(f"{role}_wavelength_m")
                        if wl:
                            _node(item, f"{role} wavelength", eng(wl, "m"))
            if rates.pairs:
                group = _node(tree, "Coincidence rates (all times)", bold=True)
                for (a, b), p in rates.pairs.items():
                    _node(group, f"{a} × {b}", eng(p["coincidence_rate_hz"], "Hz"))
            if rates.analyses:
                group = _node(tree, "Analyses", bold=True)
                for name, a in rates.analyses.items():
                    self._analysis_node(group, name, a)
        tree.expandAll()
        tree.resizeColumnToContents(0)
        tree.setUpdatesEnabled(True)

    def _analysis_node(self, parent, name: str, a: dict[str, Any]) -> None:
        kind = a.get("type", "analysis")
        chans = a.get("channels")
        if chans is None and "herald" in a:
            chans = [a["herald"], a["a"], a["b"]]
        item = _node(parent, f"{name} [{kind}]", " × ".join(chans) if chans else "")
        listed = _ANALYSIS_FIELDS.get(kind)
        shown = set()
        if listed is not None:
            for key, label, fmt in listed:
                value = a.get(key)
                if value is not None:
                    shown.add(key)
                    _node(item, label, fmt(value))
        for key, value in a.items():
            if key in shown or key in ("type", "channels", "by_role") or key.startswith("_"):
                continue
            if isinstance(value, bool | int | float):
                _node(item, key, plain(value))
            elif isinstance(value, str):
                _node(item, key, value)
        for role, d in (a.get("by_role") or {}).items():
            if isinstance(d, dict) and d.get("trace_a", 0) > 0:
                _node(
                    item,
                    f"{role} Schmidt number",
                    f"{plain(d['schmidt_a'])} / {plain(d['schmidt_b'])}",
                )


def _format_value(key: str, value: Any) -> str:
    if isinstance(value, bool | str):
        return str(value)
    if key.endswith("_W"):
        return eng(value, "W")
    if key.endswith("_Hz") or key.endswith("_hz"):
        return eng(value, "Hz")
    if key.endswith("_m"):
        return eng(value, "m")
    if key.endswith("_s"):
        return eng(value, "s")
    return plain(value, 6)
