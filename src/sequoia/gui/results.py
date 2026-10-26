import math
import re
from typing import Any

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QListWidget,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sequoia.gui.recipes import ALL_TOOLS, RECIPES, TASKS, recipe_text
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


def car_text(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "undefined: no accidental estimate"
    return plain(x)


UNDEFINED_REASONS = {
    "CAR": "undefined: no accidental estimate",
    "heralding efficiency": "undefined: no singles on the heralding detector",
    "heralded g2(0)": "undefined: no herald x a or herald x b coincidences",
    "unheralded g2(0)": "undefined: no accidental floor to normalise by",
    "two-photon interference visibility": "undefined: no distinguishable-photon rate",
    "correlation E": "undefined: no coincidences at these settings",
}


def metric_text(metric: str, value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return UNDEFINED_REASONS.get(metric, "undefined: zero denominator")
    if "efficiency" in metric:
        return percent(x)
    return plain(x)


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


def _header_tips(table: QTableWidget, tips: dict[int, str]) -> None:
    for column, tip in tips.items():
        item = table.horizontalHeaderItem(column)
        if item is not None:
            item.setToolTip(tip)


LINK_HEADERS = ["Link", "Kind", "From → To", "P end [mW]", "w_x [mm]", "w_y [mm]", "R_x [m]", "DOP"]
LINK_TIPS = {
    3: "classical power at the end of the link",
    4: "1/e^2 beam radius at the end of the link",
    6: "wavefront radius of curvature (x) at the end of the link",
    7: "degree of polarization of the classical light",
}
NO_MODE = "no Gaussian beam description for this light (e.g. after a multimode element)"
CHANNEL_HEADERS = [
    ("Channel", ""),
    ("Detector", ""),
    (
        "Recorded singles [Hz]",
        "clicks per second after dead time (non-paralyzable detector, then tagger channel)",
    ),
    ("of which pairs [Hz]", "recorded singles from pair photons, dead-time corrected"),
    (
        "of which background [Hz]",
        "recorded dark counts, classical leakage, Fock-source flux and afterpulses",
    ),
    (
        "Potential clicks [Hz]",
        "detected photons plus dark counts per second before dead time (efficiency included)",
    ),
    ("Dead-time survival", "recorded singles / potential clicks"),
    ("Dead time", "longer of the detector and tagger dead times"),
    ("Delay", "cable delay to the tagger channel"),
]


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
        ("car", "CAR (true / accidental)", lambda v: car_text(v)),
        ("car_total", "CAR ((true + acc) / acc)", plain),
        ("coincidence_rate_hz", "all-time coincidences", lambda v: eng(v, "Hz")),
        ("heralding_a_given_b", "heralding efficiency (first | second)", percent),
        ("heralding_b_given_a", "heralding efficiency (second | first)", percent),
        ("two_photon_visibility", "two-photon interference visibility", plain),
        ("distinguishable_rate_hz", "rate if photons were distinguishable", lambda v: eng(v, "Hz")),
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


TAB_HELP = {
    "Diagnostics": "Problems and warnings found in your setup. Click a row to select the part "
    "or link that causes it.",
    "Links": "The light at the end of every link: power, beam radius (w), wavefront radius, "
    "polarization purity (DOP) and, for photon-pair setups, the pair photons arriving per second "
    "and how well they couple into fibers.",
    "Readouts && probes": "What each detector and probe reads: photocurrent, incident power, "
    "beam size and so on.",
    "Rate counts": "Key metrics (CAR, heralding efficiency, heralded and unheralded g2(0), "
    "two-photon interference visibility, polarization correlation E), then per detector the "
    "potential click rate before dead time and the recorded singles after it, and the "
    "coincidence analyses in their stated full-width windows.",
    "Caustic": "Beam radius along the optical path: where the beam is focused and how large it "
    "is at each part.",
    "Polarization": "Polarization state on the chosen link: the ellipse, and the point on the "
    "Poincare sphere (drag the sphere to rotate it, double-click to reset).",
    "Spectra": "How the light on the chosen link is spread over wavelength.",
    "Histograms": "Arrival-time differences between two detectors. A peak at zero delay means "
    "photons arrive together; a flat floor is accidental coincidences.",
    "Budget": "Where light is lost along a path, and how much each real part costs compared "
    "with an ideal one.",
    "Sweeps": "Change one setting over a range and plot a result, or let the optimizer search "
    "for the best value.",
    "Monte Carlo": "Simulates the stream of single detector clicks a time tagger would record "
    "in the lab and checks it against the analytic numbers.",
    "Photon counts": "Photon-number statistics of the drawn circuit: Import from setup lists its "
    "parts and photon inputs (Fock sources, SPDC pairs); Run gives output-state probabilities, "
    "counts per second, detector click patterns, marginals and heralded distributions through the "
    "setup's detectors. Extract to circuit editor turns the circuit into editable elements. "
    "See docs/FOCK.md.",
    "CHSH": "Bell test forecast: choose the analyser parameter of each arm, two settings per "
    "arm and the four coincidence analyses; the rate engine gives counts per setting pair, "
    "E and S (raw and accidental-subtracted) with the anticipated counting spread. A forecast, "
    "not a measured violation.",
    "Compare": "Measured data: load a CSV/XLSX file, map its columns and units, fit a fringe, "
    "peak or HOM dip (or compare with a sweep of this setup) and export data, model, residuals "
    "and a JSON record.",
    "Gaussian": "Multimode Gaussian photon statistics of the pair sources at higher pump "
    "power, where several pairs can be created at once: CAR falls and heralded g2 rises with "
    "pump power. Per window; no dead time or jitter; Fock-state sources are refused.",
}


MODEL_SUMMARY = {
    "Diagnostics": "setup checks and the warnings of every engine run; Engine and Run say which "
    "model and run produced each row",
    "Links": "paraxial Gaussian beams (ABCD with M^2), Jones polarization, mean power at the link "
    "end; pair photons/s before detector efficiency",
    "Readouts && probes": "ideal non-perturbing readouts of the classical light; a probe reads "
    "the end of its link, or its position along a free-space link",
    "Rate counts": "first-order in the pair rate, non-paralyzable dead time (detector then "
    "tagger), Gaussian jitter (plus tail when set); multi-pair only via the uncorrelated "
    "accidental formula",
    "Caustic": "paraxial Gaussian envelope (1/e^2 radius, M^2 embedded Gaussian); apertures are "
    "drawn but their diffraction is not modelled",
    "Polarization": "single link: classical field or the reduced one-photon state; joint view: the "
    "two-photon state of a pair, normalised on one photon per link",
    "Spectra": "sampled spectra of the classical light, or photon rate per frequency sample of "
    "the signal/idler; no spectral phase",
    "Histograms": "rate-engine correlation function convolved with the detector and tagger "
    "jitter; flat accidental floor S_a S_b x bin width",
    "Budget": "transmission per element along one path, classical power or pair photons; the "
    "ideal-parts budget reruns the setup with each part ideal",
    "Sweeps": "reruns the selected engine at every point; Photon-count observables use the "
    "Photon counts tab's settings",
    "Monte Carlo": "simulated time tags with jitter, dead time, dark counts and afterpulses; "
    "Poisson pairs, or pulsed multimode thermal pairs for a Schmidt number K",
    "Gaussian": "multimode Gaussian states per window, multi-pair terms included; no dead time "
    "or jitter",
    "Photon counts": "independent windows, no dead time: rates are potential clicks before dead "
    "time; truncated inputs report their omitted weight",
    "CHSH": "rate engine at the four setting pairs; fair sampling, a forecast",
    "Compare": "least-squares fits with Poisson or given weights; identifiability is checked",
}
TOOL_TITLES = ("Diagnostics", "Links", "Readouts && probes", "Rate counts")
SUBJECT_COLUMN = 4
MAX_HELP_ISSUES = 3


class ResultsDock(QWidget):
    issue_activated = Signal(str)
    task_changed = Signal(str)

    SPINNER = "|/-\\"
    TAB_TITLES = TOOL_TITLES

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._outcome: SimulationOutcome | None = None
        self._busy = False
        self._spin = 0
        self._failure: str | None = None
        self.tab_errors: dict[str, str] = {}
        self._spinner = QTimer(self)
        self._spinner.setInterval(120)
        self._spinner.timeout.connect(self._tick)

        self.status = QLabel("No results yet")
        self.status.setContentsMargins(6, 3, 6, 3)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tabs = QTabWidget()
        self.task_tabs = QTabWidget()
        self.task = ALL_TOOLS
        self.tools: dict[str, QWidget] = {}
        self._labels: dict[str, str] = {}
        self._tips: dict[str, str] = {}

        self.diagnostics = _make_table(["", "Engine", "Run", "Code", "Subject", "Message"])
        _header_tips(
            self.diagnostics,
            {
                1: "the engine or tab that produced the row; 'setup' for checks of the drawn setup",
                2: "the run that produced the row; 'current' for the latest base simulation",
            },
        )
        self.diagnostics.itemSelectionChanged.connect(self._issue_selected)
        self.diagnostics.cellDoubleClicked.connect(self._issue_double_clicked)

        self.links = _make_table(LINK_HEADERS)

        self.readouts = _tree(["Name", "Value"])

        self.channels = _make_table([h for h, _ in CHANNEL_HEADERS])
        _header_tips(self.channels, {i: t for i, (_, t) in enumerate(CHANNEL_HEADERS) if t})
        self.analyses = _tree(["Analysis", "Value"])
        self.metrics = _make_table(["Metric", "Value", "Analysis", "Definition (denominator)"])
        self.counts = QSplitter(Qt.Orientation.Vertical)
        self.counts.addWidget(self.metrics)
        self.counts.addWidget(self.channels)
        self.counts.addWidget(self.analyses)

        for title, widget in zip(
            TOOL_TITLES, (self.diagnostics, self.links, self.readouts, self.counts), strict=True
        ):
            self.tools[title] = widget
            self.tabs.addTab(widget, title)

        from sequoia.gui.analysis_tabs import (
            BudgetTab,
            CausticTab,
            HistogramTab,
            PolarizationTab,
            SpectraTab,
        )
        from sequoia.gui.chsh_panel import ChshTab
        from sequoia.gui.compare_panel import CompareTab
        from sequoia.gui.fock_panel import FockTab
        from sequoia.gui.gaussian_panel import GaussianTab
        from sequoia.gui.montecarlo_panel import MonteCarloTab
        from sequoia.gui.sweep_panel import SweepTab

        self.provider = lambda: (None, None)
        self.caustic = CausticTab()
        self.polarization = PolarizationTab()
        self.spectra = SpectraTab()
        self.histograms = HistogramTab()
        self.budget = BudgetTab(lambda: self.provider())
        self.sweeps = SweepTab(lambda: self.provider())
        self.montecarlo = MonteCarloTab(lambda: self.provider())
        self.gaussian = GaussianTab(lambda: self.provider())
        self.fock = FockTab(lambda: self.provider())
        self.chsh = ChshTab(lambda: self.provider())
        self.compare = CompareTab(lambda: self.provider())
        self.sweeps.fock_settings = self.fock.sweep_settings
        for w in (self.fock.duration, self.fock.n_max_pairs):
            w.valueChanged.connect(self.sweeps.refresh_stale)
        self.fock.detector_mode.currentIndexChanged.connect(self.sweeps.refresh_stale)
        self.plot_tabs = {
            "Caustic": self.caustic,
            "Polarization": self.polarization,
            "Spectra": self.spectra,
            "Histograms": self.histograms,
            "Budget": self.budget,
            "Sweeps": self.sweeps,
            "Monte Carlo": self.montecarlo,
            "Gaussian": self.gaussian,
            "Photon counts": self.fock,
            "CHSH": self.chsh,
            "Compare": self.compare,
        }
        self.engine_issues: dict[str, list] = {}
        for title, widget in self.plot_tabs.items():
            self.tools[title] = widget
            self.tabs.addTab(widget, title)
            widget.on_issues = lambda issues, name=title: self.set_engine_issues(name, issues)
        self._labels = {title: title for title in self.tools}

        self.task_list = QListWidget()
        self.task_list.addItems(list(TASKS))
        self.task_list.setToolTip(
            "What do you want to do? Each task shows the tools it needs; "
            f"'{ALL_TOOLS}' lists every engine as tabs"
        )
        self.task_list.setMinimumWidth(150)
        self.task_list.currentTextChanged.connect(
            lambda name: self.show_task(name) if name and name != self.task else None
        )
        self.recipe = QLabel("")
        self.recipe.setWordWrap(True)
        self.recipe.setContentsMargins(8, 4, 8, 4)
        self.recipe.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        recipe_page = QWidget()
        recipe_layout = QVBoxLayout(recipe_page)
        recipe_layout.setContentsMargins(0, 0, 0, 0)
        recipe_layout.setSpacing(0)
        recipe_layout.addWidget(self.recipe)
        recipe_layout.addWidget(self.task_tabs, 1)
        self.pages = QStackedWidget()
        self.pages.addWidget(recipe_page)
        self.pages.addWidget(self.tabs)
        self.pages.setCurrentWidget(self.tabs)
        self.body = QSplitter(Qt.Orientation.Horizontal)
        self.body.addWidget(self.task_list)
        self.body.addWidget(self.pages)
        self.body.setStretchFactor(1, 1)
        self.body.setCollapsible(1, False)
        self.body.setSizes([190, 710])

        self.help = QLabel("")
        self.help.setWordWrap(True)
        self.help.setContentsMargins(8, 4, 8, 4)
        self.help.setStyleSheet(
            "color: palette(placeholder-text); background: palette(alternate-base);"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.status)
        layout.addWidget(self.body, 1)
        layout.addWidget(self.help)
        self.tabs.currentChanged.connect(self._update_help)
        self.task_tabs.currentChanged.connect(self._update_help)
        self.task_list.setCurrentRow(len(TASKS) - 1)
        self._update_help()
        self.resize(720, 360)
        self._refresh_status()

    def sizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(900, 260)

    def minimumSizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(200, 110)

    def container(self) -> QTabWidget:
        return self.tabs if self.task == ALL_TOOLS else self.task_tabs

    def current_tool(self) -> QWidget | None:
        return self.container().currentWidget()

    def tool_title(self, widget: QWidget | None) -> str | None:
        return next((t for t, w in self.tools.items() if w is widget), None)

    def task_tools(self, name: str | None = None) -> list[str]:
        name = self.task if name is None else name
        recipe = RECIPES.get(name)
        return list(recipe.tools) if recipe is not None else list(self.tools)

    def show_task(self, name: str) -> None:
        recipe = RECIPES.get(name)
        name = recipe.name if recipe is not None else ALL_TOOLS
        current = self.current_tool()
        chosen = recipe.tools if recipe is not None else ()
        for tabs in (self.tabs, self.task_tabs):
            tabs.blockSignals(True)
        self.task_tabs.clear()
        self.tabs.clear()
        for title in self.tools:
            if title not in chosen:
                self._add_tool(self.tabs, title)
        for title in chosen:
            self._add_tool(self.task_tabs, title)
        self.task = name
        target = self.container()
        if current is not None and target.indexOf(current) >= 0:
            target.setCurrentWidget(current)
        else:
            target.setCurrentIndex(0)
        for tabs in (self.tabs, self.task_tabs):
            tabs.blockSignals(False)
        self.recipe.setText(recipe_text(name))
        self.pages.setCurrentIndex(0 if recipe is not None else 1)
        if recipe is not None and recipe.polarization_view is not None:
            self.polarization.views.setCurrentIndex(recipe.polarization_view)
        self.task_list.blockSignals(True)
        self.task_list.setCurrentRow(TASKS.index(name))
        self.task_list.blockSignals(False)
        self._update_help()
        self.task_changed.emit(name)

    def show_tool(self, widget: QWidget) -> None:
        if self.container().indexOf(widget) < 0:
            title = self.tool_title(widget)
            task = next((r.name for r in RECIPES.values() if title in r.tools), ALL_TOOLS)
            self.show_task(task)
        self.container().setCurrentWidget(widget)

    def _add_tool(self, tabs: QTabWidget, title: str) -> None:
        index = tabs.addTab(self.tools[title], self._labels[title])
        tabs.setTabToolTip(index, self._tips.get(title, ""))

    def _set_label(self, title: str, text: str, tip: str = "") -> None:
        self._labels[title] = text
        self._tips[title] = tip
        widget = self.tools[title]
        for tabs in (self.tabs, self.task_tabs):
            index = tabs.indexOf(widget)
            if index >= 0:
                tabs.setTabText(index, text)
                tabs.setTabToolTip(index, tip)

    def _update_help(self, *_args) -> None:
        title = self.tool_title(self.current_tool()) or ""
        parts = [TAB_HELP.get(title, "")]
        if title in MODEL_SUMMARY:
            parts.append(f"Model: {MODEL_SUMMARY[title]}.")
        found = [i for i in self.engine_issues.get(title, []) if i.severity == "warning"]
        if found:
            shown = "; ".join(i.message.replace("\n", " ") for i in found[:MAX_HELP_ISSUES])
            more = len(found) - MAX_HELP_ISSUES
            parts.append(
                f"Approximations of the last run: {shown}"
                + (f" (+{more} more in Diagnostics)" if more > 0 else "")
            )
        text = "\n".join(p for p in parts if p)
        self.help.setText(text)
        self.help.setVisible(bool(text))

    def connect_controller(self, controller) -> None:
        controller.busy_changed.connect(self.set_busy)
        controller.finished.connect(self.set_outcome)
        controller.failed.connect(self.set_failed)
        for tab in (
            self.sweeps,
            self.montecarlo,
            self.gaussian,
            self.fock,
            self.chsh,
            self.compare,
        ):
            tab.revision_source = controller.document_key
            controller.document_edited.connect(tab.refresh_stale)

    @property
    def outcome(self) -> SimulationOutcome | None:
        return self._outcome

    def set_engine_issues(self, name: str, issues) -> None:
        self.engine_issues[name] = list(issues)
        if self._outcome is not None:
            self._fill_diagnostics(self._outcome)
        self._update_help()

    def clear(self) -> None:
        self._outcome = None
        self._failure = None
        self.engine_issues.clear()
        _fill(self.diagnostics, [])
        _fill(self.links, [])
        _fill(self.channels, [])
        _fill(self.metrics, [])
        self.readouts.clear()
        self.analyses.clear()
        self.tab_errors.clear()
        for widget in self.plot_tabs.values():
            reset = getattr(widget, "reset", None)
            if reset is not None:
                reset()
            widget.set_outcome(None)
        for title in TOOL_TITLES:
            self._set_label(title, title)
        self._update_help()
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
        self.tab_errors.clear()
        for title, name, fill in (
            (TOOL_TITLES[0], "Diagnostics", self._fill_diagnostics),
            (TOOL_TITLES[1], "Links", self._fill_links),
            (TOOL_TITLES[2], "Readouts", self._fill_readouts),
            (TOOL_TITLES[3], "Rate counts", self._fill_counts),
        ):
            try:
                fill(outcome)
            except Exception as exc:
                self.tab_errors[name] = f"{type(exc).__name__}: {exc}"
                self._set_label(title, f"{title} (!)", f"{name} tab could not be filled: {exc}")
            else:
                if title != TOOL_TITLES[0]:
                    self._set_label(title, title)
        for name, widget in self.plot_tabs.items():
            try:
                widget.set_outcome(outcome if outcome.classical is not None else None)
            except Exception as exc:
                self.tab_errors[name] = f"{type(exc).__name__}: {exc}"
        self._refresh_status()

    def shutdown(self) -> None:
        """Cancel and join the background jobs of the analysis tabs (before widgets are freed)."""
        for widget in self.plot_tabs.values():
            runner = getattr(widget, "runner", None)
            if runner is not None:
                runner.shutdown()
            dialog = getattr(widget, "optimizer", None)
            if dialog is not None:
                dialog.runner.shutdown()

    def select_link(self, link_id: str) -> None:
        for widget in (self.caustic, self.polarization, self.spectra, self.budget):
            widget.select_link(link_id)

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
        for name, message in self.tab_errors.items():
            parts.append(f"{name} tab failed: {message}")
        self.status.setText("  -  ".join(parts))

    def _fill_diagnostics(self, outcome: SimulationOutcome) -> None:
        order = {"error": 0, "warning": 1}
        tagged = [(i, i.engine or "setup", i.run_id or "current") for i in outcome.issues] + [
            (i, i.engine or name, i.run_id or "last run")
            for name, found in self.engine_issues.items()
            for i in found
        ]
        tagged.sort(
            key=lambda t: (order.get(t[0].severity, 2), t[1] != "setup", t[1].lower(), t[2])
        )
        rows = []
        for issue, engine, run in tagged:
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
            rows.append([icon_item, _NumItem(engine), _NumItem(run), code, subject, message])
        _fill(self.diagnostics, rows)
        self.diagnostics.setColumnWidth(0, 28)
        n = len(tagged)
        self._set_label(TOOL_TITLES[0], f"Diagnostics ({n})" if n else "Diagnostics")

    def _issue_subject(self, row: int) -> str | None:
        item = self.diagnostics.item(row, SUBJECT_COLUMN)
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
        headers = list(LINK_HEADERS)
        tips = dict(LINK_TIPS)
        has_coupling = classical is not None and any(
            "coupling_efficiency" in s.info for s in classical.links.values()
        )
        if has_coupling:
            tips[len(headers)] = "classical mode overlap with the fiber mode"
            headers.append("Fiber coupling")
        if rates is not None:
            tips[len(headers)] = (
                "pair photons per second arriving at the link end, before detection"
            )
            tips[len(headers) + 1] = "fiber coupling of the pair photons, per role"
            headers += ["Pair photons [1/s]", "Pair-photon coupling"]
        self.links.setColumnCount(len(headers))
        self.links.setHorizontalHeaderLabels(headers)
        _header_tips(self.links, tips)
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
                    row += [_NumItem("no light", -1.0, "no light reaches the end of this link")]
                    row += [_NumItem("") for _ in range(4)]
                else:
                    m = light.mode
                    row.append(_NumItem(plain(light.power * 1e3), light.power * 1e3))
                    if m is None:
                        row += [_NumItem("n/a", tooltip=NO_MODE) for _ in range(3)]
                    else:
                        row += [
                            _NumItem(plain(m.w_x * 1e3), m.w_x * 1e3),
                            _NumItem(plain(m.w_y * 1e3), m.w_y * 1e3),
                            _NumItem(plain(m.radius_x), m.radius_x),
                        ]
                    row.append(_NumItem(f"{light.dop:.3f}", light.dop))
                if has_coupling:
                    eta = state.info.get("coupling_efficiency")
                    row.append(
                        _NumItem("", tooltip="not a fiber coupling")
                        if eta is None
                        else _NumItem(percent(eta), eta)
                    )
                if rates is not None:
                    entry = rates.links.get(state.id)
                    if entry is None:
                        row += [_NumItem("0", 0.0, "no pair photons on this link"), _NumItem("")]
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
                incident = d.get("incident_hz")
                survival = d.get("survival")
                rows.append(
                    [
                        _NumItem(name or "-", tooltip=None if name else "not on a tagger channel"),
                        _NumItem(det_id),
                        _NumItem(eng(d["singles_hz"], "Hz"), d["singles_hz"]),
                        _NumItem(eng(d["pair_singles_hz"], "Hz"), d["pair_singles_hz"]),
                        _NumItem(eng(d["background_hz"], "Hz"), d["background_hz"]),
                        _NumItem("" if incident is None else eng(incident, "Hz"), incident),
                        _NumItem("" if survival is None else percent(survival), survival),
                        _NumItem(eng(d["dead_time_s"], "s"), d["dead_time_s"]),
                        _NumItem("not on a tagger channel")
                        if delay is None
                        else _NumItem(eng(delay, "s"), delay),
                    ]
                )
        _fill(self.channels, rows)
        from sequoia.analysis.metrics import key_metrics

        metric_rows = []
        for m in key_metrics(rates.analyses) if rates is not None else []:
            value = m["value"]
            shown = metric_text(m["metric"], value)
            metric_rows.append(
                [
                    _NumItem(m["metric"]),
                    _NumItem(shown, value),
                    _NumItem(m["analysis"]),
                    _NumItem(m["definition"], tooltip=m["definition"]),
                ]
            )
        _fill(self.metrics, metric_rows)

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
