import math

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sequoia.analysis.chsh import format_chsh, run_chsh
from sequoia.analysis.metrics import correlation_from_rates
from sequoia.analysis.sweep import candidate_targets
from sequoia.engines.cache import RunCache
from sequoia.gui.analysis_tabs import save_text
from sequoia.gui.jobs import JobRunner
from sequoia.gui.results import _fill, _make_table, _NumItem, plain
from sequoia.gui.runguard import RunGuard

OUTCOME_LABELS = ("a+ b+", "a+ b-", "a- b+", "a- b-")
DEFAULT_OUTCOMES = ("tt", "tr", "rt", "rr")
DEFAULT_SETTINGS = ((0.0, 22.5), (11.25, 33.75))
PAIR_LABELS = ("a b", "a b'", "a' b", "a' b'")


def _angle_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-360.0, 360.0)
    spin.setDecimals(3)
    spin.setSuffix(" deg")
    spin.setValue(value)
    return spin


class ChshTab(RunGuard, QWidget):
    """CHSH forecast from four analyser-setting pairs through the rate engine."""

    def __init__(self, provider, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard_init()
        self.provider = provider
        self.runner = JobRunner(self)
        self.result = None
        self.on_issues = None
        self.note = ""
        self._structure = None
        self.target_a = QComboBox()
        self.target_b = QComboBox()
        for combo in (self.target_a, self.target_b):
            combo.setMinimumWidth(200)
        self.settings = [
            [_angle_spin(v) for v in DEFAULT_SETTINGS[0]],
            [_angle_spin(v) for v in DEFAULT_SETTINGS[1]],
        ]
        self.outcomes = [QComboBox() for _ in OUTCOME_LABELS]
        self.quantity = QComboBox()
        self.quantity.addItems(["raw coincidences", "accidentals subtracted"])
        self.time = QDoubleSpinBox()
        self.time.setRange(1e-6, 1e7)
        self.time.setDecimals(3)
        self.time.setValue(10.0)
        self.time.setSuffix(" s per setting pair")
        self.sidebands = QSpinBox()
        self.sidebands.setRange(0, 100)
        self.sidebands.setValue(2)
        self.sidebands.setPrefix("sideband windows ")
        self.sidebands.setToolTip(
            "Sideband windows averaged for the accidental estimate (0: taken as exact)"
        )
        self.run_button = QPushButton("Forecast CHSH")
        self.export_button = QPushButton("Export plan...")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.table = _make_table(
            [
                "Setting pair",
                "Arm a parameter",
                "Arm b parameter",
                *[f"Counts {o} (raw)" for o in OUTCOME_LABELS],
                "E raw",
                "E accidental-subtracted",
            ]
        )

        grid = QGridLayout()
        grid.setContentsMargins(6, 4, 6, 4)
        grid.addWidget(QLabel("Arm a analyser parameter"), 0, 0)
        grid.addWidget(self.target_a, 0, 1)
        grid.addWidget(QLabel("a"), 0, 2)
        grid.addWidget(self.settings[0][0], 0, 3)
        grid.addWidget(QLabel("a'"), 0, 4)
        grid.addWidget(self.settings[0][1], 0, 5)
        grid.addWidget(QLabel("Arm b analyser parameter"), 1, 0)
        grid.addWidget(self.target_b, 1, 1)
        grid.addWidget(QLabel("b"), 1, 2)
        grid.addWidget(self.settings[1][0], 1, 3)
        grid.addWidget(QLabel("b'"), 1, 4)
        grid.addWidget(self.settings[1][1], 1, 5)
        mapping = QHBoxLayout()
        mapping.addWidget(QLabel("Coincidence analyses"))
        for label, combo in zip(OUTCOME_LABELS, self.outcomes, strict=True):
            mapping.addWidget(QLabel(label))
            mapping.addWidget(combo)
        mapping.addStretch(1)
        actions = QHBoxLayout()
        for w in (self.quantity, self.time, self.sidebands, self.run_button, self.export_button):
            actions.addWidget(w)
        actions.addWidget(self.status, 1)
        explain = QLabel(
            "Settings are the values of the chosen analyser parameters, not analysis angles: a "
            "half-wave plate at h before a PBS analyses linear polarization at 2h. '+' is the "
            "transmitted output of an arm, '-' the orthogonal one. The result is a forecast "
            "from the model with coincidences post-selected on two detections (fair sampling)."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color: palette(placeholder-text);")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(grid)
        layout.addLayout(mapping)
        layout.addLayout(actions)
        layout.addWidget(explain)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)

        self.run_button.clicked.connect(self.run)
        self.export_button.clicked.connect(self.export_plan)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled"))
        for combo in (self.target_a, self.target_b, self.quantity, *self.outcomes):
            combo.currentIndexChanged.connect(self.refresh_stale)
        for spin in (*self.settings[0], *self.settings[1], self.time, self.sidebands):
            spin.valueChanged.connect(self.refresh_stale)

    def config_key(self):
        return (
            self.target_a.currentText(),
            self.target_b.currentText(),
            tuple(s.value() for row in self.settings for s in row),
            tuple(c.currentText() for c in self.outcomes),
            self.quantity.currentIndex(),
            self.time.value(),
            self.sidebands.value(),
        )

    def set_outcome(self, outcome) -> None:
        setup, library = self.provider()
        if setup is not None:
            key = (
                tuple((i.id, i.part) for i in setup.instances),
                tuple(a.get("name") for a in setup.analyses),
            )
            if key != self._structure:
                self._structure = key
                self._load_choices(setup, library)
        if self.track_revision(outcome) and self.result is not None:
            self.status.setText(self.stale_text(self.note))

    def _load_choices(self, setup, library) -> None:
        angles = [c.target for c in candidate_targets(setup, library) if c.unit == "deg"]
        names = [a.get("name", "") for a in setup.analyses if a.get("type") == "coincidences"]
        for combo, preferred in ((self.target_a, "hwp_a"), (self.target_b, "hwp_b")):
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(angles)
            pick = previous if previous in angles else f"settings:{preferred}.angle"
            if pick in angles:
                combo.setCurrentText(pick)
            combo.blockSignals(False)
        for combo, preferred in zip(self.outcomes, DEFAULT_OUTCOMES, strict=True):
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            pick = previous if previous in names else preferred
            if pick in names:
                combo.setCurrentText(pick)
            combo.blockSignals(False)

    def reset(self) -> None:
        self.invalidate()
        self.result = None
        self.note = ""
        self._structure = None
        _fill(self.table, [])
        self.summary.setText("")
        self.status.setText("")

    def run(self) -> None:
        setup, library = self.provider()
        if setup is None or self.runner.busy:
            return
        self.set_outcome(None)
        a, b = self.target_a.currentText(), self.target_b.currentText()
        names = [c.currentText() for c in self.outcomes]
        if not a or not b or a == b:
            self.status.setText("choose two different analyser parameters, one per arm")
            return
        if not all(names) or len(set(names)) != 4:
            self.status.setText("map four different coincidence analyses to the four outcomes")
            return
        sa = [s.value() for s in self.settings[0]]
        sb = [s.value() for s in self.settings[1]]
        quantity = "total_hz" if self.quantity.currentIndex() == 0 else "true_hz"
        time, sidebands = self.time.value(), self.sidebands.value()
        self.begin_run()

        def job(should_cancel, progress):
            progress(0, 0)
            return run_chsh(
                setup,
                library,
                a,
                b,
                sa,
                sb,
                "deg",
                [names[:2], names[2:]],
                quantity,
                time,
                RunCache(),
                should_cancel,
                sidebands,
            )

        self.run_button.setEnabled(False)
        self.status.setText("running the four setting pairs...")
        self.runner.start(job)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText(text)

    def _failed(self, message: str) -> None:
        self._end(f"failed: {message.splitlines()[0]}" if self.run_is_current() else "")

    def _done(self, result) -> None:
        if not self.run_is_current():
            self._end("")
            return
        self.result = result
        t = result.integration_time
        total = result.total_rates if result.total_rates is not None else result.rates
        rows = []
        for i in range(2):
            for j in range(2):
                counts = total[i, j].ravel() * t
                e_raw = correlation_from_rates(*total[i, j].ravel())
                e_cor = (
                    result.correlations_corrected[i, j]
                    if result.correlations_corrected is not None
                    else math.nan
                )
                rows.append(
                    [
                        _NumItem(PAIR_LABELS[2 * i + j]),
                        _NumItem(plain(result.settings_a[i]), result.settings_a[i]),
                        _NumItem(plain(result.settings_b[j]), result.settings_b[j]),
                        *[_NumItem(plain(c, 6), c) for c in counts],
                        _NumItem(plain(e_raw), e_raw),
                        _NumItem(plain(e_cor), e_cor),
                    ]
                )
        _fill(self.table, rows)
        self.summary.setText(format_chsh(result).replace("\n", "<br>"))
        z = result.significance
        self.note = (
            f"S = {result.s:.4f} ({'raw' if result.quantity == 'total_hz' else 'subtracted'}), "
            + ("significance undefined" if z is None else f"|S| - 2 = {z:.1f} sigma (forecast)")
        )
        self._end(self.stale_text(self.note) if self.accept_run() else self.note)

    def plan_text(self) -> str:
        r = self.result
        if r is None:
            return ""
        lines = [format_chsh(r), "", "acquisition plan (counts expected per setting pair)"]
        lines.append(
            "pair,a_parameter_deg,b_parameter_deg," + ",".join(OUTCOME_LABELS) + ",seconds"
        )
        for i in range(2):
            for j in range(2):
                rates = r.total_rates if r.total_rates is not None else r.rates
                counts = rates[i, j].ravel() * r.integration_time
                lines.append(
                    f"{i}{j},{r.settings_a[i]:g},{r.settings_b[j]:g},"
                    + ",".join(f"{c:.6g}" for c in counts)
                    + f",{r.integration_time:g}"
                )
        lines.append("counts are raw coincidences in each analysis window")
        return "\n".join(lines) + "\n"

    def export_plan(self) -> None:
        if self.result is not None:
            save_text(self, "Export CHSH plan", "chsh_plan.txt", self.plan_text())
