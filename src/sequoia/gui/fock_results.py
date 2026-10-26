import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sequoia.fock.analysis import condition, count_variance, format_state
from sequoia.fock.circuit import FockError
from sequoia.fock.experiment import CANCELLED_SAMPLING
from sequoia.fock.from_setup import SetupResult
from sequoia.fock.run import SetupViews, ranked, setup_views
from sequoia.gui.fock_common import (
    BAR_LIMIT,
    ResultTable,
    draw_bars,
    num,
    parse_filter,
    style_plot,
)

SIGMA_TIP = (
    "Predicted spread of the count over a fixed number of independent detection windows: "
    "sqrt(N p (1 - p)) (binomial)"
)
RATE_TIP = (
    "Probability per window x windows per second. Dead time is not modelled, so this is the "
    "potential click rate, not a recorded count rate"
)
MODEL_EMPTY = (
    "Model card: run the drawn setup to see the measurement (window, windows per second, "
    "duration), the detectors, the effects included and omitted, and the normalization."
)


def pattern(state, ids) -> str:
    return " ".join(f"{d}={n}" for d, n in zip(ids, state, strict=True))


def bars_title(plot, shown: list, total: int) -> None:
    if total > BAR_LIMIT:
        rest = float(sum(p for _, p in shown[BAR_LIMIT:]))
        plot.setTitle(
            f"most probable {BAR_LIMIT} of {total} states; the rest has probability {rest:.3g}"
        )
    else:
        plot.setTitle(None)


class SetupResultsView(QWidget):
    """Result views of a drawn-setup run: output states, detector clicks, marginals, conditional
    distribution, samples, plots and notes, with the model card above them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result: SetupResult | None = None
        self.views: SetupViews | None = None
        self.model_card = QLabel(MODEL_EMPTY)
        self.model_card.setWordWrap(True)
        self.model_card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.model_card.setFrameShape(QLabel.Shape.StyledPanel)
        self.model_card.setMargin(4)
        self.states_table = ResultTable(
            [
                "State",
                "Detectors",
                "Probability\nper window",
                "Potential rate\n1/s, no dead time",
                "Expected counts\nin duration",
                "Sigma\nbinomial, fixed windows",
                "Conditional\nprobability",
            ],
            "Run to see the output states of the drawn setup.",
            (
                "Detector counts, in the order of the Detectors column",
                "Counts per detector",
                "Probability of this detector outcome in one detection window",
                RATE_TIP,
                "Expected number of windows with this outcome in the chosen duration",
                SIGMA_TIP,
                "Probability given the condition typed above (normalised over accepted states)",
            ),
        )
        self.state_filter = QLineEdit()
        self.state_filter.setPlaceholderText(
            "condition on detectors, e.g. spad_1=1, spad_2>=1 (index or detector id)"
        )
        self.state_filter.setClearButtonEnabled(True)
        self.state_summary = QLabel("")
        self.state_summary.setWordWrap(True)
        self.clicks_table = ResultTable(
            [
                "Click pattern",
                "Probability\nper window",
                "Potential rate\n1/s, no dead time",
                "Expected counts\nin duration",
                "Sigma\nbinomial",
            ],
            "Run to see the detector click patterns.",
            ("Detectors that click (xN: N counts)", "", RATE_TIP, "", SIGMA_TIP),
        )
        self.detector_table = ResultTable(
            [
                "Detector",
                "Model",
                "Efficiency",
                "Dark\nper window",
                "P(click)\nper window",
                "Mean counts\nper window",
                "Potential click rate\n1/s, no dead time",
                "Expected counts\nin duration",
                "Sigma\nbinomial",
            ],
            "Run to see each detector's click probability and rate.",
            (
                "",
                "",
                "Detector efficiency, applied once in the detector response",
                "Probability of one dark click per detection window",
                "Probability of at least one count per window",
                "Mean number of counts per window",
                RATE_TIP,
                "",
                SIGMA_TIP,
            ),
        )
        self.marginals_table = ResultTable(
            ["Quantity", "Detector / mode", "n", "P(n photons arrive)", "P(n detected)"],
            "Run to see the photon-number marginals per detector and per output mode.",
        )
        self.cond_table = QTableWidget(0, 3)
        self.cond_table.setHorizontalHeaderLabels(["Detector", "Condition", "Value"])
        self.cond_table.horizontalHeader().setStretchLastSection(True)
        self.cond_table.verticalHeader().setVisible(False)
        self.cond_result = ResultTable(
            [
                "State",
                "Detectors",
                "Conditional\nprobability",
                "Joint probability\nper window",
                "Accepted-event rate\n1/s",
            ],
            "Run, then choose a condition on at least one detector.",
            (
                "",
                "",
                "Probability given the condition (normalised over accepted states)",
                "Unnormalised probability of the accepted outcome per window",
                "Accepted events per second (joint probability x windows per second)",
            ),
        )
        self.cond_summary = QLabel("")
        self.cond_summary.setWordWrap(True)
        self.named_condition = QComboBox()
        self.named_condition.addItem("custom (table below)")
        self.named_condition.setToolTip(
            "Named heralding conditions of the setup's measurement definition: herald channels "
            "must click, veto channels must not"
        )
        self.named_summary = QLabel("")
        self.named_summary.setWordWrap(True)
        self.named_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.samples_table = ResultTable(
            ["State", "Detectors", "Shots", "Frequency", "P\nper window"],
            "Set shots above 0 and run to draw seeded samples of the detector states.",
        )
        self.samples_note = QLabel("")
        self.notes_view = QPlainTextEdit()
        self.notes_view.setReadOnly(True)
        self.notes_view.setPlaceholderText("Notes, approximations and warnings appear after a run.")
        self.setup_plot = pg.PlotWidget()
        style_plot(self.setup_plot)
        self.setup_bars: pg.BarGraphItem | None = None
        self.setup_normalized = QCheckBox("show conditional distribution")
        self.pair_plot = pg.PlotWidget()
        style_plot(self.pair_plot, "photons arriving at detectors", "probability")
        self.pair_bars: pg.BarGraphItem | None = None

        states_page = QWidget()
        sl = QVBoxLayout(states_page)
        sl.setContentsMargins(0, 0, 0, 0)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Condition"))
        srow.addWidget(self.state_filter, 1)
        sl.addLayout(srow)
        sl.addWidget(self.state_summary)
        sl.addWidget(self.states_table, 1)
        clicks_page = QSplitter(Qt.Orientation.Vertical)
        clicks_page.addWidget(self.detector_table)
        clicks_page.addWidget(self.clicks_table)
        cond_page = QWidget()
        cl = QVBoxLayout(cond_page)
        cl.setContentsMargins(0, 0, 0, 0)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Measurement condition"))
        nrow.addWidget(self.named_condition, 1)
        cl.addLayout(nrow)
        cl.addWidget(self.named_summary)
        cl.addWidget(self.cond_table)
        cl.addWidget(self.cond_summary)
        cl.addWidget(self.cond_result, 1)
        samples_page = QWidget()
        sm = QVBoxLayout(samples_page)
        sm.setContentsMargins(0, 0, 0, 0)
        sm.addWidget(self.samples_note)
        sm.addWidget(self.samples_table, 1)
        plot_page = QWidget()
        pl = QVBoxLayout(plot_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.setup_normalized)
        pl.addWidget(self.setup_plot, 2)
        pl.addWidget(self.pair_plot, 1)
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(states_page, "Output states")
        self.result_tabs.addTab(clicks_page, "Detector clicks")
        self.result_tabs.addTab(self.marginals_table, "Marginals")
        self.result_tabs.addTab(cond_page, "Conditional")
        self.result_tabs.addTab(samples_page, "Samples")
        self.result_tabs.addTab(plot_page, "Plot")
        self.result_tabs.addTab(self.notes_view, "Notes")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.model_card)
        layout.addWidget(self.result_tabs, 1)

        self.state_filter.textChanged.connect(self.show_states)
        self.setup_normalized.toggled.connect(self.show_states)
        self.named_condition.currentIndexChanged.connect(self.fill_conditional)

    @property
    def tables(self) -> tuple[ResultTable, ...]:
        return (
            self.states_table,
            self.clicks_table,
            self.detector_table,
            self.marginals_table,
            self.cond_result,
            self.samples_table,
        )

    def clear_tables(self, empty: str | None = None) -> None:
        for table in self.tables:
            table.clear_rows(empty)
        self.cond_table.setRowCount(0)

    def clear(self) -> None:
        self.result = None
        self.views = None
        self.clear_tables()
        self.notes_view.setPlainText("")
        self.model_card.setText(MODEL_EMPTY)
        self.state_summary.setText("")
        self.cond_summary.setText("")
        self.named_summary.setText("")
        self.samples_note.setText("")
        if self.setup_bars is not None:
            self.setup_plot.removeItem(self.setup_bars)
            self.setup_bars = None
        if self.pair_bars is not None:
            self.pair_plot.removeItem(self.pair_bars)
            self.pair_bars = None
        self.setup_plot.setTitle(None)

    def show_result(self, res: SetupResult, views: SetupViews | None) -> None:
        self.result = res
        self.views = views if views is not None else setup_views(res)
        v = self.views
        se = res.setup_experiment
        cnt = res.counts
        self.detector_table.set_rows(
            v.detectors,
            lambda _r, d: (
                (d.id, None),
                (d.kind, None),
                (f"{d.efficiency:.4g}", d.efficiency),
                (f"{d.dark:.3g}", d.dark),
                (num(d.p_click), d.p_click),
                (num(d.mean), d.mean),
                (f"{d.rate:.6g}", d.rate),
                (f"{d.total:.6g}", d.total),
                (f"{d.sigma:.4g}", d.sigma),
            ),
        )
        self.clicks_table.set_rows(
            v.patterns,
            lambda r, row: (
                (row[0], float(r)),
                (num(row[1]), row[1]),
                (f"{row[2]:.6g}", row[2]),
                (f"{row[3]:.6g}", row[3]),
                (f"{row[4]:.4g}", row[4]),
            ),
            lambda row: row[1],
        )
        self.marginals_table.set_rows(
            v.marginals,
            lambda _r, row: (
                (row[0], None),
                (row[1], None),
                (str(row[2]), float(row[2])),
                (num(row[3]), row[3]),
                ("" if row[4] is None else num(row[4]), -1.0 if row[4] is None else row[4]),
            ),
        )
        self.fill_conditions(res.detector_ids)
        self.fill_named(v)
        self.show_states()
        self.fill_conditional()
        self.fill_samples()
        self.fill_pair_plot()
        notes = list(se.notes)
        notes += [m for m in res.messages if m not in notes]
        if res.omitted_weight > 1e-12:
            notes.append(f"omitted input weight (not simulated): {res.omitted_weight:.4g}")
        notes.append(
            f"detection window {se.window_s:.6g} s, {cnt.windows_per_second:.6g} windows per "
            f"second, duration {cnt.duration_s:g} s"
        )
        self.notes_view.setPlainText("\n".join(f"- {n}" for n in notes))
        self.model_card.setText(self.model_text(res))

    @staticmethod
    def model_text(res: SetupResult) -> str:
        se = res.setup_experiment
        cnt = res.counts
        models = sorted({spec.model.kind for spec in se.compiled.detectors})
        text = (
            f"Measurement: one detection window = {se.window_s:.4g} s (pulse period), "
            f"{cnt.windows_per_second:.4g} windows/s, duration {cnt.duration_s:g} s "
            f"({cnt.n_windows:.4g} windows); detectors {', '.join(se.detector_ids)} "
            f"({', '.join(models)}, efficiency and dark clicks per window included). "
            "Model: photon-number circuit; loss, interference, multiplexing included; dead time, "
            "jitter, afterpulsing and spectral/temporal structure not included, so rates are "
            "potential clicks before dead time. Uncertainty: binomial over a fixed number of "
            f"windows. Normalization: resolved probability {res.mass:.6g}, omitted input weight "
            f"{res.omitted_weight:.3g}."
        )
        high = res.outside_dilute_regime
        if high:
            text = (
                "Acquisition regime not modelled: potential click rate x dead time = "
                + ", ".join(f"{d} {v:.2g}" for d, v in high.items())
                + " (dilute regime < 0.1); recorded lab counts will be lower. "
                + text
            )
        if res.status != "complete":
            text = "PARTIAL RESULT (run cancelled before it finished). " + text
        return text

    def fill_conditions(self, ids: list[str]) -> None:
        old = {}
        for r in range(self.cond_table.rowCount()):
            name = self.cond_table.item(r, 0).text()
            op = self.cond_table.cellWidget(r, 1)
            val = self.cond_table.cellWidget(r, 2)
            old[name] = (op.currentIndex(), val.value())
        self.cond_table.setRowCount(len(ids))
        for r, d in enumerate(ids):
            item = QTableWidgetItem(d)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.cond_table.setItem(r, 0, item)
            op = QComboBox()
            op.addItems(["(any)", "=", ">=", ">", "<=", "<", "!="])
            val = QSpinBox()
            val.setRange(0, 99)
            val.setValue(1)
            if d in old:
                op.setCurrentIndex(old[d][0])
                val.setValue(old[d][1])
            op.currentIndexChanged.connect(self.fill_conditional)
            val.valueChanged.connect(self.fill_conditional)
            self.cond_table.setCellWidget(r, 1, op)
            self.cond_table.setCellWidget(r, 2, val)

    def cond_spec(self) -> dict[int, tuple[str, int]]:
        spec = {}
        for r in range(self.cond_table.rowCount()):
            op = self.cond_table.cellWidget(r, 1)
            val = self.cond_table.cellWidget(r, 2)
            if op is not None and op.currentIndex() > 0:
                spec[r] = (op.currentText(), val.value())
        return spec

    def fill_named(self, views: SetupViews) -> None:
        current = self.named_condition.currentText()
        self.named_condition.blockSignals(True)
        while self.named_condition.count() > 1:
            self.named_condition.removeItem(1)
        for nc in views.conditions:
            self.named_condition.addItem(nc.name)
        index = self.named_condition.findText(current)
        self.named_condition.setCurrentIndex(max(index, 0))
        self.named_condition.blockSignals(False)
        if views.condition_error:
            self.named_summary.setText(
                f"measurement conditions not applied: {views.condition_error}"
            )
        elif views.conditions:
            self.named_summary.setText(
                "\n".join(
                    f"{nc.name}: {nc.rule}, success probability "
                    f"{nc.conditional.success_probability:.6g} per window"
                    + (f" ({nc.description})" if nc.description else "")
                    for nc in views.conditions
                )
            )
        else:
            self.named_summary.setText("the measurement defines no named conditions")

    def fill_conditional(self, *_args) -> None:
        res = self.result
        if res is None:
            return
        cnt = res.counts
        index = self.named_condition.currentIndex()
        named = self.views.conditions if self.views is not None else ()
        self.cond_table.setEnabled(index <= 0)
        head = ""
        if 0 < index <= len(named):
            nc = named[index - 1]
            cond = nc.conditional
            head = f"'{nc.name}' ({nc.rule}): "
        else:
            spec = self.cond_spec()
            if not spec:
                self.cond_summary.setText("choose a condition on at least one detector")
                self.cond_result.clear_rows()
                return
            cond = condition(res.detected, spec)
        rate = cond.success_probability * cnt.windows_per_second
        self.cond_summary.setText(
            f"{head}success probability {cond.success_probability:.6g} per window "
            f"({rate:.6g} per second, {rate * cnt.duration_s:.6g} in {cnt.duration_s:g} s); "
            f"{len(cond.joint)} detector states kept, distribution normalised"
        )
        wps = cnt.windows_per_second
        self.cond_result.set_rows(
            ranked(cond.joint),
            lambda r, sp: (
                (format_state(sp[0]), float(r)),
                (pattern(sp[0], res.detector_ids), None),
                (num(cond.probs.get(sp[0], 0.0)), cond.probs.get(sp[0], 0.0)),
                (num(sp[1]), sp[1]),
                (f"{sp[1] * wps:.6g}", sp[1] * wps),
            ),
            lambda sp: sp[1],
            mass_column=3,
        )

    def fill_samples(self) -> None:
        res = self.result
        if res is None:
            return
        if res.samples is None:
            self.samples_table.clear_rows()
            self.samples_note.setText(
                "samples were not drawn: the run was cancelled before sampling"
                if CANCELLED_SAMPLING in res.messages
                else "set shots above 0 to draw seeded samples of the detector states"
            )
            return
        sm = res.samples
        note = f"{sm.shots} shots, seed {sm.seed}"
        if sm.unresolved_probability:
            note += (
                f"; {sm.unresolved} shots fell in the unresolved input weight "
                f"({sm.unresolved_probability:.3g}, truncated pair numbers) and are not listed"
            )
        self.samples_note.setText(note)
        self.samples_table.set_rows(
            tuple(sorted(sm.counts.items(), key=lambda kv: -kv[1])),
            lambda r, sc: (
                (format_state(sc[0]), float(r)),
                (pattern(sc[0], res.detector_ids), None),
                (str(sc[1]), float(sc[1])),
                (num(sc[1] / sm.shots), sc[1] / sm.shots),
                (num(res.detected.get(sc[0], 0.0)), res.detected.get(sc[0], 0.0)),
            ),
        )

    def show_states(self, *_args) -> None:
        res = self.result
        if res is None:
            return
        ids = res.detector_ids
        cnt = res.counts
        text = self.state_filter.text()
        cond = None
        try:
            spec = parse_filter(text, ids)
        except FockError as exc:
            spec = {}
            self.state_summary.setText(str(exc))
        else:
            if spec:
                cond = condition(res.detected, spec)
                self.state_summary.setText(
                    f"success probability {cond.success_probability:.6g} "
                    f"({len(cond.joint)} states kept, conditional distribution normalised)"
                )
            else:
                pn = ", ".join(f"{n}: {v:.4g}" for n, v in self.views.photon_numbers.items())
                self.state_summary.setText(
                    f"total probability {res.mass:.12g}; photons arriving at the detectors: {pn}"
                )
        rows = self.views.states
        wps, dur, n_win = cnt.windows_per_second, cnt.duration_s, cnt.n_windows

        def cells(r, sp):
            s, p = sp
            rate = p * wps
            cp = cond.probs.get(s) if cond is not None else None
            sig = math.sqrt(count_variance(p, n_win))
            return (
                (format_state(s), float(r)),
                (pattern(s, ids), None),
                (num(p), p),
                (f"{rate:.6g}", rate),
                (f"{rate * dur:.6g}", rate * dur),
                (f"{sig:.4g}", sig),
                ("" if cp is None else num(cp), -1.0 if cp is None else cp),
            )

        self.states_table.set_rows(rows, cells, lambda sp: sp[1], mass_column=2)
        if self.setup_normalized.isChecked() and cond is not None and cond.probs:
            shown = list(ranked(cond.probs))
        else:
            shown = list(rows)
        self.setup_bars = draw_bars(self.setup_plot, self.setup_bars, shown)
        bars_title(self.setup_plot, shown, len(shown))

    def fill_pair_plot(self) -> None:
        if self.views is None:
            return
        pn = list(self.views.photon_numbers.items())
        if self.pair_bars is not None:
            self.pair_plot.removeItem(self.pair_bars)
            self.pair_bars = None
        if not pn:
            return
        brush = QApplication.palette().highlight().color()
        self.pair_bars = pg.BarGraphItem(
            x=np.array([n for n, _ in pn], dtype=float),
            height=np.array([p for _, p in pn]),
            width=0.8,
            brush=brush,
        )
        self.pair_plot.addItem(self.pair_bars)


class EditorResultsView(QWidget):
    """Output states of a circuit-editor run, with a condition filter and a bar plot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("condition on detected state, e.g. 0=1, 1>=1")
        self.filter_edit.setClearButtonEnabled(True)
        self.success_label = QLabel("")
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: palette(placeholder-text);")
        self.results_table = ResultTable(
            ["State", "Probability", "Counts\nsampled shots", "Conditional\nprobability"],
            "Run to see the output states of the circuit.",
            (
                "Detector outcome |n0,n1,...> in detector order",
                "Probability per run (per window)",
                "Number of seeded shots with this outcome (shots > 0)",
                "Probability given the condition typed above",
            ),
        )
        self.plot = pg.PlotWidget()
        style_plot(self.plot)
        self.bars: pg.BarGraphItem | None = None
        self.normalized = QCheckBox("show conditional distribution")
        self.normalized.setChecked(False)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Condition"))
        filter_row.addWidget(self.filter_edit, 1)
        filter_row.addWidget(self.normalized)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(filter_row)
        layout.addWidget(self.success_label)
        layout.addWidget(self.message_label)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.results_table)
        splitter.addWidget(self.plot)
        layout.addWidget(splitter, 1)

    def clear(self) -> None:
        self.results_table.clear_rows()
        if self.bars is not None:
            self.plot.removeItem(self.bars)
            self.bars = None
        self.plot.setTitle(None)
        self.success_label.setText("")
        self.message_label.setText("")

    def show_result(self, res, rows) -> None:
        cond = None
        text = self.filter_edit.text()
        try:
            spec = parse_filter(text)
        except FockError as exc:
            spec = {}
            self.success_label.setText(str(exc))
        msgs = list(res.messages)
        if spec:
            cond = condition(res.probabilities, spec)
            self.success_label.setText(
                f"success probability {cond.success_probability:.6g} "
                f"({len(cond.joint)} states kept, conditional distribution normalized)"
            )
        elif not text.strip():
            self.success_label.setText(f"total probability {res.mass:.12g}")
        pn = res.raw.by_photon_number()
        if len(pn) > 1:
            msgs.append("photon-number mass: " + ", ".join(f"{k}: {v:.4g}" for k, v in pn.items()))
        self.message_label.setText("; ".join(msgs))
        counts = res.samples.counts if res.samples else {}
        sampled = res.samples is not None

        def cells(i, sp):
            s, p = sp
            c = counts.get(s)
            cp = cond.probs.get(s) if cond is not None else None
            return (
                (format_state(s), float(i)),
                (f"{p:.6g}", p),
                ("" if not sampled else str(c or 0), float(c or 0)),
                ("" if cp is None else f"{cp:.6g}", -1.0 if cp is None else cp),
            )

        self.results_table.set_rows(rows, cells, lambda sp: sp[1])
        if self.normalized.isChecked() and cond is not None and cond.probs:
            shown = list(ranked(cond.probs))
        else:
            shown = list(rows)
        self.bars = draw_bars(self.plot, self.bars, shown)
        bars_title(self.plot, shown, len(shown))
