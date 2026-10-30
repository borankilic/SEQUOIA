from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sequoia.errors import Issue, SequoiaError
from sequoia.fock.experiment import (
    Experiment,
    FockResult,
    load_experiment,
    save_experiment,
)
from sequoia.fock.from_setup import (
    ComponentRow,
    Extraction,
    InputDescription,
    SetupExperiment,
    SetupResult,
)
from sequoia.fock.preflight import Estimate, estimate_experiment
from sequoia.fock.run import (
    EditorOutcome,
    SetupOutcome,
    SetupRequest,
    rejection_text,
    run_editor_experiment,
    run_setup_request,
)
from sequoia.gui.fock_common import BAR_LIMIT, parse_filter
from sequoia.gui.fock_editor import (
    DETECTOR_CHOICES,
    ELEMENT_TYPES,
    CircuitEditor,
    default_experiment,
)
from sequoia.gui.fock_results import EditorResultsView
from sequoia.gui.fock_setup import SetupWorkspace
from sequoia.gui.jobs import JobRunner
from sequoia.gui.runguard import RunGuard

__all__ = ["BAR_LIMIT", "DETECTOR_CHOICES", "ELEMENT_TYPES", "FockTab", "parse_filter"]

ESTIMATE_SETUP = "Resources: Import from setup (or Run) to see the estimate before computing."


class FockTab(RunGuard, QWidget):
    """Photon counts: the drawn circuit as a photon-number experiment, or a free circuit editor.
    The tab owns the run controls and the run lifecycle; the circuit editor, the setup workspace
    and the result views are separate widgets, and the computation is in sequoia.fock.run."""

    def __init__(self, provider=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard_init()
        self.provider = provider
        self.on_issues = None
        self.runner = JobRunner(self)
        self.result: FockResult | None = None
        self.result_rows: tuple = ()
        self.setup_exp: SetupExperiment | None = None
        self.setup_result: SetupResult | None = None
        self.extraction: Extraction | None = None
        self.components: list[ComponentRow] = []
        self.inputs: list[InputDescription] = []
        self.estimate: Estimate | None = None
        self._stale_note = ""
        self._saved_key: str | None = None
        self._result_key: str | None = None
        self._run_key: str | None = None
        self._run_estimate: Estimate | None = None
        self.confirm_replace = None

        self.mode_box = QComboBox()
        self.mode_box.addItems(["From setup", "Circuit editor"])
        self.import_setup_button = QPushButton("Import from setup")
        self.import_setup_button.setToolTip(
            "Read the drawn circuit: list its components and the photon inputs found in it. "
            "Nothing is copied; Run always uses the canvas as it is now."
        )
        self.extract_button = QPushButton("Extract to circuit editor")
        self.extract_button.setToolTip(
            "Turn the drawn circuit into editable beam splitter, phase and loss elements "
            "and open them in the circuit editor. The editor gets an independent copy: later "
            "canvas edits do not reach it and editor changes do not change the canvas."
        )
        self.export_run_button = QPushButton("Export run...")
        self.export_run_button.setToolTip(
            "Write the last run as a folder: the setup as run, the library parts it uses, the "
            "photon-number experiment, run.json and a summary"
        )
        self.inventory_button = QPushButton("Components and inputs")
        self.inventory_button.setCheckable(True)
        self.inventory_button.setToolTip("Show the component inventory and the photon inputs")
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.0, 1e9)
        self.duration.setDecimals(6)
        self.duration.setValue(1.0)
        self.duration.setPrefix("duration ")
        self.duration.setSuffix(" s")
        self.n_max_pairs = QSpinBox()
        self.n_max_pairs.setRange(0, 12)
        self.n_max_pairs.setSpecialValueText("pairs: auto")
        self.n_max_pairs.setPrefix("max pairs ")
        self.detector_mode = QComboBox()
        self.detector_mode.addItems(["setup detectors", "ideal PNR"])
        self.open_button = QPushButton("Open circuit...")
        self.save_button = QPushButton("Save circuit...")
        self.import_button = QPushButton("Import Perceval GUI circuit...")
        self.backend = QComboBox()
        self.backend.addItem("native")
        from sequoia.fock.perceval_bridge import perceval_version

        if perceval_version() is not None:
            self.backend.addItem("perceval")
        self.shots = QSpinBox()
        self.shots.setRange(0, 100_000_000)
        self.shots.setValue(0)
        self.shots.setPrefix("shots ")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(1)
        self.seed.setPrefix("seed ")
        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setStyleSheet("color: palette(placeholder-text);")
        self.status.setWordWrap(True)
        self.estimate_label = QLabel(ESTIMATE_SETUP)
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("color: palette(placeholder-text);")

        self.editor = CircuitEditor(self.shots, self.seed, self.backend)
        self.editor_results = EditorResultsView()
        self.workspace = SetupWorkspace()
        self.setup_view = self.workspace.results
        for name in (
            "n_modes",
            "brightness",
            "indist",
            "max_photons",
            "circuit_table",
            "input_table",
            "add_button",
            "remove_button",
            "up_button",
            "down_button",
            "new_type",
            "preserved_label",
            "mixture_label",
            "clear_mixture_button",
            "editor_tabs",
            "add_element",
            "remove_element",
            "move_element",
            "clear_mixture",
        ):
            setattr(self, name, getattr(self.editor, name))
        for name in (
            "filter_edit",
            "normalized",
            "success_label",
            "message_label",
            "results_table",
            "plot",
        ):
            setattr(self, name, getattr(self.editor_results, name))
        self.components_table = self.workspace.components_table
        self.inputs_table = self.workspace.inputs_table
        self.info_tabs = self.workspace.info_tabs
        for name in (
            "model_card",
            "states_table",
            "state_filter",
            "state_summary",
            "clicks_table",
            "detector_table",
            "marginals_table",
            "cond_table",
            "cond_result",
            "cond_summary",
            "samples_table",
            "samples_note",
            "notes_view",
            "setup_plot",
            "setup_normalized",
            "pair_plot",
            "result_tabs",
        ):
            setattr(self, name, getattr(self.setup_view, name))

        row_a = QHBoxLayout()
        row_a.setContentsMargins(6, 4, 6, 0)
        for w in (
            self.mode_box,
            self.import_setup_button,
            self.extract_button,
            self.inventory_button,
            self.duration,
            self.n_max_pairs,
            self.detector_mode,
            self.export_run_button,
            self.open_button,
            self.save_button,
            self.import_button,
        ):
            row_a.addWidget(w)
        row_a.addStretch(1)
        row_b = QHBoxLayout()
        row_b.setContentsMargins(6, 0, 6, 0)
        for w in (
            self.backend,
            self.shots,
            self.seed,
            self.run_button,
            self.cancel_button,
            self.progress,
        ):
            row_b.addWidget(w)
        row_b.addWidget(self.status, 1)
        row_c = QHBoxLayout()
        row_c.setContentsMargins(6, 0, 6, 4)
        row_c.addWidget(self.estimate_label, 1)

        editor_page = QSplitter()
        editor_page.addWidget(self.editor)
        editor_page.addWidget(self.editor_results)
        editor_page.setStretchFactor(0, 1)
        editor_page.setStretchFactor(1, 1)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.workspace)
        self.stack.addWidget(editor_page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row_a)
        layout.addLayout(row_b)
        layout.addLayout(row_c)
        layout.addWidget(self.stack, 1)

        self.inventory_button.toggled.connect(self.info_tabs.setVisible)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.open_button.clicked.connect(self.open_dialog)
        self.save_button.clicked.connect(self.save_dialog)
        self.import_button.clicked.connect(self.import_dialog)
        self.export_run_button.clicked.connect(self.export_run_dialog)
        self.import_setup_button.clicked.connect(self.import_from_setup)
        self.extract_button.clicked.connect(self.extract_to_editor)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.filter_edit.textChanged.connect(self._refresh_results)
        self.normalized.toggled.connect(self._refresh_results)
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.cancelled.connect(lambda: self._end("cancelled: no result (run stopped)"))
        for w in (self.duration, self.n_max_pairs, self.shots, self.seed, self.max_photons):
            w.valueChanged.connect(self.refresh_stale)
        for w in (self.detector_mode, self.backend):
            w.currentIndexChanged.connect(self.refresh_stale)
        self.editor.changed.connect(self._editor_changed)
        self.set_experiment(default_experiment())
        self._saved_key = self.editor_key()
        self.mode_box.setCurrentIndex(0 if provider is not None else 1)
        self._mode_changed()

    @property
    def setup_mode(self) -> bool:
        return self.mode_box.currentIndex() == 0

    @property
    def bars(self):
        return self.editor_results.bars

    @property
    def setup_bars(self):
        return self.setup_view.setup_bars

    @property
    def pair_bars(self):
        return self.setup_view.pair_bars

    def _mode_changed(self, *_args) -> None:
        setup = self.setup_mode
        self.stack.setCurrentIndex(0 if setup else 1)
        for w in (
            self.import_setup_button,
            self.extract_button,
            self.inventory_button,
            self.duration,
            self.n_max_pairs,
            self.export_run_button,
        ):
            w.setVisible(setup)
        self.detector_mode.setVisible(setup)
        for w in (self.open_button, self.save_button, self.import_button):
            w.setVisible(not setup)
        self._show_estimate()

    def sweep_settings(self) -> dict:
        return {
            "detectors_mode": "pnr" if self.detector_mode.currentIndex() == 1 else "setup",
            "n_max": self.n_max_pairs.value() or None,
            "duration": self.duration.value(),
        }

    def config_key(self):
        return (
            self.duration.value(),
            self.n_max_pairs.value(),
            self.detector_mode.currentIndex(),
            self.shots.value(),
            self.seed.value(),
            self.backend.currentText(),
            self.max_photons.value(),
        )

    def set_outcome(self, outcome) -> None:
        if self.track_revision(outcome) and (self.setup_result is not None or self.setup_exp):
            text = self.stale_text(self._stale_note)
            if self.setup_mode:
                self.status.setText(text)
            if self.setup_result is not None:
                self.state_summary.setText(text)

    def reset(self) -> None:
        self.invalidate()
        self.result = None
        self.result_rows = ()
        self._result_key = None
        self.editor_results.clear()
        self.setup_exp = None
        self.setup_result = None
        self.extraction = None
        self.components = []
        self.inputs = []
        self.estimate = None
        self._stale_note = ""
        if self.on_issues is not None:
            self.on_issues([])
        self.workspace.clear()
        self.status.setText("")
        self._show_estimate()

    def shutdown(self) -> None:
        self.runner.shutdown()

    def experiment(self) -> Experiment:
        return self.editor.experiment()

    def set_experiment(self, exp: Experiment) -> None:
        self.editor.set_experiment(exp)
        if self.result is not None:
            self._mark_editor_stale()
        self._show_estimate()

    def editor_key(self) -> str | None:
        return self.editor.editor_key()

    @property
    def editor_dirty(self) -> bool:
        return self.editor_key() != self._saved_key

    def _editor_changed(self) -> None:
        if self.result is not None:
            self._mark_editor_stale()
        self._show_estimate()

    def _editor_result_outdated(self) -> bool:
        return self.result is not None and self.editor_key() != self._result_key

    def _mark_editor_stale(self) -> None:
        if self._editor_result_outdated():
            self.success_label.setText(
                "outdated - the circuit or its inputs were edited since this run; run again"
            )

    def editor_estimate(self) -> Estimate | None:
        try:
            return estimate_experiment(self.experiment())
        except (SequoiaError, ValueError):
            return None

    def _show_estimate(self) -> None:
        if self.setup_mode:
            est = self.estimate
            if est is None:
                self.estimate_label.setText(ESTIMATE_SETUP)
                self.estimate_label.setToolTip("")
                return
            prefix = "Resources of the drawn setup (from the last import or run)"
        else:
            est = self.editor_estimate()
            if est is None:
                self.estimate_label.setText("Resources: the circuit or its inputs are not valid")
                self.estimate_label.setToolTip("")
                return
            prefix = "Resources"
        text = f"{prefix}: {est.summary()}"
        if est.errors:
            text = f"{prefix}: will not run - {est.errors[0]}"
        elif est.warnings:
            text += ". " + "; ".join(est.warnings)
        self.estimate_label.setText(text)
        self.estimate_label.setToolTip(
            "\n".join([est.summary(), *est.errors, *est.warnings])
            + "\nRemedies: lower max pairs, photons or shots, or reduce modes."
        )

    def _start(self, text: str) -> None:
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.import_setup_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText(text)

    def _setup_job(self, kind: str):
        setup, library = self.provider() if self.provider is not None else (None, None)
        if setup is None:
            self.status.setText("no setup is open")
            return None
        req = SetupRequest(
            kind,
            setup,
            library,
            duration=self.duration.value(),
            n_max=self.n_max_pairs.value() or None,
            detectors_mode="pnr" if self.detector_mode.currentIndex() == 1 else "setup",
            max_photons=self.max_photons.value(),
            shots=self.shots.value(),
            seed=self.seed.value(),
            backend=self.backend.currentText(),
        )
        return lambda should_cancel, progress: run_setup_request(req, should_cancel, progress)

    def import_from_setup(self) -> None:
        if self.runner.busy:
            return
        self.inventory_button.setChecked(True)
        job = self._setup_job("import")
        if job is None:
            return
        self.begin_run()
        self._start("reading the drawn circuit...")
        self.runner.start(job)

    def extract_to_editor(self) -> None:
        if self.runner.busy or not self._may_replace():
            return
        job = self._setup_job("extract")
        if job is None:
            return
        self.begin_run()
        self._start("extracting the circuit...")
        self.runner.start(job)

    def run(self) -> None:
        if self.runner.busy:
            return
        if self.setup_mode:
            job = self._setup_job("run")
            if job is None:
                return
            self.begin_run()
            self._start("running...")
            self.runner.start(job)
            return
        try:
            exp = self.experiment()
            exp.photons()
            est = estimate_experiment(exp)
        except (SequoiaError, ValueError) as exc:
            self._end(f"invalid input: {exc}")
            return
        if not est.ok:
            self.message_label.setText(rejection_text(est))
            self._end(f"not run: {est.errors[0]}")
            return
        self._run_key = self.editor_key()
        self._run_estimate = est

        def job(should_cancel, progress):
            return run_editor_experiment(exp, est, should_cancel, progress)

        self.begin_run()
        self._start("running...")
        self.runner.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _end(self, text: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.import_setup_button.setEnabled(True)
        self.extract_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText(text)

    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "failed"
        if not self.run_is_current():
            self._end("")
            return
        self.message_label.setText(first)
        self._end(f"failed: {first}")

    def _done(self, outcome) -> None:
        if not self.run_is_current():
            self._end("")
            return
        if isinstance(outcome, SetupOutcome):
            self._payload_done(outcome)
            return
        self._editor_done(outcome)

    def _editor_done(self, outcome: EditorOutcome) -> None:
        result = outcome.result
        self.result = result
        self.result_rows = outcome.ranked
        self._result_key = self._run_key
        r = result.resources
        text = (
            f"{r['output_states']} states in {r['seconds']:.3g} s ({result.backend}, "
            f"{r['permanents']} permanents)"
        )
        if outcome.status != "complete":
            text = f"partial result ({outcome.status}): {text}; samples not drawn"
        self._end(text)
        self._refresh_results()

    def _issues(self, texts: list[str]) -> None:
        if self.on_issues is not None:
            self.on_issues([Issue("warning", "photon_counts", "photon counts", t) for t in texts])

    def _payload_done(self, p: SetupOutcome) -> None:
        outdated = self.accept_run()
        self.components = p.components
        self.inputs = p.inputs
        self.workspace.show_inventory(self.components, self.inputs)
        if p.se is not None:
            self.setup_exp = p.se
        self.estimate = p.estimate
        self._show_estimate()
        warnings: list[str] = []
        if p.rejected:
            self.setup_result = None
            self.setup_view.clear()
            self.setup_view.clear_tables("Not run: the resource estimate exceeds the limits.")
            self.notes_view.setPlainText(p.error)
            self.state_summary.setText(p.error)
            self._issues([p.error])
            self._stale_note = ""
            self._end(f"not run: {p.estimate.errors[0]}")
            return
        if p.error:
            self.setup_exp = None
            self.setup_result = None
            self.setup_view.result = None
            self.setup_view.views = None
            self.setup_view.clear_tables()
            self.notes_view.setPlainText(p.error)
            self.state_summary.setText(p.error)
            self.result_tabs.setCurrentIndex(self.result_tabs.count() - 1)
            self._issues([p.error])
            self._stale_note = ""
            self._end(f"cannot use the drawn circuit: {p.error.splitlines()[0]}")
            return
        se = p.se
        if se.omitted_weight > 1e-4:
            warnings.append(
                f"input weight beyond the pair-number truncation is not simulated: "
                f"{se.omitted_weight:.3g}"
            )
        if p.kind == "run" and p.result is not None:
            self.setup_result = p.result
            if p.views is None:
                self.setup_view.clear()
                self.setup_view.clear_tables(
                    "Partial result: cancelled before the tables were filled; run again."
                )
                self.model_card.setText(self.setup_view.model_text(p.result))
            else:
                self.setup_view.show_result(p.result, p.views)
            for d, load in p.result.outside_dilute_regime.items():
                warnings.append(
                    f"'{d}': potential click rate x dead time = {load:.3g}; the shown rates are "
                    "potential clicks before dead time, not recorded counts"
                )
            r = p.result.result.resources
            note = (
                f"{len(p.result.detected)} detector states in {r['seconds']:.3g} s "
                f"({p.result.result.backend}, {r['permanents']} permanents)"
            )
            if p.result.omitted_weight > 1e-4:
                note += f"; omitted input weight {p.result.omitted_weight:.3g}"
            if p.status != "complete":
                note = f"partial result (cancelled before it finished): {note}"
            self._stale_note = note
            self._issues(warnings)
            self._end(self.stale_text(note) if outdated else note)
            if outdated:
                self.state_summary.setText(self.stale_text(note))
        elif p.kind == "extract" and p.extraction is not None:
            exp, ex = p.extraction
            self.extraction = ex
            self.set_experiment(exp)
            self._saved_key = None
            self.mode_box.setCurrentIndex(1)
            self.editor_tabs.setCurrentIndex(0)
            note = (
                f"extracted {len(ex.circuit.elements)} elements on {ex.circuit.n_modes} modes "
                f"({'verified against the exact compilation' if ex.verified else 'one block'}, "
                f"max deviation {ex.max_error:.1e})"
            )
            if ex.notes:
                note += "; " + "; ".join(ex.notes)
            note += (
                "; the editor holds a copy: later edits to the canvas do not reach it, and edits "
                "here do not change the canvas"
            )
            if outdated:
                note = "the setup changed while extracting; extract again. " + note
            self._issues(warnings + list(ex.notes))
            self._end(note)
        else:
            self.notes_view.setPlainText("\n".join(se.notes))
            self._stale_note = f"{len(p.components)} components, {len(p.inputs)} input source(s)"
            self._issues(warnings)
            self._end(self.stale_text(self._stale_note) if outdated else self._stale_note)

    def _refresh_results(self, *_args) -> None:
        res = self.result
        if res is None:
            return
        self.editor_results.show_result(res, self.result_rows)
        self._mark_editor_stale()

    def export_run_dialog(self) -> None:
        if self.setup_result is None:
            self.status.setText("run the drawn circuit first")
            return
        path = QFileDialog.getExistingDirectory(self, "Export run to folder")
        if path:
            self.export_run(path)

    def export_run(self, path: str | Path) -> bool:
        from sequoia.fock.records import write_run_bundle

        if self.setup_result is None:
            return False
        try:
            write_run_bundle(self.setup_result, path)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot export run", str(exc))
            return False
        note = f"exported run to {path}"
        if self.results_outdated():
            note += " (the run predates later edits; the bundle holds the inputs it was run on)"
        self.status.setText(note)
        return True

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open circuit experiment", "", "Circuit experiment (*.yaml *.yml *.json)"
        )
        if path:
            self.open_path(path)

    def save_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save circuit experiment", "", "Circuit experiment (*.yaml *.yml *.json)"
        )
        if path:
            self.save_path(path)

    def import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Perceval GUI circuit", "", "Perceval GUI circuit (*.json)"
        )
        if path:
            self.import_path(path)

    def _problem(self, title: str, message: str) -> None:
        self.status.setText(f"{title}: {message.splitlines()[0] if message else ''}")
        QMessageBox.warning(self, title, message)

    def _may_replace(self) -> bool:
        """Ask before an edited, unsaved circuit is replaced: save, discard or cancel."""
        if not self.editor_dirty:
            return True
        if self.confirm_replace is not None:
            answer = self.confirm_replace(self)
        else:
            box = QMessageBox.question(
                self,
                "Unsaved circuit",
                "The circuit in the editor has unsaved changes. Save them first?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            answer = {
                QMessageBox.StandardButton.Save: "save",
                QMessageBox.StandardButton.Discard: "discard",
            }.get(box, "cancel")
        if answer == "save":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save circuit experiment", "", "Circuit experiment (*.yaml *.yml *.json)"
            )
            return bool(path) and self.save_path(path)
        return answer == "discard"

    def open_path(self, path: str | Path) -> bool:
        if not self._may_replace():
            return False
        try:
            self.set_experiment(load_experiment(path))
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot open circuit", str(exc))
            return False
        self._saved_key = self.editor_key()
        self.status.setText(f"opened {Path(path).name}")
        return True

    def save_path(self, path: str | Path) -> bool:
        try:
            save_experiment(self.experiment(), path)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot save circuit", str(exc))
            return False
        self._saved_key = self.editor_key()
        self.status.setText(f"saved {Path(path).name}")
        return True

    def import_path(self, path: str | Path) -> bool:
        from sequoia.fock.import_perceval_gui import import_perceval_gui

        if not self._may_replace():
            return False
        try:
            imp = import_perceval_gui(path, strict=False)
        except (SequoiaError, OSError, ValueError) as exc:
            self._problem("Cannot import circuit", str(exc))
            return False
        n = imp.circuit.n_modes
        imp.experiment.occupations = [1 if i < min(2, n) else 0 for i in range(n)]
        self.set_experiment(imp.experiment)
        self._saved_key = None
        note = f"imported {Path(path).name}: {n} modes, {imp.n_components} components"
        if imp.warnings:
            note += f"; {len(imp.warnings)} warning(s): " + "; ".join(imp.warnings)
        self.status.setText(note)
        return True
