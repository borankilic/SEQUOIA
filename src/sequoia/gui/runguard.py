class RunGuard:
    """Ties an analysis tab's results to the inputs they were computed for.

    A run captures, when it is launched, the document key (``revision_source``: document
    identity, edit revision and library) and the tab's own settings (``config_key``). Results
    are shown as current only while both still match; a run reset or cancelled before it
    finishes is dropped (``run_is_current``).
    """

    revision_source = None

    def _guard_init(self) -> None:
        self._generation = 0
        self._pending: int | None = None
        self._launch_key = None
        self._shown_key = None

    def document_key(self):
        return self.revision_source() if self.revision_source is not None else None

    def config_key(self):
        return None

    def current_key(self):
        return (self.document_key(), self.config_key())

    def begin_run(self) -> None:
        self._pending = self._generation
        self._launch_key = self.current_key()

    def run_is_current(self) -> bool:
        return self._pending == self._generation

    def accept_run(self) -> bool:
        """Mark the finished run's results as shown; True when its inputs changed meanwhile."""
        self._shown_key = self._launch_key
        return self.results_outdated()

    def results_outdated(self) -> bool:
        return self._shown_key is not None and self._shown_key != self.current_key()

    def invalidate(self) -> None:
        self._generation += 1
        self._shown_key = None
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.cancel()

    def track_revision(self, _outcome=None) -> bool:
        return self.results_outdated()

    def refresh_stale(self, *_args) -> None:
        self.set_outcome(None)

    def stale_text(self, text: str) -> str:
        return f"outdated - the setup or the settings changed since this run. {text}".strip()


def card_html(rows) -> str:
    from html import escape

    return "<br>".join(f"<b>{escape(a)}:</b> {escape(b)}" for a, b in rows)


class RunRecords:
    """Export, pin and compare the stored run of an analysis tab (not its current widgets)."""

    def _records_init(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel, QPushButton

        self.record = None
        self.baseline = None
        self.comparison = None
        self.compare_dialog = None
        self.model_card = QLabel("")
        self.model_card.setWordWrap(True)
        self.model_card.setTextFormat(Qt.TextFormat.RichText)
        self.model_card.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.model_card.setVisible(False)
        self.export_run_button = QPushButton("Export run...")
        self.export_run_button.setToolTip(
            "Write the stored run (its own setup, library parts, measurement, settings, data "
            "and warnings) to a folder that reruns without your library"
        )
        self.pin_button = QPushButton("Pin as baseline")
        self.compare_button = QPushButton("Compare with baseline")
        self.export_run_button.clicked.connect(self.export_run_dialog)
        self.pin_button.clicked.connect(self.pin_baseline)
        self.compare_button.clicked.connect(lambda: self.compare_with_baseline())
        self._update_run_buttons()

    def run_buttons(self) -> list:
        return [self.export_run_button, self.pin_button, self.compare_button]

    def _update_run_buttons(self) -> None:
        has = self.record is not None
        self.export_run_button.setEnabled(has)
        self.pin_button.setEnabled(has)
        self.compare_button.setEnabled(has and self.baseline is not None)

    def set_record(self, record, card=None) -> None:
        self.record = record
        rows = list(card if card is not None else (record.model if record is not None else []))
        self.model_card.setText(card_html(rows))
        self.model_card.setVisible(bool(rows))
        self._update_run_buttons()

    def clear_record(self) -> None:
        self.set_record(None, [])

    def export_run_dialog(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        if self.record is None:
            return
        path = QFileDialog.getExistingDirectory(self, "Export run to folder")
        if path:
            self.export_run(path)

    def export_run(self, path) -> bool:
        from PySide6.QtWidgets import QMessageBox

        from sequoia.runs import write_bundle

        if self.record is None:
            return False
        try:
            write_bundle(self.record, path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot export run", f"{path}\n\n{exc}")
            return False
        note = f"exported run {self.record.run_id} to {path}"
        if self.results_outdated():
            note += " (the run predates later edits; the bundle holds the inputs it was run on)"
        self.status.setText(note)
        return True

    def pin_baseline(self) -> None:
        if self.record is None:
            return
        self.baseline = self.record
        self._update_run_buttons()
        self.status.setText(f"pinned {self.record.run_id} as the baseline")

    def compare_with_baseline(self, show: bool = True):
        from sequoia.runs import compare_records

        if self.record is None or self.baseline is None:
            return None
        self.comparison = compare_records(self.baseline, self.record)
        if show:
            self._show_comparison(self.comparison)
        return self.comparison

    def _show_comparison(self, comparison) -> None:
        from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

        from sequoia.gui.results import _fill, _make_table, _NumItem
        from sequoia.runs import _short

        dialog = QDialog(self)
        dialog.setWindowTitle("Compare with baseline")
        inputs = QLabel(
            "Inputs that differ:\n"
            + (
                "\n".join(comparison.input_differences)
                or "none: same setup, settings, measurement and library parts"
            )
        )
        inputs.setWordWrap(True)
        tab = _make_table(["Output", "Baseline", "Current", "Difference"])
        _fill(
            tab,
            [
                [_NumItem(k), _NumItem(_short(a)), _NumItem(_short(b)), _NumItem(_short(d))]
                for k, a, b, d in comparison.rows
            ],
        )
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"baseline {comparison.baseline_id}  vs  {comparison.current_id}"))
        layout.addWidget(inputs)
        layout.addWidget(tab, 1)
        dialog.resize(760, 480)
        self.compare_dialog = dialog
        dialog.show()
