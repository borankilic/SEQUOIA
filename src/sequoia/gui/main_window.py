from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import sequoia.components.builtin  # noqa: F401
from sequoia.errors import SequoiaError
from sequoia.gui.canvas import CanvasScene, CanvasView, plan_layout
from sequoia.gui.canvas.items import CELL
from sequoia.gui.document import SetupDocument
from sequoia.gui.inspector import InspectorPanel
from sequoia.gui.palette import Palette
from sequoia.gui.results import ResultsDock, eng, ids_for_issue
from sequoia.gui.simulation import SimulationController, SimulationOutcome
from sequoia.library.store import Library

ORGANIZATION = "SEQUOIA"
APPLICATION = "SEQUOIA"
MAX_RECENT = 8
OVERLAP_PX = 10.0
FILE_FILTER = "Setup files (*.yaml *.yml);;All files (*)"

HINT = (
    "Empty setup\n\n"
    "Drag a part from the palette onto the canvas, or double-click it.\n"
    "Connect ports by dragging from one port to another."
)


def make_settings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


class MainWindow(QMainWindow):
    def __init__(
        self,
        library: Library,
        setup_path: Path | str | None = None,
        settings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.library = library
        self.settings = settings if settings is not None else make_settings()
        self.confirm_discard: Callable[[MainWindow], str] = self._ask_discard
        self.ask_open_path: Callable[[], str | None] = self._dialog_open_path
        self.ask_save_path: Callable[[], str | None] = self._dialog_save_path
        self.ask_library_path: Callable[[], str | None] = self._dialog_library_path
        self.show_error: Callable[[str, str], None] = self._dialog_error
        self._fitted = False
        self._closing = False
        self._bound = False

        self.document = SetupDocument(library)
        self.scene = CanvasScene(self.document)
        self.view = CanvasView(self.scene)
        self.palette = Palette(library)
        self.inspector = InspectorPanel(self.document)
        self.results = ResultsDock()
        self.controller = SimulationController(self.document, self)
        self.results.connect_controller(self.controller)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_status_bar()
        self._wire()
        self._bind_document()
        self._default_state = self.saveState()
        self._default_geometry = self.saveGeometry()
        self._restore_settings()
        self._update_recent_menu()
        self.controller.start()
        if setup_path is not None:
            self.open_path(setup_path)

    def _build_central(self) -> None:
        central = QWidget()
        box = QVBoxLayout(central)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.overlap_bar = QFrame()
        self.overlap_bar.setFrameShape(QFrame.Shape.StyledPanel)
        self.overlap_bar.setAutoFillBackground(True)
        row = QHBoxLayout(self.overlap_bar)
        row.setContentsMargins(8, 3, 4, 3)
        row.addWidget(QLabel("Some parts overlap on the canvas."))
        self.overlap_button = QPushButton("Auto layout")
        self.overlap_button.setFlat(True)
        self.overlap_button.clicked.connect(self._offer_accepted)
        row.addWidget(self.overlap_button)
        row.addStretch(1)
        dismiss = QToolButton()
        dismiss.setText("×")
        dismiss.setAutoRaise(True)
        dismiss.clicked.connect(self.overlap_bar.hide)
        row.addWidget(dismiss)
        self.overlap_bar.hide()
        box.addWidget(self.overlap_bar)
        box.addWidget(self.view, 1)
        self.setCentralWidget(central)
        self.hint = QLabel(HINT, self.view.viewport())
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet("color: palette(placeholder-text); font-size: 14px;")
        self.view.viewport().installEventFilter(self)

    def _dock(self, widget: QWidget, title: str, name: str, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        self.palette_dock = self._dock(
            self.palette, "Parts", "dock_palette", Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.inspector_dock = self._dock(
            self.inspector, "Inspector", "dock_inspector", Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.results_dock = self._dock(
            self.results, "Results", "dock_results", Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.palette.setMinimumWidth(180)
        self.inspector.setMinimumWidth(260)
        self.results.setMinimumHeight(120)
        self.resize(1400, 900)
        self.resizeDocks([self.palette_dock, self.inspector_dock], [280, 340], Qt.Horizontal)
        self.resizeDocks([self.results_dock], [230], Qt.Vertical)

    def _action(self, text, slot, shortcut=None, checkable=False, checked=False) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            if isinstance(shortcut, list):
                action.setShortcuts(shortcut)
            else:
                action.setShortcut(shortcut)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
            action.toggled.connect(slot)
        else:
            action.triggered.connect(lambda _checked=False: slot())
        return action

    def _canvas_action(self, text, slot, shortcut=None) -> QAction:
        action = self._action(text, slot, shortcut)
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.view.addAction(action)
        return action

    def _build_actions(self) -> None:
        S = QKeySequence
        self.act_new = self._action("New", self.new_document, S.StandardKey.New)
        self.act_open = self._action("Open...", self.open_dialog, S.StandardKey.Open)
        self.act_open_library = self._action("Open Library...", self.open_library_dialog)
        self.act_save = self._action("Save", self.save, S.StandardKey.Save)
        self.act_save_as = self._action("Save As...", self.save_as, S.StandardKey.SaveAs)
        self.act_quit = self._action("Quit", self.close, S.StandardKey.Quit)
        self.act_quit.setMenuRole(QAction.MenuRole.QuitRole)

        self.act_undo = self._action("Undo", self.undo, S.StandardKey.Undo)
        self.act_redo = self._action("Redo", self.redo, [S("Ctrl+Shift+Z"), S("Ctrl+Y")])
        self.act_delete = self._canvas_action(
            "Delete", lambda: self.scene.delete_selected(), S("Delete")
        )
        self.act_copy = self._canvas_action(
            "Copy", lambda: self.scene.copy_selection(), S.StandardKey.Copy
        )
        self.act_paste = self._canvas_action(
            "Paste", lambda: self.scene.paste(), S.StandardKey.Paste
        )
        self.act_rotate = self._canvas_action(
            "Rotate", lambda: self.scene.rotate_selected(90.0), "R"
        )
        self.act_rotate_back = self._canvas_action(
            "Rotate counter-clockwise", lambda: self.scene.rotate_selected(-90.0), "Shift+R"
        )
        self.act_flip = self._canvas_action("Flip", lambda: self.scene.flip_selected(), "F")
        self.act_remove_probe = self._action("Remove probe", self.remove_selected_probes)
        self.act_remove_probe.setEnabled(False)
        self.menu_add_probe = QMenu("Add probe", self)
        self.menu_add_probe.setEnabled(False)
        self.menu_add_probe.aboutToShow.connect(self._fill_probe_menu)
        self.menu_add_instance = QMenu("Add instance", self)
        self.menu_add_instance.aboutToShow.connect(self._fill_instance_menu)
        self.act_layout = self._action("Auto layout", self.auto_layout, "Ctrl+L")
        self.act_snap = self._action(
            "Snap to grid", self.view.set_snap, "Ctrl+G", checkable=True, checked=True
        )

        self.act_zoom_in = self._action(
            "Zoom in", lambda: self.view.zoom_by(1.25), [S("Ctrl+="), S("Ctrl++")]
        )
        self.act_zoom_out = self._action("Zoom out", lambda: self.view.zoom_by(0.8), "Ctrl+-")
        self.act_fit = self._action("Fit to content", self.view.fit_to_content, "Ctrl+0")
        self.act_reset_layout = self._action("Reset layout", self.reset_layout)

        self.act_run = self._action("Run now", self.controller.run_now, "Ctrl+Return")
        self.act_autorun = self._action(
            "Auto-run", self.set_auto_run, checkable=True, checked=self.controller.auto_run
        )

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        self.recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addAction(self.act_open_library)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit = bar.addMenu("&Edit")
        for action in (self.act_undo, self.act_redo):
            edit.addAction(action)
        edit.addSeparator()
        for action in (self.act_delete, self.act_copy, self.act_paste):
            edit.addAction(action)
        edit.addSeparator()
        for action in (self.act_rotate, self.act_rotate_back, self.act_flip):
            edit.addAction(action)
        edit.addSeparator()
        edit.addMenu(self.menu_add_probe)
        edit.addAction(self.act_remove_probe)
        edit.addMenu(self.menu_add_instance)
        edit.addSeparator()
        edit.addAction(self.act_layout)
        edit.addAction(self.act_snap)

        view = bar.addMenu("&View")
        for action in (self.act_zoom_in, self.act_zoom_out, self.act_fit):
            view.addAction(action)
        view.addSeparator()
        for dock in (self.palette_dock, self.inspector_dock, self.results_dock):
            view.addAction(dock.toggleViewAction())
        view.addSeparator()
        view.addAction(self.act_reset_layout)

        run = bar.addMenu("&Run")
        run.addAction(self.act_run)
        run.addAction(self.act_autorun)

    def _build_status_bar(self) -> None:
        self.run_label = QLabel("")
        self.run_label.setContentsMargins(8, 0, 8, 0)
        self.statusBar().addPermanentWidget(self.run_label)
        self._refresh_run_label()

    def _wire(self) -> None:
        self.view.status_message.connect(self._show_message)
        self.palette.part_activated.connect(self._add_at_center)
        self.results.issue_activated.connect(self._focus_issue)
        self.controller.finished.connect(self._on_outcome)
        self.controller.failed.connect(self._on_failed)
        self.controller.busy_changed.connect(lambda _busy: self._refresh_run_label())
        self.controller.cancelled.connect(self._refresh_run_label)

    def _bind_document(self) -> None:
        self._bound = True
        doc, scene = self.document, self.scene
        scene.selection_changed.connect(self._on_selection)
        doc.dirty_changed.connect(self._update_title)
        doc.path_changed.connect(self._on_path_changed)
        doc.changed.connect(self._on_document_changed)
        doc.structure_changed.connect(self._update_hint)
        stack = doc.undo_stack
        stack.canUndoChanged.connect(self._update_undo_actions)
        stack.canRedoChanged.connect(self._update_undo_actions)
        stack.undoTextChanged.connect(self._update_undo_actions)
        stack.redoTextChanged.connect(self._update_undo_actions)
        self._update_undo_actions()
        self._update_title()
        self._update_hint()

    def _unbind_document(self) -> None:
        if not self._bound:
            return
        self._bound = False
        doc, scene = self.document, self.scene
        for signal, slot in (
            (scene.selection_changed, self._on_selection),
            (doc.dirty_changed, self._update_title),
            (doc.path_changed, self._on_path_changed),
            (doc.changed, self._on_document_changed),
            (doc.structure_changed, self._update_hint),
            (doc.undo_stack.canUndoChanged, self._update_undo_actions),
            (doc.undo_stack.canRedoChanged, self._update_undo_actions),
            (doc.undo_stack.undoTextChanged, self._update_undo_actions),
            (doc.undo_stack.redoTextChanged, self._update_undo_actions),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def eventFilter(self, obj, event) -> bool:
        if obj is self.view.viewport() and event.type() == QEvent.Type.Resize:
            self.hint.setGeometry(self.view.viewport().rect())
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self.hint.setGeometry(self.view.viewport().rect())
            QTimer.singleShot(0, self.view.fit_to_content)

    def _update_hint(self) -> None:
        self.hint.setVisible(not self.document.setup.instances)

    def _show_message(self, text: str) -> None:
        self.statusBar().showMessage(text, 6000 if text else 0)

    def _update_title(self, *_args) -> None:
        doc = self.document
        label = doc.path.name if doc.path is not None else (doc.setup.name or "untitled")
        self.setWindowTitle(f"{label}[*] - SEQUOIA")
        self.setWindowModified(doc.dirty)

    def _on_path_changed(self) -> None:
        self._update_title()
        if self.document.path is not None:
            self._add_recent(self.document.path)

    def _on_document_changed(self) -> None:
        self._update_title()
        if self.overlap_bar.isVisible() and not self._overlapping():
            self.overlap_bar.hide()

    def _update_undo_actions(self, *_args) -> None:
        try:
            stack = self.document.undo_stack
            can_undo, can_redo = stack.canUndo(), stack.canRedo()
            undo_text, redo_text = stack.undoText(), stack.redoText()
        except RuntimeError:
            return
        self.act_undo.setEnabled(can_undo)
        self.act_redo.setEnabled(can_redo)
        self.act_undo.setText(f"Undo {undo_text}".strip())
        self.act_redo.setText(f"Redo {redo_text}".strip())

    def undo(self) -> None:
        self.document.undo_stack.undo()

    def redo(self) -> None:
        self.document.undo_stack.redo()

    def set_auto_run(self, enabled: bool) -> None:
        self.controller.auto_run = bool(enabled)
        if enabled:
            self.controller.run_now()
        self._refresh_run_label()

    def _item_ids(self) -> set[str]:
        s = self.scene
        return {*s.component_items, *s.link_items, *s.probe_items}

    def _fill_probe_menu(self) -> None:
        self.scene.populate_probe_menu(self.menu_add_probe)

    def _fill_instance_menu(self) -> None:
        x, y = self.view.view_center_cells()
        point = QPointF(self.scene.snap_cells(x) * CELL, self.scene.snap_cells(y) * CELL)
        self.scene.populate_instance_menu(self.menu_add_instance, point)

    def remove_selected_probes(self) -> None:
        probes = [i for i in self.scene.selected_ids() if i in self.scene.probe_items]
        if not probes:
            return
        with self.document.macro("Remove probes"):
            for probe_id in probes:
                self.document.remove_probe(probe_id)

    def _update_selection_actions(self, ids: list[str]) -> None:
        links = [i for i in ids if i in self.scene.link_items]
        probes = [i for i in ids if i in self.scene.probe_items]
        self.menu_add_probe.setEnabled(len(ids) == 1 and len(links) == 1)
        self.act_remove_probe.setEnabled(bool(probes))

    def _on_selection(self, ids: list[str]) -> None:
        self._update_selection_actions(ids)
        if len(ids) == 1:
            item_id = ids[0]
            if item_id in self.scene.component_items:
                self.inspector.show_selection("instance", item_id)
            elif item_id in self.scene.link_items:
                self.inspector.show_selection("link", item_id)
            elif item_id in self.scene.probe_items:
                self.inspector.show_selection("probe", item_id)
            else:
                self.inspector.show_selection(None)
        else:
            self.inspector.show_selection(None)

    def _add_at_center(self, ref: str) -> str | None:
        x, y = self.view.view_center_cells()
        occupied = {tuple(i.position) for i in self.document.setup.instances}
        px, py = self.scene.snap_cells(x), self.scene.snap_cells(y)
        while (px, py) in occupied:
            px, py = px + 2.0, py + 1.0
        return self.scene.add_part(ref, (px, py))

    def _focus_issue(self, subject: str) -> None:
        known = self._item_ids()
        ids = [i for i in ids_for_issue(subject) if i in known]
        if not ids:
            return
        self.scene.select_many(ids)
        for item_id in ids:
            item = self.scene.component_items.get(item_id) or self.scene.link_items.get(item_id)
            if item is not None:
                self.view.ensureVisible(item)
                break

    def apply_highlights(self, outcome: SimulationOutcome | None) -> None:
        self.scene.clear_highlights()
        if outcome is None:
            return
        known = self._item_ids()
        for severity in ("warning", "error"):
            for issue in outcome.issues:
                if issue.severity != severity:
                    continue
                ids = [i for i in ids_for_issue(issue.subject) if i in known]
                self.scene.set_highlight(ids, severity)

    def _on_outcome(self, outcome: SimulationOutcome) -> None:
        self.apply_highlights(outcome)
        self._refresh_run_label()

    def _on_failed(self, _message: str) -> None:
        self._refresh_run_label()

    def _refresh_run_label(self, *_args) -> None:
        ctrl = self.controller
        parts = []
        if ctrl.busy:
            parts.append("Running...")
        elif not ctrl.auto_run:
            parts.append("Auto-run off")
        outcome = ctrl.last_outcome
        if outcome is not None and not ctrl.busy:
            errors = sum(i.severity == "error" for i in outcome.issues)
            warnings = sum(i.severity == "warning" for i in outcome.issues)
            state = "Invalid" if outcome.classical is None else "Done"
            parts.append(
                f"{state} in {eng(outcome.elapsed, 's', 3)} | "
                f"{errors} error{'s' * (errors != 1)}, {warnings} warning{'s' * (warnings != 1)}"
            )
        elif outcome is None and not ctrl.busy and ctrl.auto_run:
            parts.append("Ready")
        self.run_label.setText("  ".join(parts))

    def _overlapping(self) -> bool:
        rects = []
        for item in self.scene.component_items.values():
            rects.append(item.mapRectToScene(item._body_path().boundingRect()))
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                box = a.intersected(b)
                if box.width() > OVERLAP_PX and box.height() > OVERLAP_PX:
                    return True
        return False

    def _check_overlap(self) -> None:
        self.overlap_bar.setVisible(self._overlapping())

    def _offer_accepted(self) -> None:
        self.overlap_bar.hide()
        self.auto_layout()

    def auto_layout(self) -> None:
        if not self.document.setup.instances:
            return
        viewport = self.view.viewport()
        plan = plan_layout(self.document, (float(viewport.width()), float(viewport.height())))
        self.scene.apply_layout(plan.positions, plan.rotations)
        self.overlap_bar.hide()
        self.view.fit_to_content()

    def reset_layout(self) -> None:
        self.restoreState(self._default_state)
        for dock in (self.palette_dock, self.inspector_dock, self.results_dock):
            dock.show()
        self.resizeDocks([self.palette_dock, self.inspector_dock], [280, 340], Qt.Horizontal)
        self.resizeDocks([self.results_dock], [230], Qt.Vertical)

    def _ask_discard(self, _window) -> str:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText("The setup has unsaved changes.")
        save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(save)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        assert clicked is cancel or clicked is None
        return "cancel"

    def _dialog_open_path(self) -> str | None:
        start = str(self.document.path.parent) if self.document.path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open setup", start, FILE_FILTER)
        return path or None

    def _dialog_save_path(self) -> str | None:
        start = (
            str(self.document.path) if self.document.path else f"{self.document.setup.name}.yaml"
        )
        path, _ = QFileDialog.getSaveFileName(self, "Save setup", start, FILE_FILTER)
        return path or None

    def _dialog_library_path(self) -> str | None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose library folder", str(self.library.root)
        )
        return path or None

    def _dialog_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def maybe_save(self) -> bool:
        """True when it is fine to discard the current document."""
        if not self.document.dirty:
            return True
        choice = self.confirm_discard(self)
        if choice == "save":
            return self.save()
        return choice == "discard"

    def save(self) -> bool:
        if self.document.path is None:
            return self.save_as()
        return self._save_to(None)

    def save_as(self) -> bool:
        path = self.ask_save_path()
        if not path:
            return False
        if Path(path).suffix == "":
            path += ".yaml"
        return self._save_to(path)

    def _save_to(self, path) -> bool:
        try:
            self.document.save(path)
        except (SequoiaError, OSError, ValueError) as exc:
            self.show_error("Cannot save", str(exc))
            return False
        self._show_message(f"Saved {self.document.path}")
        return True

    def _install_document(self, document: SetupDocument) -> None:
        new_scene = CanvasScene(document)
        self._unbind_document()
        old_doc, old_scene = self.document, self.scene
        self.controller.stop()
        self.document = document
        self.scene = new_scene
        self.view.setScene(self.scene)
        self.scene.set_snap(self.act_snap.isChecked())
        self.inspector.set_document(document)
        self.controller.document = document
        self.controller.last_outcome = None
        self.controller.clear_cache()
        self.results.clear()
        self._bind_document()
        self.overlap_bar.hide()
        old_scene.deleteLater()
        old_doc.deleteLater()
        self.controller.start()
        self._refresh_run_label()

    def new_document(self) -> bool:
        if not self.maybe_save():
            return False
        self._install_document(SetupDocument(self.library))
        QTimer.singleShot(0, self.view.fit_to_content)
        return True

    def open_dialog(self) -> bool:
        if not self.maybe_save():
            return False
        path = self.ask_open_path()
        return bool(path) and self._load(path)

    def open_path(self, path: Path | str) -> bool:
        if not self.maybe_save():
            return False
        return self._load(path)

    def _load(self, path: Path | str) -> bool:
        try:
            document = SetupDocument.open(path, self.library)
            self._install_document(document)
        except (SequoiaError, OSError, ValueError) as exc:
            self.show_error("Cannot open", f"{path}\n\n{exc}")
            return False
        except Exception as exc:
            self.show_error("Cannot open", f"{path}\n\n{type(exc).__name__}: {exc}")
            return False
        self._add_recent(Path(path))
        self.view.fit_to_content()
        self._check_overlap()
        if self.overlap_bar.isVisible():
            self._show_message("Some parts overlap; Auto layout can spread them out.")
        return True

    def open_library_dialog(self) -> bool:
        path = self.ask_library_path()
        if not path:
            return False
        return self.set_library(Library(path))

    def set_library(self, library: Library) -> bool:
        self.library = library
        self.document.library = library
        self.document.invalidate_components()
        self.palette.library = library
        self.palette.refresh()
        self.scene.rebuild()
        self.inspector.reload()
        self.controller.clear_cache()
        if self.controller.auto_run:
            self.controller.run_now()
        return True

    def recent_files(self) -> list[str]:
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [str(p) for p in (value or [])]

    def _add_recent(self, path: Path) -> None:
        resolved = str(Path(path).resolve())
        files = [resolved] + [p for p in self.recent_files() if p != resolved]
        self.settings.setValue("recent_files", files[:MAX_RECENT])
        self._update_recent_menu()

    def clear_recent(self) -> None:
        self.settings.setValue("recent_files", [])
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        self.recent_menu.clear()
        files = self.recent_files()
        for path in files:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _c=False, p=path: self.open_path(p))
        self.recent_menu.setEnabled(bool(files))
        if files:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction("Clear menu").triggered.connect(self.clear_recent)

    def _restore_settings(self) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        snap = self.settings.value("canvas/snap", True)
        snap = snap if isinstance(snap, bool) else str(snap).lower() != "false"
        self.act_snap.setChecked(snap)
        self.view.set_snap(snap)

    def _save_settings(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("canvas/snap", self.act_snap.isChecked())
        self.settings.sync()

    def closeEvent(self, event) -> None:
        if not self.maybe_save():
            event.ignore()
            return
        self._closing = True
        self._save_settings()
        self._unbind_document()
        self.controller.shutdown()
        event.accept()
