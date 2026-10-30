import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import sequoia.components.builtin  # noqa: F401
from sequoia.cli import resolve_library
from sequoia.gui.main_window import MainWindow
from sequoia.gui.start_screen import StartScreen
from sequoia.library.store import Library


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sequoia",
        description="SEQUOIA lab simulator. Use 'sequoia run SETUP' to simulate without the GUI.",
    )
    parser.add_argument("setup", nargs="?", help="setup file to open")
    parser.add_argument("--library", help="library folder (default: ./library or near the setup)")
    return parser.parse_args(argv)


def find_library(explicit: str | None, setup_path: str | None) -> Path | None:
    if explicit:
        root = resolve_library(explicit, setup_path) if setup_path else Path(explicit)
        return root if root.is_dir() else None
    if setup_path:
        directory = Path(setup_path).resolve().parent
        for candidate_dir in (directory, *directory.parents):
            candidate = candidate_dir / "library"
            if candidate.is_dir():
                return candidate
    default = Path("library")
    return default if default.is_dir() else None


def create_window(
    argv: list[str] | None = None,
    settings: QSettings | None = None,
    setup: str | None = None,
    untitled: bool = False,
) -> MainWindow | None:
    args = parse_args(argv)
    if setup is not None:
        args.setup = setup
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("SEQUOIA")
    app.setOrganizationName("SEQUOIA")
    root = find_library(args.library, args.setup)
    if root is None:
        if args.library:
            QMessageBox.critical(None, "SEQUOIA", f"Library folder {args.library!r} not found.")
            return None
        chosen = QFileDialog.getExistingDirectory(None, "Choose the library folder")
        if not chosen:
            return None
        root = Path(chosen)
    window = MainWindow(Library(root), args.setup, settings=settings)
    window.launcher = lambda: open_start_screen(argv, settings)
    if untitled and args.setup:
        make_untitled(window, args.setup)
    return window


def make_untitled(window: MainWindow, template_path: str) -> None:
    window.document.path = None
    window.document.path_changed.emit()
    resolved = str(Path(template_path).resolve())
    window.settings.setValue(
        "recent_files", [p for p in window.recent_files() if Path(p).resolve() != Path(resolved)]
    )
    window._update_recent_menu()


def show_start_screen(
    argv: list[str] | None = None,
    settings: QSettings | None = None,
    templates_root: Path | None = None,
) -> StartScreen:
    QApplication.instance() or QApplication(sys.argv[:1])
    screen = StartScreen(templates_root, settings)
    screen.windows = []

    def launch(setup: str | None = None, untitled: bool = False, task: str | None = None) -> None:
        window = create_window(argv, settings, setup, untitled)
        if window is None:
            return
        if task is not None:
            window.show_task(task)
        screen.windows.append(window)
        window.show()
        screen.close()

    def ask_open() -> None:
        path, _ = QFileDialog.getOpenFileName(
            screen, "Open project", "", "Setup files (*.yaml *.yml);;All files (*)"
        )
        if path:
            launch(path)

    screen.new_requested.connect(launch)
    screen.open_requested.connect(ask_open)
    screen.path_requested.connect(lambda path: launch(path))
    screen.template_requested.connect(lambda path: launch(path, untitled=True))
    screen.recipe_requested.connect(
        lambda path, task: launch(path or None, untitled=bool(path), task=task)
    )
    return screen


_OPEN_SCREENS: list[StartScreen] = []


def open_start_screen(
    argv: list[str] | None = None, settings: QSettings | None = None
) -> StartScreen:
    """Show a start screen; choosing something there opens a new window."""
    screen = show_start_screen(argv, settings)
    _OPEN_SCREENS.append(screen)
    screen.show()
    screen.raise_()
    return screen


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    if parse_args(argv).setup is None:
        screen = show_start_screen(argv)
        screen.show()
        return app.exec()
    window = create_window(argv)
    if window is None:
        return 1
    window.launcher = lambda: open_start_screen(argv)
    window.show()
    code = app.exec()
    window.deleteLater()
    return code
