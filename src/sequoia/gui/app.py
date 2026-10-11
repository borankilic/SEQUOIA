import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import sequoia.components.builtin  # noqa: F401
from sequoia.cli import resolve_library
from sequoia.gui.main_window import MainWindow
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
    argv: list[str] | None = None, settings: QSettings | None = None
) -> MainWindow | None:
    args = parse_args(argv)
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
    return MainWindow(Library(root), args.setup, settings=settings)


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = create_window(argv)
    if window is None:
        return 1
    window.show()
    code = app.exec()
    window.deleteLater()
    return code
