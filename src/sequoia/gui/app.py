import sys

from PySide6.QtWidgets import QApplication

from sequoia.gui.main_window import MainWindow


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SEQUOIA")
    window = MainWindow()
    window.show()
    return app.exec()
