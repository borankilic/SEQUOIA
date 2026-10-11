from PySide6.QtCore import QSettings

from sequoia.gui.main_window import MainWindow


def test_main_window_opens(qtbot, library, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(library, settings=settings)
    qtbot.addWidget(window)
    window.show()
    assert "SEQUOIA" in window.windowTitle()
    assert window.isVisible()
