"""Start the application: python -m app [model.step | project.mns | design.mad]"""
import faulthandler
import sys
import traceback
from datetime import datetime
from pathlib import Path

from . import __name__ as _  # noqa: F401  (sets QT_API before Qt is imported)
from qtpy.QtCore import QLocale
from qtpy.QtWidgets import QApplication, QMessageBox

from .spaces import AppWindow

LOG_DIR = Path.home() / ".membrane_neuron_simulator"


def _install_error_handling():
    """Without a console, errors would vanish (and PyQt5 aborts on unhandled exceptions in
    slots). Log them to a file and show them instead; hard crashes go to crash.log."""
    LOG_DIR.mkdir(exist_ok=True)
    crash_log = open(LOG_DIR / "crash.log", "a", buffering=1)
    crash_log.write(f"\n--- session {datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
    faulthandler.enable(crash_log)

    def excepthook(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        with open(LOG_DIR / "errors.log", "a") as f:
            f.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} ---\n{text}")
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Membrane Neuron Simulator",
                                 f"An error occurred:\n\n{value}\n\nDetails were written to\n{LOG_DIR / 'errors.log'}")
    sys.excepthook = excepthook
    return crash_log


def main(argv=None):
    argv = sys.argv if argv is None else argv
    _install_error_handling()
    QLocale.setDefault(QLocale.c())  # engineering input: '.' as decimal separator everywhere
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Membrane Neuron Simulator")
    app.setStyle("Fusion")
    window = AppWindow()
    window.show()
    if len(argv) > 1:
        window.open_path(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
