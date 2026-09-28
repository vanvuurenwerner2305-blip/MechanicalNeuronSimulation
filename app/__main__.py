"""Start the application: python -m app [model.step | project.mns]"""
import sys

from . import __name__ as _  # noqa: F401  (sets QT_API before Qt is imported)
from qtpy.QtCore import QLocale
from qtpy.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QLocale.setDefault(QLocale.c())  # engineering input: '.' as decimal separator everywhere
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Membrane Neuron Simulator")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if len(argv) > 1:
        path = argv[1]
        if path.lower().endswith(".mns"):
            window.open_project(path)
        else:
            window.open_step(path)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
