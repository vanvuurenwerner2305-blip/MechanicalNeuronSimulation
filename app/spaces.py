"""
Top-level window with the two spaces of the software, each a full window of its own in a tab:

  Inputs → pre-activation     - the inputs to the pre-activation chamber (MainWindow): membranes, chambers,
                         sweeps and the neuron equation.
  Pre-activation → activation - the valve that maps the pre-activation (the pressure difference across its
                                membrane) to the activation, the gas pressure of a chosen segment of a squeezed
                                tube (ActivationWindow), saved as a design (*.mad).
  Full neuron                 - both imported and linked (FullNeuronWindow): the design's membrane replaces a
                                neuron membrane; fluid parameters, recording, sweeps; saved as *.mfn.

Each space keeps its own CAD model and project. The inactive space is hidden, so its keyboard
shortcuts (the same keys in both) do not clash.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QTabWidget

from .activation import DESIGN_SUFFIX
from .activation_window import ActivationWindow
from .full_neuron import SUFFIX as FULL_SUFFIX
from .full_neuron_window import FullNeuronWindow
from .main_window import APP_NAME, MainWindow

NEURON_TAB, ACTIVATION_TAB, FULL_TAB = "Inputs → pre-activation", "Pre-activation → activation", "Full neuron"


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1550, 950)
        self.neuron = MainWindow()
        self.activation = ActivationWindow()
        self.full = FullNeuronWindow()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for window, title in ((self.neuron, NEURON_TAB), (self.activation, ACTIVATION_TAB), (self.full, FULL_TAB)):
            window.setWindowFlags(Qt.Widget)   # embedded: menus, docks and toolbars stay inside the tab
            window.windowTitleChanged.connect(self._update_title)
            self.tabs.addTab(window, title)
        font = self.tabs.tabBar().font()   # only this tab bar (a style sheet would reach the inner tabs too)
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        self.tabs.tabBar().setFont(font)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.currentChanged.connect(self._update_title)
        self.setCentralWidget(self.tabs)
        self._update_title()

    @property
    def spaces(self):
        return (self.neuron, self.activation, self.full)

    def current(self):
        return self.tabs.currentWidget()

    def show_space(self, window):
        self.tabs.setCurrentWidget(window)

    def _update_title(self, *_):
        title = self.current().windowTitle()
        self.setWindowTitle(title if title != APP_NAME else APP_NAME)

    def open_path(self, path):
        """Open a STEP file in the current space, a .mns project in the neuron space or a design in the
        activation-function space."""
        lower = str(path).lower()
        if lower.endswith(FULL_SUFFIX):
            self.show_space(self.full)
            self.full.open_project(path)
        elif lower.endswith(DESIGN_SUFFIX):
            self.show_space(self.activation)
            self.activation.open_project(path)
        elif lower.endswith(".mns"):
            self.show_space(self.neuron)
            self.neuron.open_project(path)
        elif self.current() is not self.full:
            self.current().open_step(path)
        else:
            self.show_space(self.neuron)
            self.neuron.open_step(path)

    def closeEvent(self, event):
        for window in self.spaces:
            if window.worker is not None:
                window.worker.cancel()
                window.worker.wait(30000)
            window.viewport.close()
        self.neuron.cad.close()
        super().closeEvent(event)
