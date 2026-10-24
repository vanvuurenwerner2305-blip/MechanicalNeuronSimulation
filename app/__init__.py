"""Membrane Neuron Simulator desktop application."""
import os

# pyvistaqt/qtpy would otherwise pick whichever Qt binding it finds first; mixing bindings crashes.
# PyQt5 is the binding that ships with Anaconda; set QT_API to use another one.
os.environ.setdefault("QT_API", "pyqt5")
